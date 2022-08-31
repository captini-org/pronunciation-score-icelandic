import glob, random, torch, json, transformers,pickle
from collections import defaultdict
import soundfile as sf
from scipy import signal
import numpy as np
transformers.logging.set_verbosity(transformers.logging.ERROR)

proj_dir = '/home/caitlinr/work/pronunciation-score-icelandic/'
align_dir = proj_dir + 'setup/samromur-mfa/reference-alignment/out/'
current_wav_dir = '/home/caitlinr/scratch/capt/data/captini/audio_correct_names/'
recording_data_file = '/home/caitlinr/scratch/capt/data/captini/recording_data.tsv'
featurizer_model_path = '/home/caitlinr/scratch/dtwnbt/facebook/wav2vec2-base'


reference_feat_dir = proj_dir + '/reference-feats_w2v2-base_layer-6/'

demo_data_file = proj_dir + 'demo_recording_data.tsv'
demo_wav_dir = proj_dir + 'demo-wav/'
make_demo_set = proj_dir + 'setup/scripts-etc/local/copy_demo_audio.sh'


alignments = glob.glob(align_dir+'*/*.json')
alignable_recs = [a.rsplit('/',1)[1][:-5] for a in alignments]
recording_data = [l.split('\t') for l in open(recording_data_file,'r').read().splitlines()]
id2dat = {l[2][:-4] : l for l in recording_data[1:]}


ex_info = defaultdict(list)
for a in alignable_recs:
    a_info = id2dat[a]
    a_ex_id = a_info[3]
    ex_info[a_ex_id].append(a_info)
    
    
# get exercises with at least 11 alignable speakers
ex_ids = set([l[3] for l in recording_data])
selected_ex_ids = []
for ex in sorted(list(ex_ids)):
    recs = ex_info[ex]
    languages = [l[8].lower() for l in recs]
    l1 = languages.count('icelandic')
    if (l1 > 10) and ( (len(languages) - l1) > 10) :
        selected_ex_ids.append(ex)
print('Selected',len(selected_ex_ids),'exercises with enough recordings')

ex_info = {k:v for k,v in ex_info.items() if k in selected_ex_ids}


def w2v2_featurizer(model_path,layer):
    
    model_kwargs = {"num_hidden_layers": layer} # !!
    model = transformers.models.wav2vec2.Wav2Vec2Model.from_pretrained(model_path, **model_kwargs)
    model.eval()
    if torch.cuda.is_available():
        model.cuda()
    
    @torch.no_grad()
    def _featurizer(wav_path):
        wav, sr = sf.read(wav_path, dtype=np.float32)
        if len(wav.shape) == 2:
            wav = wav.mean(1)
        if sr != 16_000:
            new_length = int(wav.shape[0] / sr * 16_000)
            wav = signal.resample(wav, new_length)
            
        wav = torch.from_numpy(wav).unsqueeze(0)
        if torch.cuda.is_available():
            wav = wav.cuda()
        hidden_state = model(wav).last_hidden_state.squeeze(0).cpu().numpy()
        return hidden_state

    return _featurizer

# convert timestamps (seconds) to feature frame numbers
# w2v2 step size is about 20ms, or 50 frames per second
def cff(sc):
	return int(float(sc)*50)

def load_word_timestamps(alignment):
    alignment = json.load(alignment)
    words = alignment['tiers']['words']['entries']
    words = [ [cff(s),cff(e),l] for s,e,l in words if l]
    return words # a list of lists; each sublist is startFrame, endFrame, word.

def word_feats_func(refs,feats_func):
    exercise_words = [l[4] for l in refs]
    assert len(set(exercise_words)) == 1
    exercise_words = exercise_words[0].split(' ')
    exercise_words = zip([i for i in range(len(exercise_words))],exercise_words)
    exercise_words = [f'{a:03d}__{b}' for a,b in exercise_words]
    
    
    words_dict = defaultdict(dict)
    for rec in refs: #
        spk = rec[1]
        fid = rec[0]
        align_path = f'{align_dir}{spk}/{spk}-{fid}.json'
        
        with open(align_path,'r') as alignment:
            word_aligns = load_word_timestamps(alignment)
        assert ' '.join([l for s,e,l in word_aligns]) == ' '.join([w[5:] for w in exercise_words])
        
        wav_path = f'{current_wav_dir}{spk}/{spk}-{fid}.wav'
        feats = feats_func(wav_path)
        
        for i in range(len(word_aligns)):
            assert exercise_words[i][5:] == word_aligns[i][2]
            wordfeats = feats[word_aligns[i][0]:word_aligns[i][1]]
            assert wordfeats.shape[0] > 0
            words_dict[exercise_words[i]][spk] = wordfeats
    
    def undefault(d):
        # words_dict is a dict of { Word : { Speaker1 : Feat, Sp2 : Feat }, Word2 : {S1 : F, S2: F}, }   
        return {k : {k2: v2 for k2,v2 in v.items() } for k,v in d.items() }
        
    return undefault(words_dict)


w_featurizer = w2v2_featurizer(model_path = featurizer_model_path, layer = 6)

reference_feats = {}
demo_data = '\t'.join(recording_data[0])+'\n'
demo_copy = ''

for ex, recs in ex_info.items():
    l1 = [l for l in recs if l[8].lower() == 'icelandic']
    l2 = [l for l in recs if l not in l1]
    
    l1_ref = random.sample(l1,10)
    l2_ref = random.sample(l2,10)
    
    
    # 1. get w2v2 features for each word in the reference sets and save them
    reference_feats = {'L1': word_feats_func(l1_ref,w_featurizer),'L2': word_feats_func(l2_ref,w_featurizer)}
    feat_file_path = f'{reference_feat_dir}exercise_{ex}.pickle'
    
    
    with open(feat_file_path, 'wb') as handle:
        handle.write(pickle.dumps(reference_feats))
    # example: f = open('demo/exercise_####','r')
    # x = json.load(f)
    # list of different speakers feature-blocks for a word: x['L2']['002__langt'].values()

    
    # 2. write lines to copy demo-set audio files
    # and make demo-set recording data file
    test = [l for l in l2 if l not in l2_ref] + random.sample([l for l in l1 if l not in l1_ref],1)
    for l in test:
        demo_data+='\t'.join(l)+'\n'
        demo_copy += f'mkdir -p {demo_wav_dir}{l[1]}\n'
        demo_copy+= f'cp {current_wav_dir}{l[1]}/{l[2]} {demo_wav_dir}{l[1]}/{l[2]}\n'
 
     
with open(demo_data_file,'w') as handle:
    handle.write(demo_data)
with open(make_demo_set,'w') as handle:
    handle.write(demo_copy)


