import torch, transformers, string, pydub, os, json, pickle, unicodedata
import soundfile as sf
import numpy as np
from collections import defaultdict
from scipy import signal
transformers.logging.set_verbosity(transformers.logging.ERROR)



# convert timestamps (seconds) to wav2vec2 feature frame numbers
# w2v2 step size is 20ms, or 50 frames per second
def cff(sc):
	return int(float(sc)*50)


# normalise captini task text for alignment
def snorm(s):
    s = s.replace('400 (fjögur hundruð)','fjögur hundruð').replace(\
                '16. (sextándi)','sextándi').replace('Nýja-Sjálands','nýjasjálands')
    s = ''.join([c.lower() for c in s if not unicodedata.category(c).startswith("P") ])
    while '  ' in s:
        s = s.replace('  ', ' ')
    return s
    #return s.lower().translate(str.maketrans('', '', string.punctuation))


# convert defaultdicts into regular picklable dicts
def d2d(dic):
    if isinstance(dic, dict):
        dic = {k: d2d(v) for k, v in dic.items()}
    return dic


# featurize speech to wav2vec2 embedding.
# this selected model and layer MUST be the same as the featurizer for
# users' speech scored at runtime.
# if you wish to change it, these pronunciation models
# must be re-built with the new featurizer.
def w2v2_featurizer(model_path,layer):
    model_kwargs = {"num_hidden_layers": layer} 
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


# load word alignments from Montreal Forced Aligner .json format
def load_word_timestamps(alignment):
    alignment = json.load(alignment)
    words = alignment['tiers']['words']['entries']
    words = [ [cff(s),cff(e),l] for s,e,l in words if l]
    return words # a list of lists; each sublist is startFrame, endFrame, word.


# input: normalised task text,
# metadata for a set of reference recordings,
#   columns: Recording, Speaker, Text, Age, Gender, Native_Language
# corpus directory to the recordings and their alignments,
# featuriser function
def word_feats_func(normed_text,rec_list,corpus_dir,featurizer):
    task_words = normed_text.split(' ')
    task_words = zip([i for i in range(len(task_words))],task_words)
    task_words = [f'{a:03d}__{b}' for a,b in task_words]
    
    words_dict = defaultdict(dict)
    
    for rec in rec_list:
        spk = rec[1]
        fid = rec[0]
        align_path = f'{corpus_dir}{spk}/{spk}-{fid}.json'
        wav_path = f'{corpus_dir}{spk}/{spk}-{fid}.wav'
        
        with open(align_path,'r') as alignment:
            word_aligns = load_word_timestamps(alignment)
        #assert ' '.join([l for s,e,l in word_aligns]) == normed_text
        try:
            assert ' '.join([l for s,e,l in word_aligns]) == normed_text
        except:
            print('***********')
            print('normed text', normed_text)
            print(' '.join([l for s,e,l in word_aligns ] ))
            print(word_aligns)
        
        feats = featurizer(wav_path)
        
        for i in range(len(word_aligns)):
            assert task_words[i][5:] == word_aligns[i][2]
            wordfeats = feats[word_aligns[i][0]:word_aligns[i][1]]
            assert wordfeats.shape[0] > 0
            words_dict[task_words[i]][spk] = wordfeats
    
    # words_dict is a dict of { Word : { Speaker1 : Feat, Sp2 : Feat }, Word2 : {S1 : F, S2: F}, }           
    return d2d(words_dict)


# validate the list of recordings -
# does the audio file exist & is not empty,
# does the alignment file exist,
# does every word have nonzero alignment duration.
def validate(rec_list,corpus_dir):
    validated = []
    
    for rec in rec_list:
        spk = rec[1]
        fid = rec[0]
        align_path = f'{corpus_dir}{spk}/{spk}-{fid}.json'
        wav_path = f'{corpus_dir}{spk}/{spk}-{fid}.wav'

        # keep the rec in rec_list only if pass all checks
        if os.path.exists(align_path):
            with open(align_path,'r') as alignment:
                word_aligns = load_word_timestamps(alignment)
            if all([e>s for s,e,l in word_aligns]):
                wave = pydub.AudioSegment.from_wav(wav_path)
                if len(wave) >100: # duration over 0.1 seconds
                    if pydub.silence.detect_silence(wave):
                        validated.append(rec)
    return validated



# create pronunciation model for a task
# inputs:
# task ID number; task text;
# path to corpus containing speech files and MFA alignments
# metadata file for corpus
#   columns: Recording, Speaker, Text, Age, Gender, Native_Language
# speech featuriser
# directory to save
# output: pickle file keyed to task_id
def model_one_task(task_id, task_text, corpus_dir, corpus_meta_file, featurizer, save_dir):

    norm_text = snorm(task_text)

    with open(corpus_meta_file,'r') as handle:
        cmeta = handle.read().splitlines()
    cmeta = [l.split('\t') for l in cmeta[1:]]

    # get the reference recordings for this task
    task_recs_meta = [l for l in cmeta if snorm(l[2]) == norm_text]
    l1_meta = validate([l for l in task_recs_meta if l[5].lower() in ['islenska','icelandic']],corpus_dir)
    l2_meta = validate([l for l in task_recs_meta if l[5].lower() not in ['islenska','icelandic']],corpus_dir)

    if (len(l1_meta) > 0) and (len(l2_meta) > 0):
        # get w2v2 features for each word in the reference sets and save them
        reference_feats = {'L1': word_feats_func(norm_text,l1_meta,corpus_dir,featurizer),\
                               'L2': word_feats_func(norm_text,l2_meta,corpus_dir,featurizer)}
        pmodel_file_path = f'{save_dir}task_{task_id}.pickle'
        with open(pmodel_file_path, 'wb') as handle:
            handle.write(pickle.dumps(reference_feats))
        # example: f = open('scoring_models/task_####','r')
        # x = json.load(f)
        # list of different speakers feature-blocks for a word: x['L2']['002__langt'].values()
        print(f'Created scoring model for task {task_id} "{task_text}"')
    else:
        print(f'COULD NOT create scoring model for task {task_id} "{task_text}" (missing data)')


#TODO:
# quick-update existing scoring models:
# check for new speakers in corpus_dir and add them to existing task pickles



def model_all_tasks(task_list, corpus_dir, corpus_meta_file, featurizer_path, featurizer_layer, save_dir):

    featurizer = w2v2_featurizer(model_path = featurizer_path, layer = featurizer_layer)
    
    with open(task_list,'r') as handle:
        all_tasks = [l.split('\t') for l in handle.read().splitlines()]
        
    for task_id, task_text in all_tasks:
        model_one_task(task_id, task_text, corpus_dir, corpus_meta_file, featurizer, save_dir)
    


def main():
    corpus_dir = '/home/caitlinr/work/cc/captini_new/audio_correct_names/'

    featurizer_path = '/home/caitlinr/work/models/LVL/wav2vec2-large-xlsr-53-icelandic-ep10-1000h/'
    layer = 8 # or 7
        
    corpus_meta_file = './captini_metadata.tsv'
    task_db = './task-text.txt'
    
    save_dir = '../task_models_w2v2-IS-1000h/'

    model_all_tasks(task_db, corpus_dir, corpus_meta_file, featurizer_path, layer, save_dir)



if __name__ == "__main__":
    main()
