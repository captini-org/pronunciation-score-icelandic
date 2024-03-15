import sys, unicodedata, json, os, string, pickle
import torch, transformers, pydub, librosa
import soundfile as sf
import numpy as np
from collections import defaultdict
from scipy import signal
from os.path import basename, splitext, exists
sys.path.append('../pronunciation-score-icelandic')
from captinialign import makeAlign
transformers.logging.set_verbosity(transformers.logging.ERROR)




# convert defaultdicts into regular picklable dicts
def d2d(dic):
    if isinstance(dic, dict):
        dic = {k: d2d(v) for k, v in dic.items()}
    return dic


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


# access fields from expected metadata format
def rec_info(file_metadata, corpus_dir):
    speaker_id = file_metadata[1]
    file_id = file_metadata[0]
    wav_path = f'{corpus_dir}{speaker_id}/{speaker_id}-{file_id}.wav'
    aln_path = f'{corpus_dir}{speaker_id}/{speaker_id}-{file_id}.json'
    normed_text = snorm(file_metadata[3])
    return wav_path, aln_path, normed_text


# load word and phone alignments from Montreal Forced Aligner .json format
# retime phones per word
def load_timestamps(alignment):
    
    # return phones from the start to end time of one word
    def getwps(start,end,phones):
        return [[round(s-start,2),round(e-start,2),p] for s,e,p in phones if (s>=start) & (e<=end)]

    alignment = json.load(alignment)
    words = alignment['tiers']['words']['entries']
    phones = alignment['tiers']['phones']['entries']
    words = [ [cff(s),cff(e),l] for s,e,l in words if l]
    phones = [ [cff(s),cff(e),l] for s,e,l in phones if l]
    wplist = [(w,getwps(s,e,phones)) for s,e,w in words]
    return words, wplist


# validate the list of recordings -
# nonsilent, alignable, every word has nonzero duration.
def validate(rec_list,corpus_dir):
    validated = []
    
    for rec in rec_list:
        wav_path, aln_path, _ = rec_info(rec,corpus_dir)
        
        if os.path.exists(aln_path):
            with open(aln_path,'r') as alignment:
                word_aligns, _ = load_timestamps(alignment)
            if all([e>s for s,e,l in word_aligns]):
                wave = pydub.AudioSegment.from_wav(wav_path)
                if pydub.silence.detect_silence(wave):
                    validated.append(rec)
    return validated


# get recordings that have valid speech features already
def revalidate(rec_list,feats_dict):
    validated = []
    spks = []

    for w,ws in feats_dict.items():
        spks += list(ws.keys())
    spks = set(spks)

    for rec in rec_list:
        if rec[1] in spks:
            validated.append(rec)
    return validated






# force align recordings
# with pre existing alignment model and dictionary.
# requires system install of kaldi,
#  and excellent dictionary coverage of the corpus.
# implemented only to replace data temporarily lost on smallvoice,
# in normal cases use MFA to train & apply an alignment model.
def align_corpus(corpus_meta_file, corpus_dir):
    with open(corpus_meta_file,'r') as handle:
        rec_list = handle.read().splitlines()
    rec_list = [l.split('\t') for l in rec_list[1:]]
    
    for rec in rec_list:
        wav_path, aln_path, normed_text = rec_info(rec,corpus_dir)

        if not exists(aln_path):
            wav_duration = librosa.get_duration(path=wav_path)
            word_aligns, phone_aligns = makeAlign(normed_text, wav_path, wav_duration, splitext(basename(wav_path))[0])

            for i in range(len(word_aligns)):
                offset = word_aligns[i][1]
                phone_aligns[word_aligns[i][0]] = [(l,round(s+offset,2),round(e+offset,2)) for l,s,e in phone_aligns[word_aligns[i][0]]]
                
            if word_aligns:
                # close enough to mfa json format
                w = {"entries" : [ [s, e, l.split('__')[-1]] for l,s,e in word_aligns ] } 
                p = {"entries" : [ [s, e, l.split('__')[-1]] for l,s,e in [phone for word in phone_aligns.values() for phone in word] ] } 
                j = {"tiers": {"words": w, "phones": p}}
                with open(aln_path, 'w') as handle:
                    json.dump(j,handle)
            else:
                print(f"Couldn't align {basename(wav_path)} ({wav_duration})")



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





# collect word pronunciations from a task
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
        wav_path, align_path, _ = rec_info(rec,corpus_dir)
        spk = rec[1]
        
        with open(align_path,'r') as alignment:
            word_aligns, _ = load_timestamps(alignment)
        assert ' '.join([l for s,e,l in word_aligns]) == normed_text
        
        feats = featurizer(wav_path)
        
        for i in range(len(word_aligns)):
            assert task_words[i][5:] == word_aligns[i][2]
            wordfeats = feats[word_aligns[i][0]:word_aligns[i][1]]
            assert wordfeats.shape[0] > 0
            words_dict[task_words[i]][spk] = wordfeats
    
    # words_dict is a dict of { Word : { Speaker1 : Feat, Sp2 : Feat }, Word2 : {S1 : F, S2: F}, }           
    return d2d(words_dict)



# collect phone pronunciations from one task
def add_task_phones(normed_text, rec_list, corpus_dir, task_feats, lang, speaker_split, phones_dict):
    task_words = normed_text.split(' ')
    task_words = zip([i for i in range(len(task_words))],task_words)
    task_words = [f'{a:03d}__{b}' for a,b in task_words]

    for rec in rec_list:
        spk = rec[1]
        fid = rec[0]
        align_path = f'{corpus_dir}{spk}/{spk}-{fid}.json'
        
        with open(align_path,'r') as alignment:
            _, phone_aligns = load_timestamps(alignment)
        
        assert ' '.join([w for w,pinfo in phone_aligns]) == normed_text
        

        for i in range(len(phone_aligns)):
            assert task_words[i][5:] == phone_aligns[i][0]
            word_feats = task_feats[lang][task_words[i]][spk]
            word_phone_aligns = phone_aligns[i][1]

            for s,e,p in word_phone_aligns:
                #if e-s > 1:
                if (e-s > 1) and (int(spk[-1]) in speaker_split):
                    phones_dict[lang][p].append(word_feats[s:e])

    return phones_dict





# create task pronunciation model for a task
# output: pickle file keyed to task_id
def model_one_task(task_id, task_text, corpus_dir, corpus_meta_file, featurizer, save_dir):

    norm_text = snorm(task_text)
    with open(corpus_meta_file,'r') as handle:
        cmeta = handle.read().splitlines()
    cmeta = [l.split('\t') for l in cmeta[1:]]

    # get the reference recordings for this task
    task_recs_meta = [l for l in cmeta if snorm(l[3]) == norm_text]
    l1_meta = validate([l for l in task_recs_meta if l[6].lower() in ['islenska','icelandic']],corpus_dir)
    l2_meta = validate([l for l in task_recs_meta if l[6].lower() not in ['islenska','icelandic']],corpus_dir)

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

    task_status = ''
    if (len(l1_meta) >= 20) and (len(l2_meta) >= 20):
        task_status = 'complete'
    elif (len(l1_meta) >= 10) and (len(l2_meta) >= 10):
        task_status = 'basic'
    elif (len(l1_meta) >= 1) and (len(l2_meta) >= 1):
        task_status = 'not_valid'
    elif (len(l1_meta) == 0) or (len(l2_meta) == 0):
        task_status = 'absent'

    print(f'{task_id}\t{task_text}\t{len(l1_meta)}\t{len(l2_meta)}\t{task_status}')
    return l1_meta+l2_meta


# add phones from one task to pronunciation models
# output: updated version of input phones dict
def remodel_one_task(task_id, task_text, corpus_dir, corpus_meta_file, old_feats_dir, speaker_split, phones_dict):

    norm_text = snorm(task_text)

    with open(corpus_meta_file,'r') as handle:
        cmeta = handle.read().splitlines()
    cmeta = [l.split('\t') for l in cmeta[1:]]

    speech_feats_path = f'{old_feats_dir}task_{task_id}.pickle'
    try:
        with open(speech_feats_path, 'rb') as handle:
            task_feats = pickle.load(handle)

        # get the reference recordings for this task
        task_recs_meta = [l for l in cmeta if snorm(l[3]) == norm_text]
        l1_meta = revalidate([l for l in task_recs_meta if l[6].lower() in ['islenska','icelandic']],task_feats['L1'])
        l2_meta = revalidate([l for l in task_recs_meta if l[6].lower() not in ['islenska','icelandic']],task_feats['L2'])

        phones_dict1 = add_task_phones(norm_text, l1_meta, corpus_dir, task_feats, 'L1', speaker_split, phones_dict)
        phones_dict12 = add_task_phones(norm_text, l2_meta, corpus_dir, task_feats, 'L2', speaker_split, phones_dict1)

        print(f'Used task {task_id} "{task_text}"')
        return phones_dict12


    except:
        print(f'did NOT use phones from task {task_id} "{task_text}" (missing data)')
        return phones_dict



def model_all_tasks(task_list, corpus_dir, corpus_meta_file, featurizer_path, featurizer_layer, save_dir, valid_meta_file):

    featurizer = w2v2_featurizer(model_path = featurizer_path, layer = featurizer_layer)
    
    with open(task_list,'r') as handle:
        all_tasks = [l.split('\t') for l in handle.read().splitlines()]

    with open(corpus_meta_file,'r') as handle:
        hed = [l.split('\t') for l in handle.read().splitlines()][0]

    final_meta = [hed]
    for task_id, task_text in all_tasks:
        final_meta += model_one_task(task_id, task_text, corpus_dir, corpus_meta_file, featurizer, save_dir)

    with open(valid_meta_file,'w') as handle:
        handle.write('\n'.join(['\t'.join(l) for l in final_meta]))

        
# monophone version
def model_all_monophones(task_list, corpus_dir, corpus_meta_file, old_feats_dir, save_path, speaker_split):
    
    with open(task_list,'r') as handle:
        all_tasks = [l.split('\t') for l in handle.read().splitlines()]

    phones_dict = {'L1':defaultdict(list),'L2': defaultdict(list)}
    
    for task_id, task_text in all_tasks:
        phones_dict = remodel_one_task(task_id, task_text, corpus_dir, corpus_meta_file, old_feats_dir, speaker_split, phones_dict)
    for ph in sorted(list(phones_dict['L1'].keys())):
        print(ph,'L1', len(phones_dict['L1'][ph]),sep='\t')
    for ph in sorted(list(phones_dict['L2'].keys())):
        print(ph,'L2',len(phones_dict['L2'][ph]),sep='\t')
        
    with open(save_path, 'wb') as handle:
        handle.write(pickle.dumps(phones_dict))


        

def main():

    corpus_meta_file = '/Users/cati/corpora/captisr/filtered_capti_metadata.tsv'
    corpus_dir = '/Users/cati/corpora/captisr/audio_correct_names/'
    
    #align_corpus(corpus_meta_file, corpus_dir)
    
    featurizer_path = '/Users/cati/corpora/models/LVL/wav2vec2-large-xlsr-53-icelandic-ep10-1000h/'
    layer = 8
    validated_meta_file = './setup/captini_metadata.tsv'
    task_db = './setup/task-text.txt'
    task_save_dir = './setup/task_models_w2v2-IS-1000h/'

    #model_all_tasks(task_db, corpus_dir, corpus_meta_file, featurizer_path, layer, task_save_dir, validated_meta_file)

    
    #speaker_split = list(range(0,10)) # all speakers
    speaker_split = list(range(0,4)) # can split by final digit.
    # most even 3 way split is 0123 456 789.
    monophone_save_path = './models/monophones/w2v2-IS-1000h_SPLIT3.pickle'


    model_all_monophones(task_db, corpus_dir, validated_meta_file, task_save_dir, monophone_save_path, speaker_split)

    


if __name__ == "__main__":
    main()
