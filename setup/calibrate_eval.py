import glob, pickle, random, unicodedata, json, string, os
import numpy as np
from dtw import dtw

def snorm(s):
    s = s.replace('400 (fjögur hundruð)','fjögur hundruð').replace(\
                '16. (sextándi)','sextándi').replace('Nýja-Sjálands','nýjasjálands')
    s = ''.join([c.lower() for c in s if not unicodedata.category(c).startswith("P") ])
    while '  ' in s:
        s = s.replace('  ', ' ')
    return s

# convert timestamps (seconds) to wav2vec2 feature frame numbers
# w2v2 step size is 20ms, or 50 frames per second
def cff(sc):
	return int(float(sc)*50)


# load phone alignments from Montreal Forced Aligner .json format
# retime per word
def load_timestamps(alignment,counted=True):
    
    # return phones from the start to end time of one word
    def getwps(start,end,phones,counted):
        if counted:
            return [[round(p[0]-start,2),round(p[1]-start,2),f'{i:03d}__{p[2]}']
                    for i,p in enumerate(phones) if (p[0]>=start) & (p[1]<=end)]
        else:
            return [[round(s-start,2),round(e-start,2),p] for s,e,p in phones if (s>=start) & (e<=end)]


    alignment = json.load(alignment)
    words = alignment['tiers']['words']['entries']
    phones = alignment['tiers']['phones']['entries']
    words = [ [cff(s),cff(e),l] for s,e,l in words if l]
    phones = [ [cff(s),cff(e),l] for s,e,l in phones if l]
    wpdict = {f'{i:03d}__{wd[2]}':getwps(wd[0],wd[1],phones,counted) for i,wd in enumerate(words)}
    return wpdict

# read an existing monophones model
def read_set(fpath,limit_per_phone = 300, random_seed=3):
    with open(fpath,'rb') as handle:
        monophone_db = pickle.load(handle)
        
    if limit_per_phone:
        for lang, langdict in monophone_db.items():
            for phone, exemplars in langdict.items():
                random.Random(random_seed).shuffle(exemplars)
                monophone_db[lang][phone] = [e for e in exemplars if e.shape[0]>1][:limit_per_phone]
    return monophone_db

                    
def ref_dev_test_splits(task_models_dir, split_save_path, n_L1 = 20, m_L2 = 10):

    # task path -> task ID
    def _t_id(task_path):
        return int(task_path.split('_')[-1].split('.')[0])
    
    # get all speakers - some speakers can miss a word
    def _task_spks(task_dict):
        spks = []
        for w,ws in task_dict.items():
            spks += list(ws.keys())
        return list(set(spks))

    # 3 way split of speakerlist for cross validation
    # ideally at least 30 speakers
    def _split(sklist):
        random.shuffle(sklist)
        ref = sklist[:10]
        div = random.randint( (len(sklist)-10) // 2, (len(sklist)-9) // 2)
        aset = sklist[10:div+10]
        bset = sklist[div+10:]
        return {"REF": ref, "setA": aset, "setB": bset}

    # 10 ref, 10 dev, rest if any test
    # suggest n_L1 at least 25
    def _calibration_split_L1(sklist):
        random.shuffle(sklist)
        ref = sklist[:10]
        aset = sklist[10:20]
        bset = sklist[20:]
        return {"REF": ref, "DEV": aset, "TEST": bset}

    # 10 ref, rest if any test
    # suggest n_L2 at least 15
    def _calibration_split_L2(sklist):
        random.shuffle(sklist)
        ref = sklist[:10]
        bset = sklist[10:]
        return {"REF": ref, "TEST": bset}

    # presumably best performance
    # but impossible to evaluate
    def _final_split_L1(sklist):
        random.shuffle(sklist)
        div = random.randint( len(sklist) // 2, (len(sklist)+1) // 2)
        ref = sklist[:div]
        aset = sklist[div:]
        return {"REF": ref, "DEV": aset, "TEST": []}
    
    def _final_split_L2(sklist):
        return {"REF": sklist, "TEST": []}


    task_models = sorted(list(glob.glob(task_models_dir+'task_*.pickle')), key= _t_id)
    
    splits_dict = {}
    for model_path in task_models:
        with open(model_path, 'rb') as handle:
            task_feats = pickle.load(handle)
        l1_spk, l2_spk = _task_spks(task_feats['L1']), _task_spks(task_feats['L2'])
        
        if (len(l1_spk) >= n_L1) and (len(l2_spk) >= m_L2):
            task_id = _t_id(model_path)
            splits_dict[task_id] = {"L1": _final_split_L1(l1_spk), "L2": _final_split_L2(l2_spk)}

    with open(split_save_path,'w') as handle:
        json.dump(splits_dict,handle)
        
    return splits_dict



def clean(id):
    return id.split('__',1)[1]


def run_task_set(lang,split,split_key,feats_path,corpus_meta,corpus_dir,new_feats_path=None):

    def dtw_function (test_word,references_words):
        references_words = [v for v in references_words]
        def _dtw(a_feats,b_feats):
            if a_feats.shape[0] < 2 or b_feats.shape[0] < 2:
                return np.nan
            return dtw(a_feats,b_feats,keep_internals=True)
        return [_dtw(test_word,ref_word) for ref_word in references_words]

    def prepare_score(l1,l2):
        if np.isnan(l1) or np.isnan(l2):
            return("SHORT") # things were too short to score, probably
        else:
            return (l2-l1)/(l2+l1)     
        
    # average dtw cost per word, relative to reference set
    def get_word_avg(dtw_obj_list):
    # the only possibly float here is numpy.nan
        dtw_obj_list = [d for d in dtw_obj_list if not isinstance(d,float)]
        if len(dtw_obj_list) == 0:
            return(np.nan)
        else:
            return np.nanmean([d.normalizedDistance for d in dtw_obj_list])

    # average cost per phone relative to reference set
    def get_phone_avg(dtw_obj_list,word_phone_aligns):
        def _get_pcost(pstart,pend,dtw_obj):
            st = np.where(dtw_obj.index1==pstart)[0][0]
            en = np.where(dtw_obj.index1==min(max(dtw_obj.index1),max(pend-1,0)))
            en = en[0][-1]+1 #add 1 for slicing
            phone_tokens = zip(dtw_obj.index1[st:en],dtw_obj.index2[st:en])
            return np.mean([dtw_obj.localCostMatrix[tx,rx] for tx,rx in phone_tokens])
                
        dtw_obj_list = [d for d in dtw_obj_list if not isinstance(d,float)]
        if len(dtw_obj_list) == 0:
            return { ph : np.nan for s,e,ph in word_phone_aligns}

        pcosts = {ph : (np.nanmean([_get_pcost(s,e,d) for d in dtw_obj_list]) if (e-s > 1) else np.nan) for s,e,ph in word_phone_aligns}
        return pcosts
        

    # load the taskspeech
    with open(feats_path,'rb') as handle:
        feats = pickle.load(handle)

    test_spks = split_key[lang][split]
    l1_ref_spks = split_key['L1']['REF']
    l2_ref_spks = split_key['L2']['REF']

    if new_feats_path:

        new_refset = {"L1": {word : {sk: feats for sk, feats in sfeats.items() if sk in l1_ref_spks} \
            for word, sfeats in feats['L1'].items()}, \
            "L2": {word : {sk: feats for sk, feats in sfeats.items() if sk in l2_ref_spks} \
            for word, sfeats in feats['L2'].items()}}
            
        with open(new_feats_path,'wb') as handle:
            handle.write(pickle.dumps(new_refset))
        
            
    taskwords = sorted(list(feats["L1"].keys()))
    tasktext =  snorm(' '.join([w.split('__')[1] for w in taskwords]))

    test_meta = [l for l in corpus_meta if snorm(l[3]) == tasktext and l[1] in test_spks]
    assert len(test_meta) == len(test_spks)

    
    pscores = []
    wscores = []
    sscores = []

    for test_rec in test_meta:
        test_spk = test_rec[1]
        aln_path = f'{corpus_dir}{test_spk}/{test_spk}-{test_rec[0]}.json'

        with open(aln_path,'r') as handle:
            phone_aligns = load_timestamps(handle)

        l1_word_dtws = { wd :
            dtw_function(feats[lang][wd][test_spk],[v for k,v in feats['L1'][wd].items() if k in l1_ref_spks])
                for wd in taskwords }
        l1_word_avgs = {wd: get_word_avg(l1_word_dtws[wd]) for wd in taskwords}
        l1_phone_avgs = {wd : get_phone_avg(l1_word_dtws[wd], phone_aligns[wd]) for wd in taskwords}

        l2_word_dtws = { wd :
            dtw_function(feats[lang][wd][test_spk],[v for k,v in feats['L2'][wd].items() if k in l2_ref_spks])
                for wd in taskwords }
        l2_word_avgs = {wd: get_word_avg(l2_word_dtws[wd]) for wd in taskwords}
        l2_phone_avgs = {wd : get_phone_avg(l2_word_dtws[wd], phone_aligns[wd]) for wd in taskwords}

        
        final_word_scores = [(clean(w_id), prepare_score(l1_word_avgs[w_id],l2_word_avgs[w_id]))
            for w_id in taskwords]

        final_phone_scores = [x for xs in [
             [(clean(p_id), 
                prepare_score(l1_phone_avgs[w_id][p_id], l2_phone_avgs[w_id][p_id])) 
                for s,e,p_id
                in phone_aligns[w_id]]
            for w_id in taskwords] for x in xs ]

        sent_score = np.nanmean([sc for wd,sc in final_word_scores if not isinstance(sc,str)])

        pscores += [(ph, sc) for ph,sc in final_phone_scores if not isinstance(sc,str)]
        wscores += [(wd,sc) for wd,sc in final_word_scores if not isinstance(sc,str)]
        sscores += [sent_score]

    return pscores, wscores, sscores



# best results with global threshold for words
# and task threshold for phones (which ignores phone identity)
def calibrate_task(l1dev,l1test,l2test,key_save_path,log_path):

    words_l1_dev = [sc for tasks in list(l1dev['W'].values()) for sc in tasks]
    global_word_th = np.percentile([s for w,s in words_l1_dev],1)

    # initialise scoring key with global threshold for words
    task_key = {task: {"word": global_word_th} for task in l1dev['W'].keys()}

    # for eval
    p1test, p2test = 0, 0

    hed = f'Task\tL1_word\tL1_phone\tL2_word\tL2_phone\n'
    log_info=hed
    for task in sorted(list(l1dev['W'].keys()),key=int):
        print(task)

        task_phone_th = np.percentile([sc for p,sc in l1dev['P'][task] ], 1)
        task_key[task]["phone"] = task_phone_th

        # format for printing
        def pr(a,b):
            if b:
                return f"{a/len(b):.3f}"
            else:
                return ""
            
        if l1test['W'][task]:
            w1t = len([ sc for w,sc in l1test['W'][task] if sc > global_word_th])
            p1t = len([ sc for p,sc in l1test['P'][task] if sc > task_phone_th])
            p1test += p1t

            
        if l2test['W'][task]:
            w2t = len([ sc for w,sc in l2test['W'][task] if sc > global_word_th])
            p2t = len([ sc for p,sc in l2test['P'][task] if sc > task_phone_th])
            p2test += p2t

        if l1test['W'][task] or l2test['W'][task]:
            log_info += f"{task}\t{pr(w1t,l1test['W'][task])}\t{pr(p1t,l1test['P'][task])}\t"
            log_info += f"{pr(w2t,l2test['W'][task])}\t{pr(p2t,l2test['P'][task])}\n"

    words_l1_test = [sc for tasks in list(l1test['W'].values()) for sc in tasks]
    if words_l1_test:
        l1_w_test_accept = len([s for w,s in words_l1_test if s > global_word_th])/len(words_l1_test)
        l1_p_test_accept = p1test/len([sc for tasks in list(l1test['P'].values()) for sc in tasks])

    words_l2_test = [sc for tasks in list(l2test['W'].values()) for sc in tasks]
    if words_l2_test:
        l2_w_test_accept = len([s for w,s in words_l2_test if s > global_word_th])/len(words_l2_test)
        l2_p_test_accept = p2test/len([sc for tasks in list(l2test['P'].values()) for sc in tasks])

    if words_l1_test and words_l2_test:
        log_info += f"AVERAGE\t{l1_w_test_accept:.3f}\t{l1_p_test_accept:.3f}\t"
        log_info += f"{l2_w_test_accept:.3f}\t{l2_p_test_accept:.3f}"

    with open(key_save_path,'w') as handle:
        json.dump(task_key,handle)
    if log_info != hed:
        with open(log_path,'w') as handle:
            handle.write(log_info)


def task_dtw_corpus(eval_run):
    initial_task_models_dir = './setup/task_models_w2v2-IS-1000h/'
        
    split_save_path = f'./setup/split_{eval_run}.json'
    tasks_splits = ref_dev_test_splits(initial_task_models_dir,split_save_path)

    with open(split_save_path,'r') as handle:
        tasks_splits = json.load(handle)
        
    corpus_dir = '/Users/cati/corpora/captisr/audio_correct_names/'
    corpus_meta_file = './setup/captini_metadata.tsv'
    with open(corpus_meta_file,'r') as handle:
        corpus_meta = handle.read().splitlines()
    corpus_meta = [l.split('\t') for l in corpus_meta[1:]]


    print('BUILDING TASK DEV + TEST DATA...')
    new_task_models_dir = f'./models/task_models_w2v2-IS-1000h_l8_{eval_run}/'
    os.mkdir(new_task_models_dir)
    # DEV/EVAL.
    l1_dev = {"P" : {}, "W" : {}, "S": {}}
    l1_test = {"P" : {}, "W" : {}, "S": {}}
    l2_test = {"P" : {}, "W" : {}, "S": {}}
    for task, split in tasks_splits.items():
        feats_path = f'{initial_task_models_dir}task_{task}.pickle'
        print(task,'...')
        
        l1ap, l1aw, l1as = run_task_set("L1","DEV",split,feats_path,corpus_meta, corpus_dir, \
                        new_feats_path=f'{new_task_models_dir}task_{task}.pickle')
        l1_dev['P'][task] = l1ap
        l1_dev['W'][task] = l1aw
        l1_dev['S'][task] = l1as

        l1bp, l1bw, l1bs = run_task_set("L1","TEST",split,feats_path,corpus_meta, corpus_dir)
        l1_test['P'][task] = l1bp
        l1_test['W'][task] = l1bw
        l1_test['S'][task] = l1bs
        
        l2bp, l2bw, l2bs = run_task_set("L2","TEST",split,feats_path,corpus_meta, corpus_dir)
        l2_test['P'][task] = l2bp
        l2_test['W'][task] = l2bw
        l2_test['S'][task] = l2bs


    temp_dict = {"l1_dev": l1_dev, "l1_test": l1_test, "l2_test": l2_test}
    dev_save_path=f'./setup/DEV_{eval_run}.json'
    with open(dev_save_path,'w') as handle:
        json.dump(temp_dict, handle)


def calibrate_eval_task(run_path):
    print('CALIBRATING TASKS')
    dev_save_path=f'./setup/DEV_{run_path}.json'
    with open(dev_save_path,'r') as handle:
        dev_dtws = json.load(handle)
    task_key_save_path = f'./models/task_key_{run_path}.json'
    eval_log_path = f'./setup/task_eval_{run_path}.tsv'
    calibrate_task(dev_dtws["l1_dev"],dev_dtws["l1_test"],dev_dtws["l2_test"],task_key_save_path,eval_log_path)




# --------- monophones v ----------

# score speech samples (dev/test) according to provided monophone models
def run_phone(phone,users,models):
    
    def mdtw(test_token,references_tokens):
        references_tokens = [v for v in references_tokens]
        def _dtw(a_feats,b_feats):
            if a_feats.shape[0] < 2 or b_feats.shape[0] < 2:
                return np.nan
            return dtw(a_feats,b_feats,keep_internals=True)
        dtw_objs = [_dtw(test_token,ref_token) for ref_token in references_tokens]
        
        return np.nanmean([x.normalizedDistance for x in dtw_objs if not isinstance(x,float)]) 

    def prepare_score(l1,l2):
        if np.isnan(l1) or np.isnan(l2):
            print("***This probably shouldn't happen here")
            return(None)
        else:
            return (l2-l1)/(l2+l1)

    print(phone)
    dscores = [[mdtw(token,models['L1'][phone]), mdtw(token,models['L2'][phone])] for token in users[phone]]
    
    pscores = [[l1,l2,prepare_score(l1,l2)] for l1,l2 in dscores]
    return pscores


def mono_corpus_dtws(ref_path,dev_path,test_path,feats_dir,save_dir,eval_run):

    models_ref = read_set(ref_path)
    dev_set = read_set(dev_path)
    test_phones = read_set(test_path)

    print('CALCULATING DEV PHONE COSTS')
    
    phones_list = sorted(list(models_ref['L1'].keys()))
    
    dev_pscores = {phone : 
            {'L1': run_phone(phone,dev_set['L1'],models_ref), 
             'L2': run_phone(phone,dev_set['L2'],models_ref)} 
            for phone in phones_list}
    dev_dtws_path = f'{save_dir}/mono_DEV_{eval_run}.pickle'
    with open(dev_dtws_path, 'wb') as handle:
        handle.write(pickle.dumps(dev_pscores))
    return dev_dtws_path

        
def write_key(dev_dtws_path,key_path):
    with open(dev_dtws_path,'rb') as handle:
        scoreset = pickle.load(handle)
        
    phones_list = sorted(list(scoreset.keys()))
    
    l1gl = []
    for phone in phones_list:
        l1gl += scoreset[phone]['L1']
    l1cg = [c for l1, l2, c in l1gl]
    
    global_th = np.percentile(l1cg,1)

    mono_key=''
    for phone in phones_list:
        l1all = scoreset[phone]['L1']
        l1c = [c for l1, l2, c in l1all]        
        phone_th = np.percentile(l1c,1)
        
        if len(l1c) < 50:
            mono_key += f'{phone}\t{global_th}\n'
        else:
            mono_key += f'{phone}\t{phone_th}\n'
    with open(key_path, 'w') as handle:
        handle.write(mono_key)

        
def mono_eval_task(speech,files,monophones,thresholds,lang,split):
    words = list(speech[lang].keys())
    pbin = []
    wraw = []
    wbin = []

    
    def mdtw(test_token,references_tokens):
        references_tokens = [v for v in references_tokens]
        def _dtw(a_feats,b_feats):
            if a_feats.shape[0] < 2 or b_feats.shape[0] < 2:
                return np.nan
            return dtw(a_feats,b_feats,keep_internals=True)
        dtw_objs = [_dtw(test_token,ref_token) for ref_token in references_tokens]
        
        return np.nanmean([x.normalizedDistance for x in dtw_objs if not isinstance(x,float)]) 

    def prepare_score(l1,l2):
        if np.isnan(l1) or np.isnan(l2):
            return("SHORT") # things were too short to score, probably
        else:
            return (l2-l1)/(l2+l1)
            
    for word in words:
        for spk in speech[lang][word]:
            if int(spk[-1]) in split:
                with open(files[spk],'r') as alignment:
                    aligns = load_timestamps(alignment,counted=False)[word]
                wpbins = []
                wpscores = []
                for s,e,phone in aligns:
                    bscore = None
                    test_token = speech[lang][word][spk][s:e]
                    dl1, dl2 = mdtw(test_token,monophones['L1'][phone]), mdtw(test_token,monophones['L2'][phone])
                    pscore = prepare_score(dl1,dl2)
                    if pscore == "SHORT":
                        wpbins.append(1) # free 'correct' if short
                    else:
                        wpscores.append(pscore)
                        if pscore >= thresholds[phone]:
                            wpbins.append(1)
                        else: 
                            wpbins.append(0)
                if wpscores:
                    pbin += wpbins
                    wbin.append(min(wpbins))
                    wraw.append(np.nanmean(wpscores))
                
    return pbin, wbin, wraw


def monophone_test(ref_path,key_path,test_split,testable_tasks,corpus_dir,cmeta):
    ref_models = read_set(ref_path)
    
    with open(key_path,'r') as handle:
        score_key = handle.read().splitlines()
    score_key=[l.split('\t') for l in score_key]
    score_key = {phone : float(binary_threshold) for phone, binary_threshold in score_key}

    l1_p = []
    l1_w = []
    l1_w2 = []
    l2_p = []
    l2_w = []
    l2_w2 = []

    print("TESTING PHONE MODELS")
    for task in testable_tasks:
        print(task.split('_')[-1].split('.')[0])
        with open(task,'rb') as handle:
            task_speech = pickle.load(handle)
        task_words = ' '.join([w.split('__')[1] for w in sorted(list(task_speech['L1'].keys()))])
        norm_text = snorm(task_words)
        task_recs_meta = {l[1]:f'{corpus_dir}{l[1]}/{l[2][:-3]}json' for l in cmeta if snorm(l[3]) == norm_text}
        l1_pb, l1_wb, l1_wr = mono_eval_task(task_speech,task_recs_meta,ref_models,score_key,'L1',test_split)
        l2_pb, l2_wb, l2_wr = mono_eval_task(task_speech,task_recs_meta,ref_models,score_key,'L2',test_split)
        
        l1_p += l1_pb
        l1_w += l1_wb
        l1_w2 += l1_wr
        l2_p += l2_pb
        l2_w += l2_wb
        l2_w2 += l2_wr

    print('\n-MONOPHONES SUMMARY-')
    print(f'L1:\t{(100*np.mean(l1_p)):.2f}% (n={len(l1_p)}) phones')
    print(f'\t{(100*np.mean(l1_w)):.2f}% (n={len(l1_w)}) words')

    print(f'L2:\t{(100*np.mean(l2_p)):.2f}% (n={len(l2_p)}) phones')
    print(f'\t{(100*np.mean(l2_w)):.2f}% (n={len(l2_w)}) words')
    
        
def monophone_calibrate(eval_run):
    mono_ref_path = './models/monophones/w2v2-IS-1000h_SPLIT3.pickle'
    mono_dev_path = './models/monophones/w2v2-IS-1000h_SPLIT9.pickle'
    mono_test_path = './models/monophones/w2v2-IS-1000h_SPLIT6.pickle'
    sentence_feats_dir = './setup/task_models_w2v2-IS-1000h/'
    tmp_dir = './setup/'
    
    dev_dtws_path=mono_corpus_dtws(mono_ref_path,mono_dev_path,mono_test_path,sentence_feats_dir,tmp_dir, eval_run)
    
    corpus_dir = '/Users/cati/corpora/captisr/audio_correct_names/'
    corpus_meta_file = './setup/captini_metadata.tsv'
    with open(corpus_meta_file,'r') as handle:
        corpus_meta = handle.read().splitlines()
    corpus_meta = [l.split('\t') for l in corpus_meta[1:]]

    # must match test set above
    test_split = list(range(7,10))

    key_save_path = f'./models/phone_key_{eval_run}.tsv'

    write_key(dev_dtws_path,key_save_path)
    testable_tasks = sorted(list(glob.glob(sentence_feats_dir+'*.pickle')), key= lambda x: int(x.split('_')[-1][:-7]))
    monophone_test(mono_ref_path,key_save_path,test_split,testable_tasks,corpus_dir,corpus_meta)


def process():
    eval_run = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
    task_dtw_corpus(eval_run)
    calibrate_eval_task(eval_run)
    monophone_calibrate(eval_run)

process()

