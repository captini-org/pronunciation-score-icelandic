import numpy as np
import soundfile as sf
from scipy import signal
from collections import defaultdict
from dtw import dtw
import glob, torch, transformers, sys, random, pickle
transformers.logging.set_verbosity(transformers.logging.ERROR)


class PronunciationScorer():

    def __init__(self, reference_feat_dir, model_path, model_layer): 

        self.reference_feat_dir = reference_feat_dir
        self.model_path = model_path
        self.model_layer = model_layer

        self.featurize = self.w2v2_featurizer()
        
        # convert seconds (alignment timestamps) to feature frames (w2v2)
        # w2v2 step size is about 20ms, or 50 frames per second
        # change as needed if not using w2v2 featurizer
        def cff(sc):
            return int(float(sc)*50)
        self.s2f = cff
                
    # return a function from wav file path to speech embedding
    # see: https://github.com/Bartelds/neural-acoustic-distance/blob/main/extract_features.py
    def w2v2_featurizer(self):
        model_kwargs = {"num_hidden_layers": self.model_layer} # !! TODO
        model = transformers.models.wav2vec2.Wav2Vec2Model.from_pretrained(self.model_path, **model_kwargs)
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


    def task_scorer(self,exercise_id):
        reference_path = f'{self.reference_feat_dir}task_{exercise_id}.pickle'
        try:
            with open(reference_path,'rb') as handle:
                reference_sets = pickle.load(handle)
                print(reference_sets['L1'].keys())
        except FileNotFoundError:
            # Handle the case where the file is not found by returning empty output
            print(f"File not found: '{reference_path}'. Returning empty output.")
            return '', {}
        taskwords = ' '.join([w.split('__')[1] for w in sorted(list(reference_sets['L1'].keys()))])
        return(taskwords, reference_sets)

    
    def score_one(self,reference_sets,test_audio_path,word_aligns,phone_aligns):
        word_aligns = [(w,self.s2f(s),self.s2f(e)) for w,s,e in word_aligns]
        phone_aligns = {w : [(p,self.s2f(s),self.s2f(e)) for p,s,e in w_ps] 
            for w, w_ps in phone_aligns.items()}
    
        #reference_path = f'{self.reference_feat_dir}task_{exercise_id}.pickle'
        #with open(reference_path,'rb') as handle:
        #    reference_sets = pickle.load(handle)
        assert phone_aligns.keys() == reference_sets['L1'].keys()
                
        test_feats = self.featurize(test_audio_path)
        test_word_feats = {w:test_feats[s:e] for w,s,e in word_aligns}
        
        
        def dtw_function (test_word,references_words):
            references_words = [v for v in references_words]
            def _dtw(a_feats,b_feats):
                if a_feats.shape[0] < 2 or b_feats.shape[0] < 2:
                    return np.nan
                return dtw(a_feats,b_feats,keep_internals=True)
            return [_dtw(test_word,ref_word) for ref_word in references_words]
        
        
        def clean(id):
            return id.split('__',1)[1]
        
        
        # average dtw cost per word, relative to reference set
        def get_word_avg(dtw_obj_list):
            # the only possible float here is numpy.nan
            # but numpy.isnan() cannot check DTW objects
            # therefore, discard all floats, remaining objects are DTW objects
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
                return { ph : np.nan for ph,s,e in word_phone_aligns}
            
            pcosts = {ph : (np.nanmean([_get_pcost(s,e,d) for d in dtw_obj_list]) if (e-s > 1) else np.nan) for ph,s,e in word_phone_aligns}
            return pcosts
        
        
        # heuristic score:
        # native speakers should always be closest to other native speakers
        # l2 speakers may be closer to l2 or similar distance from both groups    
        def prepare_score(l1,l2):
            if np.isnan(l1) or np.isnan(l2):
                return('TOO SHORT TO SCORE')
            else:
                return (l2-l1)/(l2+l1)
            
            
        def _compare (comp_set): # call with 'L1' or 'L2' 
        
            word_dtws = { word_id : 
                dtw_function(
                    test_word_feats[word_id], 
                    reference_sets[comp_set][word_id].values()) 
                for word_id,s,e 
                in word_aligns}
                
            word_avg_costs = { word_id : 
                get_word_avg(word_dtws[word_id]) 
                for word_id,s,e 
                in word_aligns }
                
            phone_avg_costs = { word_id : 
                get_phone_avg(word_dtws[word_id], phone_aligns[word_id]) 
                for word_id,s,e 
                in word_aligns }
                
            return word_avg_costs, phone_avg_costs
          
        
        l1_w_costs, l1_p_costs = _compare('L1')
        l2_w_costs, l2_p_costs = _compare('L2')
        
        final_word_scores = [
            (clean(w_id), prepare_score(l1_w_costs[w_id],l2_w_costs[w_id]))
            for w_id,s,e 
            in word_aligns
        ]
        
        final_phone_scores = [
            (clean(w_id), [(
                clean(p_id), 
                prepare_score(l1_p_costs[w_id][p_id], l2_p_costs[w_id][p_id])) 
                for p_id,s,e 
                in phone_aligns[w_id]
                ])
            for w_id, s,e 
            in word_aligns
        ]
        
        return final_word_scores, final_phone_scores
        
        
