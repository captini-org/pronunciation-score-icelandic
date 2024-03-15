import numpy as np
import json

class FeedbackConverter():

    def __init__(self, task_key_path, phone_key_path, lower_bound_100, upper_bound_100, not_scored_value = "TOO SHORT TO SCORE"):
        self.task_key_path = task_key_path
        self.phone_key_path = phone_key_path
        self.lower_bound_100 = lower_bound_100
        self.upper_bound_100 = upper_bound_100
        self.not_scored_value = not_scored_value

        self.range_100 = self.upper_bound_100 - self.lower_bound_100

        try:
            with open(phone_key_path,'r') as handle:
                phone_key = handle.read().splitlines()
            phone_key=[l.split('\t') for l in phone_key]
            self.phone_key = {phone : float(binary_threshold) for phone, binary_threshold in phone_key}
            with open(task_key_path,'r') as handle:
                self.task_key = json.load(handle)
        except:
            raise Exception(f"At least one of the score key files {task_key_path} or {phone_key_path} couldn't be loaded.")



# feedback for task-based scoring -----

    def scale_binary_task(self,raw_score,unit,task_id):
        if raw_score == self.not_scored_value:
            return 1
        elif raw_score >= self.task_key[task_id][unit]:
            return 1
        else:
            return 0
    
    def b_list_task(self,scores_list,unit,task_id):
        return [(label, self.scale_binary_task(score,unit,task_id)) for label,score in scores_list]

    

    # scale score from interval [-1,1] to integers [0,100]
    # alternately could replace this with % of phones correct.
    def scale_100(self,raw_score):
        if raw_score == self.not_scored_value:
            return 100
        elif raw_score <= self.lower_bound_100:
            return 0
        elif raw_score >= self.upper_bound_100:
            return 100
        else:
            rescaled_score = (raw_score - self.lower_bound_100) / self.range_100
            return round(100*rescaled_score)

        

    # heuristics?
    # return 1 (correct) for phones/words that are too short to score,
    #   except when a word has score 0 and all phones in that word are too short,
    #    return 0 for all of that word's phones.
    # also, if a word has score 0 but all individual phones have score 1
    #    (as a real score, not when they are all too short),
    #    change the worst phone score to 0 so there is some corrective feedback
    # TODO turn that part off it if overcorrects native speakers

    def wordfix(self,word_phone_scores, word_score, task_id):
        if word_score == 1:
            return self.b_list_task(word_phone_scores,'phone',task_id)
        elif all([sc == self.not_scored_value for ph,sc in word_phone_scores]):
            return [(ph, 0) for ph,sc in word_phone_scores ]
        else:
            bin_scores = self.b_list_task(word_phone_scores,'phone',task_id)
            if all([sc == 1 for ph,sc in bin_scores]):
                sc_list = [1 if sc == self.not_scored_value
                               else sc for ph,sc in word_phone_scores]
                min_ix = sc_list.index(min(sc_list))
                bin_scores[min_ix] = (bin_scores[min_ix][0],0)
            return bin_scores




# feedback for fallback phone scoring -----

    def scale_binary_monophone(self,raw_score,phone_id):
        if raw_score == self.not_scored_value:
            return 1
        elif raw_score >= self.phone_key[phone_id]:
            return 1
        else:
            return 0

    def b_list_monophone(self,scores_list):
        return [(label, self.scale_binary_monophone(score,label)) for label,score in scores_list]

    # score word 0 if any phone is 0, else 1
    # TODO may cause overcorrection of native speakers,
    # consider word score by average of phone raw scores instead
    def b_wordfromphone(self,phone_bins):
        return [( word, min([b for p,b in b_phones]) ) for word, b_phones in phone_bins]

    def scale_100_monophone(self,phone_bins):
        plist = []
        for w, b_phones in phone_bins:
            plist += [b for p,b in b_phones]
        return int(100*np.nanmean(plist))



    
    # output is:
    # - one score 0-100 for the entire task
    # - a score 0/1 for each word
    # - a score 0/1 for each phone
    def convert(self,word_scores,phone_scores,task_id):

        if task_id in self.task_key.keys():
            task_fb = self.scale_100( np.nanmean([sc for wd,sc in word_scores if sc != self.not_scored_value]
                                                or 1) )
            word_fb = self.b_list_task(word_scores,'word',task_id)
            phone_fb = [(p_sc[0], self.wordfix(p_sc[1],w_fb[1],task_id) )
                        for w_fb, p_sc in zip(word_fb,phone_scores)]

        else:
            phone_fb = [(p_sc[0], self.b_list_monophone(p_sc[1]) ) for p_sc in phone_scores]
            word_fb = self.b_wordfromphone(phone_fb)
            task_fb = self.scale_100_monophone(phone_fb)
                        
        return(task_fb, word_fb, phone_fb)


