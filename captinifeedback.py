import numpy as np


class FeedbackConverter():

    def __init__(self, binary_threshold, lower_bound_100, upper_bound_100, not_scored_value = "TOO SHORT TO SCORE"):
        self.binary_threshold = binary_threshold
        self.lower_bound_100 = lower_bound_100
        self.upper_bound_100 = upper_bound_100
        self.not_scored_value = not_scored_value

        self.range_100 = self.upper_bound_100 - self.lower_bound_100



    # scale score from interval [-1,1] to binary 0/1
    def scale_binary(self,raw_score):
        if raw_score == self.not_scored_value:
            return 1
        elif raw_score >= self.binary_threshold:
            return 1
        else:
            return 0


    # TODO consider limit -
    # e.g. return at most three 0's, the 3 worst words/phones.
    def b_list(self,scores_list):
        return [(label, self.scale_binary(score)) for label,score in scores_list]

        

    # scale score from interval [-1,1] to integers [0,100]
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

    def wordfix(self,word_phone_scores, word_score):
        if word_score == 1:
            return self.b_list(word_phone_scores)
        elif all([sc == self.not_scored_value for ph,sc in word_phone_scores]):
            return [(ph, 0) for ph,sc in word_phone_scores ]
        else:
            bin_scores = self.b_list(word_phone_scores)
            if all([sc == 1 for ph,sc in bin_scores]):
                sc_list = [1 if sc == self.not_scored_value
                               else sc for ph,sc in word_phone_scores]
                min_ix = sc_list.index(min(sc_list))
                bin_scores[min_ix] = (bin_scores[min_ix][0],0)
            return bin_scores


    # FEEDBACK
    
    # input is scores from PronunciationScorer which are on interval [-1, 1]
    
    # output is:
    # - one score 0-100 for the entire task
    # - a score 0/1 for each word
    # - a score 0/1 for each phone
    
    def convert(self,word_scores,phone_scores):

        task_fb = self.scale_100( np.nanmean([sc for wd,sc in word_scores if sc != self.not_scored_value]
                                                or 1) )
        
        word_fb = self.b_list(word_scores)

        phone_fb = [(p_sc[0], self.wordfix(p_sc[1],w_fb[1]) )
                        for w_fb, p_sc in zip(word_fb,phone_scores)]

        return(task_fb, word_fb, phone_fb)
