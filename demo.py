from captinialign import makeAlign
from captiniscore import PronunciationScorer
from captinifeedback import FeedbackConverter
import librosa


# score a demo user utterance:
# process user audio from file path + its task id number
# using the given Scorer and Feedback converter,
# then show some output 
def run_demo(task_id,user_wav,scorer,fb):
    
    task_text, task_model = scorer.task_scorer(task_id)
    user_wav_duration = librosa.get_duration(path=user_wav)

    word_aligns, phone_aligns = makeAlign(
        task_text,
        user_wav,
        user_wav_duration)
        
    word_scores, phone_scores = scorer.score_one(
        task_model,
        user_wav,
        word_aligns,
        phone_aligns)

    task_feedback, word_feedback, phone_feedback = fb.convert(
        word_scores,
        phone_scores)

    score_output = {'task_feedback': task_feedback,
        'word_feedback': word_feedback, 'phone_feedback': phone_feedback,
        'word_scores': word_scores, 'phone_scores': phone_scores,
        'word_aligns': word_aligns, 'phone_aligns': phone_aligns}

    display(score_output)

    

# a way to view pronunciation score output
def display(score_output):
    
    task_feedback = score_output['task_feedback']
    word_feedback = score_output['word_feedback']
    phone_feedback = score_output['phone_feedback']
    
    print(task_feedback)
    for w_s,p_s in zip(word_feedback, phone_feedback):
        assert w_s[0] == p_s[0]
        print(f'{w_s[0]}\t{w_s[1]}')
        for i in range(len(p_s[1])):
            print(f'\t{p_s[1][i][0]}\t{p_s[1][i][1]}')

            

def main():

    # Speech embedding model and layer must match the pre-computed scoring models.
    # This featurizer path loads the model from huggingface, 
    #   which occasionally has connection problems.
    # For more stable use, download the models from huggingface
    #   and change the path to local directory such as './models/LVL/wav2vec2-large-xlsr-53-icelandic-ep10-1000h'
    speech_featurizer_path = 'carlosdanielhernandezmena/wav2vec2-large-xlsr-53-icelandic-ep10-1000h'
    speech_featurizer_layer = 8

    
    # path to pronunciation references for scoring
    scoring_models = './task_models_w2v2-IS-1000h/'

    
    # Define constants for converting pronunciation scores
    # to user feedback
    binary_threshold = -0.005
    lower_bound_100 = -0.1
    upper_bound_100 = 0.0

    
    # PronunciationScorer takes considerable time to initialise,
    #     due to loading the w2v2 featurizer.
    # After the first loading, it quickly scores each new user input speech.
    # It's faster on GPU.
    # Do not re-load a new w2v2 featurizer each time a user speaks.
    scorer = PronunciationScorer(
        scoring_models, 
        speech_featurizer_path, 
        speech_featurizer_layer)

    
    # FeedbackConverter new module to process scores into user feedback
    fb = FeedbackConverter(binary_threshold, lower_bound_100, upper_bound_100)

    
    # some examples.
    demo_info_file = 'demo.tsv'
    
    with open(demo_info_file,'r') as example_db:
        example_db = [l.split('\t') for l in example_db.read().splitlines()[1:]]
    for user_wav,task_id,user_language in example_db:
        print('\n',user_wav.split('/')[-1], user_language)
        run_demo(task_id,user_wav,scorer,fb)
    
    
if __name__ == "__main__":
    
    main()
