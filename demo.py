from captinialign import makeAlign
from captiniscore import PronunciationScorer
from random import choice


# do forced alignment
#   (word and phone level)
# on a 'new' utterance from a user,
# and score the user's pronunciation according to these units
def get_scores(exercise_id, exercise_text, speaker_id, file_id, file_duration, scorer):

    word_aligns, phone_aligns = makeAlign(
        exercise_text,
        speaker_id,
        file_id,
        file_duration)
        
    word_scores, phone_scores = scorer.score_one(
        exercise_id,
        speaker_id,
        file_id,
        word_aligns,
        phone_aligns)
        
    score_output = {'word_scores': word_scores, 
        'phone_scores': phone_scores, 'word_aligns': word_aligns,
        'phone_aligns': phone_aligns}
    return score_output

    
# score a random utterance from the demo set
# and print some output 
def run_random_demo(example,scorer):
    
    exercise_id = example[3]
    exercise_text = example[4]
    speaker_id = example[1]
    recording_id = example[0]
    recording_duration = example[13]
    print(f'Scoring exercise {exercise_id}: File {speaker_id}-{recording_id} ({example[8]})')
    
    score_output = get_scores(
        exercise_id,
        exercise_text,
        speaker_id,
        recording_id,
        recording_duration,
        scorer)

    display(score_output)


# a way to view pronunciation score output
def display(score_output):  
    word_scores = score_output['word_scores']
    phone_scores = score_output['phone_scores']
    for w_s,p_s in zip(word_scores, phone_scores):
        assert w_s[0] == p_s[0]
        print(f'{w_s[0]}\t{w_s[1]}')
        for i in range(len(p_s[1])):
            print(f'\t{p_s[1][i][0]}\t{p_s[1][i][1]}')




if __name__ == "__main__":

    # how many examples to run
    n_demos = 10

    # files provided as examples to use for score demo
    demo_info_file = './demo_recording_data.tsv'
    demo_wav_dir = './demo-wav/'
    demo_reference_sets = './reference-feats_w2v2-base_layer-6/'

    # Speech embedding model and layer must match the pre-computed reference sets.
    # This featurizer path loads the model from huggingface, 
    #   which occasionally has connection problems.
    # For more stable use, download the models
    #   from https://huggingface.co/facebook/wav2vec2-base
    #   and change the path to local directory such as './models/facebook/wav2vec2-base'
    speech_featurizer_path = 'facebook/wav2vec2-base'
    speech_featurizer_layer = 6


    scorer = PronunciationScorer(
        demo_wav_dir, 
        demo_reference_sets, 
        speech_featurizer_path, 
        speech_featurizer_layer)
    
    with open(demo_info_file,'r') as example_db:
        example_db = example_db.read().splitlines()[1:]
        for i in range(n_demos):
            example = choice(example_db).split('\t')
            run_random_demo(example,scorer)
    
    

