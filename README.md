# pronunciation-score-icelandic
#### Demo: first version of pronunciation score module for Icelandic(194 exercises), temporarily combined with alternate fallback method to handle the other 269 exercises

#### Purpose: to illustrate the captini lesson content, user interface, and interactive learning environment. Not purpose: to evaluate pronunciation accuracy. 
- The fallback system (58% of lesson content) can not be used by real students of Icelandic to practice their pronunciation, and can not be part of any academic submission/publication.

## Run
`python3 ./demo/demo.py` shows examples of the pronunciation scoring system.

First, download [this file](https://drive.google.com/file/d/1kPbGDGSAMuyEGdfW5N29fk3pEXjS2n1G/view?usp=share_link) to replace `models/monophones/w2v2-IS-1000h_SPLIT3.pickle.PLACEHOLDER.txt`

For problems (or to suggest more permanent file storage), please contact caitlinr@ru.is

## Install

A virtual environment is recommended. Cuda is optional. The following may change on different systems:
```
conda create --name captinidemo python=3.11
conda activate captinidemo
conda install pytorch::pytorch torchvision torchaudio -c pytorch
conda install conda-forge::kaldi
conda install conda-forge::dtw-python
conda install conda-forge::pydub
pip install soundfile
pip install librosa
pip install transformers
```

These instructions are for a standalone demo of pronunciation
scoring. Refer to the dockerfile to use this pronunciation scoring module as
part of the complete Captini web service.


## Setup and training
These steps are described for documentation of how this demo was
created, but they don't need to be repeated in order to run the provided demo.

#### 1. Download audio data
- The reference data is a parallel speech corpus of native (L1) and non native
(L2) speakers pronouncing each Captini exercise.
- Data was recorded on [samromur.is](https://samromur.is/). One way to
  access it is GetRecordings from
  [samromur-tools](https://github.com/cadia-lvl/samromur-tools/).
    - Copy this repo's file `setup/GetRecordingsCredentials.json` into
  samromur-tools/GetRecordings, rename it to `credentials.json`, and edit it
  to fill in missing details.
    - Add a query to `samromur-tools/GetRecordings/modules/database.py` to
  download recordings of sentences with the source "captini", spoken
  by adults.
- Some tasks were previously recorded but no longer used, so the reference
  recordings can also be filtered by `pronunciation-score-icelandic/setup/task-text.txt`

#### 2. Pronunciation dictionary
- The pronunciation dictionary is based on
    [General Pronunciation Dictionary for ASR](https://repository.clarin.is/repository/xmlui/handle/20.500.12537/199)
    using IPA phone set.
- Missing words, that are part of the Captini exercises, have
    been added. The pronunciations were confirmed by a native
    speaker/Icelandic teacher.

#### 3. Forced alignment
- The acoustic models for alignment were trained on the Train split of
Samrómur 21.05.
- This module uses Kaldi for alignment. Montreal Forced Aligner is recommended to train acoustic models for
  alignment, and to align the Captini reference data with the trained model.
- [samromur-mfa](https://github.com/cadia-lvl/samromur-mfa)
conveniently applies MFA to Samrómur data.
    - Replace the samromur-mfa run.sh file with the included
	`pronunciation-score-icelandic/setup/mfa-examples/run.sh`
	- Example files for using samromur-mfa are also provided there.
- To run live pronunciation scoring, Kaldi files for dictionary and
  pretrained aligner must be placed in
  `pronunciation_score_icelandic/alignment/captini_pretrained_aligner`
  (or equivalent).
    - After MFA training, copy these from a location like
	`~/Documents/MFA/captini_pretrained_aligner/`
    - Files that aren't in this repo don't need to be copied. MFA creates some extra files.
	- Update paths for the pretrained aligner and pronunciation dictionary name in `captinialign.py`.
- After trained alignment models are in `pronunciation_score_icelandic/alignment`, they
  can be used instead of MFA for quick alignment of new Captini
  reference recordings, with
  align_corpus function in `setup/hybridsetup.py`
- If you prefer to replace Kaldi with other alignment for live
  pronunciation scoring, then replace `captinialign.py`

#### 4. Task models
- This exemplar-based pronunciation scoring uses a parallel corpus to
build L1 and L2 pronunciation scoring models for each task. See [here](https://ieeexplore.ieee.org/document/10095033) for
method details and
[here](https://www.isca-archive.org/interspeech_2023/richter23b_interspeech.pdf)
for a variation with some results on Captini data.
- Scoring is not valid for tasks without at least 20 L1 and 10 L2 reference
  speakers, because otherwise there is not enough parallel data for
  comparing the user's pronunciations and calibrating thresholds.
- After the reference corpus is aligned, build initial task 
models and fallback monophone models with the function model\_all\_tasks in `setup/hybridsetup.py`
- Build final models and get thresholds for scoring with
  `setup/calibrate_eval.py`. It automatically evaluates the fallback
  (monophone) system, but to evaluate the main system, choose a
  different corpus split function in ref\_dev\_test\_splits().
- Speech embeddings of these scoring models must match speech
  embeddings at runtime, including the model layer specification. A reasonable place to start is [this wav2vec2
  Icelandic model](https://huggingface.co/carlosdanielhernandezmena/wav2vec2-large-xlsr-53-icelandic-ep10-1000h)
  layer 8, but the layer should be selected by evaluating performance
  for all intermediate model layers.

## Notes
The files that Kaldi creates during forced alignment are not currently deleted. 
Therefore, a lot of small files will accumulate in `alignment/new/`, or alternate
location specified in `captinialign.py`.
They are not needed for anything and should be deleted, immediately
after scoring each pronunciation or by a regular cleanup process.


## Scoring vowel length
This is not implemented within the pronunciation module itself, but
can be derived from its output. The following references explain how
to classify long vs. short vowel length, using aligned phone durations
which are provided by the pronunciation module.


- Pind, J. (1995). Speaking rate, voice-onset time, and quantity: The search for higher-order invariants for two Icelandic speech cues. Perception & Psychophysics, 57(3), 291-304.
   - See Figure 2: a linear classifier of tokens as long or short, based on the ratio of the vowel's duration and the duration of the consonant after it.
- Pind, J. (1999). Speech segment durations and quantity in Icelandic. The Journal of the Acoustical Society of America, 106(2), 1045-1053.
   - See Figure 1: a linear classifier of vowels (average) as long or short, based on the ratio of the vowel's duration and the duration of the consonant after it.
- Many works cited & discussed in the introduction of Pind 1999
- Flego, S. M., & Berkson, K. H. (2020). A Phonetic Illustration of the Sound System of Icelandic. IULC Working Papers, 20(1).

Please understand this measurement has not ever been tried for L2 Icelandic speakers, as far as known. But the figures of Pind (1995, 1999) for L1 speakers are encouraging.

