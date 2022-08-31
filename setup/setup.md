# pronunciation-score-icelandic

## To train & update general acoustic models or dictionaries used for alignment

#### About dictionary
  - The pronunciation dictionary is based on General Pronunciation Dictionary for ASR, https://repository.clarin.is/repository/xmlui/handle/20.500.12537/199
  - A few missing words, that are part of the Captini exercises, have been added.  
These are listed in ADDED_TO_DICT.txt and a native/fluent speaker must check them before release.

#### About acoustic model training data
  - The acoustic models are trained on the Train split of Samrómur 21.05, which must be downloaded. Any other training corpus must also be in the correct format for samromur-mfa.

#### 1. Install [samromur-mfa](https://github.com/cadia-lvl/samromur-mfa) according to instructions (can skip installing Sequitur G2P). 
Example:
```
conda create --name smfac
conda activate smfac
conda install -c conda-forge montreal-forced-aligner
conda install pandas
conda install -c conda-forge alive-progress 
git clone https://github.com/cadia-lvl/samromur-mfa.git
```
#### 2. Train acoustic model for the first time
  - Replace the samromur-mfa run.sh file with the included run.sh
  - Edit the samromur-mfa info.json file, found in samromur-mfa/local/info.json.  
An example is included as info_train.json, but it may need to be adjusted, according to the instructions from samromur-mfa.
  - Run samromur-mfa by `./run.sh --dictionary PATH_TO_DICTIONARY` or for better job handling, an example `runacoustic.sbatch` is included
#### 3. You can add words to the pronunciation dictionary
Edit the text file and give the correct dictionary file path to samromur-mfa.

## To prepare the CAPTinI demo recordings for scoring
#### 1. Get the original recordings and metadata of the Captini exercise speech
  - There is not yet public distribution of this
#### 2. Prepare the recordings for alignment with samromur-mfa
  - Scripts are included as a record of what was done, however this is unlikely to be reproducible for other cases without editing.  
The steps performed for this dataset were:
    - Add a normalised form of the sentence (suitable to use for alignment) to the metadata file, 
    - Create a key file associating these sentences with a unique Exercise ID
    - Perform basic checks on each audio file (e.g. not empty/silent) and remove recordings which fail
- This is done by `preprocess.py` followed by `./move_exclude.sh` (which is created by preprocess.py)
#### 3. Align the reference recordings with samromur-mfa
Use the previously trained acoustic model and dictionary. An example info.json file for this step is included as `info_align_refs.json`, along with `runrefalignment.sbatch`

#### 4. Set up to do live forced alignment
- Find the pretrained aligner that MFA made in the previous step. Default is a location like `~/Documents/MFA/captini_pretrained_aligner/`
- Copy the `dictionary` and `pretrained_aligner` folders from inside of that to `pronunciation_score_icelandic/alignment/captini_pretrained_aligner` (or equivalent)
- Not all MFA files are necessary, therefore to save space only the necessary ones are copied into this repository
- Update paths for the pretrained aligner and pronunciation dictionary name in `captinialign.py`

#### 5. Create the Captini demo
- build_demo_sets.py
  - Select only the exercises that have enough L1 and L2 recordings available
  - For each of those exercises, select a few recordings to use as input speech in the demo; the rest are the reference sets
  - Pre-compute speech features for all reference sets, and for each exercise, save these features, with their word alignments, as pickles
- This step does NOT require Montreal Forced Aligner or samromur-mfa, but it needs python `transformers`, `torch`, and `soundfile`. These can have conflicts with installing certain versions of MFA, in which case a separate environment is recommended.
- New versions of MFA occasionally change the format of .json alignment output. In this case, the function `load_word_timestamps` needs to be edited. This demo used MFA 2.0.6.


