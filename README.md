# pronunciation-score-icelandic
demo: first version of pronunciation score module for icelandic

## Use
`demo.py` shows examples of the pronunciation scoring system.

When there are problems, please contact caitlinr@ru.is

## Setup

Using a virtual environment is recommended. The following commands were used to install requirements in conda, but may change on different systems:

```
conda create --name captinidemo python=3.10
conda activate captinidemo
conda install -c conda-forge montreal-forced-aligner=2.0.0rc6
conda install -c conda-forge dtw-python 
conda install -c conda-forge transformers
```
If torch and soundfile were not automatically installed, add them as well.

Cuda is optional, it can also run on CPU.

 
If you already have Kaldi, it may not be necessary to install Montreal Forced Aligner. 
Check `shutil.which("compute-mfcc-feats")`
If it returns a valid path to run the kaldi tool (along with the other kaldi functions called in `captinialign.py`) 
then you might not need to install MFA.

## Scoring vowel length

This is not implemented within the pronunciation module itself. The following references explain how to classify long vs. short vowel length, using aligned phone durations which are provided by the pronunciation module.


- Pind, J. (1995). Speaking rate, voice-onset time, and quantity: The search for higher-order invariants for two Icelandic speech cues. Perception & Psychophysics, 57(3), 291-304.
   - See Figure 2: a linear classifier of tokens as long or short, based on the ratio of the vowel's duration and the duration of the consonant after it.
- Pind, J. (1999). Speech segment durations and quantity in Icelandic. The Journal of the Acoustical Society of America, 106(2), 1045-1053.
   - See Figure 1: a linear classifier of vowels (average) as long or short, based on the ratio of the vowel's duration and the duration of the consonant after it.
- Many works cited & discussed in the introduction of Pind 1999
- Flego, S. M., & Berkson, K. H. (2020). A Phonetic Illustration of the Sound System of Icelandic. IULC Working Papers, 20(1).

Please understand this measurement has not ever been tried for L2 Icelandic speakers, as far as I know. But the figures of Pind (1995, 1999) for L1 speakers are encouraging.



## Notes

The Kaldi acoustic models and pronunciation dictionary (lexicon) inside the `alignment` folder must NOT be 
released or distributed in any public way. 
These are for temporary demonstration purposes only, using pre-existing resources, 
compiled from various and partially unknown sources. 
Therefore, it is impossible to attribute them or gain permissions.
They must be replaced before any software publication.

The files that Kaldi creates during forced alignment are not currently deleted. 
Therefore, they will accumulate in `alignment/new/` (one example included). 
They are not needed for anything and can be safely deleted.
