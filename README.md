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
