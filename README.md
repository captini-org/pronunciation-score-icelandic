# pronunciation-score-icelandic
demo: first version of pronunciation score module for icelandic

## Use
`demo.py` shows examples of the pronunciation scoring system.

When there are problems, please contact caitlinr@ru.is

## To install

A virtual environment is recommended. The following commands can be used to install requirements, but may change on different systems:
```
conda create --name captinidemo python=3.10
conda activate captinidemo
conda install -c conda-forge dtw-python 
conda install -c conda-forge pysoundfile 
conda install -c conda-forge kaldi 
conda install -c conda-forge transformers
conda install -c conda-forge gcc=12.1.0
```
Cuda is optional, this can also run on CPU.

## Scoring vowel length

This is not implemented within the pronunciation module itself. The following references explain how to classify long vs. short vowel length, using aligned phone durations which are provided by the pronunciation module.


- Pind, J. (1995). Speaking rate, voice-onset time, and quantity: The search for higher-order invariants for two Icelandic speech cues. Perception & Psychophysics, 57(3), 291-304.
   - See Figure 2: a linear classifier of tokens as long or short, based on the ratio of the vowel's duration and the duration of the consonant after it.
- Pind, J. (1999). Speech segment durations and quantity in Icelandic. The Journal of the Acoustical Society of America, 106(2), 1045-1053.
   - See Figure 1: a linear classifier of vowels (average) as long or short, based on the ratio of the vowel's duration and the duration of the consonant after it.
- Many works cited & discussed in the introduction of Pind 1999
- Flego, S. M., & Berkson, K. H. (2020). A Phonetic Illustration of the Sound System of Icelandic. IULC Working Papers, 20(1).

Please understand this measurement has not ever been tried for L2 Icelandic speakers, as far as known. But the figures of Pind (1995, 1999) for L1 speakers are encouraging.



## Notes

Please see [about setup](https://github.com/captini-org/pronunciation-score-icelandic/blob/main/setup/setup.md) for documentation of how this demo was created. You do not need to repeat these procedures in order to run the provided demo.

The files that Kaldi creates during forced alignment are not currently deleted. 
Therefore, they will accumulate in `alignment/new/` (one example included). 
They are not needed for anything and can be safely deleted.
