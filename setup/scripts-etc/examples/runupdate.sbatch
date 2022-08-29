#!/bin/bash
#SBATCH --job-name=smrU1
#SBATCH --mem-per-cpu=4G
#SBATCH -p longrunning
#SBATCH --time=48:0:0

cd /home/caitlinr/work/pronunciation-score-icelandic/setup/samromur-mfa
source /data/tools/anaconda/etc/profile.d/conda.sh
conda activate smfac
./run.sh --dictionary /home/caitlinr/work/pronunciation-score-icelandic/setup/pdicts/fbb1/frambordabok_captini_v2.txt --pronun_dictionary /home/caitlinr/work/pronunciation-score-icelandic/setup/samromur-mfa/smr215T0/dict_update /home/caitlinr/work/pronunciation-score-icelandic/setup/samromur-mfa/smr215T0/s215Tv0.zip

