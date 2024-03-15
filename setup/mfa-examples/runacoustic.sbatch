#!/bin/bash
#SBATCH --job-name=smrT0
#SBATCH --mem-per-cpu=4G
#SBATCH -p longrunning
#SBATCH --time=48:0:0

cd /home/caitlinr/work/pronunciation-score-icelandic/setup/samromur-mfa
source /data/tools/anaconda/etc/profile.d/conda.sh
conda activate smfac
./run.sh --dictionary /home/caitlinr/work/pronunciation-score-icelandic/setup/pdicts/fbb0/frambordabok_asr_v1.txt

