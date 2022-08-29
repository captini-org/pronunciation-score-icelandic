#!/bin/bash
#SBATCH --job-name=refaln
#SBATCH --mem-per-cpu=4G
#SBATCH -p longrunning
#SBATCH --time=48:0:0

cd /home/caitlinr/work/pronunciation-score-icelandic/setup/samromur-mfa
source /data/tools/anaconda/etc/profile.d/conda.sh
conda activate smfac
./run.sh --dictionary /home/caitlinr/work/pronunciation-score-icelandic/setup/samromur-mfa/smr215T0/dict_update/frambordabok_captini_v2.dict --model /home/caitlinr/work/pronunciation-score-icelandic/setup/samromur-mfa/smr215T0/s215Tv0.zip

