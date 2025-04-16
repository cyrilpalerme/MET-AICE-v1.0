#!/bin/bash -f
#$ -N Run_UNet_AICE
#$ -l h_rt=08:00:00
#$ -S /bin/bash
#$ -pe shmem-1 1
#$ -l h_rss=40G,mem_free=40G,h_data=40G
#$ -q gpu-r8.q
#$ -l h=gpu-05.ppi.met.no
##$ -j y
#$ -m ba
#$ -o /home/cyrilp/Documents/OUT/OUT_$JOB_NAME.$JOB_ID_$TASK_ID
#$ -e /home/cyrilp/Documents/ERR/ERR_$JOB_NAME.$JOB_ID_$TASK_ID
##$ -R y
##$ -r y

module use /modules/MET/rhel8/user-modules/
module load cuda/11.6.0

source /modules/rhel8/conda/install/etc/profile.d/conda.sh
conda activate /lustre/storeB/users/cyrilp/mycondaTF

python3 "/lustre/storeB/users/cyrilp/AICE/MET_AICE_Github/Train_model/Run_UNet.py"
