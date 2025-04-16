#!/bin/bash -f
#$ -N AICE_verification
#$ -l h_rt=02:00:00
#$ -S /bin/bash
#$ -pe shmem-1 1
#$ -l h_rss=5G,mem_free=5G,h_data=5G
#$ -q research-r8.q
##$ -j y
#$ -m ba
#$ -o /home/cyrilp/Documents/OUT/OUT_$JOB_NAME.$JOB_ID_$TASK_ID
#$ -e /home/cyrilp/Documents/ERR/ERR_$JOB_NAME.$JOB_ID_$TASK_ID
##$ -R y
##$ -r y

source /modules/rhel8/conda/install/etc/profile.d/conda.sh
conda activate /modules/rhel8/conda/install/envs/production-08-2024

python3 /lustre/storeB/users/cyrilp/AICE/MET_AICE_Github/Verification/Stats_RMSE_ice_edge_distance_error_Barents_AICE_domain.py
