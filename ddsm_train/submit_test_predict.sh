#!/bin/bash
#SBATCH --job-name          pred_ddsm_full1152
#SBATCH --output            pred_ddsm_full1152.log
#SBATCH --error             pred_ddsm_full1152.log
#SBATCH --nodes             1
#SBATCH --ntasks-per-node   1
#SBATCH --cpus-per-task     1
#SBATCH --mem               8G
#SBATCH --partition         skylake-gpu
#SBATCH --gres              gpu:1
#SBATCH --time              10:00


module load openmpi/4.0.0
module load cudnn/7.0.5-cuda-9.0.176

source activate py2

cd "/fred/oz121/binyan/repos/end2end-all-conv/"

export PYTHONPATH=$PYTHONPATH:"/fred/oz121/binyan/repos/end2end-all-conv/"

srun "/fred/oz121/anaconda/envs/py2/bin/python" "ddsm_train/predict_ddsm_full1152.py" \
    --resume_from="saved_model/ddsm_full1152/resnet_1152x896_prt_addtop1_best.h5"
