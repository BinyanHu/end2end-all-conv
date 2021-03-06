#!/bin/bash
#SBATCH --job-name          ddsm_full1152
#SBATCH --output            ddsm_full1152.log
#SBATCH --error             ddsm_full1152.log
#SBATCH --nodes             1
#SBATCH --ntasks-per-node   1
#SBATCH --cpus-per-task     4
#SBATCH --mem               8G
#SBATCH --partition         skylake-gpu
#SBATCH --gres              gpu:1
#SBATCH --time              30:00


module load openmpi/4.0.0
module load cudnn/7.0.5-cuda-9.0.176

source activate py2

cd "/fred/oz121/binyan/repos/end2end-all-conv/"

export PYTHONPATH=$PYTHONPATH:"/fred/oz121/binyan/repos/end2end-all-conv/"

export NUM_CPU_CORES=4

TRAIN_DIR="data/curated_breast_imaging_ddsm/Combined_full_images/full_train_1152x896"
VAL_DIR="data/curated_breast_imaging_ddsm/Combined_full_images/full_val_1152x896"
TEST_DIR="data/curated_breast_imaging_ddsm/Combined_full_images/full_test_1152x896"
PATCH_STATE="saved_model/ddsm_patch/5cls_best_model.h5"
BEST_MODEL="saved_model/ddsm_full1152/resnet_1152x896_prt_addtop1_best.h5"
FINAL_MODEL="NOSAVE"

srun "/fred/oz121/anaconda/envs/py2/bin/python" "ddsm_train/image_clf_train.py" \
    --patch-model-state $PATCH_STATE \
    --no-resume-from \
    --img-size 1152 896 \
    --no-img-scale \
    --rescale-factor 0.003891 \
    --featurewise-center \
    --featurewise-mean 52.18 \
    --no-equalize-hist \
    --top-depths 512 512 \
    --top-repetitions 3 3 \
    --batch-size 2 \
    --train-bs-multiplier 0.5 \
    --augmentation \
    --class-list neg calc mass \
    --patch-net resnet50 \
    --nb-epoch 1 \
    --all-layer-epochs 5 \
    --no-load-val-ram \
    --no-load-train-ram \
    --optimizer adam \
    --weight-decay 0.0001 \
    --weight-decay2 0.0001 \
    --hidden-dropout 0.0 \
    --hidden-dropout2 0.0 \
    --init-learningrate 0.001 \
    --all-layer-multiplier 0.1 \
    --lr-patience 2 \
    --es-patience 10 \
    --auto-batch-balance \
    --pos-cls-weight 1.0 \
    --neg-cls-weight 1.0 \
    --best-model $BEST_MODEL \
    --final-model $FINAL_MODEL \
    $TRAIN_DIR $VAL_DIR $TEST_DIR    
