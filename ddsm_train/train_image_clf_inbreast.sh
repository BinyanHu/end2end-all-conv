#!/bin/bash

TRAIN_DIR="INbreast/train_dat_mod/train"
VAL_DIR="INbreast/train_dat_mod/test"
TEST_DIR="INbreast/train_dat_mod/val"

RESUME_FROM="ddsm_vgg16_s10_[512-512-1024]x2_hybrid.h5"
BEST_MODEL="INbreast/transferred_inbreast_best_model.h5"
FINAL_MODEL="NOSAVE"
export NUM_CPU_CORES=4

# 255/65535 = 0.003891.
python ddsm_train/image_clf_train.py \
	--no-patch-model-state \
	--resume-from $RESUME_FROM \
    --img-size 1152 896 \
    --no-img-scale \
    --rescale-factor 0.003891 \
	--featurewise-center \
    --featurewise-mean 44.33 \
    --no-equalize-hist \
    --patch-net resnet50 \
    --block-type resnet \
    --top-depths 512 512 \
    --top-repetitions 2 2 \
    --bottleneck-enlarge-factor 2 \
    --no-add-heatmap \
    --avg-pool-size 7 7 \
    --add-conv \
    --no-add-shortcut \
    --hm-strides 1 1 \
    --hm-pool-size 5 5 \
    --fc-init-units 64 \
    --fc-layers 2 \
    --batch-size 4 \
    --train-bs-multiplier 0.5 \
	--augmentation \
	--class-list neg pos \
	--nb-epoch 0 \
    --all-layer-epochs 50 \
    --load-val-ram \
    --load-train-ram \
    --optimizer adam \
    --weight-decay 0.001 \
    --hidden-dropout 0.0 \
    --weight-decay2 0.01 \
    --hidden-dropout2 0.0 \
    --init-learningrate 0.0001 \
    --all-layer-multiplier 0.01 \
	--lr-patience 2 \
	--es-patience 10 \
	--auto-batch-balance \
    --pos-cls-weight 1.0 \
	--neg-cls-weight 1.0 \
	--best-model $BEST_MODEL \
	--final-model $FINAL_MODEL \
	$TRAIN_DIR $VAL_DIR $TEST_DIR

