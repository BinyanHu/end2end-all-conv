TRAIN_DIR="/fred/oz121/repos/end2end-all-conv/INbreast/train_dat_mod/train/"
VAL_DIR="/fred/oz121/repos/end2end-all-conv/INbreast/train_dat_mod/val"
TEST_DIR="/fred/oz121/repos/end2end-all-conv/INbreast/train_dat_mod/test"
RESUME_FROM="ddsm_vgg16_s10_[512-512-1024]x2_hybrid.h5"
BEST_MODEL="INbreast/transferred_inbreast_best_model.h5"
FINAL_MODEL="NOSAVE"
export NUM_CPU_CORES=4

python ddsm_train/image_clf_train.py \
    --no-patch-model-state \
    --resume-from $RESUME_FROM \
    --img-size 1152 896 \
    --no-img-scale \
    --rescale-factor 0.003891 \
    --featurewise-center \
    --featurewise-mean 44.33 \
    --no-equalize-hist \
    --batch-size 4 \
    --train-bs-multiplier 0.5 \
    --augmentation \
    --class-list mass_ben mass_mal \
    --nb-epoch 0 \
    --all-layer-epochs 50 \
    --optimizer adam \
    --weight-decay 0.001 \
    --hidden-dropout 0.0 \
    --weight-decay2 0.01 \
    --hidden-dropout2 0.0 \
    --init-learningrate 0.0001 \
    --all-layer-multiplier 0.01 \
    --es-patience 10 \
    --auto-batch-balance \
    --best-model $BEST_MODEL \
    --final-model $FINAL_MODEL \
    $TRAIN_DIR $VAL_DIR $TEST_DIR
