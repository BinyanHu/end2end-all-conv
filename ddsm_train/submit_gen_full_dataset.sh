#!/bin/bash
#SBATCH --job-name          gen_full1152
#SBATCH --output            gen_full1152.log
#SBATCH --error             gen_full1152.log
#SBATCH --nodes             1
#SBATCH --ntasks-per-node   1
#SBATCH --cpus-per-task     1
#SBATCH --mem               4G
#SBATCH --partition         skylake
#SBATCH --time              4:00:00


module load openmpi/4.0.0
module load cudnn/7.0.5-cuda-9.0.176

source activate py2

cd "/fred/oz121/binyan/repos/end2end-all-conv/"

export PYTHONPATH=$PYTHONPATH:"/fred/oz121/binyan/repos/end2end-all-conv/"

# Mass training and validtion set
srun "/fred/oz121/anaconda/envs/py2/bin/python" ddsm_train/sample_patches_combined.py\
    "data/curated_breast_imaging_ddsm/mass_case_description_train_set.csv"\
    "data/curated_breast_imaging_ddsm/Mass-Training ROI and Cropped Images/"\
    "data/curated_breast_imaging_ddsm/Mass-Training Full Mammogram Images/"\
    "data/curated_breast_imaging_ddsm/Combined_full_images/full_train_1152x896"\
    "data/curated_breast_imaging_ddsm/Combined_full_images/full_val_1152x896"\
    --target-height=1152\
    --target-width=896\
    --patch-size=0

# Mass test set
srun "/fred/oz121/anaconda/envs/py2/bin/python" ddsm_train/sample_patches_combined.py\
    "data/curated_breast_imaging_ddsm/mass_case_description_test_set.csv"\
    "data/curated_breast_imaging_ddsm/Mass-Test ROI and Cropped Images/"\
    "data/curated_breast_imaging_ddsm/Mass-Test Full Mammogram Images/"\
    "data/curated_breast_imaging_ddsm/Combined_full_images/full_test_1152x896"\
    "data/curated_breast_imaging_ddsm/Combined_full_images/_folder_that_will_not_be_created"\
    --target-height=1152\
    --target-width=896\
    --patch-size=0\
    --val-size=0\

# Calc training and validation set
srun "/fred/oz121/anaconda/envs/py2/bin/python" ddsm_train/sample_patches_combined.py\
    "data/curated_breast_imaging_ddsm/calc_case_description_train_set.csv"\
    "data/curated_breast_imaging_ddsm/Calc-Training ROI and Cropped Images/"\
    "data/curated_breast_imaging_ddsm/Calc-Training Full Mammogram Images/"\
    "data/curated_breast_imaging_ddsm/Combined_full_images/full_train_1152x896"\
    "data/curated_breast_imaging_ddsm/Combined_full_images/full_val_1152x896"\
    --target-height=1152\
    --target-width=896\
    --patch-size=0

# Calc test set
srun "/fred/oz121/anaconda/envs/py2/bin/python" ddsm_train/sample_patches_combined.py\
    "data/curated_breast_imaging_ddsm/calc_case_description_test_set.csv"\
    "data/curated_breast_imaging_ddsm/Calc-Test ROI and Cropped Images/"\
    "data/curated_breast_imaging_ddsm/Calc-Test Full Mammogram Images/"\
    "data/curated_breast_imaging_ddsm/Combined_full_images/full_test_1152x896"\
    "data/curated_breast_imaging_ddsm/Combined_full_images/_folder_that_will_not_be_created"\
    --target-height=1152\
    --target-width=896\
    --patch-size=0\
    --val-size=0\
