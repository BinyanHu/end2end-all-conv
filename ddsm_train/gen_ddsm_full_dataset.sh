# Mass training and validtion set
python ddsm_train/sample_patches_combined.py\
    "data/curated_breast_imaging_ddsm/mass_case_description_train_set.csv"\
    "data/curated_breast_imaging_ddsm/Mass-Training ROI and Cropped Images/"\
    "data/curated_breast_imaging_ddsm/Mass-Training Full Mammogram Images/"\
    "data/curated_breast_imaging_ddsm/Combined_full_images/full_train_1152x896"\
    "data/curated_breast_imaging_ddsm/Combined_full_images/full_val_1152x896"\
    --target-height=1152\
    --target-width=896\
    --patch-size=0

# Mass test set
python ddsm_train/sample_patches_combined.py\
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
python ddsm_train/sample_patches_combined.py\
    "data/curated_breast_imaging_ddsm/calc_case_description_train_set.csv"\
    "data/curated_breast_imaging_ddsm/Calc-Training ROI and Cropped Images/"\
    "data/curated_breast_imaging_ddsm/Calc-Training Full Mammogram Images/"\
    "data/curated_breast_imaging_ddsm/Combined_full_images/full_train_1152x896"\
    "data/curated_breast_imaging_ddsm/Combined_full_images/full_val_1152x896"\
    --target-height=1152\
    --target-width=896\
    --patch-size=0

# Calc test set
python ddsm_train/sample_patches_combined.py\
    "data/curated_breast_imaging_ddsm/calc_case_description_test_set.csv"\
    "data/curated_breast_imaging_ddsm/Calc-Test ROI and Cropped Images/"\
    "data/curated_breast_imaging_ddsm/Calc-Test Full Mammogram Images/"\
    "data/curated_breast_imaging_ddsm/Combined_full_images/full_test_1152x896"\
    "data/curated_breast_imaging_ddsm/Combined_full_images/_folder_that_will_not_be_created"\
    --target-height=1152\
    --target-width=896\
    --patch-size=0\
    --val-size=0\
