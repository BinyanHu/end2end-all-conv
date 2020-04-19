# Usage

The following is a step-by-step example of training a model to classify image patches into 3 classes.



## Environment Creation

Install all the packages with:

```shell
pip install -r requirements.txt
```



Note that we are using Keras 2.0.8 which depends on TensorFlow 1.5.0. So load CUDA 9 and CuDNN 7 by:

```shell
module load cudnn/7.0.5-cuda-9.0.176
```

before running any programs.



Also, add the current project directory to `PYTHONPATH` so that the Python modules can be correctly imported.

```shell
export PYTHONPATH=$PYTHONPATH:your_path_to_repos/end2end-all-conv
```



## Data Generation

1. Download the DDSM dataset and use [convert_dicom_to_png.py](ddsm_train\convert_dicom_to_png.py) mentioned above to convert all scans to png format. Finally, put the png dataset directory into `data/`. Please check the directory tree structure to ensure that the following steps will work:

   ```
   end2end-all-conv/data/
   	curated_breast_imaging_ddsm/
   		Calc-Test Full Mammogram Images/
   		Calc-Test ROI and Cropped Images/
   		Calc-Training Full Mammogram Images/
   		Calc-Training ROI and Cropped Images/
   		Mass-Test Full Mammogram Images/
   		Mass-Test ROI and Cropped Images/
   		Mass-Training Full Mammogram Images/
   		Mass-Training ROI and Cropped Images/
   		calc_case_description_test_set.csv
   		calc_case_description_train_set.csv
   		mass_case_description_test_set.csv
   		mass_case_description_train_set.csv
   ```



2. Run [gen_ddsm_dataset.sh](ddsm_train\gen_ddsm_dataset.sh) to generate the dataset. This script will crop image patches of shape `(256, 256)` from the dataset. Modify the shell script to tune parameters like output directories and numbers of each class. In the example we set all numbers of the samples to 1 for a quick test, but it is recommended to set them larger to ensure a stable training.

   After the generation is finished, a new director `train_dat_mod` should appear in the `data/` directory:

   ```
   end2end-all-conv/data/
   	curated_breast_imaging_ddsm/
   		train_dat_mod/
   			train/
   				background/
   				mass_ben/
   				mass_mal/
   				pat_lst.txt
   			val/
   				background/
   				mass_ben/
   				mass_mal
   				pat_lst.txt
   ```

   Also, check the size and contents of the images. If there exists all-black images, go back and check the image generation step.

   ![Figure: Image patch sample](D:/Sources/Samples/end2end-all-conv/img/patch_sample.png)

   
   
   

## Model Training

### Training on Login Nodes

Run [train_image_clf_ddsm.sh](ddsm_train\train_image_clf_ddsm.sh)  to train a classification model Model with the dataset we just made. This model accepts input images of shape `(256,256)` and output classifications of the three classes: background, benign mass and malignant mass. This model can be further used for inferencing or transfer-learning.



### Training on Job Nodes

Run [submit_train_image_clf_ddsm.sh ](ddsm_train/submit_train_image_clf_ddsm.sh ) by

```shell
sbatch ddsm_train/submit_train_image_clf_ddsm.sh
```

to allocate a job node and submit the training to it. After the job starts, the outputs will be written to `train_ddsm.log`. Use

```shell
tail -f train_ddsm.log
```

to reflect the outputs in real-time.
