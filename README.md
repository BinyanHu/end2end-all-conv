[![License: CC BY-NC-ND 4.0](https://img.shields.io/badge/License-CC%20BY--NC--ND%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-nd/4.0/)

# Deep Learning to Improve Breast Cancer Detection on Screening Mammography (End-to-end Training for Whole Image Breast Cancer Screening using An All Convolutional Design)
Li Shen, Ph.D. CS

Icahn School of Medicine at Mount Sinai

New York, New York, USA

![Fig1](https://raw.githubusercontent.com/lishen/end2end-all-conv/master/ddsm_train/Fig-1%20patch%20to%20whole%20image%20conv.jpg "Convert conv net from patch to whole image")

## Introduction
This is the companion site for our paper that was originally titled "End-to-end Training for Whole Image Breast Cancer Diagnosis using An All Convolutional Design" and was retitled as "Deep Learning to Improve Breast Cancer Detection on Screening Mammography". The paper has been published [here](https://rdcu.be/bPOYf). You may also find the arXiv version [here](https://arxiv.org/abs/1708.09427). This work was initially presented at the NIPS17 workshop on machine learning for health. Access the 4-page short paper [here](https://arxiv.org/abs/1711.05775). Download the [poster](https://raw.githubusercontent.com/lishen/end2end-all-conv/master/ddsm_train/NIPS17%20ML4H%20Poster.pdf).

For our entry in the DREAM2016 Digital Mammography challenge, see this [write-up](https://www.synapse.org/LiShenDMChallenge). This work is much improved from our method used in the challenge.



## Issues Resolved

### Requirements

The project uses Python2 with several packages of old versions. The following configuration is inferred from the code and proved to work:

```
Keras==2.0.8
tensorflow-gpu==1.5.0
opencv-python
pandas
pydicom
Pillow
```



### API changes

Versions of some packages, such as OpenCV and scipy are hard to infer, so we install the latest and update the code. Known changes are:

1. `cv2.findContours` returns 2 parameters instead of 3. So each of the following code block:

   ```python
   if int(ver[0]) < 3:
   	contours,_ = cv2.findContours(img_bin.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
   else:
   	_,contours,_ = cv2.findContours(img_bin.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
   ```

   should be changed to:

   ```python
   contours,_ = cv2.findContours(img_bin.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
   ```

   

2. `scipy.misc.toimage` is deprecated and has been removed for a long time. We took the file `piltil.py` and placed it in the project directory, so that we can replace:

   ```python
   from scipy.misc import toimage
   ```

   and import it by:

   ```python
   from pilutil import toimage
   ```



### Data Preprocessing

Use [convert_dicom_to_png.py](ddsm_train\convert_dicom_to_png.py) to converting the scans from `.dcm` to `.png` format. The raw dataset is expected to be downloaded to `CBIS-DDSM`, and this script will store all the png images into a  new directory named `CBIS-DDSM-png`. See the python file for more details.



### Data Generation

The folder names and the description files we are using are different from those of the author of this project. The [patch dataset generation script](ddsm_train\sample_patches_combined.py) must be modified to adapt these changes. 

1. The index names of the description files (those ends with `.csv`) have changed. As is discussed in [issue/2](https://github.com/lishen/end2end-all-conv/issues/2#issuecomment-338938924), we should change the key string names of the python files or the column names of the csv files for reproduction. A comparison is listed below:

   | name in python code | name in description csv |
   | ------------------- | ----------------------- |
   | "side"              | "left or right breast"  |
   | "view"              | "image view"            |



2. The file paths of the images have changed, so the function `const_filename` can not be used. The file paths are also supposed to be accessible  in the description csv files. However, the information in our description files does not match the current dataset. For example, according to the description, an image which is supposed to be located at

   ```
   Mass-Training_P_00001_LEFT_CC/1.3.6.1.4.1.9590.100.1.2.422112722213189649807611434612228974994/1.3.6.1.4.1.9590.100.1.2.342386194811267636608694132590482924515/000000.dcm
   ```

   is actually located at:

   ```
   Mass-Training Full Mammogram Images/Mass-Training_P_00001_LEFT_CC/07-20-2016-DDSM-74994/1-full mammogram images-24515/000000.png
   ```

   Issues caused by the file path inconsistence are also found in [tensorflow_datasets curated_breast_imaging_ddsm](https://www.tensorflow.org/datasets/catalog/curated_breast_imaging_ddsm). 

   Therefore, we abandoned the function `const_filename` and wrote another function `get_image_and_mask` to automatically search the file path, match and load the image files.



## Issues NOT Resolved

1. The original code does not generate test set, but test set is used in the demos. If you hope to use test sets, modify [sample_patches_combined.py](ddsm_train\sample_patches_combined.py) to implement test set splitting.



## Usage

Things got easier after we have resolved some issues and rewrote some scripts. The following is a step-by-step example of training a model to classify image patches into 3 classes.



### Environment Creation

Install all the packages with:

```shell
pip install -r requirements.txt
```



Note that we are using Keras 2.0.8 which depends on TensorFlow 1.5.0. So load CUDA 9 and CuDNN 7 by:

```shell
module load cudnn/7.0.5-cuda-9.0.176
```

before running any programs.



### Data Generation

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



2. Run [gen_ddsm_dataset.sh](ddsm_train\gen_ddsm_dataset.sh) to generate the dataset. This script will crop image patches of shape `(256, 256)` from the dataset. Modify the shell script to tune parameters like output directories and numbers of each class. In the example we set all numbers of the samples to 1 for a quick test, but it is recommended to set them larger to ensure model convergence.

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

   ![Figure: Image patch sample](img\patch_sample.png)

   

### Model Training

Run [train_image_clf_ddsm.sh](ddsm_train\train_image_clf_ddsm.sh)  to train a classification model Model with the dataset we just made. This model accepts input images of shape `(256,256)` and output classifications of the three classes: background, benign mass and malignant mass. This model can be used for inferencing or transfer-learning.



## Whole image model downloads

A few best whole image models are available for downloading at this Google Drive [folder](https://drive.google.com/open?id=0B1PVLadG_dCKV2pZem5MTjc1cHc). YaroslavNet is the DM challenge top-performing team's [method](https://www.synapse.org/#!Synapse:syn9773040/wiki/426908). Here is a table for individual downloads:

| Database  | Patch Classifier  | Top Layers (two blocks)  | Single AUC  | Augmented AUC  | Link  |
|---|---|---|---|---|---|
| DDSM  | Resnet50  | \[512-512-1024\]x2  | 0.86  | 0.88  | [download](https://drive.google.com/open?id=0B1PVLadG_dCKSUJYdzNyZjVsZHc)  |
| DDSM  | VGG16  | 512x1  | 0.83  | 0.86  | [download](https://drive.google.com/open?id=0B1PVLadG_dCKYnREWlJQZ2JaSDQ)  |
| DDSM  | VGG16  | \[512-512-1024\]x2  | 0.85  | 0.88  | [download](https://drive.google.com/open?id=0B1PVLadG_dCKdVQzbDRLNTZ4TXM)  |
| DDSM | YaroslavNet | heatmap + max pooling + FC16-8 + shortcut | 0.83 | 0.86 | [download](https://drive.google.com/open?id=0B1PVLadG_dCKVk9RM1dMeTkwcTg) |
| INbreast  | VGG16  | 512x1  | 0.92  | 0.94  | [download](https://drive.google.com/open?id=0B1PVLadG_dCKN0ZxNFdCRWxHRFU)  |
| INbreast  | VGG16  | \[512-512-1024\]x2  | 0.95  | 0.96  | [download](https://drive.google.com/open?id=0B1PVLadG_dCKUnQwYVhOd2NfQlk)  |

- Inference level augmentation is obtained by horizontal and vertical flips to generate 4 predictions.
- The listed scores are single model AUC and prediction averaged AUC.
- 3 Model averaging on DDSM gives AUC of 0.91
- 2 Model averaging on INbreast gives AUC of 0.96.

## Patch classifier model downloads
Several patch classifier models (i.e. patch state) are also available for downloading at this Google Drive [folder](https://drive.google.com/open?id=0B1PVLadG_dCKV2pZem5MTjc1cHc). Here is a table for individual download:

| Model  | Train Set | Accuracy | Link |
|---|---|---|---|
| Resnet50  | S10  | 0.89  | [download](https://drive.google.com/open?id=0B1PVLadG_dCKMTc2RGV1NGF6bm8) |
| VGG16  | S10  | 0.84  | [download](https://drive.google.com/open?id=0B1PVLadG_dCKZUotelJBTzkwaGM) |
| VGG19  | S10  | 0.79  | [download](https://drive.google.com/open?id=0B1PVLadG_dCKUHFpeTgwaVFYS1E) |
| YaroslavNet (Final) | S10 | 0.89 | [download](https://drive.google.com/open?id=0B1PVLadG_dCKSlR4V1QwTGdsbGs) |
| Resnet50  | S30  | 0.91  | [download](https://drive.google.com/open?id=0B1PVLadG_dCKSW1RZ0NzOVBJWHc) |
| VGG16  | S30  | 0.86  | [download](https://drive.google.com/open?id=0B1PVLadG_dCKbjdodnp2SlR2WFU) |
| VGG19  | S30  | 0.89  | [download](https://drive.google.com/open?id=0B1PVLadG_dCKNUpGc3Q1dWJ5OGM) |

With patch classifier models, you can convert them into any whole image classifier by adding convolutional, FC and heatmap layers on top and see for yourself.

## A bit explanation of this repository's file structure
- The **.py** files under the root directory are Python modules to be imported.
- You shall set the `PYTHONPATH` variable like this: `export PYTHONPATH=$PYTHONPATH:your_path_to_repos/end2end-all-conv` so that the Python modules can be imported.
- The code for patch sampling, patch classifier and whole image training are under the [ddsm_train](./ddsm_train) folder.
- [sample_patches_combined.py](./ddsm_train/sample_patches_combined.py) is used to sample patches from images and masks.
- [patch_clf_train.py](./ddsm_train/patch_clf_train.py) is used to train a patch classifier.
- [image_clf_train.py](./ddsm_train/image_clf_train.py) is used to train a whole image classifier, either on top of a patch classifier or from another already trained whole image classifier (i.e. finetuning).
- There are multiple shell scripts under the [ddsm_train](./ddsm_train) folder to serve as examples.

## Transfer learning is as easy as 1-2-3
In order to transfer a model to your own data, follow these easy steps.
### Determine the rescale factor
The rescale factor is used to rescale the pixel intensities so that the max value is 255. For PNG format, the max value is 65535, so the rescale factor is 255/65535 = 0.003891. If your images are already in the 255 scale, set rescale factor to 1.
### Calculate the pixel-wise mean
This is simply the mean pixel intensity of your train set images.
### Image size
This is currently fixed at 1152x896 for the models in this study. However, you can change the image size when converting from a patch classifier to a whole image classifier.
### Finetune
Now you can finetune a model on your own data for cancer predictions! You may check out this shell [script](ddsm_train/train_image_clf_inbreast.sh). Alternatively, copy & paste from here:
```shell
TRAIN_DIR="INbreast/train"
VAL_DIR="INbreast/val"
TEST_DIR="INbreast/test"
RESUME_FROM="ddsm_vgg16_s10_[512-512-1024]x2_hybrid.h5"
BEST_MODEL="INbreast/transferred_inbreast_best_model.h5"
FINAL_MODEL="NOSAVE"
export NUM_CPU_CORES=4

python image_clf_train.py \
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
    --es-patience 10 \
    --auto-batch-balance \
    --best-model $BEST_MODEL \
    --final-model $FINAL_MODEL \
    $TRAIN_DIR $VAL_DIR $TEST_DIR
```
Some explanations of the arguments:
- The batch size for training is the product of `--batch-size` and `--train-bs-multiplier`. Because training uses roughtly twice (both forward and back props) the GPU memory of testing, `--train-bs-multiplier` is set to 0.5 here.
- For model finetuning, only the second stage of the two-stage training is used here. So `--nb-epoch` is set to 0.
- `--load-val-ram` and `--load-train-ram` will load the image data from the validation and train sets into memory. You may want to turn off these options if you don't have sufficient memory. When turned off, out-of-core training will be used.
- `--weight-decay` and `--hidden-dropout` are for stage 1. `--weight-decay2` and `--hidden-dropout2` are for stage 2.
- The learning rate for stage 1 is `--init-learningrate`. The learning rate for stage 2 is the product of `--init-learningrate` and `--all-layer-multiplier`.

## Computational environment
The research in this study is carried out on a Linux workstation with 8 CPU cores and a single NVIDIA Quadro M4000 GPU with 8GB memory. The deep learning framework is Keras 2 with Tensorflow as the backend. 
### About Keras version
It is known that Keras >= 2.1.0 can give errors due an API change. See issue [#7](https://github.com/lishen/end2end-all-conv/issues/7). Use Keras with version < 2.1.0. For example, Keras=2.0.8 is known to work.




