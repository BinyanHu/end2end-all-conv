# Summary of Issues

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
h5py
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

   

2. `scipy.misc.toimage` is deprecated and has been removed for a long time. We took the file `pilutil.py` and placed it in the project directory, so that we can replace:

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

   Issues caused by the file path inconsistences are also found in [tensorflow_datasets curated_breast_imaging_ddsm](https://www.tensorflow.org/datasets/catalog/curated_breast_imaging_ddsm). 

   Therefore, we abandoned the function `const_filename` and wrote another function `get_image_and_mask` to automatically search the file path, match and load the image files.



## Issues NOT Resolved

1. The original code does not generate test set, but test set is used in the demos. If you hope to use test sets, modify [sample_patches_combined.py](ddsm_train\sample_patches_combined.py) to implement test set splitting.