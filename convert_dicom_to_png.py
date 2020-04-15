import cv2 as cv
import os
import pydicom

src_data_dir = "/fred/oz121/CBIS-DDSM/"
dst_data_dir = "/fred/oz121/CBIS-DDSM-png/"
if not os.path.exists(dst_data_dir):
    os.mkdir(dst_data_dir)


for cur_dir, sub_folders, file_names in os.walk(src_data_dir):
    # print(cur_dir)
    # print(sub_folders)
    # print(file_names)
    # print()
    for file_name in file_names:
        if file_name.endswith(".dcm"):
            src_file_path = os.path.join(cur_dir, file_name)
            image = pydicom.read_file(src_file_path).pixel_array  # read dicom image and get image array
            dst_file_path = src_file_path.replace(src_data_dir, dst_data_dir).replace(".dcm", ".png")
            dst_dir = os.path.dirname(dst_file_path)
            if not os.path.exists(dst_dir):
                os.makedirs(dst_dir)
            print(src_file_path)
            print(image.shape)
            print(dst_file_path)
            print()
            # cv.imwrite(src_file_path[:-3]+"png", image)  # write png image
