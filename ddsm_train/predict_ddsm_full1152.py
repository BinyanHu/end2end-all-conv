import argparse
import os

from keras.models import load_model
import numpy as np

from dm_image import read_img_for_pred

parser = argparse.ArgumentParser(description="DM image clf testing")
parser.add_argument("--image_path", type=str, default=None)
parser.add_argument("--resume_from", type=str, default="saved_model/ddsm_full1152/resnet_1152x896_prt_addtop1_best.h5")
# parser.add_argument("--rescale-factor", dest="rescale_factor", type=float, default=None)
# parser.add_argument("--featurewise-mean", dest="featurewise_mean", type=float, default=52.16)

ARGS = parser.parse_args()
image_path = ARGS.image_path
resume_from = ARGS.resume_from
rescale_factor = 0.003891
featurewise_mean = 52.18

print "Load saved model:", resume_from + '.'

model = load_model(resume_from)

print "Done."

if image_path is None:
    num_samples = 0
    num_correct = 0
    image_dir = "data/curated_breast_imaging_ddsm/Combined_full_images/full_test_1152x896/pos"
    for image_file_name in os.listdir(image_dir):
        image_path = os.path.join(image_dir, image_file_name)
        image = read_img_for_pred(
            image_path,
            equalize_hist=False,
            data_format="default",
            dup_3_channels=True,
            target_size=(1152, 896),
            target_scale=None,
            gs_255=None,
            rescale_factor=rescale_factor
        )

        # image = read_resize_img(
        #     ARGS.image_path,
        #     (1152, 896),
        #     target_scale=4095,
        #     gs_255=False
        # )

        print "On reading", image.mean(), "+-", image.std()

        image -= featurewise_mean

        print "On centering", image.mean(), "+-", image.std()

        images = image[None, ...]
        probs = model.predict(images)
        class_ = np.argmax(probs[0])

        print "prediction:", class_
        print ""

        num_samples += 1
        if class_ == 1:
            num_correct += 1

    print num_correct, "/", num_samples
    print "Total accuracy:", float(num_correct) / float(num_samples)
