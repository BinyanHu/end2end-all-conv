import argparse

from keras.models import load_model

from dm_image import read_resize_img, read_img_for_pred

parser = argparse.ArgumentParser(description="DM image clf testing")
parser.add_argument("resume_from", type=str)
parser.add_argument("image_path", type=str)
# parser.add_argument("--rescale-factor", dest="rescale_factor", type=float, default=None)
# parser.add_argument("--featurewise-mean", dest="featurewise_mean", type=float, default=52.16)

ARGS = parser.parse_args()
image_path = ARGS.image_path
resume_from = ARGS.resume_from
rescale_factor = 0.003891
featurewise_mean = 52.18

print "Load saved model:", resume_from + '.',
print "Done."

patch_model = load_model(resume_from, compile=False)


image = read_img_for_pred(
    image_path,
    equalize_hist=False,
    data_format="default",
    dup_3_channels=False,
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

image -= featurewise_mean

pred = image.predict(image)

print pred
