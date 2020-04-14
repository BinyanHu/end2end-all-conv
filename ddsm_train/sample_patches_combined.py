import argparse
import os
import sys

import cv2
import numpy as np
import pandas as pd
import pydicom as dicom
from sklearn.model_selection import train_test_split

from dm_image import add_img_margins, crop_img, read_resize_img
from dm_preprocess import DMImagePreprocessor as imprep
from pilutil import toimage

#### Define some functions to use ####


def const_filename(pat, side, view, directory, abn_type, abn_id=None):
    # P_00001, LEFT, CC
    token_list = []
    token_list.append(('Calc' if abn_type == 'calc' else 'Mass') + '-Training')
    token_list.extend([pat, side, view])
    if abn_id:
        token_list.append(str(abn_id))
    folder_name = "_".join(token_list)
    directory = os.path.join(directory, folder_name)
    for cur_dir, sub_folders, file_names in os.walk(directory):
        if len(file_names) > 1:
            return os.path.join(cur_dir, "000001.png")
        else:
            for file_name in file_names:
                if file_name.endswith(".png"):
                    return os.path.join(cur_dir, file_name)
    raise AssertionError("Could not find the png file at {!r}.".format(directory))


def crop_val(v, minv, maxv):
    v = v if v >= minv else minv
    v = v if v <= maxv else maxv
    return v


def overlap_patch_roi(patch_center, patch_size, roi_mask,
                      add_val=1000, cutoff=.5):
    x1, y1 = (patch_center[0] - patch_size/2,
              patch_center[1] - patch_size/2)
    x2, y2 = (patch_center[0] + patch_size/2,
              patch_center[1] + patch_size/2)
    x1 = crop_val(x1, 0, roi_mask.shape[1])
    y1 = crop_val(y1, 0, roi_mask.shape[0])
    x2 = crop_val(x2, 0, roi_mask.shape[1])
    y2 = crop_val(y2, 0, roi_mask.shape[0])
    roi_area = (roi_mask > 0).sum()
    roi_patch_added = roi_mask.copy()
    roi_patch_added[y1:y2, x1:x2] += add_val
    patch_area = (roi_patch_added >= add_val).sum()
    inter_area = (roi_patch_added > add_val).sum().astype('float32')
    return (inter_area/roi_area > cutoff or inter_area/patch_area > cutoff)


def create_blob_detector(roi_size=(128, 128), blob_min_area=3,
                         blob_min_int=.5, blob_max_int=.95, blob_th_step=10):
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = blob_min_area
    params.maxArea = roi_size[0]*roi_size[1]
    params.filterByCircularity = False
    params.filterByColor = False
    params.filterByConvexity = False
    params.filterByInertia = False
    # blob detection only works with "uint8" images.
    params.minThreshold = int(blob_min_int*255)
    params.maxThreshold = int(blob_max_int*255)
    params.thresholdStep = blob_th_step
    ver = (cv2.__version__).split('.')
    if int(ver[0]) < 3:
        return cv2.SimpleBlobDetector(params)
    else:
        return cv2.SimpleBlobDetector_create(params)


def sample_patches(img, roi_mask, out_dir, img_id, abn_id, pos, patch_size=256,
                   pos_cutoff=.75, neg_cutoff=.35,
                   nb_bkg=100, nb_abn=100, start_sample_nb=0, abn_type='calc',
                   bkg_dir='background',
                   calc_pos_dir='calc_mal', calc_neg_dir='calc_ben',
                   mass_pos_dir='mass_mal', mass_neg_dir='mass_ben',
                   verbose=False):
    if pos:
        if abn_type == 'calc':
            roi_out = os.path.join(out_dir, calc_pos_dir)
        else:
            roi_out = os.path.join(out_dir, mass_pos_dir)
    else:
        if abn_type == 'calc':
            roi_out = os.path.join(out_dir, calc_neg_dir)
        else:
            roi_out = os.path.join(out_dir, mass_neg_dir)
    bkg_out = os.path.join(out_dir, bkg_dir)
    basename = '_'.join([img_id, str(abn_id)])

    img = add_img_margins(img, patch_size/2)
    roi_mask = add_img_margins(roi_mask, patch_size/2)
    # Get ROI bounding box.
    roi_mask_8u = roi_mask.astype('uint8')
    ver = (cv2.__version__).split('.')
    contours, _ = cv2.findContours(roi_mask_8u.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cont_areas = [cv2.contourArea(cont) for cont in contours]
    idx = np.argmax(cont_areas)  # find the largest contour.
    rx, ry, rw, rh = cv2.boundingRect(contours[idx])
    if verbose:
        M = cv2.moments(contours[idx])
        try:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            print "ROI centroid=", (cx, cy)
            sys.stdout.flush()
        except ZeroDivisionError:
            cx = rx + int(rw/2)
            cy = ry + int(rh/2)
            print "ROI centroid=Unknown, use b-box center=", (cx, cy)
            sys.stdout.flush()

    rng = np.random.RandomState(12345)
    # Sample abnormality first.
    sampled_abn = 0
    nb_try = 0
    while sampled_abn < nb_abn:
        filename = basename + "_%04d" % (sampled_abn) + ".png"
        fullname = os.path.join(roi_out, filename)
        if os.path.exists(fullname):
            print "already exists:", fullname
            sampled_abn += 1
            continue

        if nb_abn > 1:
            x = rng.randint(rx, rx + rw)
            y = rng.randint(ry, ry + rh)
            nb_try += 1
            if nb_try >= 1000:
                print "Nb of trials reached maximum, decrease overlap cutoff by 0.05"
                sys.stdout.flush()
                pos_cutoff -= .05
                nb_try = 0
                if pos_cutoff <= .0:
                    raise Exception("overlap cutoff becomes non-positive, "
                                    "check roi mask input.")
        else:
            x = cx
            y = cy

        # import pdb; pdb.set_trace()
        if nb_abn == 1 or overlap_patch_roi((x, y), patch_size, roi_mask,
                                            cutoff=pos_cutoff):
            patch = img[y - patch_size/2:y + patch_size/2,
                        x - patch_size/2:x + patch_size/2]
            patch = patch.astype('int32')
            patch_img = toimage(patch, high=patch.max(), low=patch.min(),
                                mode='I')
            # patch = patch.reshape((patch.shape[0], patch.shape[1], 1))
            # import pdb; pdb.set_trace()
            patch_img.save(fullname)
            sampled_abn += 1
            nb_try = 0
            if verbose:
                print "sampled an", abn_id, "patch at (x,y) center=", (x, y)
                sys.stdout.flush()

    # Sample background.
    sampled_bkg = start_sample_nb
    while sampled_bkg < start_sample_nb + nb_bkg:
        filename = basename + "_%04d" % (sampled_bkg) + ".png"
        fullname = os.path.join(bkg_out, filename)
        if os.path.exists(fullname):
            print "already exists:", fullname
            sampled_bkg += 1
            continue

        x = rng.randint(patch_size/2, img.shape[1] - patch_size/2)
        y = rng.randint(patch_size/2, img.shape[0] - patch_size/2)
        if not overlap_patch_roi((x, y), patch_size, roi_mask, cutoff=neg_cutoff):
            patch = img[y - patch_size/2:y + patch_size/2,
                        x - patch_size/2:x + patch_size/2]
            patch = patch.astype('int32')
            patch_img = toimage(patch, high=patch.max(), low=patch.min(),
                                mode='I')
            patch_img.save(fullname)
            sampled_bkg += 1
            if verbose:
                print "sampled a bkg patch at (x,y) center=", (x, y)
                sys.stdout.flush()


def sample_hard_negatives(img, roi_mask, out_dir, img_id, abn_id,
                          patch_size=256, neg_cutoff=.35, nb_bkg=100,
                          start_sample_nb=0,
                          bkg_dir='background', verbose=False):
    '''WARNING: the definition of hns may be problematic.
    There has been study showing that the context of an ROI is also useful
    for classification.
    '''
    bkg_out = os.path.join(out_dir, bkg_dir)
    basename = '_'.join([img_id, str(abn_id)])

    img = add_img_margins(img, patch_size/2)
    roi_mask = add_img_margins(roi_mask, patch_size/2)
    # Get ROI bounding box.
    roi_mask_8u = roi_mask.astype('uint8')
    ver = (cv2.__version__).split('.')
    contours, _ = cv2.findContours(roi_mask_8u.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cont_areas = [cv2.contourArea(cont) for cont in contours]
    idx = np.argmax(cont_areas)  # find the largest contour.
    rx, ry, rw, rh = cv2.boundingRect(contours[idx])
    if verbose:
        M = cv2.moments(contours[idx])
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        print "ROI centroid=", (cx, cy)
        sys.stdout.flush()

    rng = np.random.RandomState(12345)
    # Sample hard negative samples.
    sampled_bkg = start_sample_nb
    while sampled_bkg < start_sample_nb + nb_bkg:
        x1, x2 = (rx - patch_size/2, rx + rw + patch_size/2)
        y1, y2 = (ry - patch_size/2, ry + rh + patch_size/2)
        x1 = crop_val(x1, patch_size/2, img.shape[1] - patch_size/2)
        x2 = crop_val(x2, patch_size/2, img.shape[1] - patch_size/2)
        y1 = crop_val(y1, patch_size/2, img.shape[0] - patch_size/2)
        y2 = crop_val(y2, patch_size/2, img.shape[0] - patch_size/2)
        x = rng.randint(x1, x2)
        y = rng.randint(y1, y2)
        if not overlap_patch_roi((x, y), patch_size, roi_mask, cutoff=neg_cutoff):
            patch = img[y - patch_size/2:y + patch_size/2,
                        x - patch_size/2:x + patch_size/2]
            patch = patch.astype('int32')
            patch_img = toimage(patch, high=patch.max(), low=patch.min(),
                                mode='I')
            filename = basename + "_%04d" % (sampled_bkg) + ".png"
            fullname = os.path.join(bkg_out, filename)
            patch_img.save(fullname)
            sampled_bkg += 1
            if verbose:
                print "sampled a hns patch at (x,y) center=", (x, y)
                sys.stdout.flush()


def sample_blob_negatives(img, roi_mask, out_dir, img_id, abn_id, blob_detector,
                          patch_size=256, neg_cutoff=.35, nb_bkg=100,
                          start_sample_nb=0,
                          bkg_dir='background', verbose=False):
    bkg_out = os.path.join(out_dir, bkg_dir)
    basename = '_'.join([img_id, str(abn_id)])

    img = add_img_margins(img, patch_size/2)
    roi_mask = add_img_margins(roi_mask, patch_size/2)
    # Get ROI bounding box.
    roi_mask_8u = roi_mask.astype('uint8')
    ver = (cv2.__version__).split('.')
    contours, _ = cv2.findContours(roi_mask_8u.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cont_areas = [cv2.contourArea(cont) for cont in contours]
    idx = np.argmax(cont_areas)  # find the largest contour.
    rx, ry, rw, rh = cv2.boundingRect(contours[idx])
    if verbose:
        M = cv2.moments(contours[idx])
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        print "ROI centroid=", (cx, cy)
        sys.stdout.flush()

    # Sample blob negative samples.
    key_pts = blob_detector.detect((img/img.max()*255).astype('uint8'))
    rng = np.random.RandomState(12345)
    key_pts = rng.permutation(key_pts)
    sampled_bkg = 0
    for kp in key_pts:
        if sampled_bkg >= nb_bkg:
            break
        x, y = int(kp.pt[0]), int(kp.pt[1])
        if not overlap_patch_roi((x, y), patch_size, roi_mask, cutoff=neg_cutoff):
            patch = img[y - patch_size/2:y + patch_size/2,
                        x - patch_size/2:x + patch_size/2]
            patch = patch.astype('int32')
            patch_img = toimage(patch, high=patch.max(), low=patch.min(),
                                mode='I')
            filename = basename + "_%04d" % (start_sample_nb + sampled_bkg) + ".png"
            fullname = os.path.join(bkg_out, filename)
            patch_img.save(fullname)
            if verbose:
                print "sampled a blob patch at (x,y) center=", (x, y)
                sys.stdout.flush()
            sampled_bkg += 1
    return sampled_bkg

#### End of function definition ####


def run(description_path, roi_mask_dir, image_dir,
        train_out_dir, val_out_dir,
        target_height=4096, target_width=None, patch_size=256,
        segment_breast=True,
        nb_bkg=30, nb_abn=30, nb_hns=15,
        pos_cutoff=.75, neg_cutoff=.35, val_size=.1,
        bkg_dir='background', calc_pos_dir='calc_mal', calc_neg_dir='calc_ben',
        mass_pos_dir='mass_mal', mass_neg_dir='mass_ben', verbose=True):

    # Print info for book-keeping.
    print "Pathology file=", description_path
    print "ROI mask dir=", roi_mask_dir
    print "Full image dir=", image_dir
    print "Train out dir=", train_out_dir
    print "Val out dir=", val_out_dir
    print "==="
    sys.stdout.flush()

    # Read ROI mask table with pathology.
    description_df = pd.read_csv(description_path, header=0)
    patient_ids = description_df["patient_id"]

    description_df = description_df.set_index(['patient_id', 'left or right breast', 'image view'])
    description_df.sort_index(inplace=True)

    # Read train set patient IDs and subset the table.
    patient_ids = patient_ids.values.ravel()
    if len(patient_ids) > 1:
        path_df = description_df.loc[patient_ids.tolist()]
    else:
        locs = description_df.index.get_loc(patient_ids[0])
        path_df = description_df.iloc[locs]

    # Determine the labels for patients.
    pat_labs = []
    for pat in patient_ids:
        pathologies = path_df.loc[pat]['pathology']
        malignant = 0
        for pathology in pathologies:
            if pathology.startswith('MALIGNANT'):
                malignant = 1
                break
        pat_labs.append(malignant)

    # Split patient list into train and val lists.
    def write_pat_list(fn, pat_list):
        with open(fn, 'w') as f:
            for pat in pat_list:
                f.write(str(pat) + "\n")
            f.close()

    if val_size > 0:
        # import pdb; pdb.set_trace()
        patient_ids, pat_val, labs_train, labs_val = train_test_split(
            patient_ids, pat_labs, stratify=pat_labs, test_size=val_size,
            random_state=12345)
        if len(pat_val) > 1:
            val_df = description_df.loc[pat_val.tolist()]
        else:
            locs = description_df.index.get_loc(pat_val[0])
            val_df = description_df.iloc[locs]
        if not os.path.exists(val_out_dir):
            os.makedirs(val_out_dir)
        write_pat_list(os.path.join(val_out_dir, 'pat_lst.txt'), pat_val.tolist())
    if len(patient_ids) > 1:
        train_df = description_df.loc[patient_ids.tolist()]
    else:
        locs = description_df.index.get_loc(patient_ids[0])
        train_df = description_df.iloc[locs]
    if not os.path.exists(train_out_dir):
        os.makedirs(train_out_dir)
    write_pat_list(os.path.join(train_out_dir, 'pat_lst.txt'), patient_ids.tolist())
    # Create a blob detector.
    blob_detector = create_blob_detector(roi_size=(patch_size, patch_size))

    #### Define a functin to sample patches.
    def do_sampling(pat_df, out_dir):
        for pat, side, view in pat_df.index.unique():
            cur_desc = description_df.loc[pat].loc[side].loc[view]
            abn_ids = cur_desc['abnormality id']
            pathologies = cur_desc['pathology']
            abn_types = cur_desc['abnormality type']
            if isinstance(cur_desc, pd.Series):
                abn_ids = [abn_ids]
                pathologies = [pathologies]
                abn_types = [abn_types]

            # Read mask image(s).
            bkg_sampled = False
            for abn_id, pathology, abn_type in zip(abn_ids, pathologies, abn_types):
                # NOTE csv not reliable due to formatting error.
                # image_path = cur_desc["image file path"]
                # mask_path = cur_desc["ROI mask file path"]
                image_path = const_filename(pat, side, view, image_dir, abn_type)
                mask_path = const_filename(pat, side, view, roi_mask_dir, abn_type, abn_id)

                # import pdb; pdb.set_trace()
                if target_width is None:
                    full_img = read_resize_img(image_path, target_height=target_height)
                else:
                    full_img = read_resize_img(image_path, target_size=(target_height, target_width))
                img_id = '_'.join([pat, side, view])
                print "ID:%s, read image of size=%s" % (img_id, full_img.shape),
                if segment_breast:
                    full_img, bbox = imprep.segment_breast(full_img)
                    print "size after segmentation=%s" % (str(full_img.shape))
                sys.stdout.flush()

                if target_width is None:
                    mask_img = read_resize_img(mask_path, target_height=target_height, gs_255=True)
                else:
                    mask_img = read_resize_img(mask_path, target_size=(target_height, target_width), gs_255=True)
                if segment_breast:
                    mask_img = crop_img(mask_img, bbox)
                # sample using mask and full image.
                nb_hns_ = nb_hns if not bkg_sampled else 0
                if nb_hns_ > 0:
                    hns_sampled = sample_blob_negatives(
                        full_img, mask_img, out_dir, img_id,
                        abn_id, blob_detector, patch_size, neg_cutoff,
                        nb_hns_, 0, bkg_dir, verbose)
                else:
                    hns_sampled = 0
                pos = pathology.startswith('MALIGNANT')
                nb_bkg_ = nb_bkg - hns_sampled if not bkg_sampled else 0
                sample_patches(full_img, mask_img, out_dir, img_id, abn_id, pos,
                               patch_size, pos_cutoff, neg_cutoff,
                               nb_bkg_, nb_abn, hns_sampled, abn_type,
                               bkg_dir, calc_pos_dir, calc_neg_dir,
                               mass_pos_dir, mass_neg_dir, verbose)
                bkg_sampled = True

    #####
    print "Sampling for train set"
    sys.stdout.flush()
    do_sampling(train_df, train_out_dir)
    print "Done."
    #####
    if val_size > 0.:
        print "Sampling for val set"
        sys.stdout.flush()
        do_sampling(val_df, val_out_dir)
        print "Done."


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Sample patches for DDSM images")
    parser.add_argument("description_path", type=str)
    parser.add_argument("roi_mask_dir", type=str)
    parser.add_argument("image_dir", type=str)
    parser.add_argument("train_out_dir", type=str)
    parser.add_argument("val_out_dir", type=str)
    parser.add_argument("--target-height", dest="target_height", type=int, default=4096)
    parser.add_argument("--target-width", dest="target_width", type=int, default=None)
    parser.add_argument("--no-target-width", dest="target_width", action="store_const", const=None)
    parser.add_argument("--segment-breast", dest="segment_breast", action="store_true")
    parser.add_argument("--no-segment-breast", dest="segment_breast", action="store_false")
    parser.set_defaults(segment_breast=True)
    parser.add_argument("--patch-size", dest="patch_size", type=int, default=256)
    parser.add_argument("--nb-bkg", dest="nb_bkg", type=int, default=30)
    parser.add_argument("--nb-abn", dest="nb_abn", type=int, default=30)
    parser.add_argument("--nb-hns", dest="nb_hns", type=int, default=15)
    parser.add_argument("--pos-cutoff", dest="pos_cutoff", type=float, default=.75)
    parser.add_argument("--neg-cutoff", dest="neg_cutoff", type=float, default=.35)
    parser.add_argument("--val-size", dest="val_size", type=float, default=.1)
    parser.add_argument("--bkg-dir", dest="bkg_dir", type=str, default="background")
    parser.add_argument("--calc-pos-dir", dest="calc_pos_dir", type=str, default="calc_mal")
    parser.add_argument("--calc-neg-dir", dest="calc_neg_dir", type=str, default="calc_ben")
    parser.add_argument("--mass-pos-dir", dest="mass_pos_dir", type=str, default="mass_mal")
    parser.add_argument("--mass-neg-dir", dest="mass_neg_dir", type=str, default="mass_ben")
    parser.add_argument("--verbose", dest="verbose", action="store_true")
    parser.add_argument("--no-verbose", dest="verbose", action="store_false")
    parser.set_defaults(verbose=True)

    args = parser.parse_args()
    run_opts = dict(
        target_height=args.target_height,
        target_width=args.target_width,
        segment_breast=args.segment_breast,
        patch_size=args.patch_size,
        nb_bkg=args.nb_bkg,
        nb_abn=args.nb_abn,
        nb_hns=args.nb_hns,
        pos_cutoff=args.pos_cutoff,
        neg_cutoff=args.neg_cutoff,
        val_size=args.val_size,
        bkg_dir=args.bkg_dir,
        calc_pos_dir=args.calc_pos_dir,
        calc_neg_dir=args.calc_neg_dir,
        mass_pos_dir=args.mass_pos_dir,
        mass_neg_dir=args.mass_neg_dir,
        verbose=args.verbose
    )
    print "\n>>> Model training options: <<<\n", run_opts, "\n"
    run(args.description_path, args.roi_mask_dir,
        args.image_dir, args.train_out_dir, args.val_out_dir, **run_opts)
