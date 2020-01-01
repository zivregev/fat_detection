import cv2
import numpy as np
import shape_finder
import classifier

param_dtype = {'names': ['id', 'size', 'centroid_y', 'centroid_x', 'min_pt_y', 'min_pt_x', 'max_pt_y', 'max_pt_x', 'skew'],
               'formats': [int, int, int, int, int, int, int, int, float]}
    # np.dtype([('id', int), ('size', int), ('centroid_y', int), ('centroid_x', int), ('min_pt_y', int), ('min_pt_x', int), ('max_pt_y', int), ('max_pt_x', int), ('skew', float)])


threshold_max = 255
threshold_min = 0
threshold_step = 5


def setup_img(img, use_otsu=True, num_of_contours_to_drop=2):
    threshold = None
    if not use_otsu:
        threshold = choose_threshold(img)
    return shape_finder.find_structure(setup_mask(img, threshold, use_otsu=use_otsu), num_of_contours_to_drop=num_of_contours_to_drop)


def choose_threshold(img):
    best_threshold = None
    best_num_of_features = 0
    for threshold in range(threshold_min, threshold_max, threshold_step):
        mask = setup_mask(img, threshold, use_otsu=False)
        num_of_features = len(np.unique(shape_finder.label_shapes(mask)))
        if num_of_features > best_num_of_features:
            best_num_of_features = num_of_features
            best_threshold = threshold
    return best_threshold


def setup_mask(img, threshold, use_otsu):
    bw_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if use_otsu:
        _, bw_mask = cv2.threshold(bw_img, 0, True, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        _, bw_mask = cv2.threshold(bw_img, threshold, True, cv2.THRESH_BINARY_INV)
    corrected_mask = np.zeros(shape=bw_mask.shape, dtype=np.int)
    corrected_mask[np.where(bw_mask == 0)] = 1
    return corrected_mask


def guess_shapes(img):
    shapes, contours, params = setup_img(img)
    clx = classifier.drop_skewed_shapes(params, np.concatenate(classifier.drop_larger_shapes(params)))
    return shapes, contours, params, clx


def get_num_of_white_pxls(img):
    return np.count_nonzero(setup_mask(img, threshold=None, use_otsu=True))

img_file = "imgs/010007.tif"
# img_file = "imgs/edited.png"
img = cv2.imread(img_file)
out_name = "split_twice"
out_file = f"imgs/out/{out_name}.png"
# process_img = timer(process_img)
# remove_contours = timer(remove_contours)
# find_shapes_and_contours = timer(find_shapes_and_contours)
# color_shapes_and_contours = timer(color_shapes_and_contours)
# find_centroids = timer