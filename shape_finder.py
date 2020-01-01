import numpy as np
from scipy import spatial
import scipy.ndimage
import scipy.signal


# In all funcs, space should be an array of 0 and 1, with 1 where a white pixel was detected in
# the original image

eight_way_connectivity_structure = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
four_way_connectivity_structure = [[0, 1, 0], [1, 1, 1], [0, 1, 0]]
interior_kernel = np.asarray([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.int8)
basic_params_dtype = {'names': ['id', 'size', 'centroid_y', 'centroid_x', 'min_pt_y', 'min_pt_x', 'max_pt_y', 'max_pt_x'], 'formats': [int, int, int, int, int, int, int, int]}


def map_ids_to_idxs(mask_with_ids):
    ar_ = mask_with_ids.flatten()
    ids, cnt = np.unique(ar_, return_counts=True)
    perm = ar_.argsort()
    if ids[0] == 0:
        s_idx = cnt[0]
        ids, cnt = ids[1:], cnt[1:]
    else:
        s_idx = 0
    idx_map = []
    for i in range(len(ids)):
        e_idx = s_idx + cnt[i]
        idx_map.append(np.unravel_index(perm[s_idx:e_idx], mask_with_ids.shape))
        s_idx = e_idx
    return ids, idx_map


def clean_up_superfluous_contours(space):
    removable_contours = find_removeable_contours(find_contours_no_id(space))
    while 1 in removable_contours:
        space[removable_contours > 0] = 0
        removable_contours = find_removeable_contours(find_contours_no_id(space))


def set_proper_contours(space, num_of_contours_to_drop=1):
    clean_up_superfluous_contours(space)
    for i in range(num_of_contours_to_drop):
        remove_contours(space)
    clean_up_superfluous_contours(space)


def find_structure(space, num_of_contours_to_drop, max_split_depth=3, min_coverage=0.6):
    space_ = np.copy(space)
    set_proper_contours(space_, num_of_contours_to_drop)
    base_shapes, _ = find_shapes_and_contours(space_, save_parameters=False)
    base_pxls = np.count_nonzero(base_shapes)
    shapes, contours, params = None, None, None
    for split_depth in range(1, max_split_depth + 1):
        current_shapes, current_contours, current_params = find_split_shapes_structure(space_, split_depth)
        shapes, contours, params = current_shapes, current_contours, current_params
        pxl_coverage = np.count_nonzero(current_shapes)
        print(f"coverage at depth {split_depth}: {pxl_coverage / base_pxls}")
        if pxl_coverage / base_pxls < min_coverage:
            break
    return shapes, contours, params


def find_split_shapes_structure(space, split_depth):
    processed_masks = []
    for mask in build_contourless_masks(space, split_depth):
        processed_masks.append(find_shapes_and_contours(mask, save_parameters=True))
    # unsplit_shapes, _, _ = processed_masks[0]
    # num_of_detected_px = np.count_nonzero(unsplit_shapes)
    # print(f"wp: {num_of_detected_px}")
    for i in range(len(processed_masks) - 1, 0, -1):
        shapes, contours, params = processed_masks[i - 1]
        subshapes, subcontours, subparams = processed_masks[i]
        split_ids = split_shape_ids(shapes, subshapes)
        split_shapes_id_array(shapes, subshapes, subcontours, split_ids)
        reassign_split_contours(contours, subparams, split_ids)
        processed_masks[i-1] = (shapes, contours, calculate_basic_params(shapes, contours))
        # coverage = np.count_nonzero(shapes)
        # print(f"coverage {i}: {coverage / num_of_detected_px}")
    return processed_masks[0]


def get_non_trivial_submask_shape_ids(mask_shape_idxs, submask_shapes, percent_of_subshape_to_include=0.01):
    shape_in_submask = submask_shapes[mask_shape_idxs]
    shape_in_submask = shape_in_submask[shape_in_submask > 0]
    submask_ids_in_shape, counts = np.unique(shape_in_submask, return_counts=True)
    nontrivial_shapes_in_submask = \
        submask_ids_in_shape[np.where(counts > percent_of_subshape_to_include * len(shape_in_submask.flatten()))]
    return nontrivial_shapes_in_submask


def find_split_shapes(mask_shapes, submask_shapes):
    mask_idx_map = map_ids_to_idxs(mask_shapes)
    split_shapes = {}
    for shape_id, shape_idxs in zip(*mask_idx_map):
        nontrivial_shapes_in_submask = get_non_trivial_submask_shape_ids(shape_idxs, submask_shapes)
        if len(nontrivial_shapes_in_submask) > 0:
            split_shapes[shape_id] = nontrivial_shapes_in_submask
    return split_shapes


def assign_new_ids(split_ids, next_id):
    new_id_mapping = {}
    for org_id in split_ids:
        subshape_ids = split_ids[org_id]
        new_id_mapping[org_id] = (subshape_ids, list(range(next_id, next_id + len(subshape_ids))))
        next_id += len(subshape_ids)
    return new_id_mapping


def split_shape_ids(shapes_mask, shapes_submask):
    split_ids = find_split_shapes(shapes_mask, shapes_submask)
    next_id = np.max(shapes_mask) + 1
    return assign_new_ids(split_ids, next_id)


def reassign_split_contours(contours, subparams, split_ids):
    old_contour_ids, old_contours_idxs = map_ids_to_idxs(contours)
    old_contour_mapping = {old_contour_ids[i] : old_contours_idxs[i] for i in range(len(old_contour_ids))}
    for old_id in split_ids.keys():
        sub_ids, new_ids = split_ids[old_id]
        updated_ids = new_ids[0] #default to one of the subshapes
        if len(new_ids) > 1:
            old_pts = list(zip(*old_contour_mapping[old_id]))
            sub_centroids = subparams[np.where(np.isin(subparams['id'], sub_ids))]
            if len(sub_centroids) > 0:
                sub_centroids = np.column_stack((sub_centroids['centroid_x'], sub_centroids['centroid_y']))
                distance_tree = spatial.cKDTree(sub_centroids)
                _, idx_of_nearest_subshape = distance_tree.query(old_pts)
                updated_ids = np.array(new_ids)[idx_of_nearest_subshape]
        contours[old_contour_mapping[old_id]] = updated_ids


def split_shapes_id_array(shapes, subshapes, subcontours, split_ids):
    shapes[np.where(np.isin(shapes, list(split_ids.keys())))] = 0
    subshape_idx_map = {id: idxs for id, idxs in zip(*map_ids_to_idxs(subshapes))}
    subcontours_idx_map = {id: idxs for id, idxs in zip(*map_ids_to_idxs(subcontours))}
    for org_id in split_ids:
        subshape_ids, new_ids = split_ids[org_id]
        for subshape_id, new_id in zip(subshape_ids, new_ids):
            shapes[subshape_idx_map[subshape_id]] = new_id
            shapes[subcontours_idx_map[subshape_id]] = new_id


def remove_contours_from_copy(space):
    space_ = np.copy(space)
    remove_contours(space_)
    return space_


def remove_contours(space):
    space[find_contours_no_id(space) > 0] = 0


def build_contourless_masks(space, split_depth):
    last = np.copy(space)
    for i in range(1 + split_depth):
        yield last
        last = remove_contours_from_copy(last)


def find_shapes_and_contours(space, save_parameters=False):
    shapes = label_shapes(space)
    unassigned_contours = find_contours_no_id(space)
    contours = shapes * unassigned_contours
    if save_parameters:
        return shapes, contours, calculate_basic_params(shapes, contours)
    else:
        return shapes, contours


def label_shapes(space, cc_structure=eight_way_connectivity_structure):
    labeled, _ = scipy.ndimage.measurements.label(space, structure=cc_structure)
    return labeled


def find_contours_no_id(space):
    inverted_space = np.zeros(shape=space.shape, dtype=np.int)
    inverted_space[space == 0] = 1
    convolution_result = scipy.signal.convolve(inverted_space, interior_kernel, mode='same')
    convolution_result[convolution_result > 1] = 1
    convolution_result = convolution_result - inverted_space
    convolution_result[convolution_result < 0] = 0
    return convolution_result


def convolve_space_with_kernel_threshold(space, kernel, threshold):
    convolution_result = scipy.signal.convolve(space, kernel, mode='same')
    convolution_result[convolution_result <= threshold] = 0
    convolution_result[convolution_result > 0] = 1
    return convolution_result


def find_superfluous_contours(contours):
    superfluous_contours_kernel = np.asarray([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
    return convolve_space_with_kernel_threshold(contours, superfluous_contours_kernel, 7)
    # convolution_result = scipy.signal.convolve(contours, superfluous_contours_kernel, mode='same')
    # convolution_result[convolution_result <= 7] = 0
    # convolution_result[convolution_result > 0] = 1
    # return convolution_result


def find_single_pxl_contours(superfluous_contours):
    single_pxl_contours_kernel = np.asarray([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    return convolve_space_with_kernel_threshold(superfluous_contours, single_pxl_contours_kernel, 0)



def find_removeable_contours(contours):
    superfluous_contours = find_superfluous_contours(contours)
    single_pxl_contours = find_single_pxl_contours(superfluous_contours)
    superfluous_contours[single_pxl_contours > 0] = 1
    return superfluous_contours


def calculate_basic_params(shapes, contours):
    parameters = []
    shape_ids, shape_idxs = map_ids_to_idxs(shapes)
    contour_ids, contour_idxs = map_ids_to_idxs(contours)
    shape_sizes_ids, sizes = np.unique(shapes[shapes > 0], return_counts=True)
    # assume that each shape has interior and contour
    for i in range(len(shape_sizes_ids)):
        shape_id = shape_sizes_ids[i]
        # print(shape_id)
        shape = shape_idxs[np.where(shape_ids == shape_id)[0][0]]
        contour = np.asarray(list(zip(*contour_idxs[np.where(contour_ids == shape_id)[0][0]])))
        size = sizes[i]
        x_s, y_s = shape
        centroid = (sum(x_s) / size, sum(y_s) / size)
        distances = np.linalg.norm(contour - centroid, axis=1)
        extreme_pts = (contour[distances.argmin()], contour[distances.argmax()])
        extreme_pts = [coord for pt in list(map(list, extreme_pts)) for coord in pt]
        parameters.append((shape_id, size, *centroid[::-1], *extreme_pts[::-1]))
    return np.array(parameters, dtype=basic_params_dtype)