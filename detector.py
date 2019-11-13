import cv2
import numpy as np
import scipy.signal
from scipy import spatial
from matplotlib import pyplot as plt
import itertools
import time
import enum
import sklearn.neighbors

basic_params_dtype = {'names': ['id', 'size', 'centroid_y', 'centroid_x', 'min_pt_y', 'min_pt_x', 'max_pt_y', 'max_pt_x'], 'formats': [int, int, int, int, int, int, int, int]}

param_dtype = {'names': ['id', 'size', 'centroid_y', 'centroid_x', 'min_pt_y', 'min_pt_x', 'max_pt_y', 'max_pt_x', 'skew'],
               'formats': [int, int, int, int, int, int, int, int, float]}
    # np.dtype([('id', int), ('size', int), ('centroid_y', int), ('centroid_x', int), ('min_pt_y', int), ('min_pt_x', int), ('max_pt_y', int), ('max_pt_x', int), ('skew', float)])

total_time = {}


def timer(func):
    total_time[func] = 0
    def _inner(*args, **kwargs):
        start = time.time()
        res = func(*args, **kwargs)
        end = time.time()
        total_time[func] += end - start
        # print(f"{func.__name__} elapsed: {end - start}")
        return res
    return _inner


def show_img(img):
    dpi = 80
    height, width, *_ = img.shape
    figsize = width/float(dpi), height/float(dpi)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.imshow(img, cmap="gray")
    plt.show()
    plt.close(fig)


def save_fig(img, name):
    dpi = 80
    height, width, *_ = img.shape
    figsize = width / float(dpi), height / float(dpi)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.imshow(img, cmap="gray")
    plt.savefig(name)
    plt.close(fig)


def get_px_neighbours(px, bounds=None):
    deltas = [-1, 0, 1]
    for d_y, d_x in itertools.product(deltas, deltas):
        y = px[0] + d_y
        x = px[1] + d_x
        if (not bounds) or (0 <= y < bounds[0] and 0 <= x < bounds[1]):
            yield (y, x)


@timer
def find_centroids(shapes, num_of_shapes):
    centroid_mask = np.zeros(shapes.shape[:2], dtype=int)
    for shape in range(1, num_of_shapes+1):
        pts = np.argwhere(shapes == shape)
        x_center, y_center = map(int, pts.sum(0)/len(pts))
        centroid_mask[int(x_center), int(y_center)] = shape
    return centroid_mask


@timer
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


@timer
def find_split_shapes_structure(bw_mask, num_of_splitting_passes, num_of_contours_to_drop):
    processed_masks = []
    for mask in build_contourless_masks(bw_mask, num_of_splitting_passes, num_of_contours_to_drop):
        processed_masks.append(find_shapes_and_contours(mask, save_shape_parameters=True))
    for i in range(len(processed_masks) - 1, 0, -1):
        shapes, contours, params = processed_masks[i - 1]
        subshapes, subcontours, subparams = processed_masks[i]
        split_ids = split_shape_ids(shapes, subshapes)
        split_shapes_id_array(shapes, subshapes, subcontours, split_ids)
        reassign_split_contours(contours, subparams, split_ids)
        processed_masks[i-1] = (shapes, contours, calculate_basic_params(shapes, contours))
    return processed_masks[0]


@timer
def get_non_trivial_submask_shape_ids(mask_shape_idxs, submask_shapes, percent_of_subshape_to_include=0.01):
    shape_in_submask = submask_shapes[mask_shape_idxs]
    shape_in_submask = shape_in_submask[shape_in_submask > 0]
    submask_ids_in_shape, counts = np.unique(shape_in_submask, return_counts=True)
    nontrivial_shapes_in_submask = \
        submask_ids_in_shape[np.where(counts > percent_of_subshape_to_include * len(shape_in_submask.flatten()))]
    return nontrivial_shapes_in_submask


@timer
def find_split_shapes(mask_shapes, submask_shapes):
    mask_idx_map = map_ids_to_idxs(mask_shapes)
    split_shapes = {}
    for shape_id, shape_idxs in zip(*mask_idx_map):
        nontrivial_shapes_in_submask = get_non_trivial_submask_shape_ids(shape_idxs, submask_shapes)
        if len(nontrivial_shapes_in_submask) > 0:
            split_shapes[shape_id] = nontrivial_shapes_in_submask
    return split_shapes


@timer
def assign_new_ids(split_ids, next_id):
    new_id_mapping = {}
    for org_id in split_ids:
        subshape_ids = split_ids[org_id]
        new_id_mapping[org_id] = (subshape_ids, list(range(next_id, next_id + len(subshape_ids))))
        next_id += len(subshape_ids)
    return new_id_mapping


@timer
def split_shape_ids(shapes_mask, shapes_submask):
    split_ids = find_split_shapes(shapes_mask, shapes_submask)
    next_id = np.max(shapes_mask) + 1
    return assign_new_ids(split_ids, next_id)


# @timer
# def split_id_based_array(ar, sub_ar, split_ids):
#     ar[np.where(np.isin(ar, list(split_ids.keys())))] = 0
#     sub_ar_idx_map = {id: idxs for id, idxs in zip(*map_ids_to_idxs(sub_ar))}
#     for org_id in split_ids:
#         subshape_ids, new_ids = split_ids[org_id]
#         for subshape_id, new_id in zip(subshape_ids, new_ids):
#             ar[sub_ar_idx_map[subshape_id]] = new_id


@timer
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


@timer
def split_shapes_id_array(shapes, subshapes, subcontours, split_ids):
    shapes[np.where(np.isin(shapes, list(split_ids.keys())))] = 0
    subshape_idx_map = {id: idxs for id, idxs in zip(*map_ids_to_idxs(subshapes))}
    subcontours_idx_map = {id: idxs for id, idxs in zip(*map_ids_to_idxs(subcontours))}
    for org_id in split_ids:
        subshape_ids, new_ids = split_ids[org_id]
        for subshape_id, new_id in zip(subshape_ids, new_ids):
            shapes[subshape_idx_map[subshape_id]] = new_id
            shapes[subcontours_idx_map[subshape_id]] = new_id


@timer
def color_centroids(img, centroids):
    idxs = [idx for i in [0, 1, 2] for idx in list(centroids.values())]
    x_s = [idx[1] for idx in list(centroids.values())]
    y_s = [idx[0] for idx in list(centroids.values())]
    z_s = [0, 1, 2]
    img[y_s, x_s] = [255, 0, 0]


@timer
def color_contour_extremes(img, shape_params):
    for i in shape_params['id']:
        color_single_shape_extremes(img, shape_params, i)


@timer
def color_single_shape_extremes(img, params, shape_num):
    shape_params = params[np.where(params['id'] == shape_num)]
    centroid = (shape_params['centroid_y'], shape_params['centroid_x'])
    closest_pt = (shape_params['min_pt_y'], shape_params['min_pt_x'])
    farthest_pt = (shape_params['max_pt_y'], shape_params['max_pt_x'])
    cv2.line(img, centroid, farthest_pt, [255, 255, 0])
    cv2.line(img, centroid, closest_pt, [0, 255, 255])


@timer
def remove_contours_from_copy(space):
    space_ = np.copy(space)
    _, contours = find_shapes_and_contours(space_)
    space_[contours > 0] = True
    return space_


@timer
def build_contourless_masks(space, num_of_masks, num_of_contours_to_drop):
    #drop first set of contours
    # masks = [remove_contours_from_copy(space)]
    last = np.copy(space)
    for i in range(num_of_contours_to_drop):
        last = remove_contours_from_copy(last)
    yield last
    for i in range(0, num_of_masks):
        yield remove_contours_from_copy(last)



@timer
def find_shapes_and_contours(space, save_shape_parameters=False):
    shapes = np.zeros(space.shape, int)
    contours = np.zeros(space.shape, int)
    curr_shape = 0
    for idx, val in np.ndenumerate(space):
        if not val and shapes[idx] == 0:
            curr_shape += 1
            fringe = [idx]
            while len(fringe) > 0:
                curr_px = fringe.pop()
                is_contour = False
                for y, x in get_px_neighbours(curr_px, space.shape):
                    if space[y, x] == 0:
                        if shapes[y, x] == 0:
                            shapes[y, x] = curr_shape
                            fringe.append((y, x))
                    else:
                        is_contour = True
                if is_contour:
                    contours[curr_px] = curr_shape
    if save_shape_parameters:
        return shapes, contours, calculate_basic_params(shapes, contours)
    else:
        return shapes, contours


@timer
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



@timer
def color_img_by_id_map(img, id_mapping, ids, color=(0, 100, 0)):
    idxs = np.where(np.isin(id_mapping, ids))
    img[idxs + (0,)] = color[0]
    img[idxs + (1,)] = color[1]
    img[idxs + (2,)] = color[2]


@timer
def color_shapes_and_contours(img, shapes, contours, shape_params=None):
    num_of_shapes = len(np.unique(shapes[shapes > 0]))
    shape_idxs = np.where(shapes > 0)
    img[shape_idxs + (0,)] = (255/float(num_of_shapes)) * shapes[shape_idxs]
    img[shape_idxs + (1,)] = 0
    img[shape_idxs + (2,)] = (255/float(num_of_shapes)) * (num_of_shapes - shapes[shape_idxs])
    contour_idxs = np.where(contours > 0)
    img[contour_idxs + (0,)] = 0
    img[contour_idxs + (1,)] = 100
    img[contour_idxs + (2,)] = 0
    if shape_params is not None:
        color_contour_extremes(img, shape_params)


@timer
def head_tail_breaks(shapes_sizes, key_idx = 1, iterations=100, tol=0.3):
    m = np.mean(shapes_sizes[:, key_idx])
    tail = shapes_sizes[np.where(shapes_sizes[:, key_idx] < m)]
    head = shapes_sizes[np.where(shapes_sizes[:, key_idx] >= m)]
    if len(head) / len(shapes_sizes) < tol:
        if iterations > 0:
            return head_tail_breaks(head, key_idx, iterations - 1, tol) + [tail[:, 0]]
        else:
            return [head[:, 0], tail[:, 0]]
    else:
        return [shapes_sizes[:, 0]]


@timer
def cluster_shapes_by(shape_params, column_name, cluster_func, *args, **kwargs):
    by_column = np.sort(shape_params, order=[column_name])[::-1]
    clusters = cluster_func(shape_params[column_name], *args, **kwargs)
    classified = []
    starting_idx = 0
    for cluster_len in clusters:
        end_idx = starting_idx + cluster_len
        classified.append(np.array(by_column['id'][starting_idx:end_idx], dtype=int))
        starting_idx = end_idx
    return classified


@timer
def cluster_by_kde(sizes, bandwidth=2.0, samples=6000):
    s_d = np.linspace(0.8 * np.min(sizes), 1.2 * np.max(sizes), samples)
    kde = sklearn.neighbors.KernelDensity(bandwidth=bandwidth, kernel="gaussian").fit(sizes.reshape(-1, 1))
    logprob = kde.score_samples(s_d.reshape(-1, 1))
    mins = s_d[scipy.signal.argrelextrema(logprob, np.less)]
    mins = list(mins)
    mins = [-float('inf')] + mins + [float('inf')]
    # clustered = [shapes_sizes[(shapes_sizes[:, 1] >= mins[i - 1]) * (shapes_sizes[:, 1] < mins[i]), 0] for i in range(1, len(mins))]
    clusters = [np.count_nonzero((sizes >= mins[i-1]) & (sizes < mins[i])) for i in range(1, len(mins))]
    clusters.reverse()
    return clusters

@timer
def classify_shapes(shapes, shape_params, num_of_shapes, skew_tol=0.2):
    return cluster_shapes_by(shape_params, 'size', cluster_by_kde)
    # for i in num_of_shapes:
    #     try:
    #         min_dist, max_dist = extremes[i]
    #         if float(min_dist)/float(max_dist) > skew_tol:
    #             # elongated, probably sinusoid
    #             classified[i] = ShapeTypes.sinusoid
    #         else:
    #
    #     except KeyError:
    #         #only a contour, sinusoid
    #         classified[i] = ShapeTypes.sinusoid


@timer
def color_classified_shapes(img, shapes, extremes, centroids, classified):
    num_of_shape_types = len(classified)
    for shape_type in range(num_of_shape_types):
        idxs = np.where(np.isin(shapes, classified[shape_type]))
        img[idxs + (0,)] = (255 / float(num_of_shape_types)) * shape_type
        img[idxs + (1,)] = 0
        img[idxs + (2,)] = (255 / float(num_of_shape_types)) * (num_of_shape_types - shape_type)
        for shape_num in classified[shape_type]:
            color_single_shape_extremes(img, extremes, centroids, shape_num)


@timer
def setup_img(img, num_of_splitting_passes = 1, num_of_contours_to_drop=0):
    bw_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bw_mask = cv2.threshold(bw_img, 175, True, cv2.ADAPTIVE_THRESH_GAUSSIAN_C)
    bw_mask = np.asarray(bw_mask, dtype=bool)
    return find_split_shapes_structure(bw_mask, num_of_splitting_passes, num_of_contours_to_drop)


@timer
def ellipse_rotations(centers, vertices):
    maj_ax = centers - vertices
    rotations = np.degrees(np.arctan2(maj_ax[:, 1], maj_ax[:, 0]))
    return rotations


@timer
def ellipse_axes(centers, vertices, covertices):
    maj_ax = centers - vertices
    min_ax = centers - covertices
    maj_ax_norm = np.linalg.norm(maj_ax, axis=1)
    min_ax_norm = np.linalg.norm(min_ax, axis=1)
    ax_norms = np.rint(np.column_stack((maj_ax_norm, min_ax_norm)))
    return ax_norms


@timer
def get_ellipse_params(params):
    ids = params['id']
    centers = np.column_stack((params['centroid_y'], params['centroid_x']))
    vertices = np.column_stack((params['max_pt_y'], params['max_pt_x']))
    covertices = np.column_stack((params['min_pt_y'], params['min_pt_x']))
    return np.column_stack((ids, centers, ellipse_axes(centers, vertices, covertices), ellipse_rotations(centers, vertices)))


@timer
def ellipse_params_from_row(row):
    return int(row[0]), tuple(row[[1, 2]].astype(int)), tuple(row[[3, 4]].astype(int)), row[5]


@timer
def draw_ellipse(img, id, center, axes, rotation):
    cv2.ellipse(img, center, axes, rotation, 0, 360, id, -1)


@timer
def find_ellipses(shape, params):
    ellipses = np.zeros(shape=shape, dtype=int)
    draw_ellipses(ellipses, params)
    return ellipses

@timer
def draw_ellipses(img, params):
    for id, center, axes, rotation in np.apply_along_axis(ellipse_params_from_row, arr=get_ellipse_params(params), axis=1):
        draw_ellipse(img, id, center, axes, rotation)


@timer
def process_img(img_file, out_file):
    img = cv2.imread(img_file)
    bw_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bw_mask = cv2.threshold(bw_img, 175, True, cv2.ADAPTIVE_THRESH_GAUSSIAN_C)
    bw_mask = np.asarray(bw_mask, dtype=bool)
    num_of_splitting_passes = 2
    shapes, contours, params = find_split_shapes_structure(bw_mask, num_of_splitting_passes)
    color_img = np.copy(img)
    color_shapes_and_contours(color_img, shapes, contours, params, True)
    show_img(color_img)
    save_fig(color_img, out_file)
        # color_img = np.copy(img)
        # approx_circles = shape_params[np.where(shape_params['skew'] > 0.2)]['id']
        # idxs = np.where(np.isin(shapes, approx_circles))
        # color_img[idxs + (0,)] = 0
        # color_img[idxs + (1,)] = 100
        # color_img[idxs + (2,)] = 0
        # for s in approx_circles:
        #     color_single_shape_extremes(color_img, shape_params, s)
        # save_fig(color_img, f"imgs/out/g_{i}.png")
    # remove_contours(bw_mask, 1)
    # 2nd pass, grab shapes
    #
    # colored_img = np.copy(img)
    # color_shapes_and_contours(colored_img, shapes, contours, num_of_shapes)
    # centroids
    # color_centroids(img, centroids)
    # extremes
    # closest, farthest = calc_contour_extremes(contours, centroids, num_of_shapes)
    # color_contour_extremes(colored_img, extremes, centroids, num_of_shapes)
    # show_img(colored_img)
    # classified = classify_shapes(shapes, shape_params, num_of_shapes)
    # print(len(classified))
    # color_classified_shapes(colored_img, shapes, extremes, centroids, classified)
    # show_img(colored_img)
    # print(shape_params[0])


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

if __name__ == "main":
    process_img(img_file, out_file)
    for func in total_time:
        print(f"total time in {func.__name__}: {total_time[func]}")