import cv2
import numpy as np
import scipy.signal
from matplotlib import pyplot as plt
import itertools
import time
import enum
import sklearn.neighbors

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
        if len(nontrivial_shapes_in_submask) > 1:
            split_shapes[shape_id] = nontrivial_shapes_in_submask
    return split_shapes


@timer
def assign_new_ids(split_ids, next_id):
    for org_id in split_ids:
        subshape_ids = split_ids[org_id]
        split_ids[org_id] = (subshape_ids, list(range(next_id, next_id + len(subshape_ids))))
        next_id += len(subshape_ids)


@timer
def split_shapes(shapes_mask, shapes_submask, contours_mask, contours_submask):
    split_ids = find_split_shapes(shapes_mask, shapes_submask)
    next_id = np.max(shapes_mask) + 1
    assign_new_ids(split_ids, next_id)
    split_id_based_array(shapes_mask, shapes_submask, split_ids)
    split_id_based_array(contours_mask, contours_submask, split_ids)
    return split_ids


@timer
def split_id_based_array(ar, sub_ar, split_ids):
    ar[np.where(np.isin(ar, list(split_ids.keys())))] = 0
    sub_ar_idx_map = {id: idxs for id, idxs in zip(*map_ids_to_idxs(sub_ar))}
    for org_id in split_ids:
        subshape_ids, new_ids = split_ids[org_id]
        for subshape_id, new_id in zip(subshape_ids, new_ids):
            ar[sub_ar_idx_map[subshape_id]] = new_id


@timer
def update_split_shape_params(params, sub_params, split_ids):
    removed_ids, _tmp = zip(*list(split_ids.items()))
    submask_ids, new_ids = map(np.concatenate, map(np.asarray, zip(*_tmp)))
    #remove shapes w/ no prms, & sort by ascending id
    submask_ids_with_params = np.isin(submask_ids, sub_params['id'])
    submask_ids = submask_ids[submask_ids_with_params]
    new_ids = new_ids[submask_ids_with_params]
    sorted_ids = np.argsort(submask_ids)
    submask_ids = submask_ids[sorted_ids]
    new_ids = new_ids[sorted_ids]
    subshape_rows = sub_params[np.where(np.isin(sub_params['id'], submask_ids))]
    subshape_rows['id'] = new_ids[np.where(np.isin(submask_ids, sub_params['id']))]
    return np.append(np.delete(params, np.where(np.isin(params['id'], removed_ids))), subshape_rows)


@timer
def color_centroids(img, centroids):
    idxs = [idx for i in [0, 1, 2] for idx in list(centroids.values())]
    x_s = [idx[1] for idx in list(centroids.values())]
    y_s = [idx[0] for idx in list(centroids.values())]
    z_s = [0, 1, 2]
    img[y_s, x_s] = [255, 0, 0]
    # for (y, x) in idxs:
    #     img[(y, x)] = [255, 0, 0]
    # print(idxs[0])
    # print(img[idxs[0]])

@timer
def calc_contour_extremes(contours, centroids, num_of_shapes):
    closest = {}
    farthest = {}
    # print(centroids[numpy.where(centroids > 0)])
    for i in range(1, num_of_shapes + 1):
        shape_pts = np.argwhere(contours == i)
        try:
            centroid = centroids[i]
            distances = np.linalg.norm(shape_pts - centroid, axis=1)
            farthest[i] = shape_pts[distances.argmax()]
            closest[i] = shape_pts[distances.argmin()]
        except KeyError:
            pass
    return closest, farthest


@timer
def color_contour_extremes(img, shape_params):
    for i in shape_params['id']:
        color_single_shape_extremes(img, shape_params, i)


@timer
def color_single_shape_extremes(img, shape_params, shape_num):
    params = shape_params[np.where(shape_params['id'] == shape_num)]
    centroid = (params['centroid_x'], params['centroid_y'])
    closest_pt = (params['min_pt_x'], params['min_pt_y'])
    farthest_pt = (params['max_pt_x'], params['max_pt_y'])
    cv2.line(img, centroid, farthest_pt, [255, 255, 0])
    cv2.line(img, centroid, closest_pt, [0, 255, 255])


@timer
def remove_contours_from_copy(space):
    space_ = np.copy(space)
    _, contours = find_shapes_and_contours(space_)
    space_[contours > 0] = True
    return space_


@timer
def build_contourless_masks(space, num_of_masks=2):
    #drop first set of contours
    masks = [remove_contours_from_copy(space)]
    for i in range(1, num_of_masks):
        masks.append(remove_contours_from_copy(masks[-1]))
    return masks



@timer
def find_shapes_and_contours(space, save_shape_parameters=False):
    labels = np.zeros(space.shape, int)
    contours = np.zeros(space.shape, int)
    if save_shape_parameters:
        shape_parameters = []
    curr_shape = 0
    for idx, val in np.ndenumerate(space):
        if not val and labels[idx] == 0:
            curr_shape += 1
            fringe = [idx]
            centroid_accum = [0, 0, 0]
            contour = []
            size = 0
            while len(fringe) > 0:
                curr_px = fringe.pop()
                is_contour = False
                if len(curr_px)  < 2:
                    print( curr_px)
                for y, x in get_px_neighbours(curr_px, space.shape):
                    if space[y, x] == 0:
                        if labels[y, x] == 0:
                            labels[y, x] = curr_shape
                            fringe.append((y, x))
                            size += 1
                    else:
                        is_contour = True
                if is_contour:
                    contours[curr_px] = curr_shape
                    if save_shape_parameters:
                        contour.append(curr_px)
                elif save_shape_parameters:
                    centroid_accum[0] += curr_px[0]
                    centroid_accum[1] += curr_px[1]
                    centroid_accum[2] += 1
            if save_shape_parameters:
                if centroid_accum[2] > 0:
                    y_centroid, x_centroid = [int(coord/centroid_accum[2]) for coord in centroid_accum[:2]]
                    centroid = (y_centroid, x_centroid)
                    contour = np.asarray(contour, dtype=int)
                    distances = np.linalg.norm(contour - centroid, axis=1)
                    extreme_pts = (contour[distances.argmin()], contour[distances.argmax()])
                    radii = list(map(lambda p: np.linalg.norm(centroid - p), extreme_pts))
                    skew = radii[0] / radii[1]
                    extreme_pts = [coord for pt in list(map(list, extreme_pts)) for coord in pt]
                    shape_parameters.append((curr_shape, size, *centroid, *extreme_pts, skew))
    if save_shape_parameters:
        return labels, contours, np.array(shape_parameters, dtype=param_dtype)
        # return labels, contours, shape_parameters, curr_shape
    else:
        return labels, contours


@timer
def color_img_by_id_map(img, id_mapping, ids, color=(0, 100, 0)):
    idxs = np.where(np.isin(id_mapping, ids))
    img[idxs + (0,)] = color[0]
    img[idxs + (1,)] = color[1]
    img[idxs + (2,)] = color[2]


@timer
def color_shapes_and_contours(img, shapes, contours, shape_params=None, color_extremes=False):
    num_of_shapes = len(np.unique(shapes[shapes > 0]))
    shape_idxs = np.where(shapes > 0)
    img[shape_idxs + (0,)] = (255/float(num_of_shapes)) * shapes[shape_idxs]
    img[shape_idxs + (1,)] = 0
    img[shape_idxs + (2,)] = (255/float(num_of_shapes)) * (num_of_shapes - shapes[shape_idxs])
    contour_idxs = np.where(contours > 0)
    img[contour_idxs + (0,)] = 0
    img[contour_idxs + (1,)] = 100
    img[contour_idxs + (2,)] = 0
    if color_extremes:
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
def find_split_shapes_structure(bw_mask, num_of_splitting_passes=2):
    processed_masks = []
    for mask in build_contourless_masks(bw_mask, num_of_splitting_passes):
        processed_masks.append(find_shapes_and_contours(mask, save_shape_parameters=True))
    for i in range(len(processed_masks) - 1, 0, -1):
        shapes, contours, params = processed_masks[i - 1]
        subshapes, subcontours, subparams = processed_masks[i]
        split_ids = split_shapes(shapes, subshapes, contours, subcontours)
        params = update_split_shape_params(params, subparams, split_ids)
    return shapes, contours, params


@timer
def setup_img(img):
    bw_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bw_mask = cv2.threshold(bw_img, 175, True, cv2.ADAPTIVE_THRESH_GAUSSIAN_C)
    bw_mask = np.asarray(bw_mask, dtype=bool)
    num_of_splitting_passes = 2
    return find_split_shapes_structure(bw_mask, num_of_splitting_passes)


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
    centers = np.column_stack((params['centroid_x'], params['centroid_y']))
    vertices = np.column_stack((params['max_pt_x'], params['max_pt_y']))
    covertices = np.column_stack((params['min_pt_x'], params['min_pt_y']))
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


# img_file = "imgs/010007.tif"
img_file = "imgs/edited.png"
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