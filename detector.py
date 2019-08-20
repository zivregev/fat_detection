import cv2
import numpy as np
from matplotlib import pyplot as plt
import itertools
import time
import enum

class ShapeTypes(enum.Enum):
    fat_cell = 0
    sinusoid = 1
    blood_vessel = 2
    other = 4

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


def save_fig(img, name):
    dpi = 80
    height, width, *_ = img.shape
    figsize = width / float(dpi), height / float(dpi)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.imshow(img, cmap="gray")
    plt.savefig(name)
    plt.clf()


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
def color_contour_extremes(img, extremes, centroids, num_of_shapes):
    for i in range(1, num_of_shapes):
        try:
            centroid = tuple(centroids[i][::-1])
            closest_pt, farthest_pt = map(lambda pt: tuple(pt[::-1]), extremes[i])
            cv2.line(img, centroid, farthest_pt, [255, 255, 0])
            cv2.line(img, centroid, closest_pt, [0, 255, 255])
        except KeyError:
            pass


@timer
def remove_contours(space, passes=1):
    for i in range(passes):
        _, contours, _ = find_shapes_and_contours(space)
        space[contours > 0] = True


@timer
def find_shapes_and_contours(space, save_shape_parameters=False):
    labels = np.zeros(space.shape, int)
    contours = np.zeros(space.shape, int)
    if save_shape_parameters:
        centroids = {}
        extremes = {}
    curr_shape = 0
    for idx, val in np.ndenumerate(space):
        if not val and labels[idx] == 0:
            curr_shape += 1
            fringe = [idx]
            centroid_accum = [0, 0, 0]
            contour = []
            while len(fringe) > 0:
                curr_px = fringe.pop()
                is_contour = False
                for y, x in get_px_neighbours(curr_px, space.shape):
                    if space[y, x] == 0:
                        if labels[y, x] == 0:
                            labels[y, x] = curr_shape
                            fringe.append((y, x))
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
            if save_shape_parameters and centroid_accum[2] > 0:
                y_centroid, x_centroid = [int(coord/centroid_accum[2]) for coord in centroid_accum[:2]]
                centroid = (y_centroid, x_centroid)
                centroids[curr_shape] = centroid
                contour = np.asarray(contour, dtype= int)
                distances = np.linalg.norm(contour - centroid, axis=1)
                extremes[curr_shape] = (contour[distances.argmin()], contour[distances.argmax()])
    if save_shape_parameters:
        return labels, contours, centroids, extremes, curr_shape
    else:
        return labels, contours, curr_shape


@timer
def color_shapes_and_contours(img, shapes, contours, num_of_shapes):
    shape_idxs = np.where(shapes > 0)
    img[shape_idxs + (0,)] = (255/float(num_of_shapes)) * shapes[shape_idxs]
    img[shape_idxs + (1,)] = 0
    img[shape_idxs + (2,)] = (255/float(num_of_shapes)) * (num_of_shapes - shapes[shape_idxs])
    contour_idxs = np.where(contours > 0)
    img[contour_idxs + (0,)] = 0
    img[contour_idxs + (1,)] = 100
    img[contour_idxs + (2,)] = 0


@timer
def head_tail_breaks(shapes, iterations=10, tol=0.3):
    m = np.mean(shapes[:, 1])
    tail = shapes[np.where(shapes[:, 1] < m)]
    head = shapes[np.where(shapes[:, 1] >= m)]
    if len(head) / len(shapes) < tol:
        if iterations > 0:
            return head_tail_breaks(head, iterations - 1, tol) + [tail[:, 0]]
        else:
            return [head[:, 0], tail[:, 0]]
    else:
        return [shapes[:, 0]]


@timer
def classify_shapes(shapes, centroids, extremes, num_of_shapes, skew_tol=0.2, size_tol=0.9):
    shapes_sizes = list(map(tuple, zip(*np.unique(shapes[shapes > 0], return_counts=True))))
    shapes_sizes.sort(key=lambda shape: shape[1], reverse=True)
    shapes_sizes = np.asarray(shapes_sizes, dtype=int)
    clustered = head_tail_breaks(shapes_sizes)
    extremes = np.asarray([extremes[shapes[0]] for shape in shapes_sizes])
    classified = {i: clustered[i][:, 0] for i in range(len(clustered))}
    return classified
    # size_diffs = shape_sizes[:-1] / shape_sizes[1:]
    # plt.stem(range(len(size_diffs)), size_diffs)
    # plt.show()
    # plt.stem(range(len(shape_sizes)), shape_sizes)
    # plt.show()
    # largest_shape = max(shape_sizes, key=lambda k: shape_sizes[k])
    # print(largest_shape, shape_sizes[largest_shape])
    # print(dict(zip(shapes, px_count)))
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
def color_classified_shapes(img, shapes, classified):
    num_of_shape_types = len(classified)
    for shape_type in classified:
        idxs = np.where(np.isin(shapes, classified[shape_type]))
        img[idxs + (0,)] = (255 / float(num_of_shape_types)) * shape_type
        img[idxs + (1,)] = 0
        img[idxs + (2,)] = (255 / float(num_of_shape_types)) * (num_of_shape_types - shape_type)


@timer
def process_img(img_file, outfile):
    img = cv2.imread(img_file)
    bw_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bw_mask = cv2.threshold(bw_img, 175, True, cv2.ADAPTIVE_THRESH_GAUSSIAN_C)
    bw_mask = np.asarray(bw_mask, dtype=bool)
    # first pass, try to remove some contours
    remove_contours(bw_mask, 1)
    # 2nd pass, grab shapes
    shapes, contours, centroids, extremes, num_of_shapes = find_shapes_and_contours(bw_mask, save_shape_parameters=True)
    colored_img = np.copy(img)
    # color_shapes_and_contours(colored_img, shapes, contours, num_of_shapes)
    # centroids
    # color_centroids(img, centroids)
    # extremes
    # closest, farthest = calc_contour_extremes(contours, centroids, num_of_shapes)
    # color_contour_extremes(img, extremes, centroids, num_of_shapes)
    # show_img(colored_img)
    classified = classify_shapes(shapes, centroids, extremes, num_of_shapes)
    color_classified_shapes(colored_img, shapes, classified)
    show_img(colored_img)


img_file = "imgs/010007.tif"
out_name = "tmp"
out_file = f"imgs/out/{out_name}.png"
# process_img = timer(process_img)
# remove_contours = timer(remove_contours)
# find_shapes_and_contours = timer(find_shapes_and_contours)
# color_shapes_and_contours = timer(color_shapes_and_contours)
# find_centroids = timer

process_img(img_file, out_file)

for func in total_time:
    print(f"total time in {func.__name__}: {total_time[func]}")