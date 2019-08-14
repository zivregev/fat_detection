import cv2
import numpy
import scipy.spatial
from matplotlib import pyplot as plt
import itertools
import time


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
    centroid_mask = numpy.zeros(shapes.shape[:2], dtype=int)
    for shape in range(1, num_of_shapes+1):
        pts = numpy.argwhere(shapes == shape)
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
        shape_pts = numpy.argwhere(contours == i)
        try:
            centroid = centroids[i]
            distances = numpy.linalg.norm(shape_pts - centroid, axis=1)
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
def find_shapes_and_contours(space, save_centroids=False):
    labels = numpy.zeros(space.shape, int)
    contours = numpy.zeros(space.shape, int)
    if save_centroids:
        centroids = {}
        extremes = {}
    curr_shape = 0
    for idx, val in numpy.ndenumerate(space):
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
                    if save_centroids:
                        contour.append(curr_px)
                elif save_centroids:
                    centroid_accum[0] += curr_px[0]
                    centroid_accum[1] += curr_px[1]
                    centroid_accum[2] += 1
            if centroid_accum[2] > 0 and save_centroids:
                y_centroid, x_centroid = [int(coord/centroid_accum[2]) for coord in centroid_accum[:2]]
                centroid = (y_centroid, x_centroid)
                centroids[curr_shape] = centroid
                contour = numpy.asarray(contour, dtype= int)
                distances = numpy.linalg.norm(contour - centroid, axis=1)
                extremes[curr_shape] = (contour[distances.argmin()], contour[distances.argmax()])
    if save_centroids:
        return labels, contours, centroids, extremes, curr_shape
    else:
        return labels, contours, curr_shape


@timer
def color_shapes_and_contours(img, shapes, contours, num_of_shapes):
    shape_idxs = numpy.where(shapes > 0)
    img[shape_idxs + (0,)] = (255/float(num_of_shapes)) * shapes[shape_idxs]
    img[shape_idxs + (1,)] = 0
    img[shape_idxs + (2,)] = (255/float(num_of_shapes)) * (num_of_shapes - shapes[shape_idxs])
    contour_idxs = numpy.where(contours > 0)
    img[contour_idxs + (0,)] = 0
    img[contour_idxs + (1,)] = 100
    img[contour_idxs + (2,)] = 0


@timer
def process_img(img_file, outfile):
    img = cv2.imread(img_file)
    bw_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bw_mask = cv2.threshold(bw_img, 175, True, cv2.ADAPTIVE_THRESH_GAUSSIAN_C)
    bw_mask = numpy.asarray(bw_mask, dtype=bool)
    # first pass, try to remove some contours
    remove_contours(bw_mask, 1)
    # 2nd pass, grab shapes
    shapes, contours, centroids, extremes, num_of_shapes = find_shapes_and_contours(bw_mask, save_centroids=True)
    color_shapes_and_contours(img, shapes, contours, num_of_shapes)
    # centroids
    color_centroids(img, centroids)
    # extremes
    # closest, farthest = calc_contour_extremes(contours, centroids, num_of_shapes)
    color_contour_extremes(img, extremes, centroids, num_of_shapes)
    show_img(img)


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