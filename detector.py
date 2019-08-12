import cv2
import numpy
import scipy.spatial
from matplotlib import pyplot as plt
import itertools
import time

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


def build_contour(interior):
    contour = numpy.zeros(interior.shape)
    for px in zip(*numpy.where(interior == 1)):
        for y, x in get_px_neighbours(px, interior.shape):
            if interior[y, x] == 0:
                contour[y, x] = 1
    # print(f"shape: {numpy.where(interior == 1)}, contour: {numpy.where(contour == 1)}")
    # contour = []
    # for px in interior:
    #     for y, x in get_px_neighbours(px, space.shape):
    #         if (y, x) not in interior:
    #             contour.append((y, x))
    return contour


def break_shape_at_contours(shape, contour):
    shapes = []
    interior = zip(*numpy.where(numpy.logical_and(shape == 1, contour == 0)))
    for idx in interior:
        if shape[idx] == 1:
            sub_shape = numpy.zeros(shape.shape)
            fringe = [idx]
            while len(fringe) > 0:
                curr_px = fringe.pop()
                for y, x in get_px_neighbours(curr_px, shape.shape):
                    if shape[y, x] == 1:
                        fringe.append((y, x))
                        shape[y, x] = 0
                sub_shape[curr_px] = 1
            shapes.append((sub_shape, build_contour(sub_shape)))
    # while len(interior) > 0:
    #     sub_shape = []
    #     fringe = [interior.pop()]
    #     while len(fringe) > 0:
    #         curr_px = fringe.pop()
    #         for y, x in get_px_neighbours(curr_px, space.shape):
    #             if (y, x) in interior:
    #                 fringe.append((y, x))
    #                 sub_shape.append((y, x))
    #                 interior.remove((y, x))
    #     if len(sub_shape) > 0:
    #         shapes.append((sub_shape, build_contour(space, sub_shape)))
    return shapes


# assumes that space[starting_px] == 0
def get_contiguous_shape(space, starting_px, visited):
    shape = numpy.zeros(space.shape)
    contour = numpy.zeros(space.shape)
    fringe = [starting_px]
    while len(fringe) > 0:
        curr_px = fringe.pop()
        is_contour = False
        for y, x in get_px_neighbours(curr_px, space.shape):
            if space[y, x] == 0:
                if visited[y, x] == 0:
                    fringe.append((y, x))
                    visited[y, x] = 1
            else:
                is_contour = True
        if is_contour:
            contour[curr_px] = 1
        else:
            shape[curr_px] = 1
    return shape, contour


def find_centroid(contour):
    return tuple([int((1.0/len(contour)) * sum(x)) for x in zip(*contour)])


def calc_contour_extremes(contour):
    centroid = find_centroid(contour)
    closest = float('inf')
    farthest = -float('inf')
    for px in contour:
        d = scipy.spatial.distance.euclidean(px,centroid)
        if d < closest:
            closest = d
        if d > farthest:
            farthest = d
    return closest, farthest


def guess_whether_round(contour, tol=0.2):
   closest, farthest = calc_contour_extremes(contour)
   # print(f"{closest}, {farthest}")
   if closest/farthest > tol:
       return True
   else:
       return False


def process_img(imgfile, outfile):
    img = cv2.imread(img_file)
    bw_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bw_mask = cv2.threshold(bw_img, 175, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C)
    visited = numpy.zeros(bw_mask.shape)
    shapes = numpy.zeros(bw_mask.shape)
    contours = numpy.zeros(bw_mask.shape)
    shape_num = 1
    for y in range(bw_mask.shape[0]):
        for x in range(bw_mask.shape[1]):
            if visited[y, x] == 0:
                if bw_mask[y, x] == 0:
                    for shape, contour in break_shape_at_contours(*get_contiguous_shape(bw_mask, (y, x), visited)):
                        shapes[shape == 1] = shape_num
                        contours[contour == 1] = shape_num
                        shape_num += 1
                    # shape, contour = get_contiguous_shape(bw_mask, (y, x), visited)
                    # shapes[shape == 1] = shape_num
                    # shapes[contour == 1] = shape_num + 0.5
                    # shape_num += 1
                    # shapes.extend(break_shape_at_contours(bw_mask, *get_contiguous_shape(bw_mask, (y, x), visited)))
    for i in range(1, shape_num):
        img[numpy.where(shapes == i)] = [255*float(i)/float(shape_num), 0, 255*float(shape_num-i)/float(shape_num)]
        img[numpy.where(contours == i)] = [0, 100, 0]
    # for i, shape in enumerate(shapes):
    #     c = [255*float(i)/float(len(shapes)), 0, 255*float(len(shapes)-i)/float(len(shapes))]
    #     print(len(shape))
    #     interior, contour = shape
    #
    #     img[interior == 1] = c
    #     img[contour == 1] = [0, 100, 0]
        # c = [0, 0, 0] if guess_whether_round(contour) else [255, 255, 255]
        # print(c)

        # for y, x in interior:
        #     img[y, x] = c
        #
        # for y, x in contour:
        #     img[y, x] = [0, 100, 0]
        #
        # centroid_y, centroid_x = find_centroid(contour)
        # img[centroid_y, centroid_x] = [0, 0, 0]
    show_img(img)

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

img_file = "imgs/010007.tif"
out_name = "tmp"
out_file = f"imgs/out/{out_name}.png"
process_img = timer(process_img)
break_shape_at_contours = timer(break_shape_at_contours)
get_contiguous_shape = timer(get_contiguous_shape)
build_contour = timer(build_contour)

process_img(img_file, out_file)

for func in total_time:
    print(f"total time in {func.__name__}: {total_time[func]}")