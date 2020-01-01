from matplotlib import pyplot as plt
import cv2
import numpy as np


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


def color_centroids(img, centroids):
    idxs = [idx for i in [0, 1, 2] for idx in list(centroids.values())]
    x_s = [idx[1] for idx in list(centroids.values())]
    y_s = [idx[0] for idx in list(centroids.values())]
    z_s = [0, 1, 2]
    img[y_s, x_s] = [255, 0, 0]


def color_contour_extremes(img, shape_params):
    for i in shape_params['id']:
        color_single_shape_extremes(img, shape_params, i)


def color_single_shape_extremes(img, params, shape_num):
    shape_params = params[np.where(params['id'] == shape_num)]
    centroid = tuple(map(int, (shape_params['centroid_y'], shape_params['centroid_x'])))
    closest_pt = tuple(map(int, (shape_params['min_pt_y'], shape_params['min_pt_x'])))
    farthest_pt = tuple(map(int, (shape_params['max_pt_y'], shape_params['max_pt_x'])))
    cv2.line(img, centroid, farthest_pt, [255, 255, 0])
    cv2.line(img, centroid, closest_pt, [0, 255, 255])


def color_img_by_id_map(img, id_mapping, ids, color=(0, 100, 0)):
    idxs = np.where(np.isin(id_mapping, ids))
    img[idxs + (0,)] = color[0]
    img[idxs + (1,)] = color[1]
    img[idxs + (2,)] = color[2]
    return img


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
    return img


def color_classified_shapes(img, shapes, params, classified):
    num_of_shape_types = len(classified)
    for shape_type in range(num_of_shape_types):
        idxs = np.where(np.isin(shapes, classified[shape_type]))
        img[idxs + (0,)] = (255 / float(num_of_shape_types)) * shape_type
        img[idxs + (1,)] = 0
        img[idxs + (2,)] = (255 / float(num_of_shape_types)) * (num_of_shape_types - shape_type)
        for shape_num in classified[shape_type]:
            color_single_shape_extremes(img, params, shape_num)
    return img