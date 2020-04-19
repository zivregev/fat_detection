import numpy as np
import sklearn.neighbors
import scipy.signal
import cv2
from enum import Enum


skew_dtype = {'names': ['id', 'skew'], 'formats': [int, float]}

class ConfirmedType(Enum):
    INCLUDE = 1
    EXCLUDE = 0

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


def cluster_array_by_column(shape_params, column_name, cluster_func, *args, **kwargs):
    by_column = np.sort(shape_params, order=[column_name])[::-1]
    clusters = cluster_func(shape_params[column_name], *args, **kwargs)
    classified = []
    starting_idx = 0
    for cluster_len in clusters:
        end_idx = starting_idx + cluster_len
        classified.append(np.array(by_column['id'][starting_idx:end_idx], dtype=int))
        starting_idx = end_idx
    return classified


def cluster_by_kde(sizes, bandwidth=2.0, samples=6000):
    print(len(sizes))
    s_d = np.linspace(0.8 * np.min(sizes), 1.2 * np.max(sizes), samples)
    kde = sklearn.neighbors.KernelDensity(bandwidth=guess_bandwidth(sizes), kernel="gaussian").fit(sizes.reshape(-1, 1))
    logprob = kde.score_samples(s_d.reshape(-1, 1))
    mins = s_d[scipy.signal.argrelextrema(logprob, np.less)]
    mins = list(mins)
    mins = [-float('inf')] + mins + [float('inf')]
    # clustered = [shapes_sizes[(shapes_sizes[:, 1] >= mins[i - 1]) * (shapes_sizes[:, 1] < mins[i]), 0] for i in range(1, len(mins))]
    clusters = [np.count_nonzero((sizes >= mins[i-1]) & (sizes < mins[i])) for i in range(1, len(mins))]
    clusters.reverse()
    return clusters


def guess_bandwidth(sizes):
    #using silverman's rule
    return max(1.06 * np.std(sizes) * (len(sizes) ** (-1.0/5.0)), 0.1)


def classify_shapes(shape_params, by):
    return cluster_array_by_column(shape_params, by, cluster_by_kde)

def classify_by_size(params, min_shape_size=5):
    non_trivial_shapes = params[params['size'] > min_shape_size]
    return top_down_cluster(params, non_trivial_shapes['id'], by='size')



def drop_larger_shapes(params, min_shape_size=5, min_to_max_size_in_cluster_ration=5, confirmed_shapes=None):
    non_trivial_shapes = params[params['size'] > min_shape_size]
    clx = top_down_cluster(params, non_trivial_shapes['id'], by='size')
    #remove singleton clusters
    clx = [c for c in clx if len(c) > 1]
    #drop all clusters that are ten times as large as max of the most numerous cluster
    most_numerous_type = clx[np.argmax([len(c) for c in clx])]
    max_size_in_most_numerous = max(params[np.where(np.isin(params['id'], most_numerous_type))]['size'])
    clx = [c for c in clx if min(params[np.where(np.isin(params['id'], c))]['size']) <
           min_to_max_size_in_cluster_ration * max_size_in_most_numerous]

    return clx

def apply_confirmed_shapes_to_clustering(clx, confirmed_shapes):
    confirmed_included = []
    confirmed_excluded = []
    unconfirmed = []
    for c in clx:
        include = False
        exclude = False
        for shape_id in confirmed_shapes:
            shape_type = confirmed_shapes[shape_id]
            if shape_id in c:
                if shape_type == ConfirmedType.INCLUDE:
                    include = True
                else:
                    exclude = True
            if include and exclude:
                break
        if (include and exclude) or ((not include) and (not exclude)):
            unconfirmed.append(c)
        elif include:
            confirmed_included.append(c)
        else:
            confirmed_excluded.append(c)
    return confirmed_included, confirmed_excluded, unconfirmed




def classify(params, confirmed_shapes=None):



def drop_skewed_shapes(params, non_trivial_ids):
    centroids = np.stack([params['centroid_y'], params['centroid_x']])
    max_pts = np.stack([params['max_pt_y'], params['max_pt_x']])
    min_pts = np.stack([params['min_pt_y'], params['min_pt_x']])
    max_axis = np.linalg.norm(max_pts - centroids, axis=0)
    min_axis = np.linalg.norm(min_pts - centroids, axis=0)
    skews = min_axis / max_axis
    skews[min_axis == 0.0] = 0.0
    true_max = np.max(skews[max_axis > 0.0])
    skews[max_axis == 0.0] = true_max + 1
    skews_ = []
    for id_idx in range(len(params)):
        skews_.append((params['id'][id_idx], skews[id_idx]))
    skews = np.asarray(skews_, dtype=skew_dtype)
    #drop all shapes with zeroed skew (i.e., lines)
    skews = skews[skews['skew'] > 0.0]
    #drop all shapes with inf skew (i.e., > true_max)
    skews = skews[skews['skew'] <= true_max]
    clx = top_down_cluster(skews, non_trivial_ids, min_cluster_size=2, depth=8, by='skew')
    #remove singleton clusters
    clx = [c for c in clx if len(c) > 1]
    return clx


def top_down_cluster(params, ids_to_cluster, min_cluster_size=2, depth=8, clx_to_ids_ratio=0.8, by='size'):
    clx = classify_shapes(params[np.where(np.isin(params['id'], ids_to_cluster))], by=by)
    if depth > 0:
        if len(clx) / len(ids_to_cluster) < clx_to_ids_ratio and len(clx) > 1:
            sub_clx = []
            for cluster in clx:
                if len(cluster) > min_cluster_size:
                    sub_clx += top_down_cluster(params, cluster, min_cluster_size, depth - 1, clx_to_ids_ratio, by=by)
                else:
                    sub_clx.append(cluster)
            return sub_clx
    return clx


def ellipse_rotations(centers, vertices):
    maj_ax = centers - vertices
    rotations = np.degrees(np.arctan2(maj_ax[:, 1], maj_ax[:, 0]))
    return rotations


def ellipse_axes(centers, vertices, covertices):
    maj_ax = centers - vertices
    min_ax = centers - covertices
    maj_ax_norm = np.linalg.norm(maj_ax, axis=1)
    min_ax_norm = np.linalg.norm(min_ax, axis=1)
    ax_norms = np.rint(np.column_stack((maj_ax_norm, min_ax_norm)))
    return ax_norms


def get_ellipse_params(params):
    ids = params['id']
    centers = np.column_stack((params['centroid_y'], params['centroid_x']))
    vertices = np.column_stack((params['max_pt_y'], params['max_pt_x']))
    covertices = np.column_stack((params['min_pt_y'], params['min_pt_x']))
    return np.column_stack((ids, centers, ellipse_axes(centers, vertices, covertices), ellipse_rotations(centers, vertices)))


def ellipse_params_from_row(row):
    return int(row[0]), tuple(row[[1, 2]].astype(int)), tuple(row[[3, 4]].astype(int)), row[5]


def draw_ellipse(img, id, center, axes, rotation):
    cv2.ellipse(img, center, axes, rotation, 0, 360, id, -1)


def find_ellipses(shape, params):
    ellipses = np.zeros(shape=shape, dtype=int)
    draw_ellipses(ellipses, params)
    return ellipses


def draw_ellipses(img, params):
    for id, center, axes, rotation in np.apply_along_axis(ellipse_params_from_row, arr=get_ellipse_params(params), axis=1):
        draw_ellipse(img, id, center, axes, rotation)