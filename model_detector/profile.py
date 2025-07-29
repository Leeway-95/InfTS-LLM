import numpy as np
from numba import njit

@njit(fastmath=True)
def _calc_neigh_pos(knn):
    # Calculate neighbor positions for knn matrix
    ind, val = np.zeros(knn.shape[0], np.int32), np.zeros(knn.shape[0] * knn.shape[1], np.int32)
    counts = np.zeros(ind.shape[0], np.int32)
    ptr = np.zeros(ind.shape[0], np.int32)
    for idx in range(knn.shape[0]):
        for kdx in range(knn.shape[1]):
            pos = knn[idx, kdx]
            counts[pos] += 1
    for idx in range(1, ind.shape[0]):
        ind[idx] = ind[idx - 1] + counts[idx - 1]
    for idx, neigh in enumerate(knn):
        for nn in neigh:
            pos = ind[nn] + ptr[nn]
            val[pos] = idx
            ptr[nn] += 1
    return ind, val

@njit(fastmath=True)
def _init_labels(knn, offset):
    # Initialize labels for knn classification
    n_timepoints, k_neighbours = knn.shape
    y_true = np.concatenate((
        np.zeros(offset, dtype=np.int32),
        np.ones(n_timepoints - offset, dtype=np.int32),
    ))
    neigh_pos = _calc_neigh_pos(knn)
    knn_labels = np.zeros(shape=(k_neighbours, n_timepoints), dtype=np.int32)
    for i_neighbor in range(k_neighbours):
        neighbours = knn[:, i_neighbor]
        knn_labels[i_neighbor] = y_true[neighbours]
    ones = np.sum(knn_labels, axis=0)
    zeros = k_neighbours - ones
    y_pred = np.asarray(ones > zeros, dtype=np.int32)
    return (zeros, ones), neigh_pos, y_true, y_pred

@njit(fastmath=True)
def _init_conf_matrix(y_true, y_pred):
    # Initialize confusion matrix from true/pred labels
    tp = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 1) & (y_pred == 0))
    fn = np.sum((y_true == 0) & (y_pred == 1))
    tn = np.sum((y_true == 1) & (y_pred == 1))
    conf_matrix = np.array([tp, fp, fn, tn], dtype=np.int32)
    return conf_matrix

@njit(fastmath=True)
def _update_conf_matrix(old_true, old_pred, new_true, new_pred, conf_matrix):
    # Update confusion matrix incrementally
    conf_matrix[0] -= (not old_true and not old_pred) - (not new_true and not new_pred)
    conf_matrix[1] -= (old_true and not old_pred) - (new_true and not new_pred)
    conf_matrix[2] -= (not old_true and old_pred) - (not new_true and new_pred)
    conf_matrix[3] -= (old_true and old_pred) - (new_true and new_pred)
    return conf_matrix

@njit(fastmath=True)
def _update_labels(split_idx, excl_zone, neigh_pos, knn_counts, y_true, y_pred, conf_matrix):
    # Update labels during change point detection
    np_ind, np_val = neigh_pos
    excl_start, excl_end = excl_zone
    knn_zeros, knn_ones = knn_counts
    ind = np_val[np_ind[split_idx]:np_ind[split_idx + 1]]
    if ind.shape[0] > 0:
        ind = np.append(ind, split_idx)
    else:
        ind = np.array([split_idx])
    for pos in ind:
        if pos != split_idx:
            knn_zeros[pos] += 1
            knn_ones[pos] -= 1
        in_excl_zone = excl_end > pos >= excl_start
        zeros, ones = knn_zeros[pos], knn_ones[pos]
        label = zeros < ones
        if not in_excl_zone:
            conf_matrix = _update_conf_matrix(y_true[pos],
                                              y_pred[pos],
                                              y_true[pos],
                                              label, conf_matrix)
        y_pred[pos] = label
    y_true[split_idx] = 0
    conf_matrix = _update_conf_matrix(y_true[excl_end], y_pred[excl_end],
                                      y_true[excl_start], y_pred[excl_start],
                                      conf_matrix)
    return y_true, y_pred, conf_matrix

@njit(fastmath=True)
def _fast_profile(knn, window_size, score, offset):
    # Compute ClaSP profile for change point detection
    n_timepoints = knn.shape[0]
    profile = np.full(shape=n_timepoints, fill_value=-np.inf, dtype=np.float32)
    knn_counts, neigh_pos, y_true, y_pred = _init_labels(knn, offset)
    conf_matrix = _init_conf_matrix(y_true, y_pred)
    excl_zone = np.array([offset, offset + window_size])
    excl_conf_matrix = _init_conf_matrix(y_true[excl_zone[0]:excl_zone[1]],
                                         y_pred[excl_zone[0]:excl_zone[1]])
    conf_matrix = conf_matrix - excl_conf_matrix
    for split_idx in range(offset, n_timepoints - offset):
        profile[split_idx] = score(conf_matrix)
        _update_labels(
            split_idx,
            excl_zone,
            neigh_pos,
            knn_counts,
            y_true,
            y_pred,
            conf_matrix
        )
        excl_zone += 1
    return profile

def calc_class(ts_stream, score, offset, return_knn=False):
    # Calculate classification profile from time series stream
    knn = ts_stream.knns[ts_stream.lbound:ts_stream.knn_insert_idx] - ts_stream.lbound
    knn = np.clip(knn, 0, knn.shape[0] - 1)
    profile = _fast_profile(knn, ts_stream.window_size, score, offset)
    if return_knn:
        return profile, knn
    return profile