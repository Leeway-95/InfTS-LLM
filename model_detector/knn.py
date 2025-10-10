import numpy as np
from numba import njit, objmode
from utils.tools import _rolling_knn, _sliding_dot, _argkmin, _roll_numba, _mean, _std

def _knn(knn_insert_idx, l, fill, window_size, dot_rolled, first, time_series, means, stds, similarity, csum, csumsq, dcsum, exclusion_radius, k_neighbours, lbound):
    # 计算时间序列的K最近邻
    idx = knn_insert_idx
    start_idx = lbound - 1
    valid_dist = slice(start_idx, l)
    dist = np.full(shape=l, fill_value=np.inf, dtype=np.float64)
    if first:
        dot_rolled[valid_dist] = _sliding_dot(time_series[idx:idx + window_size], time_series[-fill:])
    else:
        dot_rolled = dot_rolled + time_series[idx + window_size - 1] * time_series[window_size - 1:]
        if start_idx >= 0:
            dot_rolled[start_idx] = np.dot(time_series[start_idx:start_idx + window_size], time_series[idx:idx + window_size])
    rolled_dist = None
    csumsq_win = csumsq[window_size:] - csumsq[:-window_size]
    ed = -2 * dot_rolled + csumsq_win + csumsq_win[idx]
    ce = dcsum[window_size:] - dcsum[:-window_size]
    ce = np.where(ce == 0, 1.0, ce)
    last_ce = np.repeat(ce[idx], ce.shape[0])
    last_ce = np.where(last_ce == 0, 1.0, last_ce)
    with objmode(cf="float64[:]"):
        max_vals = np.maximum(ce, last_ce)
        min_vals = np.minimum(ce, last_ce)
        cf = max_vals / min_vals
    rolled_dist = ed * cf
    if rolled_dist is not None:
        dist[valid_dist] = rolled_dist[valid_dist]
    excl_range = slice(max(0, idx - exclusion_radius), min(idx + exclusion_radius, l))
    dist[excl_range] = np.max(dist)
    knns = _argkmin(dist, k_neighbours, lbound)
    dot_rolled -= time_series[idx] * time_series[:l]
    return dot_rolled, dist, knns


def _roll_all(time_series, timepoint, csum, csumsq, fill, dcsum, window_size, means, stds):
    # 更新所有滚动计算的数组
    time_series = _roll_numba(time_series, -1, timepoint)
    csum = _roll_numba(csum, -1, csum[-1] + timepoint)
    csumsq = _roll_numba(csumsq, -1, csumsq[-1] + timepoint ** 2)
    if fill > 1:
        dcsum = _roll_numba(dcsum, -1, dcsum[-1] + np.square(timepoint - time_series[-2]))
    if fill >= window_size:
        means = _roll_numba(means, -1, _mean(len(time_series) - window_size, csum, window_size))
        stds = _roll_numba(stds, -1, _std(len(time_series) - window_size, csumsq, csum, window_size))
    return time_series, csum, csumsq, dcsum, means, stds

class TimeSeriesStream:
    # 时间序列流处理类，用于实时计算K最近邻
    def __init__(self, window_size, n_timepoints, k_neighbours=3, similarity="pearson"):
        self.window_size = window_size
        self.exclusion_radius = int(window_size / 2)
        self.n_timepoints = n_timepoints
        self.k_neighbours = k_neighbours
        self.similarity = similarity
        self.lbound = 0
        self.time_series = np.full(shape=n_timepoints, fill_value=np.nan, dtype=np.float64)
        self.l = n_timepoints - window_size + 1
        self.knn_insert_idx = self.l - self.exclusion_radius - self.k_neighbours - 1
        self.csum = np.full(shape=n_timepoints + 1, fill_value=0, dtype=np.float64)
        self.csumsq = np.full(shape=n_timepoints + 1, fill_value=0, dtype=np.float64)
        self.dcsum = np.full(shape=n_timepoints + 1, fill_value=0, dtype=np.float64)
        self.means = np.full(shape=self.l, fill_value=np.nan, dtype=np.float64)
        self.stds = np.full(shape=self.l, fill_value=np.nan, dtype=np.float64)
        self.dists = np.full(shape=(self.l, k_neighbours), fill_value=np.inf, dtype=np.float64)
        self.knns = np.full(shape=(self.l, k_neighbours), fill_value=-1, dtype=np.int64)
        self.dot_rolled = None
        self.fill = 0
        self.knn_fill = 0

    def knn(self):
        # 计算当前时间点的K最近邻
        first = self.dot_rolled is None
        if self.dot_rolled is None:
            self.dot_rolled = np.full(shape=self.l, fill_value=np.inf, dtype=np.float64)
        self.dot_rolled, dist, knns = _knn(self.knn_insert_idx, self.l, self.fill, self.window_size, self.dot_rolled, first, self.time_series, self.means, self.stds, self.similarity, self.csum, self.csumsq, self.dcsum, self.exclusion_radius, self.k_neighbours, self.lbound)
        return dist, knns

    def update(self, timepoint):
        # 更新时间序列流并计算新的统计量
        if not np.isfinite(timepoint):
            timepoint = 0.0
        self.fill = min(self.fill + 1, self.l)
        self.time_series, self.csum, self.csumsq, self.dcsum, self.means, self.stds = _roll_all(self.time_series, timepoint, self.csum, self.csumsq, self.fill, self.dcsum, self.window_size, self.means, self.stds)
        if self.fill < self.window_size + self.exclusion_radius + self.k_neighbours:
            return self
        self._update_knn()
        return self

    def _update_knn(self):
        # 更新K最近邻结果
        if self.knn_fill > 0:
            self.dists = _roll_numba(self.dists, -1, np.full(shape=self.dists.shape[1], fill_value=np.inf, dtype=np.float64))
            self.knns = _roll_numba(self.knns, -1)
            self.knns[self.knn_insert_idx - self.knn_fill:self.knn_insert_idx] -= 1
            self.knns[-1, :] = np.full(shape=self.dists.shape[1], fill_value=-1, dtype=np.int64)
        dist, knn = self.knn()
        self.knns, self.dists, self.lbound, self.knn_fill = _rolling_knn(self.dists, self.knns, dist, knn, self.knn_insert_idx, self.knn_fill, self.l, self.k_neighbours, self.lbound)
        return self