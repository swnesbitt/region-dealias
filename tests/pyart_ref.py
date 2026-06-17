"""Reference oracle: Py-ART's region-dealiasing per-sweep algorithm, in pure
Python/numpy, for the non-reference-anchored path the app uses.

The Python pieces are transcribed verbatim from
``pyart/correct/region_dealias.py`` and the Cython ``_fast_edge_finder.pyx``
(translated to a Python loop). Used as the parity oracle for the Rust port so
tests don't require building the full pyart package. CI additionally checks
against the installed pyart.
"""

import numpy as np
import scipy.ndimage as ndimage


def _find_sweep_interval_splits(nyquist, interval_splits, velocities, nsweep):
    add_start = add_end = 0
    interval = (2.0 * nyquist) / (interval_splits)
    if len(velocities) != 0:
        max_vel = velocities.max()
        min_vel = velocities.min()
        if max_vel > nyquist or min_vel < -nyquist:
            add_start = int(np.ceil((max_vel - nyquist) / (interval)))
            add_end = int(np.ceil(-(min_vel + nyquist) / (interval)))
    start = -nyquist - add_start * interval
    end = nyquist + add_end * interval
    num = interval_splits + 1 + add_start + add_end
    return np.linspace(start, end, num, endpoint=True)


def _find_regions(vel, gfilter, limits):
    mask = ~gfilter
    label = np.zeros(vel.shape, dtype=np.int32)
    nfeatures = 0
    for lmin, lmax in zip(limits[:-1], limits[1:]):
        inp = (lmin <= vel) & (vel < lmax) & mask
        limit_label, limit_nfeatures = ndimage.label(inp)
        limit_label[np.nonzero(limit_label)] += nfeatures
        label += limit_label
        nfeatures += limit_nfeatures
    return label, nfeatures


def _fast_edge_finder(labels, data, rays_wrap_around, max_gap_x, max_gap_y, total_nodes):
    """Pure-Python transcription of the Cython routine."""
    l_index, n_index, l_velo, n_velo = [], [], [], []

    def add_edge(label, neighbor, vel, nvel):
        if neighbor == label or neighbor == 0:
            return
        l_index.append(label)
        n_index.append(neighbor)
        l_velo.append(vel)
        n_velo.append(nvel)

    right = labels.shape[0] - 1
    bottom = labels.shape[1] - 1
    for x_index in range(labels.shape[0]):
        for y_index in range(labels.shape[1]):
            label = labels[x_index, y_index]
            if label == 0:
                continue
            vel = data[x_index, y_index]

            # left
            x_check = x_index - 1
            if x_check == -1 and rays_wrap_around:
                x_check = right
            if x_check != -1:
                neighbor = labels[x_check, y_index]
                nvel = data[x_check, y_index]
                if neighbor == 0:
                    for _ in range(max_gap_x):
                        x_check -= 1
                        if x_check == -1:
                            if rays_wrap_around:
                                x_check = right
                            else:
                                break
                        neighbor = labels[x_check, y_index]
                        nvel = data[x_check, y_index]
                        if neighbor != 0:
                            break
                add_edge(label, neighbor, vel, nvel)

            # right
            x_check = x_index + 1
            if x_check == right + 1 and rays_wrap_around:
                x_check = 0
            if x_check != right + 1:
                neighbor = labels[x_check, y_index]
                nvel = data[x_check, y_index]
                if neighbor == 0:
                    for _ in range(max_gap_x):
                        x_check += 1
                        if x_check == right + 1:
                            if rays_wrap_around:
                                x_check = 0
                            else:
                                break
                        neighbor = labels[x_check, y_index]
                        nvel = data[x_check, y_index]
                        if neighbor != 0:
                            break
                add_edge(label, neighbor, vel, nvel)

            # top
            y_check = y_index - 1
            if y_check != -1:
                neighbor = labels[x_index, y_check]
                nvel = data[x_index, y_check]
                if neighbor == 0:
                    for _ in range(max_gap_y):
                        y_check -= 1
                        if y_check == -1:
                            break
                        neighbor = labels[x_index, y_check]
                        nvel = data[x_index, y_check]
                        if neighbor != 0:
                            break
                add_edge(label, neighbor, vel, nvel)

            # bottom
            y_check = y_index + 1
            if y_check != bottom + 1:
                neighbor = labels[x_index, y_check]
                nvel = data[x_index, y_check]
                if neighbor == 0:
                    for _ in range(max_gap_y):
                        y_check += 1
                        if y_check == bottom + 1:
                            break
                        neighbor = labels[x_index, y_check]
                        nvel = data[x_index, y_check]
                        if neighbor != 0:
                            break
                add_edge(label, neighbor, vel, nvel)

    indices = (np.array(l_index, dtype=np.int32), np.array(n_index, dtype=np.int32))
    velocities = (np.array(l_velo, dtype=np.float64), np.array(n_velo, dtype=np.float64))
    return indices, velocities


def _edge_sum_and_count(labels, num_masked_gates, data, rays_wrap_around, max_gap_x, max_gap_y):
    total_nodes = labels.shape[0] * labels.shape[1] - num_masked_gates
    if rays_wrap_around:
        total_nodes += labels.shape[0] * 2
    indices, velocities = _fast_edge_finder(
        labels.astype("int32"), data.astype("float32"),
        rays_wrap_around, max_gap_x, max_gap_y, total_nodes,
    )
    index1, index2 = indices
    vel1, vel2 = velocities
    count = np.ones_like(vel1, dtype=np.int32)
    if len(vel1) == 0:
        return ([], []), [], ([], [])
    order = np.lexsort((index1, index2))
    index1 = index1[order]; index2 = index2[order]
    vel1 = vel1[order]; vel2 = vel2[order]; count = count[order]
    unique_mask = (index1[1:] != index1[:-1]) | (index2[1:] != index2[:-1])
    unique_mask = np.append(True, unique_mask)
    index1 = index1[unique_mask]; index2 = index2[unique_mask]
    (unique_inds,) = np.nonzero(unique_mask)
    vel1 = np.add.reduceat(vel1, unique_inds, dtype=vel1.dtype)
    vel2 = np.add.reduceat(vel2, unique_inds, dtype=vel2.dtype)
    count = np.add.reduceat(count, unique_inds, dtype=count.dtype)
    return (index1, index2), count, (vel1, vel2)


def _combine_regions(region_tracker, edge_tracker):
    status, extra = edge_tracker.pop_edge()
    if status:
        return True
    node1, node2, weight, diff, edge_number = extra
    rdiff = int(np.round(diff))
    node1_size = region_tracker.get_node_size(node1)
    node2_size = region_tracker.get_node_size(node2)
    if node1_size > node2_size:
        base_node, merge_node = node1, node2
    else:
        base_node, merge_node = node2, node1
        rdiff = -rdiff
    if rdiff != 0:
        region_tracker.unwrap_node(merge_node, rdiff)
        edge_tracker.unwrap_node(merge_node, rdiff)
    region_tracker.merge_nodes(base_node, merge_node)
    edge_tracker.merge_nodes(base_node, merge_node, edge_number)
    return False


class _RegionTracker:
    def __init__(self, region_sizes):
        nregions = len(region_sizes) + 1
        self.node_size = np.zeros(nregions, dtype="int32")
        self.node_size[1:] = region_sizes[:]
        self.regions_in_node = np.zeros(nregions, dtype="object")
        for i in range(nregions):
            self.regions_in_node[i] = [i]
        self.unwrap_number = np.zeros(nregions, dtype="int32")

    def merge_nodes(self, node_a, node_b):
        regions_to_merge = self.regions_in_node[node_b]
        self.regions_in_node[node_a].extend(regions_to_merge)
        self.regions_in_node[node_b] = []
        self.node_size[node_a] += self.node_size[node_b]
        self.node_size[node_b] = 0

    def unwrap_node(self, node, nwrap):
        if nwrap == 0:
            return
        regions_to_unwrap = self.regions_in_node[node]
        self.unwrap_number[regions_to_unwrap] += nwrap

    def get_node_size(self, node):
        return self.node_size[node]


class _EdgeTracker:
    def __init__(self, indices, edge_count, velocities, nyquist_interval, nnodes):
        nedges = int(len(indices[0]) / 2)
        self.node_alpha = np.zeros(nedges, dtype=np.int32)
        self.node_beta = np.zeros(nedges, dtype=np.int32)
        self.sum_diff = np.zeros(nedges, dtype=np.float32)
        self.weight = np.zeros(nedges, dtype=np.int32)
        self._common_finder = np.zeros(nnodes, dtype=np.bool_)
        self._common_index = np.zeros(nnodes, dtype=np.int32)
        self._last_base_node = -1
        self.edges_in_node = np.zeros(nnodes, dtype="object")
        for i in range(nnodes):
            self.edges_in_node[i] = []
        edge = 0
        idx1, idx2 = indices
        vel1, vel2 = velocities
        for i, j, count, vel, nvel in zip(idx1, idx2, edge_count, vel1, vel2):
            if i < j:
                continue
            self.node_alpha[edge] = i
            self.node_beta[edge] = j
            self.sum_diff[edge] = (vel - nvel) / nyquist_interval
            self.weight[edge] = count
            self.edges_in_node[i].append(edge)
            self.edges_in_node[j].append(edge)
            edge += 1
        self.priority_queue = []

    def merge_nodes(self, base_node, merge_node, foo_edge):
        self.weight[foo_edge] = -999
        self.edges_in_node[merge_node].remove(foo_edge)
        self.edges_in_node[base_node].remove(foo_edge)
        self._common_finder[merge_node] = False
        edges_in_merge = list(self.edges_in_node[merge_node])
        if self._last_base_node != base_node:
            self._common_finder[:] = False
            edges_in_base = list(self.edges_in_node[base_node])
            for edge_num in edges_in_base:
                if self.node_beta[edge_num] == base_node:
                    self._reverse_edge_direction(edge_num)
                assert self.node_alpha[edge_num] == base_node
                neighbor = self.node_beta[edge_num]
                self._common_finder[neighbor] = True
                self._common_index[neighbor] = edge_num
        for edge_num in edges_in_merge:
            if self.node_beta[edge_num] == merge_node:
                self._reverse_edge_direction(edge_num)
            assert self.node_alpha[edge_num] == merge_node
            self.node_alpha[edge_num] = base_node
            neighbor = self.node_beta[edge_num]
            if self._common_finder[neighbor]:
                base_edge_num = self._common_index[neighbor]
                self._combine_edges(base_edge_num, edge_num, merge_node, neighbor)
            else:
                self._common_finder[neighbor] = True
                self._common_index[neighbor] = edge_num
        edges = self.edges_in_node[merge_node]
        self.edges_in_node[base_node].extend(edges)
        self.edges_in_node[merge_node] = []
        self._last_base_node = int(base_node)

    def _combine_edges(self, base_edge, merge_edge, merge_node, neighbor_node):
        self.weight[base_edge] += self.weight[merge_edge]
        self.weight[merge_edge] = -999.0
        self.sum_diff[base_edge] += self.sum_diff[merge_edge]
        self.edges_in_node[merge_node].remove(merge_edge)
        self.edges_in_node[neighbor_node].remove(merge_edge)

    def _reverse_edge_direction(self, edge):
        old_alpha = int(self.node_alpha[edge])
        old_beta = int(self.node_beta[edge])
        self.node_alpha[edge] = old_beta
        self.node_beta[edge] = old_alpha
        self.sum_diff[edge] = -1.0 * self.sum_diff[edge]

    def unwrap_node(self, node, nwrap):
        if nwrap == 0:
            return
        for edge in self.edges_in_node[node]:
            weight = self.weight[edge]
            if node == self.node_alpha[edge]:
                self.sum_diff[edge] += weight * nwrap
            else:
                assert self.node_beta[edge] == node
                self.sum_diff[edge] += -weight * nwrap

    def pop_edge(self):
        edge_num = np.argmax(self.weight)
        node1 = self.node_alpha[edge_num]
        node2 = self.node_beta[edge_num]
        weight = self.weight[edge_num]
        diff = self.sum_diff[edge_num] / (float(weight))
        if weight < 0:
            return True, None
        return False, (node1, node2, weight, diff, edge_num)


def ref_sweep_folds(vel, gfilter, nyquist, rays_wrap_around,
                    interval_splits=3, skip_between_rays=100,
                    skip_along_ray=100, centered=True):
    """Return per-gate fold counts for one sweep (pyart reference path)."""
    sdata = np.asarray(vel, dtype=np.float32)
    sfilter = np.asarray(gfilter, dtype=bool)
    nyquist_interval = nyquist * 2.0
    folds = np.zeros(sdata.shape, dtype=np.int32)

    valid_sdata = sdata[~sfilter]
    limits = _find_sweep_interval_splits(nyquist, interval_splits, valid_sdata, 0)
    labels, nfeatures = _find_regions(sdata, sfilter, limits)
    if nfeatures < 2:
        return folds
    bincount = np.bincount(labels.ravel())
    num_masked_gates = bincount[0]
    region_sizes = bincount[1:]
    indices, edge_count, velos = _edge_sum_and_count(
        labels, num_masked_gates, sdata, rays_wrap_around,
        skip_between_rays, skip_along_ray)
    if len(edge_count) == 0:
        return folds
    region_tracker = _RegionTracker(region_sizes)
    edge_tracker = _EdgeTracker(indices, edge_count, velos, nyquist_interval, nfeatures + 1)
    while True:
        if _combine_regions(region_tracker, edge_tracker):
            break
    if centered:
        gates_dealiased = region_sizes.sum()
        total_folds = np.sum(region_sizes * region_tracker.unwrap_number[1:])
        sweep_offset = int(round(float(total_folds) / gates_dealiased))
        if sweep_offset != 0:
            region_tracker.unwrap_number -= sweep_offset
    return np.take(region_tracker.unwrap_number, labels).astype(np.int32)
