//! Dynamic network reduction that assigns a fold (unwrap) count to each region
//! — a faithful port of pyart's `_RegionTracker`, `_EdgeTracker` and
//! `_combine_regions`. Float dtypes are preserved exactly: `sum_diff` is float32,
//! widened to float64 only for the `diff = sum_diff / weight` division;
//! `weight*nwrap` adjustments are done in float64 then stored back to float32.

use crate::edges::Edges;
use crate::util::rint_i64;

const NEG: i32 = -999;

struct Tracker {
    // region tracker
    node_size: Vec<i32>,
    regions_in_node: Vec<Vec<i32>>,
    unwrap_number: Vec<i32>,
    // edge tracker
    node_alpha: Vec<i32>,
    node_beta: Vec<i32>,
    sum_diff: Vec<f32>,
    weight: Vec<i32>,
    edges_in_node: Vec<Vec<i32>>,
    common_finder: Vec<bool>,
    common_index: Vec<i32>,
    last_base_node: i32,
}

fn vec_remove(v: &mut Vec<i32>, val: i32) {
    if let Some(pos) = v.iter().position(|&e| e == val) {
        v.remove(pos);
    }
}

impl Tracker {
    fn reverse_edge(&mut self, e: usize) {
        let a = self.node_alpha[e];
        self.node_alpha[e] = self.node_beta[e];
        self.node_beta[e] = a;
        self.sum_diff[e] = -self.sum_diff[e];
    }

    fn combine_edges(&mut self, base_edge: usize, merge_edge: usize, merge_node: i32, neighbor: i32) {
        self.weight[base_edge] += self.weight[merge_edge];
        self.weight[merge_edge] = NEG;
        self.sum_diff[base_edge] += self.sum_diff[merge_edge]; // float32 + float32
        vec_remove(&mut self.edges_in_node[merge_node as usize], merge_edge as i32);
        vec_remove(&mut self.edges_in_node[neighbor as usize], merge_edge as i32);
    }

    fn edge_unwrap_node(&mut self, node: i32, nwrap: i32) {
        if nwrap == 0 {
            return;
        }
        let edges = self.edges_in_node[node as usize].clone();
        for e in edges {
            let e = e as usize;
            let w = self.weight[e] as i64;
            let delta = if node == self.node_alpha[e] {
                (w * nwrap as i64) as f64
            } else {
                (-w * nwrap as i64) as f64
            };
            self.sum_diff[e] = ((self.sum_diff[e] as f64) + delta) as f32;
        }
    }

    fn edge_merge_nodes(&mut self, base_node: i32, merge_node: i32, foo_edge: usize) {
        self.weight[foo_edge] = NEG;
        vec_remove(&mut self.edges_in_node[merge_node as usize], foo_edge as i32);
        vec_remove(&mut self.edges_in_node[base_node as usize], foo_edge as i32);
        self.common_finder[merge_node as usize] = false;

        let edges_in_merge = self.edges_in_node[merge_node as usize].clone();

        if self.last_base_node != base_node {
            for v in self.common_finder.iter_mut() {
                *v = false;
            }
            let edges_in_base = self.edges_in_node[base_node as usize].clone();
            for en in edges_in_base {
                let en = en as usize;
                if self.node_beta[en] == base_node {
                    self.reverse_edge(en);
                }
                let neighbor = self.node_beta[en];
                self.common_finder[neighbor as usize] = true;
                self.common_index[neighbor as usize] = en as i32;
            }
        }

        for en in edges_in_merge {
            let en = en as usize;
            if self.node_beta[en] == merge_node {
                self.reverse_edge(en);
            }
            self.node_alpha[en] = base_node;
            let neighbor = self.node_beta[en];
            if self.common_finder[neighbor as usize] {
                let base_edge = self.common_index[neighbor as usize] as usize;
                self.combine_edges(base_edge, en, merge_node, neighbor);
            } else {
                self.common_finder[neighbor as usize] = true;
                self.common_index[neighbor as usize] = en as i32;
            }
        }

        let remaining = self.edges_in_node[merge_node as usize].clone();
        self.edges_in_node[base_node as usize].extend(remaining);
        self.edges_in_node[merge_node as usize].clear();
        self.last_base_node = base_node;
    }

    fn region_unwrap_node(&mut self, node: i32, nwrap: i32) {
        if nwrap == 0 {
            return;
        }
        for &r in &self.regions_in_node[node as usize] {
            self.unwrap_number[r as usize] += nwrap;
        }
    }

    fn region_merge_nodes(&mut self, a: i32, b: i32) {
        let moved = std::mem::take(&mut self.regions_in_node[b as usize]);
        self.regions_in_node[a as usize].extend(moved);
        self.node_size[a as usize] += self.node_size[b as usize];
        self.node_size[b as usize] = 0;
    }

    /// `np.argmax(weight)` — index of the first maximum.
    fn argmax_weight(&self) -> usize {
        let mut best = 0usize;
        let mut bestw = self.weight[0];
        for e in 1..self.weight.len() {
            if self.weight[e] > bestw {
                bestw = self.weight[e];
                best = e;
            }
        }
        best
    }
}

/// Run the network reduction; returns `unwrap_number` indexed by region label
/// (length `nfeatures + 1`, index 0 = masked/background).
pub fn network_reduce(
    edges: &Edges,
    region_sizes: &[i32],
    nfeatures: i32,
    nyquist_interval: f64,
    centered: bool,
) -> Vec<i32> {
    let nnodes = (nfeatures + 1) as usize;

    // --- region tracker init ---
    let mut node_size = vec![0i32; nnodes];
    node_size[1..].copy_from_slice(region_sizes);
    let regions_in_node: Vec<Vec<i32>> = (0..nnodes as i32).map(|i| vec![i]).collect();
    let unwrap_number = vec![0i32; nnodes];

    // --- edge tracker init ---
    let nedges = edges.i.len() / 2;
    let mut node_alpha = vec![0i32; nedges];
    let mut node_beta = vec![0i32; nedges];
    let mut sum_diff = vec![0f32; nedges];
    let mut weight = vec![0i32; nedges];
    let mut edges_in_node: Vec<Vec<i32>> = vec![Vec::new(); nnodes];

    let mut edge = 0usize;
    for k in 0..edges.i.len() {
        let i = edges.i[k];
        let j = edges.j[k];
        if i < j {
            continue;
        }
        node_alpha[edge] = i;
        node_beta[edge] = j;
        sum_diff[edge] = ((edges.vel1[k] - edges.vel2[k]) / nyquist_interval) as f32;
        weight[edge] = edges.count[k];
        edges_in_node[i as usize].push(edge as i32);
        edges_in_node[j as usize].push(edge as i32);
        edge += 1;
    }

    let mut t = Tracker {
        node_size,
        regions_in_node,
        unwrap_number,
        node_alpha,
        node_beta,
        sum_diff,
        weight,
        edges_in_node,
        common_finder: vec![false; nnodes],
        common_index: vec![0i32; nnodes],
        last_base_node: -1,
    };

    if nedges == 0 {
        return t.unwrap_number;
    }

    // --- combine loop ---
    loop {
        let edge_num = t.argmax_weight();
        let w = t.weight[edge_num];
        if w < 0 {
            break;
        }
        let node1 = t.node_alpha[edge_num];
        let node2 = t.node_beta[edge_num];
        let diff = (t.sum_diff[edge_num] as f64) / (w as f64);
        let rdiff0 = rint_i64(diff) as i32;

        let node1_size = t.node_size[node1 as usize];
        let node2_size = t.node_size[node2 as usize];
        let (base, merge, rdiff) = if node1_size > node2_size {
            (node1, node2, rdiff0)
        } else {
            (node2, node1, -rdiff0)
        };

        if rdiff != 0 {
            t.region_unwrap_node(merge, rdiff);
            t.edge_unwrap_node(merge, rdiff);
        }
        t.region_merge_nodes(base, merge);
        t.edge_merge_nodes(base, merge, edge_num);
    }

    // --- centering: shift so the average gate fold is ~0 ---
    if centered {
        let gates_dealiased: i64 = region_sizes.iter().map(|&s| s as i64).sum();
        if gates_dealiased > 0 {
            let total_folds: i64 = (0..nfeatures as usize)
                .map(|r| region_sizes[r] as i64 * t.unwrap_number[r + 1] as i64)
                .sum();
            let sweep_offset = rint_i64(total_folds as f64 / gates_dealiased as f64) as i32;
            if sweep_offset != 0 {
                for v in t.unwrap_number.iter_mut() {
                    *v -= sweep_offset;
                }
            }
        }
    }

    t.unwrap_number
}
