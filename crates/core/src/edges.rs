//! Edge enumeration between labeled regions — a faithful port of pyart's
//! Cython `_fast_edge_finder` plus the `_edge_sum_and_count` dedup
//! (`lexsort((index1, index2))` then `add.reduceat`).

/// Unique directed edges between regions, ordered by `(index2, index1)`
/// ascending — the order pyart feeds into the edge tracker.
pub struct Edges {
    pub i: Vec<i32>,    // index1 (label side)
    pub j: Vec<i32>,    // index2 (neighbour side)
    pub count: Vec<i32>,
    pub vel1: Vec<f64>, // summed label-side velocities
    pub vel2: Vec<f64>, // summed neighbour-side velocities
}

pub fn edge_sum_and_count(
    labels: &[i32],
    data: &[f32],
    nx: usize,
    ny: usize,
    wrap: bool,
    max_gap_x: i64,
    max_gap_y: i64,
) -> Edges {
    let right = nx as i32 - 1;
    let bottom = ny as i32 - 1;
    let at = |x: i32, y: i32| -> usize { (x as usize) * ny + (y as usize) };

    // raw directed edges in discovery (raster) order
    let mut raw: Vec<(i32, i32, f64, f64)> = Vec::new();
    let mut add_edge = |label: i32, neighbor: i32, vel: f64, nvel: f64| {
        if neighbor == label || neighbor == 0 {
            return;
        }
        raw.push((label, neighbor, vel, nvel));
    };

    for x in 0..nx as i32 {
        for y in 0..ny as i32 {
            let label = labels[at(x, y)];
            if label == 0 {
                continue;
            }
            let vel = data[at(x, y)] as f64;

            // left
            let mut xc = x - 1;
            if xc == -1 && wrap {
                xc = right;
            }
            if xc != -1 {
                let mut neighbor = labels[at(xc, y)];
                let mut nvel = data[at(xc, y)] as f64;
                if neighbor == 0 {
                    for _ in 0..max_gap_x {
                        xc -= 1;
                        if xc == -1 {
                            if wrap {
                                xc = right;
                            } else {
                                break;
                            }
                        }
                        neighbor = labels[at(xc, y)];
                        nvel = data[at(xc, y)] as f64;
                        if neighbor != 0 {
                            break;
                        }
                    }
                }
                add_edge(label, neighbor, vel, nvel);
            }

            // right
            let mut xc = x + 1;
            if xc == right + 1 && wrap {
                xc = 0;
            }
            if xc != right + 1 {
                let mut neighbor = labels[at(xc, y)];
                let mut nvel = data[at(xc, y)] as f64;
                if neighbor == 0 {
                    for _ in 0..max_gap_x {
                        xc += 1;
                        if xc == right + 1 {
                            if wrap {
                                xc = 0;
                            } else {
                                break;
                            }
                        }
                        neighbor = labels[at(xc, y)];
                        nvel = data[at(xc, y)] as f64;
                        if neighbor != 0 {
                            break;
                        }
                    }
                }
                add_edge(label, neighbor, vel, nvel);
            }

            // top
            let mut yc = y - 1;
            if yc != -1 {
                let mut neighbor = labels[at(x, yc)];
                let mut nvel = data[at(x, yc)] as f64;
                if neighbor == 0 {
                    for _ in 0..max_gap_y {
                        yc -= 1;
                        if yc == -1 {
                            break;
                        }
                        neighbor = labels[at(x, yc)];
                        nvel = data[at(x, yc)] as f64;
                        if neighbor != 0 {
                            break;
                        }
                    }
                }
                add_edge(label, neighbor, vel, nvel);
            }

            // bottom
            let mut yc = y + 1;
            if yc != bottom + 1 {
                let mut neighbor = labels[at(x, yc)];
                let mut nvel = data[at(x, yc)] as f64;
                if neighbor == 0 {
                    for _ in 0..max_gap_y {
                        yc += 1;
                        if yc == bottom + 1 {
                            break;
                        }
                        neighbor = labels[at(x, yc)];
                        nvel = data[at(x, yc)] as f64;
                        if neighbor != 0 {
                            break;
                        }
                    }
                }
                add_edge(label, neighbor, vel, nvel);
            }
        }
    }

    if raw.is_empty() {
        return Edges {
            i: vec![],
            j: vec![],
            count: vec![],
            vel1: vec![],
            vel2: vec![],
        };
    }

    // dedup: stable lexsort by (index2, index1) then sum duplicates per group.
    // Stable sort preserves discovery order within a group so the float64
    // accumulation order matches numpy's add.reduceat.
    let mut order: Vec<usize> = (0..raw.len()).collect();
    order.sort_by(|&a, &b| (raw[a].1, raw[a].0).cmp(&(raw[b].1, raw[b].0)));

    let mut edges = Edges {
        i: vec![],
        j: vec![],
        count: vec![],
        vel1: vec![],
        vel2: vec![],
    };
    let mut k = 0usize;
    while k < order.len() {
        let (i0, j0) = (raw[order[k]].0, raw[order[k]].1);
        let mut s1 = 0.0f64;
        let mut s2 = 0.0f64;
        let mut c = 0i32;
        while k < order.len() && raw[order[k]].0 == i0 && raw[order[k]].1 == j0 {
            s1 += raw[order[k]].2;
            s2 += raw[order[k]].3;
            c += 1;
            k += 1;
        }
        edges.i.push(i0);
        edges.j.push(j0);
        edges.vel1.push(s1);
        edges.vel2.push(s2);
        edges.count.push(c);
    }
    edges
}
