//! Connected-component labeling that reproduces `scipy.ndimage.label` with the
//! default 4-connectivity structure, **including its label numbering**: regions
//! are numbered in order of the first pixel encountered in C-order (row-major)
//! raster scan. The downstream algorithm depends on that exact numbering.

fn find(parent: &mut [i32], mut x: i32) -> i32 {
    // path halving
    while parent[x as usize] != x {
        parent[x as usize] = parent[parent[x as usize] as usize];
        x = parent[x as usize];
    }
    x
}

/// Label a boolean mask (`inp`, C-order, shape `nx × ny`) with 4-connectivity.
/// Returns `(labels, nfeatures)`; `labels[i] == 0` for background.
pub fn label4(inp: &[bool], nx: usize, ny: usize) -> (Vec<i32>, i32) {
    let n = nx * ny;
    let mut parent: Vec<i32> = vec![-1; n];
    for i in 0..n {
        if inp[i] {
            parent[i] = i as i32;
        }
    }
    // union pass: each foreground pixel unions with its already-seen
    // left (same row, prev col) and up (prev row, same col) neighbours.
    for x in 0..nx {
        for y in 0..ny {
            let i = x * ny + y;
            if !inp[i] {
                continue;
            }
            if y > 0 && inp[i - 1] {
                let ri = find(&mut parent, i as i32);
                let rl = find(&mut parent, (i - 1) as i32);
                if ri != rl {
                    parent[rl as usize] = ri;
                }
            }
            if x > 0 && inp[i - ny] {
                let ri = find(&mut parent, i as i32);
                let ru = find(&mut parent, (i - ny) as i32);
                if ri != ru {
                    parent[ru as usize] = ri;
                }
            }
        }
    }
    // relabel by first-encounter in raster order -> matches scipy numbering.
    let mut out = vec![0i32; n];
    let mut root_label = vec![0i32; n];
    let mut next = 0i32;
    for i in 0..n {
        if !inp[i] {
            continue;
        }
        let r = find(&mut parent, i as i32) as usize;
        if root_label[r] == 0 {
            next += 1;
            root_label[r] = next;
        }
        out[i] = root_label[r];
    }
    (out, next)
}
