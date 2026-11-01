//! Sequential `f64` reference of the Mamba-1 selective scan and its adjoint.
//!
//! The definition the CubeCL kernels in [`crate::tensor::ops::selective_scan`] are
//! tested against; plain loops, no device. Shapes as there: `x, dt : [B, T, D]`,
//! `a : [D, N]`, `bm, cm : [B, T, N]`, `skip : [D]`, all row-major.

/// Problem sizes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SelectiveDims {
    /// Batch.
    pub batch: usize,
    /// Time steps.
    pub seq: usize,
    /// Channels.
    pub dim: usize,
    /// State columns per channel.
    pub state: usize,
}

/// Inputs of one reference scan.
#[derive(Debug, Clone, Copy)]
pub struct SelectiveInputs<'a> {
    /// `[B, T, D]`.
    pub x: &'a [f64],
    /// `[B, T, D]`, positive.
    pub dt: &'a [f64],
    /// `[D, N]`, negative.
    pub a: &'a [f64],
    /// `[B, T, N]`.
    pub bm: &'a [f64],
    /// `[B, T, N]`.
    pub cm: &'a [f64],
    /// `[D]`.
    pub skip: Option<&'a [f64]>,
}

/// Gradients of the reference scan.
#[derive(Debug, Clone, Default)]
pub struct SelectiveRefGrads {
    /// `[B, T, D]`.
    pub dx: Vec<f64>,
    /// `[B, T, D]`.
    pub ddt: Vec<f64>,
    /// `[D, N]`.
    pub da: Vec<f64>,
    /// `[B, T, N]`.
    pub dbm: Vec<f64>,
    /// `[B, T, N]`.
    pub dcm: Vec<f64>,
    /// `[D]`.
    pub dskip: Vec<f64>,
}

fn check(d: SelectiveDims, inp: &SelectiveInputs<'_>) {
    let act = d.batch * d.seq * d.dim;
    let bc = d.batch * d.seq * d.state;
    assert_eq!(inp.x.len(), act, "x");
    assert_eq!(inp.dt.len(), act, "dt");
    assert_eq!(inp.a.len(), d.dim * d.state, "a");
    assert_eq!(inp.bm.len(), bc, "bm");
    assert_eq!(inp.cm.len(), bc, "cm");
    if let Some(s) = inp.skip {
        assert_eq!(s.len(), d.dim, "skip");
    }
}

/// States `h[b, t, d, n]` (after step `t`), `[B, T, D, N]`.
pub fn selective_states(d: SelectiveDims, inp: &SelectiveInputs<'_>) -> Vec<f64> {
    check(d, inp);
    let mut h = vec![0.0; d.batch * d.seq * d.dim * d.state];
    for b in 0..d.batch {
        for ch in 0..d.dim {
            let mut s = vec![0.0f64; d.state];
            for t in 0..d.seq {
                let row = b * d.seq + t;
                let xv = inp.x[row * d.dim + ch];
                let dtv = inp.dt[row * d.dim + ch];
                for n in 0..d.state {
                    let decay = (dtv * inp.a[ch * d.state + n]).exp();
                    s[n] = decay * s[n] + dtv * inp.bm[row * d.state + n] * xv;
                    h[(row * d.dim + ch) * d.state + n] = s[n];
                }
            }
        }
    }
    h
}

/// The scan output `y`, `[B, T, D]`.
pub fn selective_scan_reference(d: SelectiveDims, inp: &SelectiveInputs<'_>) -> Vec<f64> {
    let h = selective_states(d, inp);
    let mut y = vec![0.0; d.batch * d.seq * d.dim];
    for row in 0..d.batch * d.seq {
        for ch in 0..d.dim {
            let mut acc = 0.0;
            for n in 0..d.state {
                acc += inp.cm[row * d.state + n] * h[(row * d.dim + ch) * d.state + n];
            }
            let skip = inp.skip.map_or(0.0, |s| s[ch]);
            y[row * d.dim + ch] = acc + skip * inp.x[row * d.dim + ch];
        }
    }
    y
}

/// The analytic adjoint for `dy = dL/dy`, written straight from the recurrence
/// (`g[t] = Cm[t] dy[t] + a[t+1] g[t+1]`), independent of the kernel's
/// segmentation.
pub fn selective_scan_reference_backward(
    d: SelectiveDims,
    inp: &SelectiveInputs<'_>,
    dy: &[f64],
) -> SelectiveRefGrads {
    check(d, inp);
    assert_eq!(dy.len(), d.batch * d.seq * d.dim, "dy");
    let h = selective_states(d, inp);
    let mut out = SelectiveRefGrads {
        dx: vec![0.0; dy.len()],
        ddt: vec![0.0; dy.len()],
        da: vec![0.0; d.dim * d.state],
        dbm: vec![0.0; d.batch * d.seq * d.state],
        dcm: vec![0.0; d.batch * d.seq * d.state],
        dskip: vec![0.0; d.dim],
    };
    for b in 0..d.batch {
        for ch in 0..d.dim {
            let mut g = vec![0.0f64; d.state];
            for t in (0..d.seq).rev() {
                let row = b * d.seq + t;
                let i = row * d.dim + ch;
                let (xv, dtv, gy) = (inp.x[i], inp.dt[i], dy[i]);
                let mut gx = 0.0;
                let mut gdt = 0.0;
                for n in 0..d.state {
                    let an = inp.a[ch * d.state + n];
                    let decay = (dtv * an).exp();
                    let hp = if t == 0 { 0.0 } else { h[((row - 1) * d.dim + ch) * d.state + n] };
                    let ht = h[(row * d.dim + ch) * d.state + n];
                    let bv = inp.bm[row * d.state + n];
                    let cv = inp.cm[row * d.state + n];
                    g[n] += cv * gy;
                    gx += g[n] * dtv * bv;
                    gdt += g[n] * (an * decay * hp + bv * xv);
                    out.da[ch * d.state + n] += g[n] * dtv * decay * hp;
                    out.dbm[row * d.state + n] += g[n] * dtv * xv;
                    out.dcm[row * d.state + n] += gy * ht;
                    g[n] *= decay;
                }
                let skip = inp.skip.map_or(0.0, |s| s[ch]);
                out.dx[i] = gx + skip * gy;
                out.ddt[i] = gdt;
                out.dskip[ch] += gy * xv;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(seed: &mut u64) -> f64 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*seed >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    }

    struct Problem {
        d: SelectiveDims,
        x: Vec<f64>,
        dt: Vec<f64>,
        a: Vec<f64>,
        bm: Vec<f64>,
        cm: Vec<f64>,
        skip: Vec<f64>,
    }

    impl Problem {
        fn new(d: SelectiveDims, seed: u64) -> Self {
            let mut s = seed;
            let act = d.batch * d.seq * d.dim;
            let bc = d.batch * d.seq * d.state;
            Self {
                d,
                x: (0..act).map(|_| lcg(&mut s)).collect(),
                dt: (0..act).map(|_| 0.05 + 0.5 * (lcg(&mut s) + 1.0)).collect(),
                a: (0..d.dim * d.state).map(|_| -0.1 - 2.0 * (lcg(&mut s) + 1.0)).collect(),
                bm: (0..bc).map(|_| lcg(&mut s)).collect(),
                cm: (0..bc).map(|_| lcg(&mut s)).collect(),
                skip: (0..d.dim).map(|_| lcg(&mut s)).collect(),
            }
        }
        fn inputs(&self) -> SelectiveInputs<'_> {
            SelectiveInputs {
                x: &self.x,
                dt: &self.dt,
                a: &self.a,
                bm: &self.bm,
                cm: &self.cm,
                skip: Some(&self.skip),
            }
        }
    }

    /// The analytic adjoint against central finite differences of
    /// `L = sum(w * y)`, for every input.
    #[test]
    fn reference_backward_matches_finite_differences() {
        let d = SelectiveDims {
            batch: 2,
            seq: 6,
            dim: 3,
            state: 4,
        };
        let p = Problem::new(d, 7);
        let mut s = 99u64;
        let w: Vec<f64> = (0..d.batch * d.seq * d.dim).map(|_| lcg(&mut s)).collect();
        let loss = |p: &Problem| -> f64 {
            selective_scan_reference(p.d, &p.inputs())
                .iter()
                .zip(&w)
                .map(|(y, w)| y * w)
                .sum()
        };
        let g = selective_scan_reference_backward(d, &p.inputs(), &w);
        let eps = 1e-6;
        let fields: [(&str, &Vec<f64>, fn(&mut Problem) -> &mut Vec<f64>); 6] = [
            ("x", &g.dx, |p| &mut p.x),
            ("dt", &g.ddt, |p| &mut p.dt),
            ("a", &g.da, |p| &mut p.a),
            ("bm", &g.dbm, |p| &mut p.bm),
            ("cm", &g.dcm, |p| &mut p.cm),
            ("skip", &g.dskip, |p| &mut p.skip),
        ];
        for (name, analytic, field) in fields {
            for i in 0..analytic.len() {
                let mut plus = Problem::new(d, 7);
                field(&mut plus)[i] += eps;
                let mut minus = Problem::new(d, 7);
                field(&mut minus)[i] -= eps;
                let fd = (loss(&plus) - loss(&minus)) / (2.0 * eps);
                assert!(
                    (fd - analytic[i]).abs() <= 1e-6 * (1.0 + fd.abs()),
                    "{name}[{i}]: finite difference {fd} vs analytic {}",
                    analytic[i]
                );
            }
        }
    }

    /// `y[t]` depends on inputs at `<= t` only.
    #[test]
    fn reference_is_causal() {
        let d = SelectiveDims {
            batch: 1,
            seq: 9,
            dim: 2,
            state: 3,
        };
        let base = Problem::new(d, 3);
        let y0 = selective_scan_reference(d, &base.inputs());
        for cut in 0..d.seq {
            let mut p = Problem::new(d, 3);
            for t in cut + 1..d.seq {
                for ch in 0..d.dim {
                    p.x[t * d.dim + ch] += 1.0;
                    p.dt[t * d.dim + ch] *= 1.5;
                }
                for n in 0..d.state {
                    p.bm[t * d.state + n] -= 0.7;
                    p.cm[t * d.state + n] += 0.3;
                }
            }
            let y = selective_scan_reference(d, &p.inputs());
            for t in 0..=cut {
                for ch in 0..d.dim {
                    assert_eq!(y[t * d.dim + ch], y0[t * d.dim + ch], "t {t} cut {cut}");
                }
            }
        }
    }
}
