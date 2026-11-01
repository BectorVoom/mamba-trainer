//! Plain-Rust `f64` reference of the Mamba-1 encoder, reading the weights of a
//! live [`MambaEncoder`]. The device forward is tested against it.

use mamba3::backend::FloatElem;
use mamba3::cubecl::prelude::Runtime;
use mamba3::nn::{Linear, Param, RmsNorm};
use mamba3::ssm::selective::{SelectiveDims, SelectiveInputs, selective_scan_reference};

use crate::mamba::{MambaBlock, MambaEncoder};

fn host<R: Runtime, E: FloatElem>(p: &Param<R, E>) -> Vec<f64> {
    p.value().to_f32().into_iter().map(f64::from).collect()
}

/// `[rows, in] -> [rows, out]` with the layer's `[in, out]` weight.
fn linear<R: Runtime, E: FloatElem>(l: &Linear<R, E>, x: &[f64], rows: usize) -> Vec<f64> {
    let (i, o) = (l.in_features(), l.out_features());
    let w = host(l.weight());
    let b = l.bias().map(host);
    let mut out = vec![0.0; rows * o];
    for r in 0..rows {
        for c in 0..o {
            let mut acc = b.as_ref().map_or(0.0, |b| b[c]);
            for k in 0..i {
                acc += x[r * i + k] * w[k * o + c];
            }
            out[r * o + c] = acc;
        }
    }
    out
}

fn rms_norm<R: Runtime, E: FloatElem>(n: &RmsNorm<R, E>, x: &[f64], rows: usize) -> Vec<f64> {
    let d = n.dim();
    let w = n.weight().map(host);
    let mut out = vec![0.0; rows * d];
    for r in 0..rows {
        let row = &x[r * d..(r + 1) * d];
        let ms = row.iter().map(|v| v * v).sum::<f64>() / d as f64;
        let inv = 1.0 / (ms + n.eps() as f64).sqrt();
        for c in 0..d {
            out[r * d + c] = row[c] * inv * w.as_ref().map_or(1.0, |w| w[c]);
        }
    }
    out
}

fn silu(v: f64) -> f64 {
    v / (1.0 + (-v).exp())
}

fn softplus(v: f64) -> f64 {
    v.max(0.0) + (-v.abs()).exp().ln_1p()
}

/// One block, `[B, T, d_model]`.
pub fn block_reference<R: Runtime, E: FloatElem>(
    blk: &MambaBlock<R, E>,
    x: &[f64],
    batch: usize,
    seq: usize,
) -> Vec<f64> {
    let rows = batch * seq;
    let di = blk.d_inner;
    let n = blk.d_state;
    let xn = rms_norm(&blk.norm, x, rows);
    let uz = linear(&blk.in_proj, &xn, rows);
    // Depthwise causal conv: tap j multiplies the input delayed by k-1-j.
    let k = blk.conv.kernel_size();
    let cw = host(blk.conv.weight());
    let cb = blk.conv.bias().map(host);
    let mut u = vec![0.0; rows * di];
    for b in 0..batch {
        for t in 0..seq {
            for c in 0..di {
                let mut acc = cb.as_ref().map_or(0.0, |v| v[c]);
                for j in 0..k {
                    let delay = k - 1 - j;
                    if t >= delay {
                        acc += cw[j * di + c] * uz[((b * seq + t - delay) * 2 * di) + c];
                    }
                }
                u[(b * seq + t) * di + c] = silu(acc);
            }
        }
    }
    let xp = linear(&blk.x_proj, &u, rows);
    let width = blk.rank + 2 * n;
    let mut dtr = vec![0.0; rows * blk.rank];
    let mut bm = vec![0.0; rows * n];
    let mut cm = vec![0.0; rows * n];
    for r in 0..rows {
        dtr[r * blk.rank..(r + 1) * blk.rank].copy_from_slice(&xp[r * width..r * width + blk.rank]);
        bm[r * n..(r + 1) * n].copy_from_slice(&xp[r * width + blk.rank..r * width + blk.rank + n]);
        cm[r * n..(r + 1) * n].copy_from_slice(&xp[r * width + blk.rank + n..(r + 1) * width]);
    }
    let dt_lin = linear(&blk.dt_proj, &dtr, rows);
    let bias = host(&blk.dt_bias);
    let dt: Vec<f64> = dt_lin.iter().enumerate().map(|(i, v)| softplus(v + bias[i % di])).collect();
    let a: Vec<f64> = host(&blk.a_log).into_iter().map(|v| -v.exp()).collect();
    let skip = host(&blk.d_skip);
    let y = selective_scan_reference(
        SelectiveDims {
            batch,
            seq,
            dim: di,
            state: n,
        },
        &SelectiveInputs {
            x: &u,
            dt: &dt,
            a: &a,
            bm: &bm,
            cm: &cm,
            skip: Some(&skip),
        },
    );
    let mut gated = vec![0.0; rows * di];
    for r in 0..rows {
        for c in 0..di {
            gated[r * di + c] = y[r * di + c] * silu(uz[r * 2 * di + di + c]);
        }
    }
    let out = linear(&blk.out_proj, &gated, rows);
    x.iter().zip(out).map(|(a, b)| a + b).collect()
}

/// The encoder's per-step states, `[B, T, F] -> [B, T, d_model]`.
pub fn encoder_reference<R: Runtime, E: FloatElem>(
    enc: &MambaEncoder<R, E>,
    x: &[f64],
    batch: usize,
    seq: usize,
) -> Vec<f64> {
    let rows = batch * seq;
    let mut h = linear(&enc.input, x, rows);
    for blk in &enc.blocks {
        h = block_reference(blk, &h, batch, seq);
    }
    rms_norm(&enc.norm, &h, rows)
}
