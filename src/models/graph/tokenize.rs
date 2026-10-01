//! The host tokeniser: the test oracle of the device token kernel
//! (GRAPH_MAMBA_PLAN.md §2.3).
//!
//! [`tokens_host`] samples the same random-walk tokens as
//! [`crate::tensor::ops::graph::walk_tokens`] with ordinary loops: the same
//! hash, the same visit order, the same weights. It is not on any training
//! path — tokens are sampled on the device, every step — and exists so that the
//! kernel can be checked bit for bit.

use crate::tensor::ops::IGNORE;
use crate::tensor::ops::graph::WalkShape;

/// The host twin of [`crate::tensor::ops::random::hash_u32`].
#[inline]
pub fn hash_u32_host(index: u32, seed_lo: u32, seed_hi: u32) -> u32 {
    let mut h = index ^ seed_lo;
    h ^= h >> 16;
    h = h.wrapping_mul(0x7feb352d);
    h ^= h >> 15;
    h = h.wrapping_mul(0x846ca68b);
    h ^= seed_hi;
    h ^= h >> 16;
    h
}

/// Tokens sampled on the host, in the layout of
/// [`crate::tensor::ops::graph::Tokens`].
#[derive(Debug, Clone, PartialEq)]
pub struct HostTokens {
    /// `[rows · L · C]` node ids, visit order, [`IGNORE`] padded.
    pub node: Vec<u32>,
    /// `[rows · L · C]` weights.
    pub w: Vec<f32>,
    /// `[rows · L · 3]`: `ln(1 + |T|)`, `ln(1 + |E_T|)`, walk length over `m`.
    pub stats: Vec<f32>,
}

/// Sample the tokens of `nodes` — one row each, [`IGNORE`] for an absent row —
/// on the graph `(adj_off, adj_col)`, exactly as the device kernel does.
pub fn tokens_host(
    adj_off: &[u32],
    adj_col: &[u32],
    nodes: &[u32],
    shape: WalkShape,
    seed: (u32, u32),
    counter: u32,
) -> HostTokens {
    let (hops, walks, repeats) = (shape.hops as u32, shape.walks as u32, shape.repeats as u32);
    let (len, cap) = (shape.len(), shape.cap());
    let mut out = HostTokens {
        node: vec![IGNORE; nodes.len() * len * cap],
        w: vec![0.0; nodes.len() * len * cap],
        stats: vec![0.0; nodes.len() * len * 3],
    };
    let row_of = |u: u32| &adj_col[adj_off[u as usize] as usize..adj_off[u as usize + 1] as usize];

    for (r, &v) in nodes.iter().enumerate() {
        if v == IGNORE {
            continue;
        }
        for t in 0..len as u32 {
            let pos = r * len + t as usize;
            let base = pos * cap;
            let hop = if t < hops * repeats {
                hops - t / repeats
            } else {
                0
            };
            let slots = &mut out.node[base..base + cap];
            slots[0] = v;
            let mut count = 1usize;
            if hop > 0 {
                let k0 = hash_u32_host(v, seed.0, seed.1);
                let k1 = hash_u32_host(
                    k0.wrapping_add(counter.wrapping_mul(0x9E3779B1)),
                    0x85EBCA6B,
                    0xC2B2AE35,
                );
                for k in 0..walks {
                    let k2 = hash_u32_host(
                        k1.wrapping_add((t * walks + k).wrapping_mul(0x27D4EB2F)),
                        0x165667B1,
                        0x9E3779B9,
                    );
                    let mut u = v;
                    for q in 0..hop {
                        let row = row_of(u);
                        if row.is_empty() {
                            continue;
                        }
                        let draw = hash_u32_host(
                            k2.wrapping_add(q.wrapping_mul(0x85EBCA77)),
                            0xC2B2AE3D,
                            0x27D4EB2F,
                        );
                        u = row[(draw % row.len() as u32) as usize];
                        if !slots[..count].contains(&u) {
                            slots[count] = u;
                            count += 1;
                        }
                    }
                }
            }

            let mut twice_edges = 0u32;
            if shape.sgc {
                for i in 0..count {
                    let row = row_of(slots[i]);
                    let inside = (0..count)
                        .filter(|&j| j != i && row.binary_search(&slots[j]).is_ok())
                        .count() as u32;
                    out.w[base + i] = (1 + inside) as f32;
                    twice_edges += inside;
                }
                let total = (count as u32 + twice_edges) as f32;
                for i in 0..count {
                    out.w[base + i] /= total;
                }
            } else {
                for i in 0..count {
                    out.w[base + i] = 1.0 / count as f32;
                }
            }
            out.stats[pos * 3] = (1.0 + count as f32).ln();
            out.stats[pos * 3 + 1] = (1.0 + 0.5 * twice_edges as f32).ln();
            out.stats[pos * 3 + 2] = hop as f32 / hops as f32;
        }
    }
    out
}
