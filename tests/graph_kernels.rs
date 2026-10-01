//! GM2: the device store and the graph data-path kernels against host loops.
//!
//! Every test holds [`LOCK`]: the launch, read and upload counters are
//! process-wide, and the tests that pin them must not run beside the others.

#![cfg(feature = "backend")]

use mamba3::autograd::Var;
use mamba3::backend::{
    DType, Device, FloatElem, launch_count, read_count, reset_launch_count, reset_read_count,
    reset_upload_count, supports_dtype, upload_count,
};
use mamba3::backends::Auto;
use mamba3::models::graph::{
    CanonicalGraph, CanonicalLabels, CanonicalizeOptions, FeatureTable, Features, GraphData,
    GraphStore, Labels, NodeOrder, SPLIT_TEST, SPLIT_TRAIN, SPLIT_VAL, Splits, TargetStore,
    canonicalize, hash_u32_host, tokens_host,
};
use mamba3::tensor::ops::graph::{
    BatchRows, EpochTable, SignFlip, WalkShape, batch_rows_graphs, batch_rows_subset, confusion,
    edge_inputs, hash_ids, masked_mean, masked_mean_backward, node_inputs, pad_ragged,
    safe_class_targets, safe_float_targets, segment_broadcast, segment_pool, token_features,
    unpad_ragged, walk_tokens,
};
use mamba3::tensor::ops::index::IdTensor;
use mamba3::tensor::ops::IGNORE;
use mamba3::tensor::Tensor;

type R = Auto;

static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn lock() -> std::sync::MutexGuard<'static, ()> {
    LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn dev() -> Device<R> {
    Device::<R>::default()
}

/// A small deterministic generator.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }

    fn below(&mut self, n: usize) -> usize {
        self.next() as usize % n
    }

    /// In `[-1, 1)`.
    fn unit(&mut self) -> f32 {
        self.next() as f32 / (1u64 << 30) as f32 - 1.0
    }
}

/// Float features: `dim` values in `[-1, 1)` per row.
fn float_features(rows: usize, dim: usize, seed: u64) -> Features {
    let mut rng = Lcg(seed);
    Features::Float {
        dim,
        data: (0..rows * dim).map(|_| rng.unit()).collect(),
    }
}

/// `edges` random undirected edges inside `lo..hi`.
fn random_edges(lo: usize, hi: usize, edges: usize, rng: &mut Lcg, out: &mut (Vec<u32>, Vec<u32>)) {
    if hi - lo < 2 {
        return;
    }
    for _ in 0..edges {
        let a = lo + rng.below(hi - lo);
        let b = lo + rng.below(hi - lo);
        if a != b {
            out.0.push(a as u32);
            out.1.push(b as u32);
        }
    }
}

/// One random graph; node `n - 1` is left isolated.
fn random_graph(n: usize, edges: usize, seed: u64) -> GraphData {
    let mut rng = Lcg(seed);
    let mut pairs = (Vec::new(), Vec::new());
    random_edges(0, n - 1, edges, &mut rng, &mut pairs);
    GraphData::new(n, pairs.0, pairs.1, float_features(n, 6, seed + 1))
}

/// Many graphs of the given sizes, about two edges per node.
fn many_graphs(sizes: &[usize], seed: u64) -> GraphData {
    let mut rng = Lcg(seed);
    let mut pairs = (Vec::new(), Vec::new());
    let mut ptr = vec![0u32];
    let mut at = 0;
    for &size in sizes {
        random_edges(at, at + size, 2 * size, &mut rng, &mut pairs);
        at += size;
        ptr.push(at as u32);
    }
    let mut data = GraphData::new(at, pairs.0, pairs.1, float_features(at, 4, seed + 1));
    data.graph_ptr = ptr;
    data
}

fn canon_of<E: FloatElem>(data: &GraphData, order: NodeOrder, reverse: bool) -> CanonicalGraph<E> {
    canonicalize::<E>(
        &data.view(),
        &CanonicalizeOptions {
            order,
            symmetrize: true,
            reverse_index: reverse,
        },
    )
    .unwrap()
}

fn upload<E: FloatElem>(canon: &CanonicalGraph<E>, device: &Device<R>) -> GraphStore<R, E> {
    GraphStore::upload(canon.clone(), device).unwrap()
}

/// The whole graph as one batch, in canonical order.
fn all_rows<E: FloatElem>(store: &GraphStore<R, E>) -> BatchRows<R> {
    batch_rows_subset(store.adjacency(), 1, 0, (0, 0), 1).unwrap()
}

fn float_table<E: FloatElem>(table: &FeatureTable<E>) -> (usize, Vec<f32>) {
    match table {
        FeatureTable::Float { dim, data } => (*dim, E::slice_to_f32(data)),
        FeatureTable::Categorical { .. } => panic!("expected float features"),
    }
}

fn assert_close(actual: &[f32], expected: &[f32], tol: f32, what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}: length");
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - e).abs() <= tol * (1.0 + e.abs()),
            "{what}: index {i} got {a}, want {e}"
        );
    }
}

/// Run `f`, asserting it issued `launches` launches, no read and no upload.
fn counted<T>(launches: usize, what: &str, f: impl FnOnce() -> T) -> T {
    reset_launch_count();
    reset_read_count();
    reset_upload_count();
    let out = f();
    assert_eq!(launch_count(), launches, "{what}: launches");
    assert_eq!(read_count(), 0, "{what}: reads");
    assert_eq!(upload_count(), 0, "{what}: uploads");
    out
}

// ---------------------------------------------------------------------------
// Hash
// ---------------------------------------------------------------------------

#[test]
fn host_hash_equals_the_kernel() {
    let _guard = lock();
    let device = dev();
    let mut rng = Lcg(7);
    let mut ids: Vec<u32> = (0..10_000).map(|_| rng.next() ^ (rng.next() << 1)).collect();
    ids[0] = 0;
    ids[1] = u32::MAX;
    for seed in [(0u32, 0u32), (0x1234_5678, 0x9abc_def0), (u32::MAX, 1)] {
        let on_device = hash_ids(
            &IdTensor::from_slice(&ids, vec![ids.len()], &device).unwrap(),
            seed,
        )
        .to_vec();
        for (i, &id) in ids.iter().enumerate() {
            assert_eq!(on_device[i], hash_u32_host(id, seed.0, seed.1), "id {id}");
        }
    }
}

// ---------------------------------------------------------------------------
// G1
// ---------------------------------------------------------------------------

#[test]
fn whole_graph_batches_match_a_concatenation() {
    let _guard = lock();
    let device = dev();
    let sizes = [5usize, 9, 3, 12, 7, 4, 10];
    let data = many_graphs(&sizes, 11);
    let canon = canon_of::<f32>(&data, NodeOrder::default(), true);
    let store = upload(&canon, &device);
    let batches = vec![vec![2u32, 0, 5], vec![3, 1], vec![4, 6]];
    let table = EpochTable::build(&batches, &canon.graph_ptr, &canon.edge_ptr, 8, &device).unwrap();
    assert_eq!(table.len(), 3);
    assert_eq!(table.graphs(), 8);
    let (row_cap, edge_cap) = (32usize, 128usize);
    let dst = canon.adj_dst.as_ref().unwrap();

    for (index, batch) in batches.iter().enumerate() {
        let slots = table.slots(index).unwrap();
        let rows = counted(1, "batch_rows_graphs", || {
            batch_rows_graphs(store.adjacency(), &slots, row_cap, Some(edge_cap)).unwrap()
        });
        // Brute force: concatenate the graphs.
        let mut want_gid = Vec::new();
        let mut want_slot = Vec::new();
        let mut want_len = vec![0u32; 8];
        let mut want_eid = Vec::new();
        let mut want_src = Vec::new();
        let mut want_dst = Vec::new();
        for (slot, &g) in batch.iter().enumerate() {
            let (a, b) = (canon.graph_ptr[g as usize], canon.graph_ptr[g as usize + 1]);
            let first_row = want_gid.len() as u32;
            want_len[slot] = b - a;
            for node in a..b {
                want_gid.push(node);
                want_slot.push(slot as u32);
            }
            for e in canon.edge_ptr[g as usize]..canon.edge_ptr[g as usize + 1] {
                want_eid.push(e);
                want_src.push(canon.adj_col[e as usize] - a + first_row);
                want_dst.push(dst[e as usize] - a + first_row);
            }
        }
        assert_eq!(slots.rows_used(), want_gid.len());
        assert_eq!(slots.edges_used(), want_eid.len());
        want_gid.resize(row_cap, IGNORE);
        want_slot.resize(row_cap, IGNORE);
        for v in [&mut want_eid, &mut want_src, &mut want_dst] {
            v.resize(edge_cap, IGNORE);
        }
        assert_eq!(rows.gid().to_vec(), want_gid, "gid of batch {index}");
        assert_eq!(rows.row_graph().to_vec(), want_slot, "slots of batch {index}");
        assert_eq!(rows.lengths().to_vec(), want_len, "lengths of batch {index}");
        let edges = rows.edges().unwrap();
        assert_eq!(edges.eid().to_vec(), want_eid, "edge ids of batch {index}");
        assert_eq!(edges.src().to_vec(), want_src, "edge sources of batch {index}");
        assert_eq!(edges.dst().to_vec(), want_dst, "edge destinations of batch {index}");

        // Without edge rows the node part is the same.
        let plain = batch_rows_graphs(store.adjacency(), &slots, row_cap, None).unwrap();
        assert_eq!(plain.gid().to_vec(), want_gid);
        assert!(plain.edges().is_none());
    }

    // Capacities are checked on the host.
    let slots = table.slots(0).unwrap();
    assert!(batch_rows_graphs(store.adjacency(), &slots, 8, None).is_err());
    assert!(batch_rows_graphs(store.adjacency(), &slots, 32, Some(4)).is_err());
    assert!(table.slots(3).is_err());
    assert!(EpochTable::<R>::build(&[vec![9]], &canon.graph_ptr, &canon.edge_ptr, 8, &device).is_err());
    assert!(
        EpochTable::<R>::build(&[vec![0; 9]], &canon.graph_ptr, &canon.edge_ptr, 8, &device)
            .is_err()
    );
}

#[test]
fn node_subsets_partition_the_nodes() {
    let _guard = lock();
    let device = dev();
    let n = 103;
    let canon = canon_of::<f32>(&random_graph(n, 250, 3), NodeOrder::default(), false);
    let store = upload(&canon, &device);
    for (parts, sequences) in [(4usize, 1usize), (4, 3), (1, 1), (7, 4), (103, 2)] {
        let blocks = n.div_ceil(parts);
        let mut seen = vec![0u32; n];
        for part in 0..parts {
            let rows = counted(1, "batch_rows_subset", || {
                batch_rows_subset(store.adjacency(), parts, part, (17, 99), sequences).unwrap()
            });
            assert_eq!(rows.rows(), blocks.div_ceil(sequences) * sequences);
            assert_eq!(rows.groups(), sequences);
            let gid = rows.gid().to_vec();
            let slot = rows.row_graph().to_vec();
            let row_of = rows.row_of().unwrap().to_vec();
            let lengths = rows.lengths().to_vec();
            let real: Vec<u32> = gid.iter().copied().filter(|&v| v != IGNORE).collect();
            // Ascending, and absent rows only at the end.
            assert!(real.windows(2).all(|w| w[0] < w[1]), "ascending");
            assert_eq!(&gid[..real.len()], &real[..], "absent rows come last");
            assert!(real.len() == blocks || real.len() == blocks - 1);
            for (r, &v) in gid.iter().enumerate() {
                assert_eq!(slot[r], if v == IGNORE { IGNORE } else { 0 });
                if v != IGNORE {
                    seen[v as usize] += 1;
                    assert_eq!(row_of[v as usize] as usize, r);
                    assert_eq!(v as usize / parts, r, "one node of every block");
                }
            }
            assert_eq!(
                row_of.iter().filter(|&&r| r != IGNORE).count(),
                real.len(),
                "row_of names exactly the batch's nodes"
            );
            for (k, &len) in lengths.iter().enumerate() {
                let want = (0..real.len()).filter(|r| r % sequences == k).count();
                assert_eq!(len as usize, want, "length of sequence {k}");
            }
        }
        assert!(seen.iter().all(|&c| c == 1), "every node exactly once");
    }
    // Another epoch seed, another partition.
    let a = batch_rows_subset(store.adjacency(), 4, 0, (17, 99), 1).unwrap().gid().to_vec();
    let b = batch_rows_subset(store.adjacency(), 4, 0, (18, 99), 1).unwrap().gid().to_vec();
    assert_ne!(a, b);
    assert!(batch_rows_subset(store.adjacency(), 0, 0, (0, 0), 1).is_err());
    assert!(batch_rows_subset(store.adjacency(), 4, 4, (0, 0), 1).is_err());
    assert!(batch_rows_subset(store.adjacency(), 104, 0, (0, 0), 1).is_err());
    assert!(batch_rows_subset(store.adjacency(), 4, 0, (0, 0), 0).is_err());
}

#[test]
fn a_launch_past_the_single_axis_cube_limit_runs() {
    let _guard = lock();
    let device = dev();
    // More than 65,535 cubes of 256 units: on wgpu the count spreads over Y.
    let n = 65_535 * 256 + 4_097;
    let off = vec![0u32; n + 1];
    let adjacency =
        mamba3::tensor::ops::graph::Adjacency::<R>::upload(off, Vec::new(), None, None, &device)
            .unwrap();
    let parts = 64;
    let rows = batch_rows_subset(&adjacency, parts, 5, (1, 2), 1).unwrap();
    let row_of = rows.row_of().unwrap().to_vec();
    let gid = rows.gid().to_vec();
    assert_eq!(gid.len(), n.div_ceil(parts));
    for &r in &[0usize, 1, gid.len() / 2, gid.len() - 2] {
        assert_eq!(row_of[gid[r] as usize] as usize, r);
    }
    // The last node's entry was written, not left as allocated.
    let last = row_of[n - 1];
    assert!(last == IGNORE || last as usize == (n - 1) / parts);
    assert_eq!(row_of.iter().filter(|&&r| r != IGNORE).count(), {
        gid.iter().filter(|&&v| v != IGNORE).count()
    });
}

// ---------------------------------------------------------------------------
// G2
// ---------------------------------------------------------------------------

/// Nodes within `hops` of `v`.
fn ball(canon: &CanonicalGraph<f32>, v: u32, hops: usize) -> Vec<bool> {
    let mut inside = vec![false; canon.n_nodes];
    inside[v as usize] = true;
    let mut frontier = vec![v];
    for _ in 0..hops {
        let mut next = Vec::new();
        for &u in &frontier {
            for e in canon.adj_off[u as usize]..canon.adj_off[u as usize + 1] {
                let c = canon.adj_col[e as usize];
                if !inside[c as usize] {
                    inside[c as usize] = true;
                    next.push(c);
                }
            }
        }
        frontier = next;
    }
    inside
}

#[test]
fn walk_tokens_equal_the_host_tokeniser() {
    let _guard = lock();
    let device = dev();
    let canon = canon_of::<f32>(&random_graph(300, 600, 5), NodeOrder::default(), false);
    let store = upload(&canon, &device);
    let rows = all_rows(&store);
    let gid = rows.gid().to_vec();
    assert_eq!(gid, (0..300).collect::<Vec<u32>>());

    for sgc in [false, true] {
        let shape = WalkShape {
            hops: 3,
            walks: 4,
            repeats: 2,
            sgc,
        };
        let (len, cap) = (shape.len(), shape.cap());
        assert_eq!((len, cap), (7, 13));
        let tokens = counted(1, "walk_tokens", || {
            walk_tokens(store.adjacency(), &rows, shape, (0xabcd, 0x1234), 9).unwrap()
        });
        let node = tokens.node().to_vec();
        let w = tokens.w().to_f32();
        let stats = tokens.stats().to_f32();
        let host = tokens_host(&canon.adj_off, &canon.adj_col, &gid, shape, (0xabcd, 0x1234), 9);
        assert_eq!(node, host.node, "token nodes (sgc = {sgc})");
        assert_close(&w, &host.w, 1e-6, "token weights");
        assert_close(&stats, &host.stats, 1e-6, "token statistics");

        for (r, &v) in gid.iter().enumerate() {
            for t in 0..len {
                let hop = if t < len - 1 { 3 - t / 2 } else { 0 };
                let slots = &node[(r * len + t) * cap..(r * len + t + 1) * cap];
                let weights = &w[(r * len + t) * cap..(r * len + t + 1) * cap];
                let size = slots.iter().filter(|&&s| s != IGNORE).count();
                assert_eq!(slots[0], v, "every token starts with its node");
                assert!(slots[..size].iter().all(|&s| s != IGNORE), "slots are packed");
                assert!(size <= 1 + shape.walks * hop, "|T| <= 1 + M·i");
                if hop == 0 {
                    assert_eq!(size, 1, "the last token is the node itself");
                }
                let reach = ball(&canon, v, hop);
                assert!(slots[..size].iter().all(|&s| reach[s as usize]), "within {hop} hops");
                let mut sorted = slots[..size].to_vec();
                sorted.sort_unstable();
                sorted.dedup();
                assert_eq!(sorted.len(), size, "no node twice");
                let sum: f32 = weights.iter().sum();
                assert!((sum - 1.0).abs() < 1e-5, "weights sum to one, got {sum}");
                assert!(weights[size..].iter().all(|&x| x == 0.0));
                let s = &stats[(r * len + t) * 3..(r * len + t) * 3 + 3];
                assert!((s[0] - (1.0 + size as f32).ln()).abs() < 1e-5);
                assert!((s[2] - hop as f32 / 3.0).abs() < 1e-6);
                if !sgc {
                    assert_eq!(s[1], 0.0, "Mean does no adjacency tests");
                }
            }
        }
        // The isolated node (degree 0, so first in ascending-degree order).
        assert_eq!(canon.adj_off[1], 0, "node 0 is isolated");
        for t in 0..len {
            let slots = &node[t * cap..(t + 1) * cap];
            assert_eq!(slots[0], 0);
            assert!(slots[1..].iter().all(|&s| s == IGNORE));
            assert_eq!(w[t * cap], 1.0);
        }
    }
}

#[test]
fn tokens_depend_on_the_node_not_on_the_batch() {
    let _guard = lock();
    let device = dev();
    let canon = canon_of::<f32>(&random_graph(120, 300, 8), NodeOrder::default(), false);
    let store = upload(&canon, &device);
    let shape = WalkShape {
        hops: 2,
        walks: 3,
        repeats: 2,
        sgc: true,
    };
    let (len, cap) = (shape.len(), shape.cap());
    let full = all_rows(&store);
    let full_tokens = walk_tokens(store.adjacency(), &full, shape, (5, 6), 2).unwrap();
    let (full_node, full_w) = (full_tokens.node().to_vec(), full_tokens.w().to_f32());
    let part = batch_rows_subset(store.adjacency(), 3, 1, (40, 41), 1).unwrap();
    let part_tokens = walk_tokens(store.adjacency(), &part, shape, (5, 6), 2).unwrap();
    let (part_node, part_w) = (part_tokens.node().to_vec(), part_tokens.w().to_f32());
    let per_row = len * cap;
    for (r, &v) in part.gid().to_vec().iter().enumerate() {
        let got = &part_node[r * per_row..(r + 1) * per_row];
        if v == IGNORE {
            assert!(got.iter().all(|&s| s == IGNORE), "an absent row has no token");
            assert!(part_w[r * per_row..(r + 1) * per_row].iter().all(|&x| x == 0.0));
            continue;
        }
        let v = v as usize;
        assert_eq!(got, &full_node[v * per_row..(v + 1) * per_row]);
        assert_eq!(
            &part_w[r * per_row..(r + 1) * per_row],
            &full_w[v * per_row..(v + 1) * per_row]
        );
    }
}

#[test]
fn resampling_changes_the_tokens_and_steps_are_uniform() {
    let _guard = lock();
    let device = dev();
    let canon = canon_of::<f32>(&random_graph(200, 800, 21), NodeOrder::default(), false);
    let store = upload(&canon, &device);
    let rows = all_rows(&store);
    let shape = WalkShape {
        hops: 3,
        walks: 4,
        repeats: 1,
        sgc: false,
    };
    let (len, cap) = (shape.len(), shape.cap());
    let sets = |counter: u32| -> Vec<Vec<u32>> {
        let node = walk_tokens(store.adjacency(), &rows, shape, (77, 78), counter)
            .unwrap()
            .node()
            .to_vec();
        node.chunks(cap)
            .map(|slots| {
                let mut set: Vec<u32> = slots.iter().copied().filter(|&s| s != IGNORE).collect();
                set.sort_unstable();
                set
            })
            .collect()
    };
    let (a, b, c) = (sets(0), sets(1), sets(2));
    // Tokens of walk length >= 2 are positions 0 and 1 of each row.
    let (mut total, mut ab, mut bc) = (0, 0, 0);
    for r in 0..200 {
        for t in 0..2 {
            let i = r * len + t;
            if a[i].len() > 2 {
                total += 1;
                ab += (a[i] != b[i]) as usize;
                bc += (b[i] != c[i]) as usize;
            }
        }
    }
    assert!(total > 300, "the graph is connected enough to test ({total})");
    assert!(ab * 10 > total * 8, "counters 0 and 1 share {} of {total} token sets", total - ab);
    assert!(bc * 10 > total * 8, "counters 1 and 2 share {} of {total} token sets", total - bc);

    // 20,000 centres, each joined to the same five hubs: with one walk of one
    // step, a centre's token is itself and the hub it stepped to.
    let centres = 20_000usize;
    let mut src = Vec::new();
    let mut dst = Vec::new();
    for c in 0..centres as u32 {
        for hub in 0..5u32 {
            src.push(5 + c);
            dst.push(hub);
        }
    }
    let data = GraphData::new(centres + 5, src, dst, float_features(centres + 5, 1, 1));
    let canon = canon_of::<f32>(&data, NodeOrder::Given, false);
    let store = upload(&canon, &device);
    let rows = all_rows(&store);
    let one_step = WalkShape {
        hops: 1,
        walks: 1,
        repeats: 1,
        sgc: false,
    };
    let node = walk_tokens(store.adjacency(), &rows, one_step, (3, 4), 0)
        .unwrap()
        .node()
        .to_vec();
    let per_row = one_step.len() * one_step.cap();
    let mut picks = [0usize; 5];
    for c in 0..centres {
        let stepped = node[(5 + c) * per_row + 1];
        assert!(stepped < 5, "a centre steps to a hub");
        picks[stepped as usize] += 1;
    }
    for (hub, &count) in picks.iter().enumerate() {
        let share = count as f32 / centres as f32;
        assert!((share - 0.2).abs() < 0.01, "hub {hub} took {share} of the first steps");
    }
}

#[test]
fn tokens_on_a_path_are_intervals_and_self_loops_do_not_count() {
    let _guard = lock();
    let device = dev();
    // A path 0 - 1 - … - 39 in the order given.
    let n = 40u32;
    let src: Vec<u32> = (0..n - 1).collect();
    let dst: Vec<u32> = (1..n).collect();
    let data = GraphData::new(n as usize, src, dst, float_features(n as usize, 1, 1));
    let canon = canon_of::<f32>(&data, NodeOrder::Given, false);
    let store = upload(&canon, &device);
    let rows = all_rows(&store);
    let shape = WalkShape {
        hops: 3,
        walks: 30,
        repeats: 1,
        sgc: true,
    };
    let node = walk_tokens(store.adjacency(), &rows, shape, (1, 1), 0)
        .unwrap()
        .node()
        .to_vec();
    let (len, cap) = (shape.len(), shape.cap());
    for v in 0..n as usize {
        for t in 0..len {
            let mut set: Vec<u32> = node[(v * len + t) * cap..(v * len + t + 1) * cap]
                .iter()
                .copied()
                .filter(|&s| s != IGNORE)
                .collect();
            set.sort_unstable();
            assert!(
                set.windows(2).all(|w| w[1] == w[0] + 1),
                "token of node {v} is not an interval: {set:?}"
            );
            assert!(set.contains(&(v as u32)));
        }
    }

    // A star with one extra edge, with and without self-loops on 0 and 3.
    let pairs = [(0u32, 1u32), (0, 2), (0, 3), (0, 4), (1, 2)];
    let build = |loops: bool| {
        let mut src: Vec<u32> = pairs.iter().map(|p| p.0).collect();
        let mut dst: Vec<u32> = pairs.iter().map(|p| p.1).collect();
        if loops {
            src.extend([0, 3]);
            dst.extend([0, 3]);
        }
        let data = GraphData::new(5, src, dst, float_features(5, 1, 1));
        let canon = canon_of::<f32>(&data, NodeOrder::Given, false);
        let store = upload(&canon, &device);
        let rows = all_rows(&store);
        let shape = WalkShape {
            hops: 3,
            walks: 40,
            repeats: 1,
            sgc: true,
        };
        let tokens = walk_tokens(store.adjacency(), &rows, shape, (9, 9), 0).unwrap();
        (tokens.node().to_vec(), tokens.w().to_f32(), tokens.stats().to_f32(), shape)
    };
    let want = [5.0 / 15.0, 3.0 / 15.0, 3.0 / 15.0, 2.0 / 15.0, 2.0 / 15.0];
    for loops in [false, true] {
        let (node, w, stats, shape) = build(loops);
        let cap = shape.cap();
        // The longest token of node 0 (position 0): all five nodes.
        let slots = &node[..cap];
        let size = slots.iter().filter(|&&s| s != IGNORE).count();
        assert_eq!(size, 5, "120 steps on five nodes visit all of them");
        for (slot, &id) in slots[..5].iter().enumerate() {
            assert!((w[slot] - want[id as usize]).abs() < 1e-6, "loops = {loops}, node {id}");
        }
        assert!((stats[1] - (1.0f32 + 5.0).ln()).abs() < 1e-6, "|E_T| = 5 (loops = {loops})");
    }
}

// ---------------------------------------------------------------------------
// G3 / G4
// ---------------------------------------------------------------------------

/// `x ‖ pe` of the canonical graph as `f32` rows of `F + P`.
fn joined_rows<E: FloatElem>(canon: &CanonicalGraph<E>) -> (usize, Vec<f32>) {
    let (f, x) = float_table(&canon.x);
    let (p, pe) = match &canon.pe {
        Some((dim, data)) => (*dim, E::slice_to_f32(data)),
        None => (0, Vec::new()),
    };
    let mut rows = Vec::with_capacity(canon.n_nodes * (f + p));
    for i in 0..canon.n_nodes {
        rows.extend_from_slice(&x[i * f..(i + 1) * f]);
        rows.extend_from_slice(&pe[i * p..(i + 1) * p]);
    }
    (f + p, rows)
}

fn float_feature_case<E: FloatElem>(tol: f32) {
    let device = dev();
    if !supports_dtype(&device, E::DTYPE) {
        println!("skipped: the {} backend has no {}", device.name(), E::DTYPE.name());
        return;
    }
    let mut data = random_graph(90, 220, 31);
    let mut rng = Lcg(5);
    data.pe = Some((2, (0..90 * 2).map(|_| rng.unit()).collect()));
    let canon = canon_of::<E>(&data, NodeOrder::default(), false);
    let store = upload(&canon, &device);
    let (width, table) = joined_rows(&canon);
    assert_eq!(width, 8);
    let shape = WalkShape {
        hops: 2,
        walks: 3,
        repeats: 2,
        sgc: true,
    };
    let (len, cap) = (shape.len(), shape.cap());

    for rows in [
        all_rows(&store),
        batch_rows_subset(store.adjacency(), 4, 2, (1, 2), 3).unwrap(),
    ] {
        let gid = rows.gid().to_vec();
        let tokens = walk_tokens(store.adjacency(), &rows, shape, (1, 2), 3).unwrap();
        let got = counted(1, "token_features", || {
            token_features(&store.x().source(), store.pe(), &tokens, &rows, None).unwrap()
        });
        assert_eq!(got.dims(), &[rows.rows() * len, width]);
        let (node, w) = (tokens.node().to_vec(), tokens.w().to_f32());
        let mut want = vec![0.0f32; rows.rows() * len * width];
        for token in 0..rows.rows() * len {
            for c in 0..cap {
                let id = node[token * cap + c];
                if id == IGNORE {
                    break;
                }
                for j in 0..width {
                    want[token * width + j] += w[token * cap + c] * table[id as usize * width + j];
                }
            }
        }
        // Accumulated in f32, cast once: compare with the cast of the reference.
        let want: Vec<f32> = E::slice_to_f32(&E::slice_from_f32(&want));
        assert_close(&got.to_f32(), &want, tol, "token features");

        let inputs = counted(1, "node_inputs", || {
            node_inputs(&store.x().source(), store.pe(), &rows, None).unwrap()
        });
        let mut want = vec![0.0f32; rows.rows() * width];
        for (r, &v) in gid.iter().enumerate() {
            if v != IGNORE {
                want[r * width..(r + 1) * width]
                    .copy_from_slice(&table[v as usize * width..(v as usize + 1) * width]);
            }
        }
        assert_eq!(inputs.to_f32(), want, "node inputs are an exact gather");
    }
}

#[test]
fn float_features_match_host_loops() {
    let _guard = lock();
    float_feature_case::<f32>(1e-6);
}

#[test]
fn float_features_match_host_loops_in_f16() {
    let _guard = lock();
    float_feature_case::<half::f16>(2e-3);
}

#[test]
fn float_features_match_host_loops_in_bf16() {
    let _guard = lock();
    float_feature_case::<half::bf16>(1e-2);
}

#[test]
fn features_without_an_encoding() {
    let _guard = lock();
    let device = dev();
    let canon = canon_of::<f32>(&random_graph(50, 120, 2), NodeOrder::default(), false);
    let store = upload(&canon, &device);
    assert!(store.pe().is_none());
    let (width, table) = joined_rows(&canon);
    let rows = all_rows(&store);
    let got = node_inputs(&store.x().source(), None, &rows, None).unwrap();
    assert_eq!(got.dims(), &[50, width]);
    assert_eq!(got.to_f32(), table);
}

#[test]
fn categorical_features_become_weighted_counts() {
    let _guard = lock();
    let device = dev();
    let n = 60;
    let mut data = random_graph(n, 150, 41);
    let vocab = vec![3usize, 5, 2];
    let mut rng = Lcg(9);
    let ids: Vec<u32> = (0..n)
        .flat_map(|_| vocab.iter().map(|&v| rng.below(v) as u32).collect::<Vec<_>>())
        .collect();
    data.x = Features::Categorical {
        fields: 3,
        vocab: vocab.clone(),
        ids,
    };
    data.pe = Some((3, (0..n * 3).map(|_| rng.unit()).collect()));
    let canon = canon_of::<f32>(&data, NodeOrder::default(), false);
    let store = upload(&canon, &device);
    let FeatureTable::Categorical { field_offset, ids } = &canon.x else {
        panic!("expected ids");
    };
    assert_eq!(field_offset, &vec![0, 3, 8, 10]);
    let pe = &canon.pe.as_ref().unwrap().1;
    let (v, p) = (10usize, 3usize);
    let width = v + p;
    // The multi-hot (and the encoding) of a node.
    let row_of = |node: usize| -> Vec<f32> {
        let mut row = vec![0.0f32; width];
        for f in 0..3 {
            row[field_offset[f] as usize + ids[node * 3 + f] as usize] = 1.0;
        }
        row[v..].copy_from_slice(&pe[node * p..(node + 1) * p]);
        row
    };
    let rows = batch_rows_subset(store.adjacency(), 2, 1, (4, 4), 1).unwrap();
    let gid = rows.gid().to_vec();
    let inputs = counted(1, "multi-hot inputs", || {
        node_inputs(&store.x().source(), store.pe(), &rows, None).unwrap()
    });
    let mut want = vec![0.0f32; rows.rows() * width];
    for (r, &node) in gid.iter().enumerate() {
        if node != IGNORE {
            want[r * width..(r + 1) * width].copy_from_slice(&row_of(node as usize));
        }
    }
    assert_eq!(inputs.to_f32(), want);

    let shape = WalkShape {
        hops: 2,
        walks: 4,
        repeats: 1,
        sgc: true,
    };
    let (len, cap) = (shape.len(), shape.cap());
    let tokens = walk_tokens(store.adjacency(), &rows, shape, (8, 8), 1).unwrap();
    let got = counted(1, "token counts", || {
        token_features(&store.x().source(), store.pe(), &tokens, &rows, None).unwrap()
    });
    let (node, w) = (tokens.node().to_vec(), tokens.w().to_f32());
    let mut want = vec![0.0f32; rows.rows() * len * width];
    for token in 0..rows.rows() * len {
        for c in 0..cap {
            let id = node[token * cap + c];
            if id == IGNORE {
                break;
            }
            let row = row_of(id as usize);
            for j in 0..width {
                want[token * width + j] += w[token * cap + c] * row[j];
            }
        }
    }
    assert_close(&got.to_f32(), &want, 1e-6, "token counts");
    // Each field's counts of a real token sum to one.
    let out = got.to_f32();
    for token in 0..rows.rows() * len {
        if node[token * cap] == IGNORE {
            continue;
        }
        for f in 0..3 {
            let sum: f32 = out[token * width + field_offset[f] as usize
                ..token * width + field_offset[f + 1] as usize]
                .iter()
                .sum();
            assert!((sum - 1.0).abs() < 1e-5);
        }
    }
}

#[test]
fn sign_flips_are_per_slot_and_per_column() {
    let _guard = lock();
    let device = dev();
    let sizes = [6usize, 5, 7, 4, 8, 6, 5, 9];
    let mut data = many_graphs(&sizes, 51);
    let n = data.n_nodes;
    let mut rng = Lcg(77);
    data.pe = Some((4, (0..n * 4).map(|_| 0.25 + rng.unit().abs()).collect()));
    let canon = canon_of::<f32>(&data, NodeOrder::default(), false);
    let store = upload(&canon, &device);
    let table = EpochTable::build(
        &[(0..8).collect::<Vec<u32>>()],
        &canon.graph_ptr,
        &canon.edge_ptr,
        8,
        &device,
    )
    .unwrap();
    let rows = batch_rows_graphs(store.adjacency(), &table.slots(0).unwrap(), 64, None).unwrap();
    let slot = rows.row_graph().to_vec();
    let plain = node_inputs(&store.x().source(), store.pe(), &rows, None)
        .unwrap()
        .to_f32();
    let flip = SignFlip {
        start: 1,
        end: 4,
        seed: (123, 456),
    };
    let flipped = counted(1, "flipped inputs", || {
        node_inputs(&store.x().source(), store.pe(), &rows, Some(flip)).unwrap()
    })
    .to_f32();
    let width = 4 + 4;
    let mut signs = std::collections::HashMap::new();
    for r in 0..64 {
        for j in 0..width {
            let (a, b) = (plain[r * width + j], flipped[r * width + j]);
            if slot[r] == IGNORE {
                assert_eq!((a, b), (0.0, 0.0));
            } else if j < 4 + 1 {
                assert_eq!(a, b, "features and the first encoding column do not flip");
            } else {
                assert_eq!(a.abs(), b.abs());
                let sign = (b / a) as i32;
                let seen = *signs.entry((slot[r], j)).or_insert(sign);
                assert_eq!(seen, sign, "one sign per slot and column");
            }
        }
    }
    let minus = signs.values().filter(|&&s| s == -1).count();
    assert_eq!(signs.len(), 8 * 3);
    assert!(minus > 4 && minus < 20, "{minus} of 24 signs are negative");
    // Another seed, other signs; the tokens of a row flip with their row.
    let other = node_inputs(
        &store.x().source(),
        store.pe(),
        &rows,
        Some(SignFlip { seed: (124, 456), ..flip }),
    )
    .unwrap()
    .to_f32();
    assert_ne!(other, flipped);
    let shape = WalkShape {
        hops: 1,
        walks: 2,
        repeats: 1,
        sgc: false,
    };
    let tokens = walk_tokens(store.adjacency(), &rows, shape, (1, 1), 0).unwrap();
    let plain_tok = token_features(&store.x().source(), store.pe(), &tokens, &rows, None)
        .unwrap()
        .to_f32();
    let flipped_tok = token_features(&store.x().source(), store.pe(), &tokens, &rows, Some(flip))
        .unwrap()
        .to_f32();
    for r in 0..64 {
        if slot[r] == IGNORE {
            continue;
        }
        for t in 0..shape.len() {
            for j in 5..width {
                let i = (r * shape.len() + t) * width + j;
                let sign = signs[&(slot[r], j)] as f32;
                assert!((flipped_tok[i] - sign * plain_tok[i]).abs() < 1e-6);
            }
        }
    }
    assert!(
        node_inputs(
            &store.x().source(),
            store.pe(),
            &rows,
            Some(SignFlip { start: 0, end: 5, seed: (0, 0) })
        )
        .is_err()
    );
}

#[test]
fn edge_inputs_gather_edge_rows() {
    let _guard = lock();
    let device = dev();
    let sizes = [6usize, 5, 7, 4];
    let mut data = many_graphs(&sizes, 61);
    let edges = data.edge_src.len();
    let mut rng = Lcg(3);
    data.edge_attr = Some(Features::Float {
        dim: 3,
        data: (0..edges * 3).map(|_| rng.unit()).collect(),
    });
    let canon = canon_of::<f32>(&data, NodeOrder::default(), true);
    let store = upload(&canon, &device);
    let (dim, table) = float_table(canon.edge_x.as_ref().unwrap());
    let batch = vec![3u32, 1];
    let epoch = EpochTable::build(
        std::slice::from_ref(&batch),
        &canon.graph_ptr,
        &canon.edge_ptr,
        8,
        &device,
    )
    .unwrap();
    let rows = batch_rows_graphs(store.adjacency(), &epoch.slots(0).unwrap(), 16, Some(64)).unwrap();
    let eid = rows.edges().unwrap().eid().to_vec();
    let got = counted(1, "edge_inputs", || {
        edge_inputs(&store.edge_x().unwrap().source(), &rows, store.n_edges()).unwrap()
    });
    assert_eq!(got.dims(), &[64, dim]);
    let mut want = vec![0.0f32; 64 * dim];
    for (row, &e) in eid.iter().enumerate() {
        if e != IGNORE {
            want[row * dim..(row + 1) * dim]
                .copy_from_slice(&table[e as usize * dim..(e as usize + 1) * dim]);
        }
    }
    assert_eq!(got.to_f32(), want);
    assert!(eid.iter().any(|&e| e != IGNORE));
    // A batch laid out without edges cannot gather them.
    let plain = batch_rows_graphs(store.adjacency(), &epoch.slots(0).unwrap(), 16, None).unwrap();
    assert!(edge_inputs(&store.edge_x().unwrap().source(), &plain, store.n_edges()).is_err());
}

// ---------------------------------------------------------------------------
// G5 / G6
// ---------------------------------------------------------------------------

/// A batch of three graphs of an eight-graph dataset, 32 rows of capacity.
fn ragged_batch(device: &Device<R>) -> (BatchRows<R>, Vec<u32>, Vec<u32>) {
    let sizes = [5usize, 9, 3, 12, 7, 4, 10, 6];
    let data = many_graphs(&sizes, 71);
    let canon = canon_of::<f32>(&data, NodeOrder::default(), false);
    let store = upload(&canon, device);
    let epoch = EpochTable::build(
        &[vec![1u32, 2, 4]],
        &canon.graph_ptr,
        &canon.edge_ptr,
        8,
        device,
    )
    .unwrap();
    let rows = batch_rows_graphs(store.adjacency(), &epoch.slots(0).unwrap(), 32, None).unwrap();
    let slot = rows.row_graph().to_vec();
    let lengths = rows.lengths().to_vec();
    (rows, slot, lengths)
}

fn dot(a: &[f32], b: &[f32]) -> f64 {
    a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum()
}

#[test]
fn ragged_padding_round_trips_and_is_its_own_adjoint() {
    let _guard = lock();
    let device = dev();
    let (rows, slot, lengths) = ragged_batch(&device);
    assert_eq!(lengths, vec![9, 3, 7, 0, 0, 0, 0, 0]);
    let (d, nmax) = (8usize, 12usize);
    let mut rng = Lcg(1);
    let host: Vec<f32> = (0..32 * d).map(|_| rng.unit()).collect();
    let h = Tensor::<R, f32>::from_f32(&host, vec![32, d], &device).unwrap();
    let padded = counted(1, "pad_ragged", || pad_ragged(&h, &rows, nmax).unwrap());
    assert_eq!(padded.dims(), &[8, nmax, d]);
    let p = padded.to_f32();
    let mut first = 0;
    for g in 0..8 {
        for pos in 0..nmax {
            let got = &p[(g * nmax + pos) * d..(g * nmax + pos + 1) * d];
            if pos < lengths[g] as usize {
                assert_eq!(got, &host[(first + pos) * d..(first + pos + 1) * d]);
            } else {
                assert!(got.iter().all(|&v| v == 0.0), "pad positions are zero");
            }
        }
        first += lengths[g] as usize;
    }
    let back = counted(1, "unpad_ragged", || unpad_ragged(&padded, &rows).unwrap()).to_f32();
    for r in 0..32 {
        let got = &back[r * d..(r + 1) * d];
        if slot[r] == IGNORE {
            assert!(got.iter().all(|&v| v == 0.0), "absent rows come back as zeros");
        } else {
            assert_eq!(got, &host[r * d..(r + 1) * d]);
        }
    }
    // <pad x, y> = <x, unpad y> for arbitrary y, pads included.
    let y_host: Vec<f32> = (0..8 * nmax * d).map(|_| rng.unit()).collect();
    let y = Tensor::<R, f32>::from_f32(&y_host, vec![8, nmax, d], &device).unwrap();
    let unpadded = unpad_ragged(&y, &rows).unwrap().to_f32();
    let (lhs, rhs) = (dot(&p, &y_host), dot(&host, &unpadded));
    assert!((lhs - rhs).abs() < 1e-4 * (1.0 + lhs.abs()), "{lhs} vs {rhs}");

    // A padded length shorter than a graph drops the tail in both directions.
    let short = pad_ragged(&h, &rows, 4).unwrap();
    let back = unpad_ragged(&short, &rows).unwrap().to_f32();
    assert_eq!(&back[..4 * d], &host[..4 * d]);
    assert!(back[4 * d..9 * d].iter().all(|&v| v == 0.0));

    // Through the tape: the gradient of sum(pad(h) * y) w.r.t. h is unpad(y).
    let leaf = Var::traced(h.clone());
    let loss = leaf
        .pad_ragged(&rows, nmax)
        .unwrap()
        .mul(&Var::constant(y.clone()))
        .unwrap()
        .sum()
        .unwrap();
    let grads = loss.backward_retain().unwrap();
    let grad = grads.node(leaf.node().unwrap()).unwrap().to_f32();
    assert_close(&grad, &unpadded, 1e-6, "pad_ragged gradient");
    let leaf = Var::traced(y.clone());
    let loss = leaf
        .unpad_ragged(&rows)
        .unwrap()
        .mul(&Var::constant(h.clone()))
        .unwrap()
        .sum()
        .unwrap();
    let grads = loss.backward_retain().unwrap();
    let grad = grads.node(leaf.node().unwrap()).unwrap().to_f32();
    assert_close(&grad, &p, 1e-6, "unpad_ragged gradient");
}

#[test]
fn segment_pool_means_and_its_adjoint() {
    let _guard = lock();
    let device = dev();
    let (rows, slot, lengths) = ragged_batch(&device);
    let d = 8usize;
    let mut rng = Lcg(2);
    let mut host: Vec<f32> = (0..32 * d).map(|_| rng.unit()).collect();
    let used: usize = lengths.iter().sum::<u32>() as usize;
    assert_eq!(used, 19);
    for mean in [true, false] {
        let h = Tensor::<R, f32>::from_f32(&host, vec![32, d], &device).unwrap();
        let pooled = counted(1, "segment_pool", || segment_pool(&h, &rows, mean).unwrap());
        assert_eq!(pooled.dims(), &[8, d]);
        let mut want = vec![0.0f32; 8 * d];
        for r in 0..used {
            let g = slot[r] as usize;
            for j in 0..d {
                want[g * d + j] += host[r * d + j];
            }
        }
        if mean {
            for g in 0..3 {
                for j in 0..d {
                    want[g * d + j] /= lengths[g] as f32;
                }
            }
        }
        assert_close(&pooled.to_f32(), &want, 1e-5, "pooled rows");

        // Absent rows cannot reach a pooled row.
        let mut loud = host.clone();
        loud[used * d..].fill(1.0e6);
        let loud_h = Tensor::<R, f32>::from_f32(&loud, vec![32, d], &device).unwrap();
        assert_eq!(
            segment_pool(&loud_h, &rows, mean).unwrap().to_f32(),
            pooled.to_f32()
        );

        // <pool x, y> = <x, broadcast y>.
        let y_host: Vec<f32> = (0..8 * d).map(|_| rng.unit()).collect();
        let y = Tensor::<R, f32>::from_f32(&y_host, vec![8, d], &device).unwrap();
        let spread = counted(1, "segment_broadcast", || {
            segment_broadcast(&y, &rows, mean).unwrap()
        })
        .to_f32();
        assert!(spread[used * d..].iter().all(|&v| v == 0.0));
        let (lhs, rhs) = (dot(&pooled.to_f32(), &y_host), dot(&host, &spread));
        assert!((lhs - rhs).abs() < 1e-4 * (1.0 + lhs.abs()), "{lhs} vs {rhs}");

        // Finite differences through the tape.
        let leaf = Var::traced(h.clone());
        let loss = leaf
            .segment_pool(&rows, mean)
            .unwrap()
            .mul(&Var::constant(y.clone()))
            .unwrap()
            .sum()
            .unwrap();
        let base = loss.to_f32()[0];
        let grads = loss.backward_retain().unwrap();
        let grad = grads.node(leaf.node().unwrap()).unwrap().to_f32();
        assert_close(&grad, &spread, 1e-6, "segment_pool gradient");
        for &i in &[0usize, 3 * d + 2, 11 * d + 5, 18 * d + 7] {
            let eps = 1e-2;
            host[i] += eps;
            let bumped = Tensor::<R, f32>::from_f32(&host, vec![32, d], &device).unwrap();
            host[i] -= eps;
            let value = dot(&segment_pool(&bumped, &rows, mean).unwrap().to_f32(), &y_host) as f32;
            let numeric = (value - base) / eps;
            assert!(
                (numeric - grad[i]).abs() < 2e-3 * (1.0 + grad[i].abs()),
                "d/dh[{i}]: numeric {numeric}, analytic {}",
                grad[i]
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Targets, confusion, masked mean
// ---------------------------------------------------------------------------

#[test]
fn node_targets_are_masked_before_use() {
    let _guard = lock();
    let device = dev();
    let n = 70;
    let mut data = random_graph(n, 160, 81);
    let mut rng = Lcg(4);
    data.y = Labels::Node((0..n).map(|i| if i % 7 == 0 { -1 } else { rng.below(5) as i64 }).collect());
    data.masks = Some(Splits {
        train: (0..n).map(|i| i % 3 == 0).collect(),
        val: (0..n).map(|i| i % 3 == 1).collect(),
        test: (0..n).map(|i| i % 3 == 2).collect(),
    });
    let canon = canon_of::<f32>(&data, NodeOrder::default(), false);
    let store = upload(&canon, &device);
    let CanonicalLabels::Node(labels) = &canon.y else {
        panic!("expected node labels");
    };
    let TargetStore::Node { y, split } = store.targets() else {
        panic!("expected node targets");
    };
    let rows = batch_rows_subset(store.adjacency(), 3, 1, (2, 2), 2).unwrap();
    let gid = rows.gid().to_vec();
    for flags in [SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST, SPLIT_VAL | SPLIT_TEST] {
        let (ids, mask) = counted(1, "safe_class_targets", || {
            safe_class_targets::<R, f32>(y, split, &rows, false, flags).unwrap()
        });
        let (ids, mask) = (ids.to_vec(), mask.to_f32());
        let mut kept = 0;
        for (r, &v) in gid.iter().enumerate() {
            let real = v != IGNORE
                && labels[v as usize] != IGNORE
                && canon.split[v as usize] & flags != 0;
            assert_eq!(mask[r], real as u32 as f32);
            assert_eq!(ids[r], if real { labels[v as usize] } else { 0 });
            kept += real as usize;
        }
        assert!(kept > 0);
    }
    assert!(safe_class_targets::<R, f32>(y, split, &rows, true, SPLIT_TRAIN).is_err());
}

#[test]
fn graph_targets_are_masked_before_use() {
    let _guard = lock();
    let device = dev();
    let sizes = [5usize, 9, 3, 12, 7, 4];
    let masks = Splits {
        train: vec![true, true, false, true, false, true],
        val: vec![false, false, true, false, false, false],
        test: vec![false, false, false, false, true, false],
    };
    let batch = vec![4u32, 0, 3, 2, 5];

    // Classes.
    let mut data = many_graphs(&sizes, 91);
    data.y = Labels::GraphClass(vec![2, 0, 1, IGNORE, 1, 2]);
    data.masks = Some(masks.clone());
    let canon = canon_of::<f32>(&data, NodeOrder::default(), false);
    let store = upload(&canon, &device);
    let epoch =
        EpochTable::build(std::slice::from_ref(&batch), &canon.graph_ptr, &canon.edge_ptr, 8, &device)
            .unwrap();
    let rows = batch_rows_graphs(store.adjacency(), &epoch.slots(0).unwrap(), 64, None).unwrap();
    let TargetStore::GraphClass { y, split } = store.targets() else {
        panic!("expected graph classes");
    };
    let (ids, mask) = counted(1, "graph class targets", || {
        safe_class_targets::<R, f32>(y, split, &rows, true, SPLIT_TRAIN).unwrap()
    });
    // Slots: graphs 4 (test), 0 (train, 2), 3 (train, unlabelled), 2 (val), 5 (train, 2).
    assert_eq!(ids.to_vec(), vec![0, 2, 0, 0, 2, 0, 0, 0]);
    assert_eq!(mask.to_f32(), vec![0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]);

    // Float targets with missing values.
    let mut data = many_graphs(&sizes, 91);
    let nan = f32::NAN;
    data.y = Labels::Graph {
        targets: 2,
        values: vec![1.0, 2.0, 3.0, nan, 5.0, 6.0, nan, nan, 9.0, 10.0, 11.0, nan],
    };
    data.masks = Some(masks);
    let canon = canon_of::<f32>(&data, NodeOrder::default(), false);
    assert_eq!(canon.labelled, [3, 1, 1]);
    let store = upload(&canon, &device);
    let epoch =
        EpochTable::build(std::slice::from_ref(&batch), &canon.graph_ptr, &canon.edge_ptr, 8, &device)
            .unwrap();
    let rows = batch_rows_graphs(store.adjacency(), &epoch.slots(0).unwrap(), 64, None).unwrap();
    let TargetStore::Graph {
        targets,
        y,
        present,
        split,
    } = store.targets()
    else {
        panic!("expected float targets");
    };
    assert_eq!(*targets, 2);
    let (values, mask) = counted(1, "graph float targets", || {
        safe_float_targets(y, present, split, &rows, SPLIT_TRAIN | SPLIT_TEST).unwrap()
    });
    assert_eq!(values.dims(), &[8, 2]);
    // Slots: graph 4 (test), 0 (train), 3 (train, both missing), 2 (val), 5 (train).
    let want_values = [9.0, 10.0, 1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 11.0, 0.0];
    let want_mask = [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0];
    assert_eq!(&values.to_f32()[..10], &want_values);
    assert_eq!(&mask.to_f32()[..10], &want_mask);
    assert!(values.to_f32()[10..].iter().all(|&v| v == 0.0));
    assert!(mask.to_f32()[10..].iter().all(|&v| v == 0.0));
    assert!(values.to_f32().iter().all(|v| v.is_finite()));
}

#[test]
fn confusion_counts_kept_rows() {
    let _guard = lock();
    let device = dev();
    let (rows, classes) = (500usize, 6usize);
    let mut rng = Lcg(13);
    let predicted: Vec<u32> = (0..rows).map(|_| rng.below(classes) as u32).collect();
    let target: Vec<u32> = (0..rows).map(|_| rng.below(classes) as u32).collect();
    let mask: Vec<f32> = (0..rows).map(|_| (rng.below(4) != 0) as u32 as f32).collect();
    let p = IdTensor::<R>::from_slice(&predicted, vec![rows], &device).unwrap();
    let t = IdTensor::<R>::from_slice(&target, vec![rows], &device).unwrap();
    let m = Tensor::<R, f32>::from_f32(&mask, vec![rows], &device).unwrap();
    let counts = counted(1, "confusion", || confusion(&p, &t, &m, classes).unwrap()).to_vec();
    let mut want = vec![0u32; classes * classes];
    for r in 0..rows {
        if mask[r] != 0.0 {
            want[target[r] as usize * classes + predicted[r] as usize] += 1;
        }
    }
    assert_eq!(counts, want);
    assert_eq!(counts.iter().sum::<u32>() as f32, mask.iter().sum::<f32>());
}

fn masked_mean_case<E: FloatElem>(len: usize, tol: f32) {
    let device = dev();
    if !supports_dtype(&device, E::DTYPE) {
        println!("skipped: the {} backend has no {}", device.name(), E::DTYPE.name());
        return;
    }
    let mut rng = Lcg(len as u64);
    let x: Vec<f32> = (0..len).map(|_| rng.unit()).collect();
    let mask: Vec<f32> = (0..len).map(|i| (i % 5 != 0) as u32 as f32).collect();
    let xt = Tensor::<R, E>::from_f32(&x, vec![len], &device).unwrap();
    let mt = Tensor::<R, E>::from_f32(&mask, vec![len], &device).unwrap();
    let (mean, inv_count) = counted(1, "masked_mean", || masked_mean(&xt, &mt).unwrap());
    let stored = E::slice_to_f32(&E::slice_from_f32(&x));
    let kept: f32 = mask.iter().sum();
    let want: f64 = stored
        .iter()
        .zip(&mask)
        .map(|(v, m)| *v as f64 * *m as f64)
        .sum::<f64>()
        / kept as f64;
    assert!((mean.to_f32()[0] as f64 - want).abs() < tol as f64, "mean");
    assert!((inv_count.to_f32()[0] - 1.0 / kept).abs() < 1e-9);
    assert_eq!(inv_count.to_f32()[1], kept, "the count is kept in f32");
    let grad = Tensor::<R, E>::from_f32(&[2.0], vec![1], &device).unwrap();
    let back = counted(1, "masked_mean_backward", || {
        masked_mean_backward(&grad, &mt, &inv_count).unwrap()
    })
    .to_f32();
    for i in 0..len {
        let want = E::from_scalar(2.0 * mask[i] / kept).to_scalar();
        assert!((back[i] - want).abs() <= 1e-6 * want.abs() + f32::MIN_POSITIVE);
    }
    // A finite, non-zero gradient: the denominator did not round to infinity.
    assert!(back[1].is_finite() && back[1] != 0.0);

    // Through the tape, and with nothing kept.
    let leaf = Var::traced(xt.clone());
    let out = leaf.masked_mean(&mt).unwrap();
    assert_eq!(out.dims(), &[1]);
    let grads = out.backward_retain().unwrap();
    let g = grads.node(leaf.node().unwrap()).unwrap().to_f32();
    assert_eq!(g.len(), len);
    assert_eq!(g[0], 0.0);
    assert!(g[1] > 0.0);
    let none = Tensor::<R, E>::zeros(vec![len], &device);
    let (mean, inv_count) = masked_mean(&xt, &none).unwrap();
    assert_eq!(mean.to_f32()[0], 0.0);
    assert_eq!(inv_count.to_f32(), vec![1.0, 0.0]);
}

#[test]
fn masked_mean_matches_the_host() {
    let _guard = lock();
    masked_mean_case::<f32>(1000, 1e-5);
    masked_mean_case::<f32>(7, 1e-6);
}

#[test]
fn masked_mean_survives_a_count_above_f16_range() {
    let _guard = lock();
    // 100,000 elements, 80,000 kept: above f16's largest finite value.
    masked_mean_case::<half::f16>(100_000, 2e-3);
    masked_mean_case::<half::bf16>(100_000, 1e-2);
    let _ = DType::F16;
}
