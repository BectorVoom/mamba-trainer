//! Graph matching (`models::graph::matching`): the loss against a host
//! reference, finite-difference gradients of both encoders and the log scale,
//! the batch layout, a synthetic task it must learn, checkpoints and
//! evaluation determinism.

#![cfg(feature = "backend")]

use mamba3::backend::{Device, reset_upload_count, upload_count};
use mamba3::backends::Auto;
use mamba3::models::graph::matching::{LOG_SCALE_INIT, MAX_QUERY_BITS};
use mamba3::models::graph::{
    FeatureSpec, Features, GraphData, GraphDataset, GraphMambaSpec, GraphMatchConfig,
    GraphMatchModel, GraphMatchTrainer, GraphPool, GraphTaskSpec, GraphTrainConfig, MatchEpoch,
    MatchGroups, MatchQueries, MpnnKind, RegressionLoss,
};
use mamba3::nn::module::Module;
use mamba3::tensor::Tensor;
use mamba3::tensor::ops::index::read_all;

type R = Auto;
type Model = GraphMatchModel<R, f32>;
type Dataset = GraphDataset<R, f32>;

/// Serialises the tests: the upload counter is process-wide.
static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn lock() -> std::sync::MutexGuard<'static, ()> {
    LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn dev() -> Device<R> {
    Device::<R>::default()
}

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
}

const D_OUT: usize = 8;
const VOCAB: usize = 10;

/// Graphs of the given sizes, about two random edges per node, one
/// categorical node field (values from `values(graph, node)`) and one
/// categorical edge field.
fn graphs(sizes: &[usize], seed: u64, mut values: impl FnMut(usize, usize) -> u32) -> GraphData {
    let mut rng = Lcg(seed);
    let (mut src, mut dst, mut ptr, mut ids) = (Vec::new(), Vec::new(), vec![0u32], Vec::new());
    let mut at = 0usize;
    for (g, &size) in sizes.iter().enumerate() {
        // A path keeps every graph connected; random chords on top.
        for v in 1..size {
            src.extend([(at + v - 1) as u32, (at + v) as u32]);
            dst.extend([(at + v) as u32, (at + v - 1) as u32]);
        }
        for _ in 0..size {
            let (a, b) = (at + rng.below(size), at + rng.below(size));
            if a != b {
                src.extend([a as u32, b as u32]);
                dst.extend([b as u32, a as u32]);
            }
        }
        ids.extend((0..size).map(|v| values(g, v)));
        at += size;
        ptr.push(at as u32);
    }
    let edges = src.len();
    let mut data = GraphData::new(
        at,
        src,
        dst,
        Features::Categorical {
            fields: 1,
            vocab: vec![VOCAB],
            ids,
        },
    );
    data.graph_ptr = ptr;
    data.edge_attr = Some(Features::Categorical {
        fields: 1,
        vocab: vec![2],
        ids: (0..edges).map(|e| (e % 2) as u32).collect(),
    });
    data
}

fn spec(d_out: usize) -> GraphMambaSpec {
    GraphMambaSpec::new(
        FeatureSpec::Categorical { vocab: vec![VOCAB] },
        GraphTaskSpec::GraphRegression {
            targets: d_out,
            pool: GraphPool::Mean,
            loss: RegressionLoss::L1,
        },
    )
    .with_edge_features(Some(FeatureSpec::Categorical { vocab: vec![2] }))
    .with_mpnn(Some(MpnnKind::Gine))
    .with_d_model(16)
    .with_tokens(1, 2, 1)
    .with_node_layers(1)
    .with_row_quantum(16)
    .with_seed(3)
}

fn toy_dataset(device: &Device<R>) -> Dataset {
    let data = graphs(&[4, 6, 3, 5, 4, 7], 11, |g, v| ((g * 3 + v) % VOCAB) as u32);
    GraphDataset::new(&spec(D_OUT), data, device).unwrap()
}

const N_BITS: usize = 16;

/// Four queries of five entries: padding (weight 0) inside a row, a bit named
/// twice, and an all-padding query.
fn toy_queries(device: &Device<R>) -> MatchQueries<R, f32> {
    #[rustfmt::skip]
    let bits = [
        0, 3, 7, 0, 9,
        1, 1, 4, 15, 2,
        2, 5, 0, 11, 13,
        6, 6, 6, 6, 6,
    ];
    #[rustfmt::skip]
    let weights = [
        1.0, 0.5, 0.0, 0.25, 2.0,
        0.7, 0.3, 1.0, 0.0, 0.0,
        1.0, 1.0, 1.0, 0.5, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0,
    ];
    MatchQueries::new(&bits, &weights, 5, N_BITS, device).unwrap()
}

/// Group 0: graphs 0, 1, 2. Group 1: graphs 1 and 3 (graph 1 shared with
/// group 0). Group 2: graphs 4, 5, 0 and a repeated 4. Group 3: one member.
fn toy_groups() -> MatchGroups {
    MatchGroups::new(
        &[0, 1, 2, 3],
        &[0, 3, 5, 9, 10],
        &[0, 1, 2, 1, 3, 4, 5, 0, 4, 2],
    )
    .unwrap()
}

fn config(spec: GraphMambaSpec, hidden: usize) -> GraphMatchConfig {
    GraphMatchConfig::new(spec, N_BITS, hidden, 5)
}

fn dot(a: &[f32], b: &[f32]) -> f64 {
    a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum()
}

/// The mean cross entropy of each group's answer among its members, from the
/// embeddings on the host.
fn reference_loss(q: &[f32], z: &[f32], scale: f64, groups: &[Vec<u32>]) -> f64 {
    let mut total = 0.0;
    for (row, members) in groups.iter().enumerate() {
        let qr = &q[row * D_OUT..(row + 1) * D_OUT];
        let logits: Vec<f64> = members
            .iter()
            .map(|&s| scale * dot(qr, &z[s as usize * D_OUT..(s as usize + 1) * D_OUT]))
            .collect();
        let max = logits.iter().cloned().fold(f64::MIN, f64::max);
        let lse = max + logits.iter().map(|l| (l - max).exp()).sum::<f64>().ln();
        total += lse - logits[0];
    }
    total / groups.len() as f64
}

#[test]
fn queries_and_groups_are_validated() {
    let _guard = lock();
    let device = dev();
    let bad_bit = MatchQueries::<R, f32>::new(&[16], &[1.0], 1, N_BITS, &device);
    assert!(bad_bit.err().unwrap().to_string().contains("bit 16"));
    // A padding entry may name any bit.
    MatchQueries::<R, f32>::new(&[99], &[0.0], 1, N_BITS, &device).unwrap();
    let wide = vec![0u32; MAX_QUERY_BITS + 1];
    let weights = vec![1.0f32; MAX_QUERY_BITS + 1];
    assert!(MatchQueries::<R, f32>::new(&wide, &weights, wide.len(), N_BITS, &device).is_err());
    assert!(MatchQueries::<R, f32>::new(&[1], &[f32::NAN], 1, N_BITS, &device).is_err());
    assert!(
        MatchGroups::new(&[0, 1], &[0, 2, 2], &[0, 1]).is_err(),
        "an empty group"
    );
    assert!(
        MatchGroups::new(&[0], &[0, 3], &[0, 1]).is_err(),
        "a pointer past the members"
    );

    let groups = toy_groups();
    assert_eq!(
        groups.members(2),
        &[4, 5, 0],
        "the repeated member is dropped"
    );
    assert_eq!(groups.trivial(), vec![3]);

    let dataset = toy_dataset(&device);
    let queries = toy_queries(&device);
    let config = config(spec(D_OUT), 32);
    let epoch = |batches: &[Vec<u32>], training| {
        MatchEpoch::new(
            &dataset, &queries, &groups, batches, None, 0, training, &config,
        )
    };
    let refused = epoch(&[vec![0, 3]], true).err().unwrap().to_string();
    assert!(refused.contains("no negative"), "{refused}");
    epoch(&[vec![0, 3]], false).unwrap();
    assert!(
        epoch(&[vec![4]], false).is_err(),
        "a group id past the groups"
    );
    let other = MatchGroups::new(&[9], &[0, 2], &[0, 1]).unwrap();
    assert!(
        MatchEpoch::new(
            &dataset,
            &queries,
            &other,
            &[vec![0]],
            None,
            0,
            false,
            &config
        )
        .is_err(),
        "a query id past the queries"
    );
}

#[test]
fn layout_loss_and_gradients_match_the_reference() {
    let _guard = lock();
    let device = dev();
    let dataset = toy_dataset(&device);
    let queries = toy_queries(&device);
    let groups = toy_groups();
    let config = config(spec(D_OUT), 32);
    let model = Model::init(&config, &device).unwrap();
    let epoch = MatchEpoch::new(
        &dataset,
        &queries,
        &groups,
        &[vec![0, 1, 2], vec![1, 3]],
        None,
        0,
        false,
        &config,
    )
    .unwrap();

    // Layout: the union of the members in first-seen order, each graph once.
    assert_eq!(epoch.groups(), 8);
    assert_eq!(epoch.batch_graphs(0), &[0, 1, 2, 3, 4, 5]);
    assert_eq!(
        epoch.group_slots(0),
        &[vec![0, 1, 2], vec![1, 3], vec![4, 5, 0]]
    );
    assert_eq!(epoch.batch_graphs(1), &[1, 3, 2]);
    assert_eq!(epoch.group_slots(1), &[vec![0, 1], vec![2]]);
    let slots = epoch.slots();
    assert_eq!(slots, 8);

    let batch = epoch.batch(0).unwrap();
    let (ids, floats) =
        read_all(&[&batch.targets], &[&batch.legal, &batch.valid, &batch.bag]).unwrap();
    assert_eq!(ids[0], vec![0, 1, 4, 0, 0, 0, 0, 0]);
    assert_eq!(floats[1], vec![1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    let legal: Vec<Vec<usize>> = floats[0]
        .chunks(slots)
        .map(|row| (0..slots).filter(|&s| row[s] == 1.0).collect())
        .collect();
    assert_eq!(legal[0], vec![0, 1, 2]);
    assert_eq!(legal[1], vec![1, 3]);
    assert_eq!(legal[2], vec![0, 4, 5]);
    for row in &legal[3..] {
        assert_eq!(row, &vec![0], "a padding group keeps its safe target only");
    }
    // Bags: summed weights per bit, padding and padding rows zero.
    let bag = &floats[2];
    let mut want = vec![0.0f32; 8 * N_BITS];
    for (bit, w) in [(0, 1.25), (3, 0.5), (9, 2.0)] {
        want[bit] = w;
    }
    for (bit, w) in [(1, 1.0), (4, 1.0)] {
        want[N_BITS + bit] = w;
    }
    for (bit, w) in [(2, 1.0), (5, 1.0), (0, 1.0), (11, 0.5)] {
        want[2 * N_BITS + bit] = w;
    }
    for (got, want) in bag.iter().zip(&want) {
        assert!((got - want).abs() < 1e-6, "bag {got} vs {want}");
    }

    // The loss against the host reference, from the model's own embeddings.
    model.set_training(false);
    let loss = model.loss(&batch).unwrap();
    let value = loss.try_to_f32().unwrap()[0] as f64;
    let z = model
        .graph_embeddings(&batch.graph)
        .unwrap()
        .try_to_f32()
        .unwrap();
    let q = model
        .query_embeddings(&batch.bag)
        .unwrap()
        .try_to_f32()
        .unwrap();
    for row in 0..6 {
        let norm = dot(
            &z[row * D_OUT..(row + 1) * D_OUT],
            &z[row * D_OUT..(row + 1) * D_OUT],
        );
        assert!((norm - 1.0).abs() < 1e-4, "graph {row} has norm² {norm}");
    }
    let scale = model.scale().unwrap() as f64;
    assert!((scale - (LOG_SCALE_INIT as f64).exp()).abs() < 1e-3);
    let reference = reference_loss(&q, &z, scale, epoch.group_slots(0));
    assert!(
        (value - reference).abs() < 1e-4 * reference.abs().max(1.0),
        "loss {value} vs reference {reference}"
    );

    // Gradients against central differences, for both encoders and the scale.
    let params = model.named_parameters();
    let grads = loss.backward_retain().unwrap();
    let find = |name: &str| {
        params
            .iter()
            .find(|(n, _)| n == name)
            .unwrap_or_else(|| panic!("no parameter {name}"))
            .1
            .clone()
    };
    let eval = || model.loss(&batch).unwrap().try_to_f32().unwrap()[0] as f64;
    let checks: [(&str, &[usize]); 6] = [
        ("log_scale", &[0]),
        ("query.embed.weight", &[0, 1, 2, 32 + 5]),
        ("query.down.weight", &[0, 3, 17]),
        ("query.norm.weight", &[0, 7]),
        ("graph.head.out.weight", &[0, 5, 13]),
        ("graph.embed.weight", &[0, 17, 40]),
    ];
    for (name, indices) in checks {
        let param = find(name);
        let analytic = grads
            .get(param.id())
            .unwrap_or_else(|| panic!("no gradient for {name}"))
            .try_to_f32()
            .unwrap();
        assert!(
            analytic.iter().all(|g| g.is_finite()),
            "{name} has a non-finite gradient"
        );
        let shape = param.shape().dims().to_vec();
        let base = param.value().try_to_f32().unwrap();
        let h = 3e-3f32;
        for &idx in indices {
            let mut moved = base.clone();
            moved[idx] = base[idx] + h;
            param.set(Tensor::from_f32(&moved, shape.clone(), &device).unwrap());
            let up = eval();
            moved[idx] = base[idx] - h;
            param.set(Tensor::from_f32(&moved, shape.clone(), &device).unwrap());
            let down = eval();
            param.set(Tensor::from_f32(&base, shape.clone(), &device).unwrap());
            let numeric = (up - down) / (2.0 * h as f64);
            let a = analytic[idx] as f64;
            assert!(
                (a - numeric).abs() <= 3e-2 * numeric.abs() + 2e-3,
                "{name}[{idx}]: analytic {a} numeric {numeric}"
            );
        }
    }
    // Bit 6 is named only by the all-padding query: no gradient reaches it.
    let embed = grads
        .get(find("query.embed.weight").id())
        .unwrap()
        .try_to_f32()
        .unwrap();
    assert!(embed[6 * 32..7 * 32].iter().all(|&g| g == 0.0));
}

#[test]
fn zero_embeddings_stay_finite() {
    let _guard = lock();
    let device = dev();
    let dataset = toy_dataset(&device);
    let queries = toy_queries(&device);
    let groups = toy_groups();
    let config = config(spec(D_OUT), 32);
    let model = Model::init(&config, &device).unwrap();
    // A zero output layer: every query embeds to the zero vector.
    for (name, param) in model.named_parameters() {
        if name.starts_with("query.down.") {
            param.set(Tensor::zeros(param.shape(), &device));
        }
    }
    let epoch = MatchEpoch::new(
        &dataset,
        &queries,
        &groups,
        &[vec![0, 1, 2]],
        None,
        0,
        false,
        &config,
    )
    .unwrap();
    let batch = epoch.batch(0).unwrap();
    let q = model
        .query_embeddings(&batch.bag)
        .unwrap()
        .try_to_f32()
        .unwrap();
    assert!(
        q.iter().all(|&v| v == 0.0),
        "zero vectors normalise to zero"
    );
    let loss = model.loss(&batch).unwrap();
    let value = loss.try_to_f32().unwrap()[0] as f64;
    // Every member ties: the loss is the mean of ln(members).
    let want = (3f64.ln() + 2f64.ln() + 3f64.ln()) / 3.0;
    assert!((value - want).abs() < 1e-5, "loss {value} vs {want}");
    let grads = loss.backward_retain().unwrap();
    for (name, param) in model.named_parameters() {
        if let Some(grad) = grads.get(param.id()) {
            assert!(
                grad.try_to_f32().unwrap().iter().all(|g| g.is_finite()),
                "{name} has a non-finite gradient"
            );
        }
    }
}

/// A dataset in which graph `g` holds a set of marker values, and groups
/// whose query names one marker the answer holds and no decoy does.
struct Planted {
    data: GraphData,
    holds: Vec<Vec<bool>>,
}

fn planted(graph_count: usize, seed: u64) -> Planted {
    let mut rng = Lcg(seed);
    let sizes: Vec<usize> = (0..graph_count).map(|_| 4 + rng.below(4)).collect();
    let values: Vec<Vec<u32>> = sizes
        .iter()
        .map(|&size| (0..size).map(|_| rng.below(VOCAB) as u32).collect())
        .collect();
    let holds = values
        .iter()
        .map(|vs| (0..VOCAB).map(|v| vs.contains(&(v as u32))).collect())
        .collect();
    let data = graphs(&sizes, seed ^ 0xABCD, |g, v| values[g][v]);
    Planted { data, holds }
}

/// `count` groups of one answer and three decoys over `planted`; query `v`
/// names bit `v` alone.
fn planted_groups(planted: &Planted, count: usize, seed: u64) -> MatchGroups {
    let mut rng = Lcg(seed);
    let n = planted.holds.len();
    let (mut query, mut ptr, mut members) = (Vec::new(), vec![0u32], Vec::new());
    while query.len() < count {
        let v = rng.below(VOCAB);
        let with: Vec<u32> = (0..n as u32)
            .filter(|&g| planted.holds[g as usize][v])
            .collect();
        let without: Vec<u32> = (0..n as u32)
            .filter(|&g| !planted.holds[g as usize][v])
            .collect();
        if with.is_empty() || without.len() < 3 {
            continue;
        }
        members.push(with[rng.below(with.len())]);
        let mut decoys = Vec::new();
        while decoys.len() < 3 {
            let d = without[rng.below(without.len())];
            if !decoys.contains(&d) {
                decoys.push(d);
            }
        }
        members.extend(decoys);
        query.push(v as u32);
        ptr.push(members.len() as u32);
    }
    MatchGroups::new(&query, &ptr, &members).unwrap()
}

fn top1(scores: &[Vec<f32>]) -> f32 {
    let hits = scores
        .iter()
        .filter(|s| s[1..].iter().all(|&decoy| s[0] > decoy))
        .count();
    hits as f32 / scores.len() as f32
}

#[test]
fn learns_a_planted_marker() {
    let _guard = lock();
    let device = dev();
    let planted = planted(160, 21);
    let match_spec = spec(16).with_d_model(32);
    let dataset = Dataset::new(&match_spec, planted.data.clone(), &device).unwrap();
    let bits: Vec<u32> = (0..VOCAB as u32).collect();
    let queries = MatchQueries::new(&bits, &vec![1.0; VOCAB], 1, N_BITS, &device).unwrap();
    let train = planted_groups(&planted, 384, 1);
    let held_out = planted_groups(&planted, 128, 2);
    let config = GraphMatchConfig::new(match_spec, N_BITS, 64, 5);
    let model = Model::init(&config, &device).unwrap();
    let mut trainer = GraphMatchTrainer::new(&GraphTrainConfig {
        learning_rate: 3e-3,
        weight_decay: 0.01,
        ..Default::default()
    })
    .unwrap();

    let per_batch = 32;
    let eval_batches: Vec<Vec<u32>> = (0..held_out.len() as u32)
        .collect::<Vec<_>>()
        .chunks(per_batch)
        .map(<[u32]>::to_vec)
        .collect();
    let eval = MatchEpoch::new(
        &dataset,
        &queries,
        &held_out,
        &eval_batches,
        None,
        0,
        false,
        &config,
    )
    .unwrap();
    let before = top1(&model.score(&eval).unwrap());

    let mut losses = Vec::new();
    let mut order: Vec<u32> = (0..train.len() as u32).collect();
    let mut rng = Lcg(9);
    for epoch in 0..12u64 {
        for i in (1..order.len()).rev() {
            order.swap(i, rng.below(i + 1));
        }
        let batches: Vec<Vec<u32>> = order.chunks(per_batch).map(<[u32]>::to_vec).collect();
        let plan = MatchEpoch::new(
            &dataset,
            &queries,
            &train,
            &batches,
            Some(1024),
            epoch,
            true,
            &config,
        )
        .unwrap();
        trainer.train_epoch(&model, &plan).unwrap();
        losses.extend(trainer.read_losses().unwrap().iter().map(|info| info.loss));
        if epoch == 11 {
            // Shapes seen before: the per-shape caches are warm, and a step
            // uploads nothing (the epoch's tables were uploaded with it).
            reset_upload_count();
            trainer.train_epoch(&model, &plan).unwrap();
            assert_eq!(upload_count(), 0, "a training step uploads nothing");
            trainer.read_losses().unwrap();
        }
    }
    assert!(losses.iter().all(|l| l.is_finite()));
    let after = top1(&model.score(&eval).unwrap());
    let head: f32 = losses[..12].iter().sum::<f32>() / 12.0;
    let tail: f32 = losses[losses.len() - 12..].iter().sum::<f32>() / 12.0;
    println!(
        "planted marker: top-1 {before:.3} -> {after:.3}, loss {head:.3} -> {tail:.3}, scale {:.2}",
        model.scale().unwrap()
    );
    assert!(after > 0.9, "held-out top-1 {after} (from {before})");
    assert!(tail < head);
}

#[test]
fn checkpoints_and_evaluation_are_exact() {
    let _guard = lock();
    let device = dev();
    let dataset = toy_dataset(&device);
    let queries = toy_queries(&device);
    let groups = toy_groups();
    let config = config(spec(D_OUT), 32);
    let model = Model::init(&config, &device).unwrap();
    let mut trainer = GraphMatchTrainer::new(&GraphTrainConfig::default()).unwrap();
    let batches = vec![vec![0, 1, 2], vec![1, 2]];
    for epoch in 0..3 {
        let plan = MatchEpoch::new(
            &dataset, &queries, &groups, &batches, None, epoch, true, &config,
        )
        .unwrap();
        assert_eq!(trainer.train_epoch(&model, &plan).unwrap(), 2);
    }
    let infos = trainer.read_losses().unwrap();
    assert_eq!(infos.len(), 6);
    assert!(
        model.graph().is_training(),
        "training leaves the model in training mode"
    );

    let eval_batches = vec![vec![0, 1, 2, 3], vec![2]];
    let eval = MatchEpoch::new(
        &dataset,
        &queries,
        &groups,
        &eval_batches,
        None,
        0,
        false,
        &config,
    )
    .unwrap();
    let first = model.score(&eval).unwrap();
    let second = model.score(&eval).unwrap();
    assert_eq!(first, second, "evaluation is deterministic");
    assert!(model.graph().is_training(), "scoring restores the mode");
    assert_eq!(
        first.iter().map(Vec::len).collect::<Vec<_>>(),
        vec![3, 2, 3, 1, 3],
        "one score per distinct member"
    );

    // The scores are the scaled cosines of the standalone embeddings.
    let z = model.embed_graphs(&dataset, None, 4).unwrap();
    let q = model.embed_queries(&queries, 3).unwrap();
    assert_eq!(z.len(), 6 * D_OUT);
    assert_eq!(q.len(), 4 * D_OUT);
    let scale = model.scale().unwrap() as f64;
    for (g, scores) in [0usize, 1, 2, 3].iter().zip(&first) {
        let qg = &q[groups.query(*g) as usize * D_OUT..][..D_OUT];
        for (&m, &score) in groups.members(*g).iter().zip(scores) {
            let want = scale * dot(qg, &z[m as usize * D_OUT..][..D_OUT]);
            assert!(
                (score as f64 - want).abs() < 1e-4,
                "group {g} member {m}: {score} vs {want}"
            );
        }
    }
    let some = model.embed_graphs(&dataset, Some(&[5, 0, 5]), 2).unwrap();
    for (row, &g) in [5usize, 0, 5].iter().enumerate() {
        for c in 0..D_OUT {
            let (a, b) = (some[row * D_OUT + c], z[g * D_OUT + c]);
            assert!(
                (a - b).abs() < 1e-5,
                "graph {g} embeds the same in any batch"
            );
        }
    }

    let dir = std::env::temp_dir().join(format!("graph_match_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("matcher.ckpt");
    model.save(&path, trainer.step_count()).unwrap();
    let (loaded, step) = Model::load(&path, &device).unwrap();
    assert_eq!(step, 6);
    assert_eq!(loaded.config(), model.config());
    assert_eq!(
        loaded.score(&eval).unwrap(),
        first,
        "a reloaded model scores the same"
    );
    assert_eq!(loaded.scale().unwrap(), model.scale().unwrap());
    std::fs::remove_dir_all(&dir).ok();
}
