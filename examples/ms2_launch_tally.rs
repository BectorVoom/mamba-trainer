//! Per-op launch tally of one warmed MS2 `generate` call and one warmed
//! training step: which op, at which site, issues how many launches (and,
//! with `--timed`, how much drained wall time each site costs).
//!
//! ```text
//! cargo run --release --no-default-features --features cpu --example ms2_launch_tally -- \
//!   --mode generate --b 8 --k 8 --n 128 --top 60 [--timed] [--label ms2.step] [--slots 16] [--steps 22]
//! ```
//!
//! The model, the synthetic spectra and the configurations are the ones of
//! `profile_ms2_substructure`, so counts agree with that driver. `--timed`
//! drains the device before every launch (it serialises the queue): the split
//! between sites is meaningful, the total is not.
//!
//! `--time N --rounds R` replaces the tally with a wall-clock comparison:
//! each round times `N` back-to-back calls with one device drain at the end
//! (a pipelined average, which is what a loaded machine can measure). For
//! `generate` every round times the composed reference step and the fused
//! step in turn, in one process, so the two see the same machine state; for
//! `train` it times the step alone (compare two binaries by interleaving
//! their runs). `--fused-only` skips the composed reference (for sampling
//! the fused path with a profiler). `--stages N` prints the host clock at
//! the stage boundaries of `N` calls, undrained: issuing against waiting.

use mamba3::backend::{
    Device, flush_launch_timer, launch_tally_detailed, launch_time_tally, reset_launch_tally,
    set_launch_timer, start_launch_tally, stop_launch_tally,
};
use mamba3::backends::Auto;
use mamba3::models::ms2::chem::{Composition, composition_mass, element_index};
use mamba3::models::ms2::contract::{
    GenerationConfig, ModelConfig, SPECTRUM_SCHEMA_VERSION, SpectrumBatch,
};
use mamba3::models::ms2::dataset::ExportSpectrum;
use mamba3::models::ms2::experiment::{ExperimentSet, ExperimentSpectrum, SpectrumDomain};
use mamba3::models::ms2::formula::FormulaTable;
use mamba3::models::ms2::formula_head::DeviceFormulaTable;
use mamba3::models::ms2::generate::{GenerateStage, GenerationWorkspace, Ms2Model};
use mamba3::models::ms2::train::{Ms2Trainer, TrainConfig};
use mamba3::tensor::ops::ms2::Ms2Constants;
use mamba3::tensor::ops::random::Rng;

type R = Auto;
type E = f32;

fn fixture_comps() -> Vec<Composition> {
    let fallback = vec![
        [6, 6, 0, 0, 0, 0, 0, 0, 0, 0],
        [3, 7, 1, 2, 0, 0, 0, 0, 0, 0],
        [6, 12, 0, 6, 0, 0, 0, 0, 0, 0],
        [10, 8, 0, 0, 0, 0, 0, 0, 0, 0],
    ];
    let Ok(text) = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/ms2/chemistry_v0.json"
    )) else {
        return fallback;
    };
    let Ok(value) = serde_json::from_str::<serde_json::Value>(&text) else {
        return fallback;
    };
    let mut out = Vec::new();
    if let Some(molecules) = value.get("molecules").and_then(|m| m.as_array()) {
        for m in molecules.iter().take(4) {
            let mut c: Composition = [0; 10];
            let mut ok = false;
            if let Some(obj) = m.get("formula").and_then(|f| f.as_object()) {
                for (symbol, count) in obj {
                    if let Some(e) = element_index(symbol) {
                        c[e] = count.as_u64().unwrap_or(0) as u16;
                        ok = true;
                    }
                }
            }
            if ok {
                out.push(c);
            }
        }
    }
    if out.len() < 4 { fallback } else { out }
}

fn precursor_of(comp: &Composition) -> u32 {
    composition_mass(comp).unwrap() + 1_007_825 - 549
}

fn spectra_batch(comps: &[Composition], n_raw: usize, seed: u64, id_base: u64) -> SpectrumBatch {
    let b = comps.len();
    let precursors: Vec<u32> = comps.iter().map(precursor_of).collect();
    let mut rng = Rng::seeded(seed);
    let mut peak_id = vec![u32::MAX; b * n_raw];
    let mut mz = vec![0u32; b * n_raw];
    let mut intensity = vec![0.0f32; b * n_raw];
    for bi in 0..b {
        for i in 0..n_raw {
            peak_id[bi * n_raw + i] = i as u32;
            let f = rng.uniform_vec(1, 60_000_000.0, (precursors[bi] - 5_000_000) as f32)[0] as u32;
            mz[bi * n_raw + i] = f.max(50_000_001);
            intensity[bi * n_raw + i] = 0.5 + 2.0 * rng.uniform_vec(1, 0.0, 1.0)[0];
        }
    }
    SpectrumBatch {
        schema_version: SPECTRUM_SCHEMA_VERSION,
        n_raw: n_raw as u32,
        spectrum_id: (0..b as u64).map(|i| id_base + i).collect(),
        raw_peak_count: vec![n_raw as u32; b],
        peak_count: vec![n_raw as u32; b],
        peak_id,
        mz_udalton: mz,
        intensity,
        intensity_scale: 0,
        mz_uncertainty_udalton: vec![50; b],
        precursor_mz_udalton: precursors,
        precursor_uncertainty_udalton: vec![50; b],
        adduct: vec![1; b],
        polarity: vec![1; b],
        collision_energy_ev: vec![30.0; b],
        collision_energy_known: vec![1; b],
        energy_count: vec![1; b],
        fragment_tolerance_ppm_tenths: vec![0; b],
        precursor_tolerance_ppm_tenths: vec![0; b],
        instrument_class: vec![0; b],
    }
}

fn experiment_set(comps: &[Composition], n_raw: usize, seed: u64) -> ExperimentSet {
    use mamba3::models::ms2::graph::MolGraph;
    let batch = spectra_batch(comps, n_raw, seed, 9000);
    let spectra = (0..comps.len())
        .map(|i| {
            let base = i * n_raw;
            ExperimentSpectrum {
                molecule: i,
                spectrum: ExportSpectrum {
                    row: i as u64,
                    spectrum_id: batch.spectrum_id[i],
                    adduct: 1,
                    polarity: 1,
                    precursor_mz_udalton: batch.precursor_mz_udalton[i],
                    precursor_uncertainty_udalton: 50,
                    raw_peak_count: batch.raw_peak_count[i],
                    peak_id: batch.peak_id[base..base + n_raw].to_vec(),
                    mz_udalton: batch.mz_udalton[base..base + n_raw].to_vec(),
                    intensity: batch.intensity[base..base + n_raw]
                        .iter()
                        .map(|&v| v as f64)
                        .collect(),
                    mz_uncertainty_udalton: 50,
                    collision_energy_ev: 30.0,
                    collision_energy_known: 1,
                    energy_count: 1,
                    instrument_class: 0,
                },
                parent: MolGraph::new(Vec::new(), Vec::new()).expect("empty graph builds"),
                parent_composition: comps[i],
                labels: None,
                domain: SpectrumDomain::InDomainUnlabeled,
            }
        })
        .collect();
    ExperimentSet {
        name: "tally-synthetic".to_string(),
        source_sha256: "synthetic".to_string(),
        molecules: (0..comps.len()).map(|i| format!("mol{i}")).collect(),
        spectra,
    }
}

/// Print the tally of the region just run: per label totals, then the top
/// rows (optionally restricted to one label), with timed milliseconds when a
/// launch timer is installed.
fn report(title: &str, top: usize, label: Option<&str>, divisor: usize, timed: bool) {
    let rows = launch_tally_detailed();
    let total: usize = rows.iter().map(|r| r.count).sum();
    println!("== {title}: {total} launches");
    let mut by_label: std::collections::BTreeMap<String, usize> = Default::default();
    for r in &rows {
        *by_label.entry(r.label.clone()).or_insert(0) += r.count;
    }
    for (l, c) in &by_label {
        println!("   label {l:<24} {c:>6}");
    }
    let times: std::collections::HashMap<String, f64> = if timed {
        flush_launch_timer();
        launch_time_tally().into_iter().collect()
    } else {
        Default::default()
    };
    let mut by_op: std::collections::BTreeMap<String, (usize, f64)> = Default::default();
    let mut shown = 0usize;
    for r in &rows {
        if label.is_some_and(|l| l != r.label) {
            continue;
        }
        let key = format!("{} / {} / {}", r.label, r.op, r.site);
        let ms = times.get(&key).copied().unwrap_or(0.0);
        let e = by_op.entry(r.op.clone()).or_insert((0, 0.0));
        e.0 += r.count;
        e.1 += ms;
        if shown < top {
            if timed {
                println!(
                    "   {:>6} ({:>7.2}/unit) {:>9.3} ms  {} / {} / {}",
                    r.count,
                    r.count as f64 / divisor as f64,
                    ms,
                    r.label,
                    r.op,
                    r.site
                );
            } else {
                println!(
                    "   {:>6} ({:>7.2}/unit)  {} / {} / {}",
                    r.count,
                    r.count as f64 / divisor as f64,
                    r.label,
                    r.op,
                    r.site
                );
            }
            shown += 1;
        }
    }
    let mut ops: Vec<(String, (usize, f64))> = by_op.into_iter().collect();
    ops.sort_by(|a, b| b.1.0.cmp(&a.1.0).then_with(|| a.0.cmp(&b.0)));
    println!("-- by op ({}):", label.unwrap_or("all labels"));
    for (op, (count, ms)) in ops {
        if timed {
            println!("   {count:>6} ({:>7.2}/unit) {ms:>9.3} ms  {op}", count as f64 / divisor as f64);
        } else {
            println!("   {count:>6} ({:>7.2}/unit)  {op}", count as f64 / divisor as f64);
        }
    }
}

fn main() {
    let mut mode = "both".to_string();
    let mut n = 128usize;
    let mut b = 8usize;
    let mut k = 8u32;
    let mut top = 80usize;
    let mut timed = false;
    let mut label: Option<String> = None;
    let mut time_calls = 0usize;
    let mut rounds = 5usize;
    let mut slots = 2usize;
    let mut data: Option<String> = None;
    let mut table_path: Option<String> = None;
    let mut steps: Option<u32> = None;
    let mut fused_only = false;
    let mut stages = 0usize;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        let mut next = || args.next().expect("flag value");
        match arg.as_str() {
            "--mode" => mode = next(),
            "--n" => n = next().parse().expect("--n"),
            "--b" => b = next().parse().expect("--b"),
            "--k" => k = next().parse().expect("--k"),
            "--top" => top = next().parse().expect("--top"),
            "--label" => label = Some(next()),
            "--timed" => timed = true,
            "--time" => time_calls = next().parse().expect("--time"),
            "--rounds" => rounds = next().parse().expect("--rounds"),
            "--slots" => slots = next().parse().expect("--slots"),
            "--data" => data = Some(next()),
            "--table" => table_path = Some(next()),
            "--steps" => steps = Some(next().parse().expect("--steps")),
            "--fused-only" => fused_only = true,
            "--stages" => stages = next().parse().expect("--stages"),
            other => panic!("unknown flag {other}"),
        }
    }
    let device = Device::<R>::default();
    println!("backend: {} b {b} k {k} n {n} timed {timed}", device.name());
    let base_comps = fixture_comps();
    let host_table =
        FormulaTable::from_compositions(base_comps.clone().into_iter()).expect("table builds");
    let comps: Vec<Composition> = (0..b).map(|i| base_comps[i % base_comps.len()]).collect();
    let install_timer = |device: &Device<R>| {
        if timed {
            // A drain is a synchronisation and a read: on wgpu the first alone
            // returns before the queue has run, and a site would be charged
            // for its predecessors' work.
            let d = device.clone();
            let probe = mamba3::tensor::Tensor::<R, E>::zeros(vec![1], device);
            set_launch_timer(Some(Box::new(move || {
                d.synchronize();
                let _ = probe.to_data();
            })));
        }
    };

    if mode == "generate" || mode == "both" {
        let table = DeviceFormulaTable::<R, E>::upload(&host_table, &device).expect("table uploads");
        let mut config = ModelConfig::v0();
        config.formula_table.rows = table.rows as u32;
        config.formula_table.sha256 = table.sha256.clone();
        let mut rng = Rng::seeded(1000 + n as u64 * 131 + b as u64);
        let model = Ms2Model::<R, E>::init(&config, &device, &mut rng).expect("model inits");
        let constants = Ms2Constants::new(&device);
        let batch = spectra_batch(&comps, n, 5000 + n as u64, 7000);
        let mut gen_config = GenerationConfig { trajectories: k, ..GenerationConfig::default() };
        if let Some(steps) = steps {
            gen_config.max_steps = steps;
        }
        let decode_steps = (gen_config.max_steps as usize - 1).max(1);
        let mut workspace = GenerationWorkspace::new();
        if time_calls > 0 {
            let mut composed_ws = GenerationWorkspace::new();
            composed_ws.composed_step = true;
            for _ in 0..3 {
                for ws in [&mut composed_ws, &mut workspace] {
                    model
                        .generate(&batch, &table, &gen_config, ws, &constants)
                        .expect("warmup runs");
                }
            }
            device.synchronize();
            let mut ms = [Vec::new(), Vec::new()];
            for _ in 0..rounds {
                for (slot, ws) in [&mut composed_ws, &mut workspace].into_iter().enumerate() {
                    if fused_only && slot == 0 {
                        ms[slot].push(f64::NAN);
                        continue;
                    }
                    let started = std::time::Instant::now();
                    for _ in 0..time_calls {
                        model
                            .generate(&batch, &table, &gen_config, ws, &constants)
                            .expect("timed generate runs");
                    }
                    device.synchronize();
                    ms[slot].push(started.elapsed().as_secs_f64() * 1e3 / time_calls as f64);
                }
            }
            for (name, v) in ["composed", "fused"].iter().zip(ms.iter_mut()) {
                v.sort_by(|a, b| a.total_cmp(b));
                println!(
                    "generate {name:<8} ms/call: median {:.2} min {:.2} max {:.2} over {rounds} rounds of {time_calls}",
                    v[v.len() / 2],
                    v[0],
                    v[v.len() - 1]
                );
            }
            println!(
                "generate speedup (median composed / median fused): {:.2}x",
                ms[0][ms[0].len() / 2] / ms[1][ms[1].len() / 2]
            );
        }
        for _ in 0..3 {
            let out = model
                .generate(&batch, &table, &gen_config, &mut workspace, &constants)
                .expect("warmup runs");
            out.validate().expect("warmup validates");
        }
        if stages > 0 {
            // Host clock at each stage boundary, no drain: how long the host
            // takes to issue a call against how long it then waits for it.
            let mut marks = [0.0f64; 5];
            for _ in 0..stages {
                let started = std::time::Instant::now();
                let mut hook = |stage: GenerateStage| {
                    let slot = match stage {
                        GenerateStage::AfterSearch => 0,
                        GenerateStage::AfterDecoderInit => 1,
                        GenerateStage::AfterValidate => 2,
                        GenerateStage::AfterReadout => 3,
                        _ => return,
                    };
                    marks[slot] += started.elapsed().as_secs_f64() * 1e3;
                };
                model
                    .generate_with_hook(
                        &batch,
                        &table,
                        &gen_config,
                        &mut workspace,
                        &constants,
                        Some(&mut hook),
                    )
                    .expect("staged generate runs");
                marks[4] += started.elapsed().as_secs_f64() * 1e3;
            }
            let n = stages as f64;
            println!(
                "generate host clock, ms into the call: search issued {:.2}, decoder init {:.2}, decode loop and validation {:.2}, readout back {:.2}, returned {:.2}",
                marks[0] / n,
                marks[1] / n,
                marks[2] / n,
                marks[3] / n,
                marks[4] / n
            );
        }
        device.synchronize();
        start_launch_tally();
        reset_launch_tally();
        install_timer(&device);
        mamba3::backend::reset_transfer_counters();
        let out = model
            .generate(&batch, &table, &gen_config, &mut workspace, &constants)
            .expect("tallied generate runs");
        out.validate().expect("tallied output validates");
        device.synchronize();
        println!(
            "generate: {} device buffers created",
            mamba3::backend::allocation_calls()
        );
        report(
            &format!("generate (unit = one of {decode_steps} decode steps)"),
            top,
            label.as_deref(),
            decode_steps,
            timed,
        );
        set_launch_timer(None);
        stop_launch_tally();
    }

    if mode == "train" || mode == "both" {
        let train_config = TrainConfig {
            batch: b,
            slots,
            lambda_assign: 0.0,
            ..TrainConfig::default()
        };
        // `--data` (with `--table`): labeled spectra of a real export, eight
        // batches in file order taken in turn, so slot fill and trace lengths
        // are the data's. Otherwise the synthetic unlabeled spectra.
        let (set, table, batches): (ExperimentSet, FormulaTable, Vec<Vec<usize>>) = match &data {
            Some(path) => {
                let set = ExperimentSet::load(
                    std::path::Path::new(path),
                    &mamba3::models::ms2::targets::RecipeLimits::V0,
                )
                .expect("export loads");
                let text = std::fs::read_to_string(table_path.as_ref().expect("--data needs --table"))
                    .expect("table reads");
                let table = FormulaTable::from_json(&text).expect("table parses");
                let batches: Vec<Vec<usize>> = set
                    .labeled()
                    .chunks(b)
                    .filter(|c| c.len() == b)
                    .take(8)
                    .map(|c| c.to_vec())
                    .collect();
                assert!(!batches.is_empty(), "export holds fewer than {b} labeled spectra");
                (set, table, batches)
            }
            None => (
                experiment_set(&comps, n, 6000 + n as u64),
                host_table.clone(),
                vec![(0..b).collect()],
            ),
        };
        let mut trainer = Ms2Trainer::<R, E>::new(&ModelConfig::v0(), &table, &train_config, &device)
            .expect("trainer builds");
        for _ in 0..3 {
            for indices in &batches {
                let _ = trainer.step(&set, indices).expect("warmup runs");
            }
        }
        device.synchronize();
        if time_calls > 0 {
            let mut ms = Vec::new();
            let mut turn = 0usize;
            for _ in 0..rounds {
                let started = std::time::Instant::now();
                for _ in 0..time_calls {
                    let _ = trainer
                        .step(&set, &batches[turn % batches.len()])
                        .expect("timed step runs");
                    turn += 1;
                }
                device.synchronize();
                ms.push(started.elapsed().as_secs_f64() * 1e3 / time_calls as f64);
            }
            ms.sort_by(|a, b| a.total_cmp(b));
            println!(
                "train ms/step: median {:.2} min {:.2} max {:.2} over {rounds} rounds of {time_calls}",
                ms[ms.len() / 2],
                ms[0],
                ms[ms.len() - 1]
            );
        }
        let indices = batches[0].clone();
        start_launch_tally();
        reset_launch_tally();
        install_timer(&device);
        let _ = trainer.step(&set, &indices).expect("tallied step runs");
        device.synchronize();
        report("train step (unit = one step)", top, label.as_deref(), 1, timed);
        set_launch_timer(None);
        stop_launch_tally();
    }
}
