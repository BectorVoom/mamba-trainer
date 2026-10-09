//! Motif decoding: the stack machine against the Lean reference, the token
//! layout, and the beam step's cache gather.
//!
//! `lean/MotifDecoder/test/expected*.txt` are the outputs of the Lean
//! executable `motifcheck` (which runs the definitions the proofs are about)
//! on `vocab.txt`, `sequences.txt` and `budgets.txt` beside them. The first
//! two tests require this machine to print the same lines.

#![cfg(feature = "backend")]

use mamba3::backend::Device;
use mamba3::backends::Auto;
use mamba3::models::ms2::completion_fingerprint::{FingerprintBatch, SparseFingerprint};
use mamba3::models::ms2::completion_spectrum::{SpectrumBatch, SpectrumEvidence};
use mamba3::models::ms2::motif_model::{MotifModel, MotifModelConfig};
use mamba3::nn::Module;
use mamba3::models::ms2::motif::{
    ATOM_BASE, BOND_BASE, END, Formula, Layout, MAX_GRAPH_ATOMS, MOTIF_BASE, MotifMachine,
    MotifVocab, PREFIX_LEN, Phase, beam_search, gather_lm_cache, text,
};
use mamba3::prelude::*;
use mamba3::tensor::ops::index::IdTensor;

type R = Auto;

static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn fixture(name: &str) -> String {
    let path = format!("{}/lean/MotifDecoder/test/{name}", env!("CARGO_MANIFEST_DIR"));
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"))
}

fn vocab() -> MotifVocab {
    text::vocab(&fixture("vocab.txt")).unwrap()
}

#[test]
fn machine_prints_what_the_lean_reference_prints() {
    let vocab = vocab();
    let sequences = fixture("sequences.txt");
    let expected = fixture("expected.txt");
    let lines: Vec<&str> = sequences.lines().collect();
    let want: Vec<&str> = expected.lines().collect();
    assert_eq!(lines.len(), want.len());
    assert!(want.iter().any(|l| l.starts_with("OK")) && want.iter().any(|l| l.starts_with("ERR")));
    for (i, (line, want)) in lines.iter().zip(&want).enumerate() {
        assert_eq!(&text::check(&vocab, None, line), want, "sequence {i}: {line:?}");
    }
}

#[test]
fn budgeted_machine_prints_what_the_lean_reference_prints() {
    let vocab = vocab();
    let sequences = fixture("sequences.txt");
    let budgets = fixture("budgets.txt");
    let expected = fixture("expected_budget.txt");
    let budgets: Vec<Option<Formula>> = budgets.lines().map(|l| text::budget(l).unwrap()).collect();
    assert_eq!(sequences.lines().count(), expected.lines().count());
    let mut stricter = 0;
    for (i, (line, want)) in sequences.lines().zip(expected.lines()).enumerate() {
        let budget = budgets.get(i).copied().flatten();
        assert_eq!(text::check(&vocab, budget.as_ref(), line), want, "sequence {i}");
        stricter += usize::from(text::check(&vocab, None, line) != want);
    }
    assert!(stricter > 0, "the budget fixture rejects nothing the plain machine accepts");
}

#[test]
fn allowed_tokens_is_the_mask_and_apply_follows_it() {
    let vocab = vocab();
    // Ethanol-like chain from the fixture: C, then C, then O.
    let tokens = [
        MOTIF_BASE,
        ATOM_BASE,
        BOND_BASE + 1,
        MOTIF_BASE,
        ATOM_BASE,
        ATOM_BASE,
        BOND_BASE + 1,
        MOTIF_BASE + 1,
        ATOM_BASE,
        END,
        END,
        END,
    ];
    let mut machine = MotifMachine::new();
    let mut allowed = Vec::new();
    for &token in &tokens {
        machine.allowed_tokens(&vocab, None, &mut allowed);
        // The listed tokens are exactly the ones `allowed` accepts.
        for candidate in 0..MOTIF_BASE + vocab.len() as u32 + 2 {
            assert_eq!(
                allowed.contains(&candidate),
                machine.allowed(&vocab, None, candidate),
                "token {candidate} in {:?}",
                machine.phase
            );
        }
        assert!(allowed.contains(&token));
        // A refused token is an error and changes nothing.
        let before = (machine.free.clone(), machine.bonds.clone(), machine.phase);
        if let Some(&refused) = [END, ATOM_BASE + 31, BOND_BASE + 3, MOTIF_BASE + 99]
            .iter()
            .find(|t| !allowed.contains(t))
        {
            assert!(machine.apply(&vocab, None, refused).is_err());
            assert_eq!(before, (machine.free.clone(), machine.bonds.clone(), machine.phase));
        }
        machine.apply(&vocab, None, token).unwrap();
    }
    assert_eq!(machine.phase, Phase::Done);
    assert_eq!(machine.elements.len(), 3);
    assert_eq!(machine.bonds, vec![(0, 1, 1), (1, 2, 1)]);
    // C2H6O: the free valences left are the hydrogens.
    assert_eq!(machine.formula(), [2, 6, 0, 1, 0, 0, 0, 0, 0, 0]);
    machine.allowed_tokens(&vocab, None, &mut allowed);
    assert!(allowed.is_empty(), "nothing is allowed after the end");

    // With the exact formula as budget the same sequence is accepted, and a
    // formula one hydrogen short refuses only the final END.
    let exact: Formula = [2, 6, 0, 1, 0, 0, 0, 0, 0, 0];
    assert!(MotifMachine::run(&vocab, Some(&exact), &tokens).is_ok());
    let short: Formula = [2, 5, 0, 1, 0, 0, 0, 0, 0, 0];
    assert_eq!(MotifMachine::run(&vocab, Some(&short), &tokens).err(), Some(tokens.len() - 1));
    // A dead state is flagged as soon as the last heavy atom is placed.
    let mut machine = MotifMachine::new();
    for &token in &tokens[..9] {
        assert!(!machine.dead(&short));
        machine.apply(&vocab, Some(&short), token).unwrap();
    }
    assert!(machine.dead(&short));
    assert!(!machine.dead(&exact));
}

#[test]
fn prefix_orders_formula_and_fingerprint_levels() {
    let layout = Layout::new(5);
    assert_eq!(layout.n_out, MOTIF_BASE + 5);
    assert!(layout.bos >= layout.n_out && layout.vocab_size == layout.fingerprint_base + 4096);
    let formula: Formula = [2, 6, 0, 1, 0, 0, 0, 0, 0, 200];
    let fingerprint = SparseFingerprint {
        entries: vec![(3, 0.2), (7, 0.9), (9, 0.3), (11, 1.0), (40, 0.12)],
    };
    let prefix = layout.prefix(&formula, &fingerprint);
    assert_eq!(prefix[0], layout.bos);
    assert_eq!(*prefix.last().unwrap(), layout.sep);
    // One count token per element; counts clamp at 127.
    assert_eq!(prefix[1], layout.count_base + 2);
    assert_eq!(prefix[2], layout.count_base + 128 + 6);
    assert_eq!(prefix[10], layout.count_base + 9 * 128 + 127);
    let bit = |b: u32| layout.fingerprint_base + b;
    assert_eq!(
        prefix[11..],
        [bit(7), bit(11), layout.level, bit(9), layout.level, bit(3), bit(40), layout.sep]
    );
    // No fingerprint leaves only the separators; the longest prefix fits.
    let empty = layout.prefix(&formula, &SparseFingerprint { entries: Vec::new() });
    assert_eq!(empty.len(), 1 + 10 + 2 + 1);
    let full = SparseFingerprint {
        entries: (0..300u16).map(|b| (b, 0.6)).collect(),
    };
    assert_eq!(layout.prefix(&formula, &full).len(), PREFIX_LEN);
    assert!(prefix.iter().all(|&id| id < layout.vocab_size));
}

#[test]
fn gathered_cache_continues_from_the_parent_row() {
    let _lock = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let device = Device::<R>::default();
    let model = Mamba3LmConfig::builder()
        .vocab_size(48)
        .d_model(32)
        .n_layers(2)
        .with_ssm(|s| {
            s.n_heads = 2;
            s.n_groups = 2;
            s.head_dim = 32;
            s.d_state = 8;
            s.chunk_size = 8;
        })
        .seed(5)
        .build()
        .unwrap()
        .init::<R, f32>(&device)
        .unwrap();
    model.eval();
    let _guard = mamba3::autograd::no_grad();
    // Three rows with different histories.
    let histories: [[u32; 4]; 3] = [[1, 2, 3, 4], [9, 8, 7, 6], [20, 21, 22, 23]];
    let flat: Vec<u32> = histories.iter().flatten().copied().collect();
    let mut cache = model.empty_cache(3, &device);
    let ids = IdTensor::from_slice(&flat, vec![3, 4], &device).unwrap();
    model.forward_cached(&ids, &mut cache).unwrap();
    // Children: two from row 2, one from row 0, one from row 1.
    let parents = [2u32, 0, 2, 1];
    let next = [30u32, 31, 32, 33];
    let parent_ids = IdTensor::from_slice(&parents, vec![4], &device).unwrap();
    let mut gathered = gather_lm_cache(&cache, &parent_ids, 3).unwrap();
    let step = IdTensor::from_slice(&next, vec![4, 1], &device).unwrap();
    let stepped = model.forward_cached(&step, &mut gathered).unwrap().to_f32();
    // Reference: each child's full sequence through the uncached model.
    for (child, (&parent, &token)) in parents.iter().zip(&next).enumerate() {
        let mut sequence = histories[parent as usize].to_vec();
        sequence.push(token);
        let ids = IdTensor::from_slice(&sequence, vec![1, 5], &device).unwrap();
        let full = model.forward(&ids, false).unwrap().to_f32();
        let want = &full[4 * 48..5 * 48];
        let got = &stepped[child * 48..(child + 1) * 48];
        let worst = want.iter().zip(got).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(worst < 1e-4, "child {child} differs from its parent's continuation by {worst}");
    }
    // A parent outside the rows is refused rather than read.
    let bad = IdTensor::from_slice(&[3u32], vec![1], &device).unwrap();
    assert!(gather_lm_cache(&cache, &bad, 3).is_err());
}

/// Logits that prefer, in every state, the listed tokens in the listed order
/// and treat every other token alike.
fn preferring(n_out: usize, rows: usize, order: &[u32]) -> Vec<f32> {
    let mut row = vec![0.0f32; n_out];
    for (rank, &token) in order.iter().enumerate() {
        row[token as usize] = 10.0 - rank as f32;
    }
    row.repeat(rows)
}

#[test]
fn beam_keeps_a_completion_a_dead_end_outscores() {
    let vocab = vocab();
    let n_out = (MOTIF_BASE + vocab.len() as u32) as usize;
    // Methane: after the root, `ATOM 0` outscores `END` but leads nowhere,
    // because no motif fits the budget any more. A beam of one must still
    // return the molecule.
    let methane: Formula = [1, 4, 0, 0, 0, 0, 0, 0, 0, 0];
    let order = [MOTIF_BASE, ATOM_BASE, END];
    let mut steps = 0;
    let found = beam_search(&vocab, &methane, n_out, preferring(n_out, 1, &order), 1, 16, |parents, tokens| {
        assert_eq!(parents.len(), tokens.len());
        steps += 1;
        Ok(preferring(n_out, tokens.len(), &order))
    })
    .unwrap();
    assert_eq!(found.finished.len(), 1);
    assert_eq!(found.finished[0].0, vec![MOTIF_BASE, END]);
    assert_eq!(steps, 1, "one step after the root, then the closing END");
    assert_eq!(found.row_steps, 1);
    // The score is the masked log-probability: the root is one of the motifs
    // that fit CH4 (only carbon), END is one of two allowed tokens.
    let fits = 1.0f64;
    let root = 10.0 - (10.0f64.exp() + (fits - 1.0)).ln();
    let end = 8.0 - (9.0f64.exp() + 8.0f64.exp()).ln();
    assert!((found.finished[0].1 - (root + end)).abs() < 1e-4, "{}", found.finished[0].1);

    // Ethanol-sized budget, wider beam: every returned sequence is accepted
    // by the machine for that formula, they are distinct, and best first.
    let c2h6o: Formula = [2, 6, 0, 1, 0, 0, 0, 0, 0, 0];
    let order = [MOTIF_BASE, ATOM_BASE, BOND_BASE + 1, MOTIF_BASE + 1, END];
    let found = beam_search(&vocab, &c2h6o, n_out, preferring(n_out, 1, &order), 8, 40, |_, tokens| {
        Ok(preferring(n_out, tokens.len(), &order))
    })
    .unwrap();
    assert!(found.finished.len() >= 2, "ethanol and dimethyl ether both fit C2H6O");
    for pair in found.finished.windows(2) {
        assert!(pair[0].1 >= pair[1].1);
        assert_ne!(pair[0].0, pair[1].0);
    }
    for (tokens, _) in &found.finished {
        let machine = MotifMachine::run(&vocab, Some(&c2h6o), tokens).unwrap();
        assert_eq!(machine.formula(), c2h6o);
    }
    // A width of zero and a wrong first row are refused.
    assert!(beam_search(&vocab, &c2h6o, n_out, vec![0.0; n_out], 0, 4, |_, t| Ok(vec![0.0; t.len() * n_out])).is_err());
    assert!(beam_search(&vocab, &c2h6o, n_out, vec![0.0; 3], 2, 4, |_, t| Ok(vec![0.0; t.len() * n_out])).is_err());
    // An output size that does not cover the vocabulary is refused, not indexed.
    let short = MOTIF_BASE as usize;
    assert!(beam_search(&vocab, &c2h6o, short, vec![0.0; short], 2, 4, |_, t| Ok(vec![0.0; t.len() * short])).is_err());
    // NaN logits drop the row instead of poisoning scores.
    let found = beam_search(&vocab, &c2h6o, n_out, vec![f32::NAN; n_out], 2, 4, |_, t| Ok(vec![0.0; t.len() * n_out])).unwrap();
    assert!(found.finished.is_empty());
    // Negative infinity is probability zero: after the methane root, END at
    // 0 and ATOM 0 at -inf still complete methane, with END's probability 1.
    let found = beam_search(&vocab, &methane, n_out, preferring(n_out, 1, &[MOTIF_BASE]), 4, 16, |_, tokens| {
        let mut row = vec![0.0f32; n_out];
        row[ATOM_BASE as usize] = f32::NEG_INFINITY;
        Ok(row.repeat(tokens.len()))
    })
    .unwrap();
    assert_eq!(found.finished.len(), 1);
    assert_eq!(found.finished[0].0, vec![MOTIF_BASE, END]);
    assert!((found.finished[0].1 - 0.0).abs() < 1e-6);
    // A positive-infinite allowed logit drops the row.
    let found = beam_search(&vocab, &methane, n_out, preferring(n_out, 1, &[MOTIF_BASE]), 4, 16, |_, tokens| {
        let mut row = vec![0.0f32; n_out];
        row[END as usize] = f32::INFINITY;
        Ok(row.repeat(tokens.len()))
    })
    .unwrap();
    assert!(found.finished.is_empty());
}

#[test]
fn beam_breaks_ties_by_row_then_token_and_follows_its_parents() {
    let vocab = vocab();
    let n_out = (MOTIF_BASE + vocab.len() as u32) as usize;
    // All logits equal: the kept continuations are the lowest tokens, and
    // every `advance` call names, for each kept row, the row it continues.
    let c2h6o: Formula = [2, 6, 0, 1, 0, 0, 0, 0, 0, 0];
    let mut calls: Vec<(Vec<u32>, Vec<u32>)> = Vec::new();
    let mut previous_rows = 1usize;
    let found = beam_search(&vocab, &c2h6o, n_out, vec![0.0; n_out], 2, 40, |parents, tokens| {
        assert!(parents.iter().all(|&p| (p as usize) < previous_rows), "a parent past the live rows");
        previous_rows = parents.len();
        calls.push((parents.to_vec(), tokens.to_vec()));
        Ok(vec![0.0; tokens.len() * n_out])
    })
    .unwrap();
    // The root step: the two motifs that fit C2H6O with the lowest ids.
    assert_eq!(calls[0].0, vec![0, 0]);
    assert_eq!(calls[0].1, vec![MOTIF_BASE, MOTIF_BASE + 1]);
    assert!(calls.iter().all(|(parents, tokens)| parents.len() == tokens.len() && parents.len() <= 2));
    assert!(!found.finished.is_empty());
    // Deterministic: the same search again gives the same sequences.
    let again = beam_search(&vocab, &c2h6o, n_out, vec![0.0; n_out], 2, 40, |_, tokens| Ok(vec![0.0; tokens.len() * n_out])).unwrap();
    assert_eq!(found.finished, again.finished);
}

#[test]
fn beam_returns_a_sequence_of_exactly_the_token_limit() {
    let vocab = vocab();
    let n_out = (MOTIF_BASE + vocab.len() as u32) as usize;
    let methane: Formula = [1, 4, 0, 0, 0, 0, 0, 0, 0, 0];
    let order = [MOTIF_BASE, END];
    // `MOTIF, END` is two tokens: a limit of two must return it after one
    // advance, and a limit of one cannot.
    let mut advances = 0;
    let found = beam_search(&vocab, &methane, n_out, preferring(n_out, 1, &order), 4, 2, |_, tokens| {
        advances += 1;
        Ok(preferring(n_out, tokens.len(), &order))
    })
    .unwrap();
    assert_eq!(found.finished.len(), 1);
    assert_eq!(found.finished[0].0, vec![MOTIF_BASE, END]);
    assert_eq!(advances, 1, "the logits after the last allowed token are never asked for");
    let mut advances = 0;
    let found = beam_search(&vocab, &methane, n_out, preferring(n_out, 1, &order), 4, 1, |_, tokens| {
        advances += 1;
        Ok(preferring(n_out, tokens.len(), &order))
    })
    .unwrap();
    assert!(found.finished.is_empty());
    assert_eq!(advances, 0);

    // The encoder model with two positions writes that two-token sequence.
    let _lock = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let device = Device::<R>::default();
    let model: MotifModel<R, f32> = MotifModel::init(
        &MotifModelConfig {
            n_out,
            d_model: 64,
            layers: 1,
            attention_heads: 4,
            d_state: 8,
            max_tokens: 2,
            fingerprint_slots: 16,
            spectrum_slots: 0,
            seed: 1,
        },
        &device,
    )
    .unwrap();
    model.set_training(false);
    let _guard = mamba3::autograd::no_grad();
    let empty = SparseFingerprint { entries: Vec::new() };
    let fingerprints = FingerprintBatch::build(std::slice::from_ref(&empty), 16).unwrap();
    let conditioning = model.encode(&[methane], &fingerprints, None, &device).unwrap();
    let mut state = model.start(&conditioning, &device).unwrap();
    let start = IdTensor::from_slice(&[model.start_token()], vec![1], &device).unwrap();
    let first = model.step(&mut state, &start, &device).unwrap().to_f32();
    let limit = model.config().max_tokens;
    let found = beam_search(&vocab, &methane, n_out, first, 4, limit, |parents, tokens| {
        let parent_ids = IdTensor::from_slice(parents, vec![parents.len()], &device)?;
        model.gather(&mut state, &parent_ids)?;
        let ids = IdTensor::from_slice(tokens, vec![tokens.len()], &device)?;
        Ok(model.step(&mut state, &ids, &device)?.to_f32())
    })
    .unwrap();
    // Methane's budget admits one root and then only END.
    assert_eq!(found.finished.len(), 1);
    assert_eq!(found.finished[0].0, vec![MOTIF_BASE, END]);
}

#[test]
fn machine_refuses_to_grow_past_its_bound() {
    let vocab = vocab();
    // A carbon chain: every attachment adds one atom.
    let mut machine = MotifMachine::new();
    machine.apply(&vocab, None, MOTIF_BASE).unwrap();
    while machine.elements.len() < MAX_GRAPH_ATOMS {
        for token in [ATOM_BASE, BOND_BASE + 1, MOTIF_BASE, ATOM_BASE] {
            machine.apply(&vocab, None, token).unwrap();
        }
    }
    assert_eq!(machine.elements.len(), MAX_GRAPH_ATOMS);
    assert_eq!(machine.counts[0] as usize, MAX_GRAPH_ATOMS);
    // The cached sums still describe the graph.
    assert_eq!(machine.free_sum, machine.free.iter().map(|&f| u32::from(f)).sum::<u32>());
    assert_eq!(machine.formula()[1] as u32, machine.free_sum);
    machine.apply(&vocab, None, ATOM_BASE).unwrap();
    machine.apply(&vocab, None, BOND_BASE + 1).unwrap();
    let mut allowed = Vec::new();
    machine.allowed_tokens(&vocab, None, &mut allowed);
    assert!(allowed.is_empty(), "no motif may take the graph past the bound");
    assert!(machine.apply(&vocab, None, MOTIF_BASE).is_err());
}

#[test]
fn text_formats_refuse_what_the_lean_reference_refuses() {
    assert!(text::vocab("2\n1 6 4 0\n").is_err(), "fewer motifs than the count");
    assert!(text::vocab("1\n1 6 4 0\n1 8 2 0\n").is_err(), "more motifs than the count");
    assert!(text::vocab("1\n1 6 4 1 0 0 1\n").is_err(), "a bond from an atom to itself");
    assert!(text::vocab("1\n1 1 1 0\n").is_err(), "hydrogen is not a motif atom");
    assert!(text::vocab("1\n2 6 6 3 3 0\n").is_err(), "two atoms with no bond are not one motif");
    assert!(text::vocab("1\n2 6 6 3 3 1 0 1 1\n").is_ok());
    assert_eq!(text::budget("-").unwrap(), None);
    assert_eq!(text::budget("").unwrap(), None);
    assert!(text::budget("4 6").is_err(), "an element without a count");
    assert_eq!(text::budget("4 6 1 6 9").unwrap(), Some([1, 4, 0, 0, 0, 0, 0, 0, 0, 0]));
    assert_eq!(text::token("E"), END);
    assert_eq!(text::token("A31"), ATOM_BASE + 31);
    for word in ["A32", "B0", "B4", "M99999999999", "X1", "", "E1"] {
        assert_eq!(text::token(word), text::REJECTED, "{word:?}");
    }
}

fn small_model(device: &Device<R>, spectrum_slots: usize) -> MotifModel<R, f32> {
    MotifModel::init(
        &MotifModelConfig {
            n_out: 48,
            d_model: 64,
            layers: 2,
            attention_heads: 4,
            d_state: 8,
            max_tokens: 12,
            fingerprint_slots: 16,
            spectrum_slots,
            seed: 3,
        },
        device,
    )
    .unwrap()
}

fn evidence() -> SpectrumEvidence {
    SpectrumEvidence {
        peaks: vec![(42_033_733, 0.2), (84_080_559, 1.0), (120_044_000, 0.4)],
        precursor_mz: 180_065_000,
        adduct: 1,
        neutral_mass: 179_057_724,
    }
}

#[test]
fn encoder_model_steps_as_it_teaches_and_follows_gathered_parents() {
    let _lock = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let device = Device::<R>::default();
    let model = small_model(&device, 8);
    model.set_training(false);
    let _guard = mamba3::autograd::no_grad();
    let formula: Formula = [9, 8, 0, 4, 0, 0, 0, 0, 0, 0];
    let fingerprint = SparseFingerprint {
        entries: vec![(3, 0.2), (7, 0.9), (900, 1.0), (4000, 0.4)],
    };
    let spectrum = evidence();
    let fingerprints = FingerprintBatch::build(std::slice::from_ref(&fingerprint), 16).unwrap();
    let spectra = SpectrumBatch::build(&[Some(&spectrum)], 8).unwrap();
    let conditioning = model.encode(&[formula], &fingerprints, Some(&spectra), &device).unwrap();
    assert_eq!(conditioning.memory.dims(), vec![1, 1 + 16 + 8, 64]);

    // Teacher pass over one sequence against the stepped pass.
    let tokens: [u32; 6] = [40, 5, 2, 37, 5, 1];
    let mut inputs = vec![model.start_token()];
    inputs.extend_from_slice(&tokens[..5]);
    let ids = IdTensor::from_slice(&inputs, vec![1, 6], &device).unwrap();
    let teacher = model.logits(&ids, &conditioning).unwrap().to_f32();
    let mut state = model.start(&conditioning, &device).unwrap();
    for (position, &input) in inputs.iter().enumerate() {
        let token = IdTensor::from_slice(&[input], vec![1], &device).unwrap();
        let stepped = model.step(&mut state, &token, &device).unwrap().to_f32();
        let want = &teacher[position * 48..(position + 1) * 48];
        let worst = want.iter().zip(&stepped).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(worst < 2e-4, "position {position}: step differs from teacher by {worst}");
    }

    // Beam shape: two rows diverge, then are reordered and duplicated; each
    // child must match the teacher pass of its own full history.
    let mut state = model.start(&conditioning, &device).unwrap();
    let start = IdTensor::from_slice(&[model.start_token()], vec![1], &device).unwrap();
    model.step(&mut state, &start, &device).unwrap();
    let grow = IdTensor::from_slice(&[0u32, 0], vec![2], &device).unwrap();
    model.gather(&mut state, &grow).unwrap();
    assert_eq!(state.rows(), 2);
    let first = IdTensor::from_slice(&[40u32, 41], vec![2], &device).unwrap();
    model.step(&mut state, &first, &device).unwrap();
    let parents = [1u32, 0, 1];
    let next = [5u32, 6, 7];
    model.gather(&mut state, &IdTensor::from_slice(&parents, vec![3], &device).unwrap()).unwrap();
    let stepped = model
        .step(&mut state, &IdTensor::from_slice(&next, vec![3], &device).unwrap(), &device)
        .unwrap()
        .to_f32();
    for (child, (&parent, &token)) in parents.iter().zip(&next).enumerate() {
        let history = [model.start_token(), [40u32, 41][parent as usize], token];
        let ids = IdTensor::from_slice(&history, vec![1, 3], &device).unwrap();
        let full = model.logits(&ids, &conditioning).unwrap().to_f32();
        let want = &full[2 * 48..3 * 48];
        let got = &stepped[child * 48..(child + 1) * 48];
        let worst = want.iter().zip(got).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(worst < 2e-4, "child {child} differs from its own history by {worst}");
    }
    assert!(model.gather(&mut state, &IdTensor::from_slice(&[3u32], vec![1], &device).unwrap()).is_err());
}

#[test]
fn encoder_model_reads_its_conditioning_and_ignores_batch_mates() {
    let _lock = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let device = Device::<R>::default();
    let model = small_model(&device, 8);
    model.set_training(false);
    let _guard = mamba3::autograd::no_grad();
    let formula: Formula = [9, 8, 0, 4, 0, 0, 0, 0, 0, 0];
    let other: Formula = [3, 8, 0, 1, 0, 0, 0, 0, 0, 0];
    let fp = SparseFingerprint { entries: vec![(7, 0.9), (900, 1.0)] };
    let empty = SparseFingerprint { entries: Vec::new() };
    let spectrum = evidence();
    let inputs = [model.start_token(), 40, 5, 2];
    let logits_of = |formulas: &[Formula], fps: &[SparseFingerprint], spectra: &[Option<&SpectrumEvidence>]| {
        let b = formulas.len();
        let conditioning = model
            .encode(
                formulas,
                &FingerprintBatch::build(fps, 16).unwrap(),
                Some(&SpectrumBatch::build(spectra, 8).unwrap()),
                &device,
            )
            .unwrap();
        let ids = IdTensor::from_slice(&inputs.repeat(b), vec![b, 4], &device).unwrap();
        model.logits(&ids, &conditioning).unwrap().to_f32()
    };
    let alone = logits_of(&[formula], std::slice::from_ref(&fp), &[Some(&spectrum)]);
    let differs = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
    // Each input changes the output.
    assert!(differs(&alone, &logits_of(&[other], std::slice::from_ref(&fp), &[Some(&spectrum)])) > 1e-4);
    assert!(differs(&alone, &logits_of(&[formula], std::slice::from_ref(&empty), &[Some(&spectrum)])) > 1e-4);
    assert!(differs(&alone, &logits_of(&[formula], std::slice::from_ref(&fp), &[None])) > 1e-4);
    // A query's logits do not depend on what else is in the batch.
    let pair = logits_of(&[other, formula], &[empty.clone(), fp.clone()], &[None, Some(&spectrum)]);
    assert!(differs(&alone, &pair[alone.len()..]) < 2e-4);
    // Every parameter is reached by the loss.
    drop(_guard);
    model.set_training(true);
    let conditioning = model
        .encode(
            &[formula],
            &FingerprintBatch::build(std::slice::from_ref(&fp), 16).unwrap(),
            Some(&SpectrumBatch::build(&[Some(&spectrum)], 8).unwrap()),
            &device,
        )
        .unwrap();
    let ids = IdTensor::from_slice(&inputs, vec![1, 4], &device).unwrap();
    let loss = model.logits(&ids, &conditioning).unwrap().sum().unwrap();
    let grads = loss.backward().unwrap();
    let missing: Vec<String> = model
        .named_parameters()
        .into_iter()
        .filter(|(_, p)| grads.get(p.id()).is_none())
        .map(|(name, _)| name)
        .collect();
    assert!(missing.is_empty(), "parameters without a gradient: {missing:?}");
    // A model without the spectrum encoder refuses spectral evidence.
    let plain = small_model(&device, 0);
    let fps = FingerprintBatch::build(std::slice::from_ref(&fp), 16).unwrap();
    assert!(plain.encode(&[formula], &fps, Some(&SpectrumBatch::build(&[Some(&spectrum)], 8).unwrap()), &device).is_err());
    assert!(plain.encode(&[formula], &fps, None, &device).is_ok());
}
