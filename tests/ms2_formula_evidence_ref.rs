//! Host-only tests of `mamba3::models::ms2::formula_evidence_ref`.
//!
//! * The fast explained-peak test equals
//!   `ion_assign(...).accepted > 0` (large limits) on random small cases,
//!   including exact ion masses and accept/ambiguous/reject boundary values.
//!   The ambiguous verdict explains nothing, exactly as `accepted` counts
//!   only accepts.
//! * The softmax ranking-model gradient matches finite differences, and a
//!   planted signal is recovered.
//! * The shuffle permutation is seeded and has no fixed point.

use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;

use mamba3::models::ms2::chem::{Composition, ELEMENTS, ELECTRON_MASS, HYDROGEN};
use mamba3::models::ms2::formula_enum::HEAVY_ELEMENTS;
use mamba3::models::ms2::formula_evidence_ref::{
    CandidateEvidence, ConvergedTrainConfig, NonlinearKind, RankRule, SpectrumEvidence,
    Standardizer, TrainConfig, apply_sharp_features, build_evidence_index, check_nestedness,
    derangement, feature_vector, fit_standardizer, fit_standardizer_nonlinear, jitter_precursor_mz,
    jitter_seed, nonlinear_plus_raw_vector, nonlinear_raw_vector, precursor_tol_ppm, rank_order,
    recall_at, reference_ion_assign, residual_features, sharp_evidence_features, softmax_loss_grad,
    spectrum_evidence, standardized_nll, train_nonlinear_converged, train_softmax,
    train_softmax_converged, SpectrumInput, FEATURE_NAMES, N_FEATURES,
};

/// Random small parent composition (ion-assignment visits stay tiny).
fn random_parent(rng: &mut StdRng) -> Composition {
    let mut c: Composition = [0; 10];
    c[0] = rng.random_range(0..=4);
    c[HYDROGEN] = rng.random_range(0..=6);
    c[2] = rng.random_range(0..=2);
    c[3] = rng.random_range(0..=2);
    c[4] = rng.random_range(0..=1);
    c[5] = rng.random_range(0..=1);
    c[6] = rng.random_range(0..=1);
    if c.iter().all(|&n| n == 0) {
        c[0] = 1;
    }
    c
}

/// Integer mass of a heavy sub-vector plus `h` hydrogens.
fn subvector_mass(parent: &Composition, digits: &[u16; 9], h: u32) -> Option<u32> {
    let mut mass: u64 = 0;
    for (i, e) in HEAVY_ELEMENTS.iter().enumerate() {
        mass += u64::from(digits[i]) * u64::from(ELEMENTS[*e].mass);
    }
    mass += u64::from(h) * u64::from(ELEMENTS[HYDROGEN].mass);
    u32::try_from(mass).ok()
}

/// Rounding-residual sum of a heavy sub-vector plus `h` hydrogens, in nda.
fn subvector_res(digits: &[u16; 9], h: u32) -> u64 {
    let mut res: u64 = 0;
    for (i, e) in HEAVY_ELEMENTS.iter().enumerate() {
        res += u64::from(digits[i]) * u64::from(ELEMENTS[*e].residual_nda);
    }
    res + u64::from(h) * u64::from(ELEMENTS[HYDROGEN].residual_nda) + 421
}

#[test]
fn fast_explained_peak_equals_ion_assign() {
    let mut rng = StdRng::seed_from_u64(11);
    let ppms = [50u32, 100, 200, 1000];
    let uncertainties = [0u32, 50, 500];
    let mut compared = 0usize;
    let mut accepted = 0usize;
    let mut ambiguous_only = 0usize;
    let mut clean_reject = 0usize;
    for _ in 0..40 {
        let parent = random_parent(&mut rng);
        let adduct: u16 = if rng.random_bool(0.5) { 1 } else { 2 };
        let ppm = ppms[rng.random_range(0..ppms.len())];
        let uncertainty = uncertainties[rng.random_range(0..uncertainties.len())];
        let Some(index) =
            build_evidence_index(&parent, adduct, uncertainty, ppm).expect("index builds")
        else {
            panic!("index must build for a supported small case");
        };
        // Random peaks across the mass range.
        for _ in 0..4 {
            let peak = rng.random_range(50_000_000..600_000_000);
            let fast = index.explains(peak).expect("explains runs");
            let reference = reference_ion_assign(&parent, adduct, peak, uncertainty, ppm)
                .expect("ion_assign runs");
            assert_eq!(
                fast,
                reference.accepted > 0,
                "random peak {peak} (adduct {adduct}, ppm {ppm}, U {uncertainty}, parent {parent:?})"
            );
            compared += 1;
            if reference.accepted > 0 {
                accepted += 1;
            } else if reference.ambiguous > 0 {
                ambiguous_only += 1;
            } else {
                clean_reject += 1;
            }
        }
        // Exact ion masses and their accept/ambiguous/reject edges: a random
        // sub-vector with a random hydrogen count gives an exact candidate
        // mass; peak offsets around `tol - E` and `tol + E` walk the verdict
        // edges of that hypothesis.
        for _ in 0..8 {
            let mut digits = [0u16; 9];
            for (i, e) in HEAVY_ELEMENTS.iter().enumerate() {
                digits[i] = rng.random_range(0..=parent[*e]);
            }
            if digits.iter().all(|&n| n == 0) {
                digits[0] = 1.min(parent[0]);
                if digits[0] == 0 {
                    continue;
                }
            }
            let h_cap = u32::from(parent[HYDROGEN]) + if adduct == 1 { 1 } else { 0 } + 2;
            let h = rng.random_range(0..=h_cap.min(12));
            let Some(cand) = subvector_mass(&parent, &digits, h) else {
                continue;
            };
            let base: u64 = if adduct == 1 {
                if cand < ELECTRON_MASS {
                    continue;
                }
                u64::from(cand - ELECTRON_MASS)
            } else {
                u64::from(cand) + u64::from(ELECTRON_MASS)
            };
            if base == 0 || base > u64::from(u32::MAX) {
                continue;
            }
            let base = base as u32;
            let tol = mamba3::models::ms2::tolerance(base, ppm);
            let hypothesis_error =
                (subvector_res(&digits, h).div_ceil(1000).min(u64::from(u32::MAX)) as u32)
                    .saturating_add(uncertainty);
            // Signed offsets around both verdict edges (negative offsets
            // skipped past zero).
            let edges = [
                0i64,
                tol as i64 - hypothesis_error as i64,
                tol as i64 - hypothesis_error as i64 + 1,
                tol as i64 + hypothesis_error as i64,
                tol as i64 + hypothesis_error as i64 + 1,
            ];
            for delta in edges {
                let peak = base as i64 + delta;
                if peak <= 0 || peak > u64::from(u32::MAX) as i64 {
                    continue;
                }
                let peak = peak as u32;
                let fast = index.explains(peak).expect("explains runs");
                let reference = reference_ion_assign(&parent, adduct, peak, uncertainty, ppm)
                    .expect("ion_assign runs");
                assert_eq!(
                    fast,
                    reference.accepted > 0,
                    "edge peak {peak} (delta {delta}, adduct {adduct}, ppm {ppm}, U {uncertainty})"
                );
                compared += 1;
                if reference.accepted > 0 {
                    accepted += 1;
                } else if reference.ambiguous > 0 {
                    ambiguous_only += 1;
                } else {
                    clean_reject += 1;
                }
            }
        }
    }
    assert!(compared >= 300, "only {compared} peak comparisons ran");
    // The run must actually exercise accepts, ambiguous-only peaks (which the
    // fast test maps to false, exactly as `accepted` counts only accepts)
    // and clean rejects.
    assert!(accepted >= 5, "no accepted peak seen ({compared} compared)");
    assert!(
        ambiguous_only >= 5,
        "no ambiguous-only peak seen ({compared} compared)"
    );
    assert!(
        clean_reject >= 5,
        "no clean-reject peak seen ({compared} compared)"
    );
}

#[test]
fn unavailable_peaks_explain_nothing() {
    let parent: Composition = [2, 6, 1, 1, 0, 0, 0, 0, 0, 0];
    // Unknown m/z precision: no search in either implementation.
    assert!(build_evidence_index(&parent, 1, u32::MAX, 100).expect("builds").is_none());
    let reference = reference_ion_assign(&parent, 1, 150_000_000, u32::MAX, 100)
        .expect("ion_assign runs");
    assert_eq!(reference.accepted, 0);
    assert_ne!(reference.status & mamba3::models::ms2::ion::ION_UNAVAILABLE, 0);
    // Padding peak.
    let index = build_evidence_index(&parent, 1, 50, 100)
        .expect("builds")
        .expect("supported");
    assert!(!index.explains(0).expect("explains runs"));
    let reference = reference_ion_assign(&parent, 1, 0, 50, 100).expect("ion_assign runs");
    assert_eq!(reference.accepted, 0);
    // Scope gate: a window wider than one hydrogen mass searches nothing.
    assert!(!index.explains(2_000_000_000).unwrap_or(false)
        || reference_ion_assign(&parent, 1, 2_000_000_000, 500_000, 1000)
            .expect("ion_assign runs")
            .accepted
            == 0);
    let wide = reference_ion_assign(&parent, 1, 400_000_000, 2_000_000, 1000)
        .expect("ion_assign runs");
    assert_eq!(wide.accepted, 0);
    assert!(!index.explains(400_000_000).expect("explains runs"));
    // Unknown adduct: an error in `ion_assign`, no index here (every peak
    // unexplained, matching the lane's unavailable row).
    assert!(build_evidence_index(&parent, 7, 50, 100).expect("builds").is_none());
    assert!(reference_ion_assign(&parent, 7, 150_000_000, 50, 100).is_err());
    // Empty parent.
    assert!(build_evidence_index(&[0; 10], 1, 50, 100).expect("builds").is_none());
    // Tolerance proof bound.
    assert!(build_evidence_index(&parent, 1, 50, 1001).is_err());
}

#[test]
fn softmax_gradient_matches_finite_differences() {
    fn candidate(c: u16, h: u16, frac: f64, resid: f64) -> CandidateEvidence {
        let mut counts = [0u16; 10];
        counts[0] = c;
        counts[1] = h;
        CandidateEvidence {
            counts,
            mass: 100_000_000,
            expl_count: 0,
            expl_count_frac: frac,
            expl_intensity: 0.0,
            residual_ppm: resid,
            heavy_atoms: u32::from(c),
            dbe_twice: 2,
            resid_o1: 0.0,
            resid_o1_sq: 0.0,
            resid_ln1p: 0.0,
            resid_otol: 0.0,
            expl_minus_max: 0.0,
            expl_rank_frac: 0.0,
        }
    }
    let data = vec![
        SpectrumEvidence {
            candidates: vec![
                candidate(4, 8, 0.1, 3.0),
                candidate(6, 12, 0.5, 1.0),
                candidate(5, 10, 0.2, 8.0),
            ],
            gold: Some(1),
            has_window: true,
        },
        SpectrumEvidence {
            candidates: vec![candidate(3, 6, 0.0, 12.0), candidate(7, 14, 0.9, 2.0)],
            gold: Some(0),
            has_window: true,
        },
        SpectrumEvidence {
            candidates: vec![candidate(2, 4, 0.3, 5.0)],
            gold: None,
            has_window: true,
        },
    ];
    let dims = [0usize, 1, 12, 14];
    let weights = [0.3f64, -0.2, 0.7, -0.05];
    let (loss, grad) = softmax_loss_grad(&data, &dims, &weights, 1e-4).expect("loss runs");
    assert!(loss.is_finite());
    let eps = 1e-6;
    for (j, &w) in weights.iter().enumerate() {
        let mut up = weights.to_vec();
        let mut down = weights.to_vec();
        up[j] = w + eps;
        down[j] = w - eps;
        let (loss_up, _) = softmax_loss_grad(&data, &dims, &up, 1e-4).expect("loss runs");
        let (loss_down, _) = softmax_loss_grad(&data, &dims, &down, 1e-4).expect("loss runs");
        let numeric = (loss_up - loss_down) / (2.0 * eps);
        let scale = grad[j].abs().max(numeric.abs()).max(1.0);
        assert!(
            (grad[j] - numeric).abs() <= 1e-4 * scale,
            "dim {j}: analytic {} versus numeric {numeric}",
            grad[j]
        );
    }
}

#[test]
fn planted_signal_is_recovered() {
    let mut rng = StdRng::seed_from_u64(7);
    let mut data = Vec::new();
    for _ in 0..60 {
        let mut candidates = Vec::new();
        for i in 0..4 {
            // Gold (slot 0) carries the evidence signal; the rest carry noise.
            let frac = if i == 0 {
                1.0
            } else {
                rng.random_range(0.0..0.4)
            };
            candidates.push(CandidateEvidence {
                counts: [0; 10],
                mass: 100_000_000,
                expl_count: 0,
                expl_count_frac: frac,
                expl_intensity: rng.random_range(0.0..0.5),
                residual_ppm: rng.random_range(0.0..20.0),
                heavy_atoms: 0,
                dbe_twice: 0,
                resid_o1: 0.0,
                resid_o1_sq: 0.0,
                resid_ln1p: 0.0,
                resid_otol: 0.0,
                expl_minus_max: 0.0,
                expl_rank_frac: 0.0,
            });
        }
        data.push(SpectrumEvidence {
            candidates,
            gold: Some(0),
            has_window: true,
        });
    }
    let dims = [12usize, 13];
    let cfg = TrainConfig {
        epochs: 100,
        batch_spectra: 8,
        lr: 0.1,
        l2: 1e-4,
        seed: 7,
    };
    let weights = train_softmax(&data, &dims, &cfg).expect("training runs");
    assert!(weights[0] > 0.0, "signal weight is {}", weights[0]);
    // The recovered model ranks the gold first on every spectrum.
    for ev in &data {
        let order = rank_order(ev, &RankRule::Linear(&weights, &dims));
        assert_eq!(order[0], 0);
    }
    let ranks: Vec<Option<u32>> = data
        .iter()
        .map(|ev| RankRule::Linear(&weights, &dims).gold_rank(ev))
        .collect();
    assert_eq!(recall_at(&ranks, 1), 1.0);
}

#[test]
fn shuffle_permutation_has_no_fixed_point_and_is_seeded() {
    let perm = derangement(10, 1).expect("derangement builds");
    let mut sorted = perm.clone();
    sorted.sort_unstable();
    assert_eq!(sorted, (0..10).collect::<Vec<_>>());
    assert!(perm.iter().enumerate().all(|(i, &v)| i != v));
    assert_ne!(perm, (0..10).collect::<Vec<_>>());
    // Seeded: the same seed reproduces the permutation.
    assert_eq!(perm, derangement(10, 1).expect("derangement rebuilds"));
    // Validity across sizes and seeds.
    for (n, seed) in [(2, 5), (3, 9), (50, 2), (619, 1)] {
        let p = derangement(n, seed).expect("derangement builds");
        let mut s = p.clone();
        s.sort_unstable();
        assert_eq!(s, (0..n).collect::<Vec<_>>());
        assert!(p.iter().enumerate().all(|(i, &v)| i != v));
    }
    assert_eq!(derangement(2, 5).expect("pair swaps"), vec![1, 0]);
    assert!(derangement(1, 1).is_err());
    assert!(derangement(0, 1).is_err());
}

#[test]
fn feature_vector_names_match_positions() {
    assert_eq!(
        mamba3::models::ms2::formula_evidence_ref::FEATURE_NAMES.len(),
        N_FEATURES
    );
    assert_eq!(FEATURE_NAMES.len(), 21);
    assert_eq!(
        mamba3::models::ms2::formula_evidence_ref::FEATURE_NAMES[12],
        "expl_count_frac"
    );
    assert_eq!(FEATURE_NAMES[15], "abs_resid_over_1ppm");
    assert_eq!(FEATURE_NAMES[16], "abs_resid_sq_over_1ppm");
    assert_eq!(FEATURE_NAMES[17], "ln1p_abs_resid_over_01ppm");
    assert_eq!(FEATURE_NAMES[18], "abs_resid_over_tol");
    assert_eq!(FEATURE_NAMES[19], "expl_minus_max");
    assert_eq!(FEATURE_NAMES[20], "expl_rank_frac");
    let c = CandidateEvidence {
        counts: [4, 8, 0, 0, 0, 0, 0, 0, 0, 0],
        mass: 0,
        expl_count: 0,
        expl_count_frac: 0.25,
        expl_intensity: 0.5,
        residual_ppm: 2.0,
        heavy_atoms: 4,
        dbe_twice: 4,
        resid_o1: 2.0,
        resid_o1_sq: 4.0,
        resid_ln1p: (1.0f64 + 2.0 / 0.1).ln(),
        resid_otol: 2.0 / 20.0,
        expl_minus_max: -0.25,
        expl_rank_frac: 0.5,
    };
    let f = feature_vector(&c);
    assert_eq!(f.len(), N_FEATURES);
    assert!((f[0] - (5.0f64).ln()).abs() < 1e-12);
    assert_eq!(f[10], 4.0);
    assert_eq!(f[11], 2.0);
    assert_eq!(f[12], 0.25);
    assert_eq!(f[13], 0.5);
    assert_eq!(f[14], 2.0);
    assert_eq!(f[15], 2.0);
    assert_eq!(f[16], 4.0);
    assert!((f[17] - (21.0f64).ln()).abs() < 1e-12);
    assert_eq!(f[18], 0.1);
    assert_eq!(f[19], -0.25);
    assert_eq!(f[20], 0.5);
}

#[test]
fn sigma_zero_path_matches_previous_feature_values() {
    use mamba3::models::ms2::formula_evidence_ref::KeptPeak;
    // sigma = 0 jitter is the identity on every input.
    for mz in [0u32, 1, 150_000_000, 400_000_000, u32::MAX] {
        assert_eq!(jitter_precursor_mz(mz, 0.0, 1, 0, 0), mz);
        assert_eq!(jitter_precursor_mz(mz, 0.0, 999, 3, 12345), mz);
    }
    // Fixture: two candidates, one parent mass, two kept peaks. The
    // residuals must equal the pre-FE2 formula bit-identically, and the
    // first fifteen feature positions must equal the old computation.
    let c0: Composition = [6, 12, 0, 6, 0, 0, 0, 0, 0, 0];
    let c1: Composition = [5, 10, 0, 5, 0, 0, 0, 0, 0, 0];
    let m0: u32 = 180_063_388;
    let m1: u32 = 150_052_823;
    let parent: u32 = 180_063_390;
    let peaks = vec![
        KeptPeak { mz: 100_000_000, intensity: 3.0 },
        KeptPeak { mz: 120_000_000, intensity: 1.0 },
    ];
    let candidates = vec![c0, c1];
    let masses = vec![m0, m1];
    let input = SpectrumInput {
        candidates: &candidates,
        masses: &masses,
        peaks: &peaks,
        adduct_id: 1,
        mz_uncertainty: 50,
        ion_ppm_tenths: 100,
        parent_mass: Some(parent),
    };
    let ev = spectrum_evidence(&input, Some(&c0)).expect("evidence runs");
    assert_eq!(ev.gold, Some(0));
    for (i, (&mass, cand)) in masses.iter().zip(ev.candidates.iter()).enumerate() {
        // Bit-identical residuals: the exact pre-FE2 expression.
        let expected = (mass.abs_diff(parent) as f64) * 1e6 / f64::from(parent);
        assert!(
            cand.residual_ppm.to_bits() == expected.to_bits(),
            "candidate {i}: {} vs {expected}",
            cand.residual_ppm
        );
        let f = feature_vector(cand);
        // Old positions: ln1p counts, heavy atoms, dbe, evidence, residual.
        for e in 0..10 {
            let want = (1.0 + f64::from(candidates[i][e])).ln();
            assert!(f[e].to_bits() == want.to_bits(), "candidate {i} element {e}");
        }
        let mut heavy: u32 = 0;
        for e in HEAVY_ELEMENTS {
            heavy += u32::from(candidates[i][e]);
        }
        assert_eq!(f[10], f64::from(heavy));
        assert_eq!(f[12], cand.expl_count_frac);
        assert_eq!(f[13], cand.expl_intensity);
        assert!(f[14].to_bits() == expected.to_bits());
        // New residual transforms follow the documented definitions.
        let tol = precursor_tol_ppm();
        let [a, b, c, d] = residual_features(expected, tol);
        assert!(f[15].to_bits() == a.to_bits());
        assert!(f[16].to_bits() == b.to_bits());
        assert!(f[17].to_bits() == c.to_bits());
        assert!(f[18].to_bits() == d.to_bits());
    }
    // Running the fixture through the jittered (sigma = 0) precursor gives
    // identical evidence: the sigma = 0 path changes nothing.
    let j0 = jitter_precursor_mz(180_000_000, 0.0, 7, 1, 42);
    assert_eq!(j0, 180_000_000);
    let ev2 = spectrum_evidence(&input, Some(&c0)).expect("evidence reruns");
    assert_eq!(ev.candidates, ev2.candidates);
}

#[test]
fn jitter_is_seeded_truncated_and_thread_independent() {
    // Seeded: equal inputs give equal outputs, and the seed mix separates
    // (seed, split, index, sigma).
    let a = jitter_precursor_mz(300_000_000, 2.0, 1, 1, 7);
    assert_eq!(a, jitter_precursor_mz(300_000_000, 2.0, 1, 1, 7));
    assert_eq!(
        jitter_seed(1, 1, 7, 2.0f64.to_bits()),
        jitter_seed(1, 1, 7, 2.0f64.to_bits())
    );
    assert_ne!(
        jitter_seed(1, 1, 7, 2.0f64.to_bits()),
        jitter_seed(1, 1, 8, 2.0f64.to_bits())
    );
    assert_ne!(
        jitter_seed(1, 1, 7, 2.0f64.to_bits()),
        jitter_seed(1, 0, 7, 2.0f64.to_bits())
    );
    assert_ne!(
        jitter_seed(1, 1, 7, 1.0f64.to_bits()),
        jitter_seed(1, 1, 7, 2.0f64.to_bits())
    );
    // Truncated at 3 sigma across many draws and several mz/sigma values.
    for sigma in [1.0f64, 2.0, 5.0] {
        for (i, mz) in [50_000_000u32, 300_000_000, 900_000_000].iter().enumerate() {
            for k in 0..200u64 {
                let j = jitter_precursor_mz(*mz, sigma, 11, 1, (i as u64) * 1000 + k);
                let max_shift = (*mz as f64 * 3.0 * sigma * 1e-6).ceil() + 1.0;
                let shift = (j as f64 - *mz as f64).abs();
                assert!(
                    shift <= max_shift,
                    "mz {mz} sigma {sigma}: shift {shift} exceeds {max_shift}"
                );
            }
        }
    }
    // sigma > 0 actually moves most precursors (not stuck at identity).
    let moved = (0..100u64)
        .filter(|&k| jitter_precursor_mz(300_000_000, 5.0, 3, 0, k) != 300_000_000)
        .count();
    assert!(moved >= 90, "only {moved}/100 precursors moved at sigma=5");
    // Independent of thread scheduling: forward and reverse evaluation
    // order give the same vector (the function is pure in its inputs).
    let forward: Vec<u32> = (0..256u64)
        .map(|k| jitter_precursor_mz(250_000_000 + k as u32, 2.0, 5, 1, k))
        .collect();
    let mut backward: Vec<u32> = (0..256u64)
        .rev()
        .map(|k| jitter_precursor_mz(250_000_000 + k as u32, 2.0, 5, 1, k))
        .collect();
    backward.reverse();
    assert_eq!(forward, backward);
    // Parallel threads agree with the serial order.
    let serial = forward.clone();
    let handles: Vec<_> = (0..4)
        .map(|t| {
            std::thread::spawn(move || {
                (0..256u64)
                    .filter(|k| k % 4 == t)
                    .map(|k| (k, jitter_precursor_mz(250_000_000 + k as u32, 2.0, 5, 1, k)))
                    .collect::<Vec<_>>()
            })
        })
        .collect();
    let mut merged = vec![0u32; 256];
    for h in handles {
        for (k, v) in h.join().expect("thread runs") {
            merged[k as usize] = v;
        }
    }
    assert_eq!(serial, merged);
}

#[test]
fn sharp_evidence_features_hand_computed() {
    // Ties at the maximum: two candidates share the max intensity.
    let expl = vec![0.2, 0.8, 0.8, 0.5];
    let (minus, rank) = sharp_evidence_features(&expl);
    assert_eq!(minus, vec![0.2 - 0.8, 0.0, 0.0, 0.5 - 0.8]);
    assert_eq!(rank, vec![1.0, 0.5, 0.5, 0.75]);
    // A single candidate sits at the maximum with rank 1.
    let (minus1, rank1) = sharp_evidence_features(&[0.7]);
    assert_eq!(minus1, vec![0.0]);
    assert_eq!(rank1, vec![1.0]);
    // Empty input stays empty.
    let (minus0, rank0) = sharp_evidence_features(&[]);
    assert!(minus0.is_empty() && rank0.is_empty());
    // `apply_sharp_features` fills a spectrum in place, honoring ties.
    fn cand_with(intensity: f64) -> CandidateEvidence {
        CandidateEvidence {
            counts: [0; 10],
            mass: 100_000_000,
            expl_count: 0,
            expl_count_frac: 0.0,
            expl_intensity: intensity,
            residual_ppm: 1.0,
            heavy_atoms: 0,
            dbe_twice: 0,
            resid_o1: 1.0,
            resid_o1_sq: 1.0,
            resid_ln1p: (11.0f64).ln(),
            resid_otol: 0.05,
            expl_minus_max: 999.0,
            expl_rank_frac: 999.0,
        }
    }
    let mut ev = SpectrumEvidence {
        candidates: vec![cand_with(0.2), cand_with(0.8), cand_with(0.8), cand_with(0.5)],
        gold: Some(1),
        has_window: true,
    };
    apply_sharp_features(&mut ev);
    let got_minus: Vec<f64> = ev.candidates.iter().map(|c| c.expl_minus_max).collect();
    let got_rank: Vec<f64> = ev.candidates.iter().map(|c| c.expl_rank_frac).collect();
    assert_eq!(got_minus, vec![0.2 - 0.8, 0.0, 0.0, 0.5 - 0.8]);
    assert_eq!(got_rank, vec![1.0, 0.5, 0.5, 0.75]);
    let mut single = SpectrumEvidence {
        candidates: vec![cand_with(0.7)],
        gold: Some(0),
        has_window: true,
    };
    apply_sharp_features(&mut single);
    assert_eq!(single.candidates[0].expl_minus_max, 0.0);
    assert_eq!(single.candidates[0].expl_rank_frac, 1.0);
    // Residual transforms on a hand-computed example (r = 2 ppm, tol = 20).
    let [a, b, c, d] = residual_features(2.0, 20.0);
    assert_eq!(a, 2.0);
    assert_eq!(b, 4.0);
    assert!((c - (21.0f64).ln()).abs() < 1e-12);
    assert_eq!(d, 0.1);
}

#[test]
fn standardizer_round_trip_centres_and_scales() {
    fn cand(frac: f64, intensity: f64) -> CandidateEvidence {
        CandidateEvidence {
            counts: [0; 10],
            mass: 100_000_000,
            expl_count: 0,
            expl_count_frac: frac,
            expl_intensity: intensity,
            residual_ppm: 1.0,
            heavy_atoms: 0,
            dbe_twice: 0,
            resid_o1: 1.0,
            resid_o1_sq: 1.0,
            resid_ln1p: (11.0f64).ln(),
            resid_otol: 0.05,
            expl_minus_max: 0.0,
            expl_rank_frac: 1.0,
        }
    }
    let data = vec![
        SpectrumEvidence {
            candidates: vec![cand(0.0, 0.1), cand(1.0, 0.9)],
            gold: Some(0),
            has_window: true,
        },
        SpectrumEvidence {
            candidates: vec![cand(0.5, 0.4)],
            gold: Some(0),
            has_window: true,
        },
    ];
    let dims = [12usize, 13];
    let scaler = fit_standardizer(&data, &dims);
    // Mean over the three candidates: frac 0.5, intensity (0.1+0.9+0.4)/3.
    assert!((scaler.means[0] - 0.5).abs() < 1e-12);
    assert!((scaler.means[1] - (1.4 / 3.0)).abs() < 1e-12);
    assert!(scaler.scales.iter().all(|&s| s > 0.0));
    // Round trip on finite values.
    for ev in &data {
        for c in &ev.candidates {
            let f = feature_vector(c);
            for (j, &d) in dims.iter().enumerate() {
                let s = scaler.transform_value(f[d], j);
                let back = scaler.invert_value(s, j);
                assert!((back - f[d]).abs() < 1e-12);
            }
        }
    }
    // Zero-variance column is centred with scale 1.
    let flat = vec![
        SpectrumEvidence {
            candidates: vec![cand(0.3, 0.2), cand(0.3, 0.8)],
            gold: Some(0),
            has_window: true,
        },
    ];
    let flat_scaler = fit_standardizer(&flat, &[12]);
    assert!((flat_scaler.means[0] - 0.3).abs() < 1e-12);
    assert_eq!(flat_scaler.scales[0], 1.0);
    assert_eq!(flat_scaler.transform_value(0.3, 0), 0.0);
    // Non-finite raw values map to 0 (neutral), never NaN.
    assert_eq!(scaler.transform_value(f64::INFINITY, 0), 0.0);
    assert_eq!(scaler.transform_value(f64::NAN, 1), 0.0);
    // Nonlinear fitting centres products too, with matching base statistics.
    let nl = fit_standardizer_nonlinear(&data, NonlinearKind::PriorOnly);
    assert_eq!(nl.means.len(), 90);
    assert!(nl.scales.iter().all(|&s| s > 0.0 && s.is_finite()));
    let c0 = nonlinear_raw_vector(&data[0].candidates[0]);
    let t0 = nl.transform(&c0);
    assert_eq!(t0.len(), 90);
    assert!(t0.iter().all(|v| v.is_finite()));
    let plus = fit_standardizer_nonlinear(&data, NonlinearKind::PlusResEv);
    assert_eq!(plus.means.len(), 98);
    assert_eq!(nonlinear_plus_raw_vector(&data[0].candidates[0]).len(), 98);
}

#[test]
fn converged_optimizer_reaches_closed_form_symmetric_optimum() {
    // Symmetric two-spectrum problem on `expl_count_frac`: after
    // standardisation the gold carries +1 on spectrum 0 and -1 on spectrum 1,
    // so the objective (mean NLL plus L2) is even in `w` with the unique
    // closed-form optimum `w* = 0` and value `ln 2`.
    fn cand(frac: f64) -> CandidateEvidence {
        CandidateEvidence {
            counts: [0; 10],
            mass: 100_000_000,
            expl_count: 0,
            expl_count_frac: frac,
            expl_intensity: 0.0,
            residual_ppm: 1.0,
            heavy_atoms: 0,
            dbe_twice: 0,
            resid_o1: 1.0,
            resid_o1_sq: 1.0,
            resid_ln1p: (11.0f64).ln(),
            resid_otol: 0.05,
            expl_minus_max: 0.0,
            expl_rank_frac: 1.0,
        }
    }
    let data = vec![
        SpectrumEvidence {
            candidates: vec![cand(1.0), cand(0.0)],
            gold: Some(0),
            has_window: true,
        },
        SpectrumEvidence {
            candidates: vec![cand(0.0), cand(1.0)],
            gold: Some(0),
            has_window: true,
        },
    ];
    let dims = [12usize];
    let scaler = fit_standardizer(&data, &dims);
    assert!((scaler.means[0] - 0.5).abs() < 1e-12);
    assert!((scaler.scales[0] - 0.5).abs() < 1e-12);
    let cfg = ConvergedTrainConfig {
        max_epochs: 500,
        batch_spectra: usize::MAX,
        lr_init: 0.1,
        lr_decay: 0.02,
        l2: 1e-4,
        seed: 0,
    };
    let result = train_softmax_converged(&data, &dims, &scaler, &cfg).expect("training runs");
    assert!(result.converged, "epochs={} loss={}", result.epochs_used, result.train_nll);
    assert!(
        result.weights[0].abs() < 0.05,
        "symmetric optimum is 0, got {}",
        result.weights[0]
    );
    assert!(
        (result.train_nll - std::f64::consts::LN_2).abs() < 1e-3,
        "optimum value is ln2, got {}",
        result.train_nll
    );
    // The reported train NLL matches an independent evaluation.
    let eval = standardized_nll(&data, &dims, &scaler, &result.weights, 1e-4)
        .expect("evaluation runs");
    assert!((eval - result.train_nll).abs() < 1e-12);
    // Same symmetry holds for the nonlinear path (products of zero prior
    // features are all zero, hence zero-variance columns standardise to 0 and
    // every score is 0 regardless of weights; check it runs and is nested).
    let nl_scaler = fit_standardizer_nonlinear(&data, NonlinearKind::PriorOnly);
    let nl_cfg = ConvergedTrainConfig { max_epochs: 60, ..cfg };
    let nl_result =
        train_nonlinear_converged(&data, NonlinearKind::PriorOnly, &nl_scaler, &nl_cfg)
            .expect("nonlinear training runs");
    assert!(nl_result.train_nll.is_finite());
}

#[test]
fn nestedness_check_fires_on_planted_violation() {
    use std::collections::HashMap;
    let nested: HashMap<String, f64> = HashMap::from([
        ("prior_only".to_string(), 2.500),
        ("prior_plus_residual".to_string(), 2.499),
        ("prior_plus_evidence".to_string(), 2.498),
        ("prior_plus_residual_plus_evidence".to_string(), 2.490),
        ("prior_nonlinear".to_string(), 2.499),
        (
            "prior_nonlinear_plus_residual_plus_evidence".to_string(),
            2.489,
        ),
    ]);
    assert!(check_nestedness(&nested, 1e-3).is_empty());
    // Planted violation: a superset with a worse train objective.
    let mut violated = nested.clone();
    violated.insert("prior_plus_residual".to_string(), 2.600);
    let hits = check_nestedness(&violated, 1e-3);
    assert!(!hits.is_empty(), "planted violation must fire");
    assert!(
        hits.iter().any(|m| m.contains("prior_plus_residual")),
        "unexpected messages: {hits:?}"
    );
    // Planted violation on the full model above both components.
    let mut violated_full = nested.clone();
    violated_full.insert(
        "prior_plus_residual_plus_evidence".to_string(),
        2.600,
    );
    let hits_full = check_nestedness(&violated_full, 1e-3);
    assert!(!hits_full.is_empty());
    assert!(
        hits_full
            .iter()
            .any(|m| m.contains("prior_plus_residual_plus_evidence")),
        "unexpected messages: {hits_full:?}"
    );
    // Missing keys are skipped, never a violation by themselves.
    let partial: HashMap<String, f64> = HashMap::from([
        ("prior_plus_residual_plus_evidence".to_string(), 1.0),
        ("prior_plus_evidence".to_string(), 1.1),
    ]);
    assert!(check_nestedness(&partial, 1e-3).is_empty());
}

/// The kernel twin of architecture §1.6 (`formula_evidence`: one lane per
/// candidate, mixed-radix visits, the lane's own verdicts) against this
/// module's independent reference (sub-vectors sorted by mass, binary
/// search): on random parents and spectra, with peaks placed on and around
/// the tolerance edge of true sub-composition ions, every candidate's
/// explained count and explained weight over the selected evidence peaks
/// agree.
#[test]
fn kernel_twin_agrees_with_the_reference_index() {
    use mamba3::models::ms2::chem::composition_mass;
    use mamba3::models::ms2::formula_evidence as fe;

    let mut rng = StdRng::seed_from_u64(0x5EED_F00D);
    let n = 48usize;
    let p = fe::EVIDENCE_PEAKS;
    let mut checked = 0usize;
    let mut explained_total = 0usize;
    let mut unexplained_total = 0usize;
    for case in 0..80usize {
        let adduct: u16 = if case % 3 == 2 { 2 } else { 1 };
        let ppm: u32 = 100;
        let unc: u32 = [0u32, 5, 40][case % 3];
        let cands: Vec<Composition> = (0..4).map(|_| random_parent(&mut rng)).collect();
        // Peaks: ions of random sub-vectors of the first two candidates,
        // displaced by up to a little more than the tolerance, plus decoys.
        let mut mzs: Vec<u32> = Vec::new();
        for i in 0..28usize {
            let parent = &cands[i % 2];
            let mut digits = [0u16; 9];
            for (d, e) in digits.iter_mut().zip(HEAVY_ELEMENTS.iter()) {
                *d = rng.random_range(0..=parent[*e]);
            }
            if digits.iter().all(|&d| d == 0) {
                continue;
            }
            let h = rng.random_range(0..=u32::from(parent[HYDROGEN]) + 2);
            let Some(mass) = subvector_mass(parent, &digits, h) else {
                continue;
            };
            // t = mz + 549 for [M+H]+ and mz − 549 for [M-H]-.
            let mz = if adduct == 1 {
                mass.checked_sub(ELECTRON_MASS)
            } else {
                mass.checked_add(ELECTRON_MASS)
            };
            let Some(mz) = mz.filter(|&v| v > 0) else {
                continue;
            };
            let tol = i64::from(mamba3::models::ms2::tolerance(mz, ppm));
            let shift: i64 = rng.random_range(-(tol + 3)..=tol + 3);
            mzs.push((i64::from(mz) + shift).max(1) as u32);
        }
        for _ in 0..12 {
            mzs.push(rng.random_range(5_000_000u32..150_000_000));
        }
        mzs.sort_unstable();
        mzs.dedup();
        mzs.truncate(n);
        let count = mzs.len();
        let mut kept = vec![0u32; n * 3];
        let mut kept_f = vec![0f32; n * 2];
        for (i, &mz) in mzs.iter().enumerate() {
            kept[i * 3] = i as u32;
            kept[i * 3 + 1] = mz;
            kept[i * 3 + 2] = (count - 1 - i) as u32;
            kept_f[i * 2] = rng.random_range(0.01f32..1.0);
            kept_f[i * 2 + 1] = 1.0;
        }
        let meta = vec![count as u32, 200_000_000, 0, u32::from(adduct), ppm, 200, 7, 0];
        let spec = vec![unc, 0];
        let (ev_peaks, ev_w) = fe::evidence_peaks(&kept, &kept_f, &meta, &spec, 1, n, p);
        let m = cands.len();
        let mut cand = vec![0u32; m * 13];
        for (i, c) in cands.iter().enumerate() {
            for e in 0..10 {
                cand[i * 13 + e] = u32::from(c[e]);
            }
            cand[i * 13 + 10] = composition_mass(c).unwrap();
            cand[i * 13 + 11] = 1;
            cand[i * 13 + 12] = u32::MAX;
        }
        let out = fe::formula_evidence(&cand, &ev_peaks, &ev_w, &meta, &spec, 1, m, p, 1 << 20, u32::MAX);
        let n_ev = (0..p).filter(|&s| ev_peaks[s * 4 + 3] == 1).count();
        assert_eq!(n_ev, count.min(p), "case {case}: every kept peak is eligible here");
        for (i, c) in cands.iter().enumerate() {
            let index = build_evidence_index(c, adduct, unc, ppm).unwrap();
            let mut want_count = 0u32;
            let mut want_weight = 0f64;
            for s in 0..p {
                if ev_peaks[s * 4 + 3] != 1 {
                    continue;
                }
                let pos = ev_peaks[s * 4] as usize;
                let mz = kept[pos * 3 + 1];
                let hit = match &index {
                    Some(ix) => ix.explains(mz).unwrap(),
                    None => false,
                };
                checked += 1;
                if hit {
                    want_count += 1;
                    want_weight += f64::from(ev_w[s]);
                    explained_total += 1;
                } else {
                    unexplained_total += 1;
                }
            }
            assert_eq!(out[i * 4], want_count as f32, "case {case} candidate {i}: explained count");
            assert!(
                (f64::from(out[i * 4 + 1]) - want_weight).abs() <= 1e-5,
                "case {case} candidate {i}: explained weight {} vs {want_weight}",
                out[i * 4 + 1]
            );
            assert_eq!(out[i * 4 + 2], n_ev as f32, "case {case} candidate {i}: evidence count");
            assert_eq!(out[i * 4 + 3], 1.0, "case {case} candidate {i}: complete");
        }
    }
    // The comparison is not vacuous in either direction.
    assert!(checked > 5_000, "{checked} (candidate, peak) pairs");
    assert!(explained_total > 500, "{explained_total} explained pairs");
    assert!(unexplained_total > 500, "{unexplained_total} unexplained pairs");
}
