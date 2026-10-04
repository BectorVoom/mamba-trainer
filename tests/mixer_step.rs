//! K9: the fused incremental mixer step computes what the composed one does.
//!
//! `Mamba3Mixer::step_masked` takes a three-launch path between its projections
//! when nothing is being recorded. The composed step — which the scan tests
//! already tie to the windowed forward pass — is the reference: same mixer,
//! same inputs, same reset flags, every output and every piece of carried
//! state compared over several steps so an error in the state shows up even
//! if a single step's output hides it.
#![cfg(feature = "backend")]

use mamba3::autograd::{Var, no_grad};
use mamba3::backend::{Device, check_launches, launch_count, reset_launch_count};
use mamba3::backends::Auto;
use mamba3::models::mamba3::set_fused_step;
use mamba3::models::{Mamba3Mixer, Mamba3MixerConfig, MixerCache};
use mamba3::nn::Module;
use mamba3::ssm::config::{Discretization, SsmConfig, StateDynamics};
use mamba3::tensor::Tensor;
use mamba3::tensor::ops::random::Rng;

type R = Auto;

const BATCH: usize = 3;
const STEPS: usize = 6;

fn close(label: &str, got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "{label}: length");
    for (i, (a, b)) in got.iter().zip(want).enumerate() {
        assert!(
            (a - b).abs() <= 2e-4 * (1.0 + b.abs()),
            "{label}[{i}]: fused {a} vs composed {b}"
        );
    }
}

fn compare_cache(label: &str, fused: &MixerCache<R, f32>, composed: &MixerCache<R, f32>) {
    close(&format!("{label} h"), &fused.ssm.h.to_f32(), &composed.ssm.h.to_f32());
    close(
        &format!("{label} last_u"),
        &fused.ssm.last_u.to_f32(),
        &composed.ssm.last_u.to_f32(),
    );
    assert_eq!(fused.ssm.h.dims(), composed.ssm.h.dims(), "{label}: h shape");
    match (&fused.ssm.angle, &composed.ssm.angle) {
        (Some(a), Some(b)) => {
            assert_eq!(a.dims(), b.dims(), "{label}: angle shape");
            // Angles are only defined modulo a turn, and a value on the wrap
            // boundary may round either way.
            for (i, (x, y)) in a.to_f32().iter().zip(b.to_f32()).enumerate() {
                let d = (x - y).rem_euclid(2.0 * std::f32::consts::PI);
                assert!(
                    d.min(2.0 * std::f32::consts::PI - d) < 1e-4,
                    "{label} angle[{i}]: fused {x} vs composed {y}"
                );
            }
        }
        (None, None) => {}
        _ => panic!("{label}: only one path carried an angle"),
    }
    match (&fused.conv, &composed.conv) {
        (Some(a), Some(b)) => {
            assert_eq!(a.dims(), b.dims(), "{label}: history shape");
            close(&format!("{label} history"), &a.to_f32(), &b.to_f32());
        }
        (None, None) => {}
        _ => panic!("{label}: only one path carried a convolution history"),
    }
}

/// Run `STEPS` steps both ways and compare everything; returns the launches of
/// one step, `(fused, composed)`.
fn check(label: &str, ssm: SsmConfig, with_reset: bool) -> (usize, usize) {
    let device = Device::<R>::default();
    let mut rng = Rng::seeded(7);
    let d_model = ssm.d_model;
    let mixer: Mamba3Mixer<R, f32> = Mamba3MixerConfig::new(ssm).init(&device, &mut rng).unwrap();
    // Initialisation leaves several parameters at zero or one; move every one
    // of them so a misindexed bias or gain cannot pass as a no-op.
    for (k, (_, param)) in mixer.named_parameters().into_iter().enumerate() {
        let value = param.value();
        let shifted: Vec<f32> = value
            .to_f32()
            .iter()
            .enumerate()
            .map(|(i, v)| v + 0.3 * ((i * 7 + k * 13) as f32 * 0.37).sin())
            .collect();
        param.set(Tensor::from_f32(&shifted, value.shape().clone(), &device).unwrap());
    }

    let _guard = no_grad();
    let mut fused = mixer.empty_cache(BATCH, &device);
    let mut composed = mixer.empty_cache(BATCH, &device);
    let mut launches = (0, 0);
    for step in 0..STEPS {
        let data: Vec<f32> = (0..BATCH * d_model)
            .map(|i| ((i + step * 31) as f32 * 0.21).sin() * 1.5)
            .collect();
        let input = Var::constant(Tensor::from_f32(&data, vec![BATCH, 1, d_model], &device).unwrap());
        // Row 1 is reset on step 3, every row on step 4, none otherwise.
        let flags: Vec<f32> = (0..BATCH)
            .map(|b| ((step == 3 && b == 1) || step == 4) as u8 as f32)
            .collect();
        let reset = with_reset.then(|| Tensor::from_f32(&flags, vec![BATCH], &device).unwrap());

        set_fused_step(true);
        reset_launch_count();
        let (out_f, next_f) = mixer.step_masked(&input, &fused, reset.as_ref()).unwrap();
        launches.0 = launch_count();
        set_fused_step(false);
        reset_launch_count();
        let (out_c, next_c) = mixer.step_masked(&input, &composed, reset.as_ref()).unwrap();
        launches.1 = launch_count();
        set_fused_step(true);
        check_launches(&device).unwrap();

        let at = format!("{label} step {step}");
        assert_eq!(out_f.dims(), out_c.dims(), "{at}: output shape");
        close(&format!("{at} out"), &out_f.to_f32(), &out_c.to_f32());
        compare_cache(&at, &next_f, &next_c);
        fused = next_f;
        composed = next_c;
    }
    launches
}

fn base() -> SsmConfig {
    SsmConfig {
        d_model: 24,
        n_heads: 4,
        n_groups: 4,
        head_dim: 6,
        d_state: 8,
        chunk_size: 8,
        ..SsmConfig::default()
    }
}

// One test, not several: the fused/composed toggle is process-wide.
#[test]
fn fused_step_matches_composed() {
    // The default layer: rotational, learned trapezoid, convolution, norm, bias.
    let (fused, composed) = check("default", base(), true);
    if fused < composed {
        assert_eq!(fused, 5, "fused step: two projections and three kernels");
    } else {
        // A device whose bindings cannot hold the kernels keeps the composed path.
        assert_eq!(fused, composed);
    }
    check("default, no reset", base(), false);

    // A projection whose row and bands a full-width vector divides, so the
    // activation kernel runs vectorised.
    check(
        "wide",
        SsmConfig {
            d_model: 32,
            head_dim: 8,
            ..base()
        },
        true,
    );
    // Heads sharing B/C groups, and a state no wide vector divides.
    check(
        "grouped",
        SsmConfig {
            n_groups: 2,
            d_state: 6,
            ..base()
        },
        true,
    );
    // A single group across every head.
    check(
        "one group",
        SsmConfig {
            n_groups: 1,
            d_state: 16,
            ..base()
        },
        true,
    );
    // Everything optional switched off.
    check(
        "bare",
        SsmConfig {
            dynamics: StateDynamics::Real,
            discretization: Discretization::Euler,
            conv_kernel: None,
            bc_norm: false,
            bc_bias: false,
            skip_connection: false,
            ..base()
        },
        true,
    );
    // The shortest convolution, a fixed trapezoid, bias without the norm.
    check(
        "short conv",
        SsmConfig {
            discretization: Discretization::Trapezoid,
            conv_kernel: Some(2),
            bc_norm: false,
            d_state: 5,
            dynamics: StateDynamics::Real,
            ..base()
        },
        true,
    );
    // Norm without bias, a long convolution, with its bias.
    check(
        "long conv",
        SsmConfig {
            conv_kernel: Some(5),
            bc_bias: false,
            bias: true,
            ..base()
        },
        true,
    );
}

/// With the tape on, the step stays on the composed path and stays differentiable.
#[test]
fn recorded_step_keeps_its_gradient() {
    let device = Device::<R>::default();
    let mut rng = Rng::seeded(3);
    let mixer: Mamba3Mixer<R, f32> = Mamba3MixerConfig::new(base()).init(&device, &mut rng).unwrap();
    let cache = mixer.empty_cache(BATCH, &device);
    let data: Vec<f32> = (0..BATCH * 24).map(|i| (i as f32 * 0.3).cos()).collect();
    let input = Var::traced(Tensor::from_f32(&data, vec![BATCH, 1, 24], &device).unwrap());
    let (out, _) = mixer.step(&input, &cache).unwrap();
    assert!(out.is_tracked());
    let grads = out.sum().unwrap().backward().unwrap();
    let (name, param) = &mixer.named_parameters()[0];
    assert!(grads.get(param.id()).is_some(), "{name} got no gradient");
}
