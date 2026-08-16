//! The chunked scan is checked against a direct, unrolled evaluation of the
//! Mamba-3 recurrence. This is the test that matters most: everything else in the
//! crate is standard, but the reformulation in `ssm::scan` is where a mistake
//! would be silent.

#![cfg(feature = "backend")]

use mamba3::autograd::Var;
use mamba3::backend::Device;
use mamba3::backends::Auto;
use mamba3::ssm::scan::{ScanInputs, SsmState, mamba3_scan, mamba3_step, shift_left, ssd_chunked};
use mamba3::tensor::Tensor;

type R = Auto;
type V = Var<R, f32>;

fn dev() -> Device<R> {
    Device::<R>::default()
}

fn t(data: &[f32], shape: Vec<usize>) -> V {
    V::constant(Tensor::from_f32(data, shape, &dev()).unwrap())
}

/// Deterministic pseudo-random values in `[-1, 1)`.
fn noise(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 33) as f32 / (1u64 << 31) as f32) - 1.0
        })
        .collect()
}

fn assert_close(actual: &[f32], expected: &[f32], eps: f32, what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}: length mismatch");
    let mut worst = 0.0f32;
    let mut at = 0;
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        let err = (a - e).abs() / (1.0 + e.abs());
        if err > worst {
            worst = err;
            at = i;
        }
    }
    assert!(
        worst <= eps,
        "{what}: worst relative error {worst} at index {at} (got {}, want {})",
        actual[at],
        expected[at]
    );
}

/// Direct evaluation of
/// `h_t = a_t h_{t-1} + b_t B_{t-1} x_{t-1} + g_t B_t x_t`, `y_t = C_t^T h_t`.
#[allow(clippy::too_many_arguments)]
fn reference(
    batch: usize,
    seq: usize,
    heads: usize,
    head_dim: usize,
    state: usize,
    x: &[f32],
    b: &[f32],
    c: &[f32],
    dt: &[f32],
    lambda: &[f32],
    a_head: &[f32],
) -> (Vec<f32>, Vec<f32>) {
    let mut h = vec![0.0f32; batch * heads * head_dim * state];
    let mut prev_u = vec![0.0f32; batch * heads * head_dim * state];
    let mut y = vec![0.0f32; batch * seq * heads * head_dim];

    for tt in 0..seq {
        for bb in 0..batch {
            for hh in 0..heads {
                let idx = (bb * seq + tt) * heads + hh;
                let step = dt[idx];
                let lam = lambda[idx];
                let alpha = (step * a_head[hh]).exp();
                let beta = (1.0 - lam) * step * alpha;
                let gamma = lam * step;

                let x_off = ((bb * seq + tt) * heads + hh) * head_dim;
                let bc_off = ((bb * seq + tt) * heads + hh) * state;
                let h_off = (bb * heads + hh) * head_dim * state;

                for p in 0..head_dim {
                    let xv = x[x_off + p];
                    let mut acc = 0.0f32;
                    for n in 0..state {
                        let u = xv * b[bc_off + n];
                        let slot = h_off + p * state + n;
                        let new = alpha * h[slot] + beta * prev_u[slot] + gamma * u;
                        h[slot] = new;
                        prev_u[slot] = u;
                        acc += c[bc_off + n] * new;
                    }
                    y[x_off + p] = acc;
                }
            }
        }
    }
    (y, h)
}

struct Case {
    batch: usize,
    seq: usize,
    heads: usize,
    head_dim: usize,
    state: usize,
    x: Vec<f32>,
    b: Vec<f32>,
    c: Vec<f32>,
    dt: Vec<f32>,
    lambda: Vec<f32>,
    a_head: Vec<f32>,
}

impl Case {
    fn new(batch: usize, seq: usize, heads: usize, head_dim: usize, state: usize) -> Self {
        let bt = batch * seq * heads;
        // dt in (0, 0.6], lambda in (0, 1).
        let dt: Vec<f32> = noise(bt, 7).iter().map(|v| 0.05 + 0.25 * (v + 1.0)).collect();
        let lambda: Vec<f32> = noise(bt, 11).iter().map(|v| 0.5 * (v + 1.0) * 0.98 + 0.01).collect();
        let a_head: Vec<f32> = (0..heads).map(|i| -(1.0 + i as f32 * 0.7)).collect();
        Self {
            batch,
            seq,
            heads,
            head_dim,
            state,
            x: noise(batch * seq * heads * head_dim, 1),
            b: noise(batch * seq * heads * state, 2),
            c: noise(batch * seq * heads * state, 3),
            dt,
            lambda,
            a_head,
        }
    }

    fn reference(&self) -> (Vec<f32>, Vec<f32>) {
        reference(
            self.batch,
            self.seq,
            self.heads,
            self.head_dim,
            self.state,
            &self.x,
            &self.b,
            &self.c,
            &self.dt,
            &self.lambda,
            &self.a_head,
        )
    }

    fn vars(&self) -> (V, V, V, V, V, V) {
        let bt = vec![self.batch, self.seq, self.heads];
        (
            t(&self.x, vec![self.batch, self.seq, self.heads, self.head_dim]),
            t(&self.b, vec![self.batch, self.seq, self.heads, self.state]),
            t(&self.c, vec![self.batch, self.seq, self.heads, self.state]),
            t(&self.dt, bt.clone()),
            t(&self.lambda, bt),
            t(
                &self.a_head.iter().map(|v| (-v).ln()).collect::<Vec<_>>(),
                vec![self.heads],
            ),
        )
    }

    /// `(a, g, w)` as `ssd_chunked` expects them.
    fn weights(&self) -> (V, V, V) {
        let (_, _, _, dt, lambda, _) = self.vars();
        let a_head = t(&self.a_head, vec![1, 1, self.heads]);
        let a = dt.mul(&a_head).unwrap();
        let g = lambda.mul(&dt).unwrap();
        let next = lambda.rsub_scalar(1.0).mul(&dt).unwrap();
        let w = g.add(&shift_left(&next, 1).unwrap()).unwrap();
        (a, g, w)
    }
}

#[test]
fn chunked_scan_matches_the_recurrence() {
    for &chunk in &[1usize, 2, 3, 4, 8, 16] {
        let case = Case::new(2, 12, 3, 4, 6);
        let (x, b, c, _, _, _) = case.vars();
        let (a, g, w) = case.weights();
        let (y, state) = ssd_chunked(&x, &b, &c, &a, &g, &w, None, chunk, true).unwrap();
        let state = state.unwrap();

        let (want_y, want_h) = case.reference();
        assert_close(&y.to_f32(), &want_y, 2e-4, &format!("y (chunk={chunk})"));
        assert_close(
            &state.to_f32(),
            &want_h,
            2e-4,
            &format!("final state (chunk={chunk})"),
        );
    }
}

#[test]
fn chunked_scan_handles_ragged_lengths() {
    // Sequence lengths that are not multiples of the chunk exercise the padding path.
    for seq in [1usize, 5, 7, 13] {
        let case = Case::new(1, seq, 2, 3, 4);
        let (x, b, c, _, _, _) = case.vars();
        let (a, g, w) = case.weights();
        let (y, state) = ssd_chunked(&x, &b, &c, &a, &g, &w, None, 4, true).unwrap();
        let state = state.unwrap();
        let (want_y, want_h) = case.reference();
        assert_close(&y.to_f32(), &want_y, 2e-4, &format!("y (seq={seq})"));
        assert_close(&state.to_f32(), &want_h, 2e-4, &format!("state (seq={seq})"));
    }
}

#[test]
fn euler_mode_reproduces_mamba2() {
    // With lambda = 1 the trapezoidal weights collapse to `w = g = dt`, which is
    // exactly the Mamba-2 recurrence.
    let mut case = Case::new(1, 8, 2, 3, 4);
    case.lambda = vec![1.0; case.batch * case.seq * case.heads];
    let (x, b, c, _, _, _) = case.vars();
    let (a, g, w) = case.weights();
    assert_close(&w.to_f32(), &g.to_f32(), 1e-6, "euler weights");

    let (y, state) = ssd_chunked(&x, &b, &c, &a, &g, &w, None, 4, false).unwrap();
    assert!(state.is_none(), "want_state=false must not compute a state");
    let (want_y, _) = case.reference();
    assert_close(&y.to_f32(), &want_y, 2e-4, "euler y");
}

#[test]
fn split_windows_match_a_single_pass() {
    let case = Case::new(2, 12, 2, 3, 4);
    let (x, b, c, dt, lambda, a_log) = case.vars();
    let theta = t(
        &noise(case.batch * case.seq * case.heads * case.state / 2, 5),
        vec![case.batch, case.seq, case.heads, case.state / 2],
    );

    let expand = |v: &V, last: usize| {
        v.reshape(vec![case.batch, case.seq, case.heads, last, 1])
            .unwrap()
    };

    let (xf, bf, cf) = (
        expand(&x, case.head_dim),
        expand(&b, case.state),
        expand(&c, case.state),
    );

    let full = mamba3_scan(
        ScanInputs::new(&xf, &bf, &cf, &dt, &a_log, &lambda)
            .with_theta(&theta)
            .with_chunk_size(4),
    )
    .unwrap();

    // Same computation, cut in two, carrying state across the seam.
    let cut = 5;
    let rest = case.seq - cut;
    let take = |v: &V, start: usize, len: usize| v.slice(1, start, len).unwrap();

    let (x1, b1, c1, dt1, l1, th1) = (
        take(&xf, 0, cut),
        take(&bf, 0, cut),
        take(&cf, 0, cut),
        take(&dt, 0, cut),
        take(&lambda, 0, cut),
        take(&theta, 0, cut),
    );
    let first = mamba3_scan(
        ScanInputs::new(&x1, &b1, &c1, &dt1, &a_log, &l1)
            .with_theta(&th1)
            .with_chunk_size(4),
    )
    .unwrap();
    let first_state = first.state.as_ref().unwrap();

    let (x2, b2, c2, dt2, l2, th2) = (
        take(&xf, cut, rest),
        take(&bf, cut, rest),
        take(&cf, cut, rest),
        take(&dt, cut, rest),
        take(&lambda, cut, rest),
        take(&theta, cut, rest),
    );
    let second = mamba3_scan(
        ScanInputs::new(&x2, &b2, &c2, &dt2, &a_log, &l2)
            .with_theta(&th2)
            .with_chunk_size(4)
            .with_state(first_state),
    )
    .unwrap();

    let joined = mamba3::autograd::cat(&[first.y.clone(), second.y.clone()], 1).unwrap();
    assert_close(&joined.to_f32(), &full.y.to_f32(), 3e-4, "split vs full");
    assert_close(
        &second.state.unwrap().h.to_f32(),
        &full.state.unwrap().h.to_f32(),
        3e-4,
        "split final state",
    );
}

#[test]
fn step_matches_the_chunked_scan() {
    let case = Case::new(2, 9, 2, 3, 4);
    let (x, b, c, dt, lambda, a_log) = case.vars();
    let theta = t(
        &noise(case.batch * case.seq * case.heads * case.state / 2, 5),
        vec![case.batch, case.seq, case.heads, case.state / 2],
    );
    let d_skip = t(&[0.3, -0.2], vec![case.heads]);

    let expand = |v: &V, last: usize| {
        v.reshape(vec![case.batch, case.seq, case.heads, last, 1])
            .unwrap()
    };

    let (xf, bf, cf) = (
        expand(&x, case.head_dim),
        expand(&b, case.state),
        expand(&c, case.state),
    );
    let parallel = mamba3_scan(
        ScanInputs::new(&xf, &bf, &cf, &dt, &a_log, &lambda)
            .with_theta(&theta)
            .with_skip(&d_skip)
            .with_chunk_size(4),
    )
    .unwrap();

    let mut state = SsmState::zeros(
        case.batch,
        case.heads,
        case.head_dim,
        case.state,
        true,
        &dev(),
    );
    let mut steps = Vec::new();
    for step in 0..case.seq {
        let pick = |v: &V, trailing: Vec<usize>| {
            let mut dims = vec![case.batch, case.heads];
            dims.extend(trailing);
            v.slice(1, step, 1).unwrap().reshape(dims).unwrap()
        };
        let (y, next) = mamba3_step(
            &pick(&x, vec![case.head_dim, 1]),
            &pick(&b, vec![case.state, 1]),
            &pick(&c, vec![case.state, 1]),
            &pick(&dt, vec![]),
            &pick(&lambda, vec![]),
            &a_log,
            Some(&pick(&theta, vec![case.state / 2])),
            Some(&d_skip),
            &state,
        )
        .unwrap();
        state = next;
        steps.push(
            y.reshape(vec![case.batch, 1, case.heads, case.head_dim, 1])
                .unwrap(),
        );
    }
    let stepwise = mamba3::autograd::cat(&steps, 1).unwrap();
    assert_close(
        &stepwise.to_f32(),
        &parallel.y.to_f32(),
        3e-4,
        "step vs chunked",
    );
}

#[test]
fn scan_is_differentiable() {
    let case = Case::new(1, 6, 2, 3, 4);
    let (x, b, c, dt, lambda, a_log) = case.vars();
    let x = V::traced(x.tensor().clone());

    let (a, g, w) = case.weights();
    let (y, _) = ssd_chunked(
        &x.reshape(vec![case.batch, case.seq, case.heads, case.head_dim])
            .unwrap(),
        &b,
        &c,
        &a,
        &g,
        &w,
        None,
        3,
        false,
    )
    .unwrap();
    let loss = y.mul(&y).unwrap().sum().unwrap();
    let grads = loss.backward_retain().unwrap();
    let analytic = grads.node(x.node().unwrap()).unwrap().to_f32();

    // Central differences on a handful of coordinates.
    let eps = 1e-3;
    for &i in &[0usize, 5, 17, 30] {
        let run = |data: &[f32]| {
            let xd = t(data, vec![case.batch, case.seq, case.heads, case.head_dim]);
            let (yy, _) = ssd_chunked(&xd, &b, &c, &a, &g, &w, None, 3, false).unwrap();
            yy.mul(&yy).unwrap().sum().unwrap().to_f32()[0]
        };
        let mut plus = case.x.clone();
        let mut minus = case.x.clone();
        plus[i] += eps;
        minus[i] -= eps;
        let numeric = (run(&plus) - run(&minus)) / (2.0 * eps);
        assert!(
            (analytic[i] - numeric).abs() < 2e-2 * (1.0 + numeric.abs()),
            "grad[{i}]: analytic={} numeric={numeric}",
            analytic[i]
        );
    }
    let _ = (dt, lambda, a_log);
}
