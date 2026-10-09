//! Scratch arena unit tests (task T3): the per-thread scope-activated
//! recycler of device buffers behind [`with_scratch`].
//!
//! A buffer is recycled only after its tensor is dropped; a live tensor's
//! buffer is never handed out; sizes do not mix; retention is bounded;
//! nested scopes behave; stats account; and outside a scope
//! `allocation_calls` behaves as today.

#![cfg(feature = "backend")]

use mamba3::backend::{
    Device, ScratchArena, allocation_calls, reset_transfer_counters, with_scratch,
};
use mamba3::backends::Auto;
use mamba3::tensor::Tensor;
use mamba3::tensor::ops::elemwise::fill_;
use mamba3::tensor::ops::index::IdTensor;

type R = Auto;

/// Process-wide counters are perturbed by any test running beside these.
static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn serial() -> std::sync::MutexGuard<'static, ()> {
    SERIAL.lock().unwrap_or_else(|e| e.into_inner())
}

fn device() -> Device<R> {
    Device::<R>::default()
}

#[test]
fn buffer_recycled_only_after_drop() {
    let _serial = serial();
    let device = device();
    let arena = ScratchArena::new(1 << 20);
    reset_transfer_counters();
    with_scratch(&arena, || {
        let base = allocation_calls();
        let t = Tensor::<R, f32>::empty(vec![8], &device);
        assert_eq!(allocation_calls() - base, 1, "first buffer allocates");
        drop(t);
        let t2 = Tensor::<R, f32>::empty(vec![8], &device);
        assert_eq!(
            allocation_calls() - base,
            1,
            "the dropped tensor's buffer is reused without a new allocation"
        );
        let _ = t2;
    });
    let stats = arena.stats();
    assert_eq!(stats.served, 1, "one empty served from the arena");
    assert_eq!(stats.fell_through, 1, "one empty fell through");
    assert_eq!(stats.buffers, 1, "one buffer retained");
    assert_eq!(stats.bytes_held, 32, "8 f32 elements retained");
}

#[test]
fn live_tensor_buffer_never_handed_out() {
    let _serial = serial();
    let device = device();
    let arena = ScratchArena::new(1 << 20);
    with_scratch(&arena, || {
        let first = Tensor::<R, f32>::empty(vec![4], &device);
        mamba3::tensor::ops::elemwise::fill_(&first, 1.0);
        // Same size while the first tensor is alive: a fresh buffer, and
        // both contents stay correct.
        let second = Tensor::<R, f32>::empty(vec![4], &device);
        mamba3::tensor::ops::elemwise::fill_(&second, 2.0);
        assert_eq!(first.to_data(), vec![1.0; 4]);
        assert_eq!(second.to_data(), vec![2.0; 4]);
        drop(first);
        // After the drop the freed buffer serves again, still correctly.
        let third = Tensor::<R, f32>::empty(vec![4], &device);
        mamba3::tensor::ops::elemwise::fill_(&third, 3.0);
        assert_eq!(third.to_data(), vec![3.0; 4]);
        assert_eq!(second.to_data(), vec![2.0; 4]);
    });
    assert!(
        arena.stats().served > 0,
        "the freed buffer serves again: this test must observe recycling"
    );
}

#[test]
fn id_tensors_recycle_too() {
    let _serial = serial();
    let device = device();
    let arena = ScratchArena::new(1 << 20);
    reset_transfer_counters();
    with_scratch(&arena, || {
        let base = allocation_calls();
        let t = IdTensor::<R>::empty(vec![6], &device);
        assert_eq!(allocation_calls() - base, 1);
        drop(t);
        let t2 = IdTensor::<R>::empty(vec![6], &device);
        assert_eq!(allocation_calls() - base, 1, "id buffer reused after drop");
        // A live id tensor's buffer is never handed out either.
        let live = IdTensor::<R>::empty(vec![6], &device);
        assert_eq!(
            allocation_calls() - base,
            2,
            "live id buffer forces a fresh one"
        );
        let _ = (t2, live);
    });
}

#[test]
fn sizes_do_not_mix() {
    let _serial = serial();
    let device = device();
    let arena = ScratchArena::new(1 << 20);
    with_scratch(&arena, || {
        let a = Tensor::<R, f32>::empty(vec![4], &device);
        let b = Tensor::<R, f32>::empty(vec![8], &device);
        drop(a);
        drop(b);
        // Each size serves its own: both reuses hit, and a new size misses.
        let a2 = Tensor::<R, f32>::empty(vec![4], &device);
        let b2 = Tensor::<R, f32>::empty(vec![8], &device);
        let c = Tensor::<R, f32>::empty(vec![16], &device);
        let _ = (a2, b2, c);
    });
    let stats = arena.stats();
    assert_eq!(stats.buffers, 3, "one retained buffer per size");
    assert_eq!(stats.served, 2);
    assert_eq!(stats.fell_through, 3);
}

#[test]
fn retention_is_bounded_and_clearable() {
    let _serial = serial();
    let device = device();
    // Room for exactly one 32-byte buffer.
    let arena = ScratchArena::new(32);
    with_scratch(&arena, || {
        let a = Tensor::<R, f32>::empty(vec![8], &device);
        drop(a);
        // Retained: 32 bytes held.
        assert_eq!(arena.stats().bytes_held, 32);
        // A second size does not fit past the bound: plain allocation, and
        // the arena still holds only the first buffer.
        let b = Tensor::<R, f32>::empty(vec![16], &device);
        drop(b);
        let stats = arena.stats();
        assert_eq!(stats.buffers, 1, "beyond max_bytes nothing is retained");
        assert_eq!(stats.bytes_held, 32);
    });
    arena.clear();
    let stats = arena.stats();
    assert_eq!(stats.buffers, 0, "clear drops everything");
    assert_eq!(stats.bytes_held, 0);
}

#[test]
fn nested_scope_same_arena_is_noop_and_other_is_refused() {
    let _serial = serial();
    let device = device();
    let outer = ScratchArena::new(1 << 20);
    let inner = ScratchArena::new(1 << 20);
    reset_transfer_counters();
    with_scratch(&outer, || {
        let t = Tensor::<R, f32>::empty(vec![8], &device);
        drop(t);
        // Same arena nested: still the outer scope, reuse works.
        with_scratch(&outer, || {
            let base = allocation_calls();
            let _u = Tensor::<R, f32>::empty(vec![8], &device);
            assert_eq!(allocation_calls() - base, 0, "nested same arena reuses");
        });
        // A different arena nested: refused panic-free — the outer scope
        // stays active (the buffer comes from the outer arena, and the
        // inner arena retains nothing).
        with_scratch(&inner, || {
            let base = allocation_calls();
            let _u = Tensor::<R, f32>::empty(vec![8], &device);
            assert_eq!(
                allocation_calls() - base,
                0,
                "nested different arena still reuses the outer scope"
            );
        });
    });
    assert_eq!(
        inner.stats().buffers,
        0,
        "the refused arena retains nothing"
    );
    assert_eq!(inner.stats().served, 0);
    assert!(outer.stats().served >= 2, "outer served both nested reuses");
}

#[test]
fn outside_scope_allocations_behave_as_today() {
    let _serial = serial();
    let device = device();
    let arena = ScratchArena::new(1 << 20);
    // Populate the arena, leave the scope, then allocate: every empty
    // allocates exactly as before the arena existed.
    with_scratch(&arena, || {
        let t = Tensor::<R, f32>::empty(vec![8], &device);
        drop(t);
    });
    reset_transfer_counters();
    let base = allocation_calls();
    let t = Tensor::<R, f32>::empty(vec![8], &device);
    let u = Tensor::<R, f32>::empty(vec![8], &device);
    assert_eq!(allocation_calls() - base, 2, "no scope, no reuse");
    let _ = (t, u);
    assert_eq!(arena.stats().served, 0, "nothing served outside a scope");
}

#[test]
fn cross_thread_reuse_falls_through_to_fresh_buffers() {
    let _serial = serial();
    // T3F finding 1: CubeCL orders queued work per thread stream, so reuse
    // is bound to the allocating stream. Thread A allocates and drops (the
    // buffer is retained with A's stream); thread B activates the same
    // cloned arena, allocates the same size, queues a fill and drops
    // WITHOUT flushing; thread A allocates again, fills and reads back.
    // B's allocation must not receive A's buffer (else B's still-queued
    // write could overwrite A's live tensor), and A must read its own
    // value. Written against the generic runtime so it also runs on wgpu.
    let device = device();
    let arena = ScratchArena::new(1 << 20);
    with_scratch(&arena, || {
        let t = Tensor::<R, f32>::empty(vec![8], &device);
        drop(t);
    });
    let worker_arena = arena.clone();
    let worker_device = device.clone();
    let worker = std::thread::spawn(move || {
        with_scratch(&worker_arena, || {
            let t = Tensor::<R, f32>::empty(vec![8], &worker_device);
            fill_(&t, 2.0);
            // No flush or synchronisation: the write is still queued when
            // the tensor — and this scope — is dropped.
            drop(t);
        });
    });
    worker.join().expect("the worker thread runs");
    with_scratch(&arena, || {
        let t = Tensor::<R, f32>::empty(vec![8], &device);
        fill_(&t, 3.0);
        assert_eq!(
            t.to_data(),
            vec![3.0; 8],
            "A reads its own value: B's queued write went to another buffer"
        );
    });
    let stats = arena.stats();
    assert_eq!(
        stats.cross_stream_fallthroughs, 1,
        "B's allocation fell through: A's buffer lives on another stream"
    );
    assert_eq!(
        stats.served, 1,
        "only A's second allocation was served (from A's own buffer)"
    );
    assert_eq!(
        stats.fell_through, 2,
        "A's first and B's allocation each allocated"
    );
    assert_eq!(stats.buffers, 2, "both streams retained one buffer");
    assert_eq!(stats.bytes_held, 64, "two 8-f32 buffers retained");
}

#[test]
fn escaped_clones_and_views_block_reuse() {
    let _serial = serial();
    // A reshape and a clone of a scratch tensor kept alive across the next
    // `empty` of the same size: the live buffer is never handed out, and
    // every tensor reads its own values.
    let device = device();
    let arena = ScratchArena::new(1 << 20);
    reset_transfer_counters();
    with_scratch(&arena, || {
        let base = allocation_calls();
        let t = Tensor::<R, f32>::empty(vec![8], &device);
        fill_(&t, 1.0);
        let kept_clone = t.clone();
        let kept_view = t.reshape(vec![2, 4]).expect("reshape keeps 8 elements");
        let u = Tensor::<R, f32>::empty(vec![8], &device);
        fill_(&u, 2.0);
        assert_eq!(
            allocation_calls() - base,
            2,
            "the live buffer is never handed out"
        );
        assert_eq!(t.to_data(), vec![1.0; 8]);
        assert_eq!(kept_clone.to_data(), vec![1.0; 8]);
        assert_eq!(kept_view.to_data(), vec![1.0; 8]);
        assert_eq!(u.to_data(), vec![2.0; 8]);
        // Still held across one more same-size empty: still a fresh buffer.
        let v = Tensor::<R, f32>::empty(vec![8], &device);
        fill_(&v, 3.0);
        assert_eq!(allocation_calls() - base, 3);
        assert_eq!(v.to_data(), vec![3.0; 8]);
        assert_eq!(t.to_data(), vec![1.0; 8]);
        drop(kept_clone);
        drop(kept_view);
        drop(t);
        drop(u);
        // Everything dropped: the freed buffers serve again.
        let w = Tensor::<R, f32>::empty(vec![8], &device);
        assert_eq!(
            allocation_calls() - base,
            3,
            "the freed buffers serve again"
        );
        let _ = (v, w);
    });
}

#[test]
fn unwind_clears_the_scope() {
    let _serial = serial();
    // A panic inside `with_scratch` clears the activation: later
    // allocations behave as outside a scope, and the arena stays usable.
    let device = device();
    let arena = ScratchArena::new(1 << 20);
    let before_served = arena.stats().served;
    let out = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        with_scratch(&arena, || {
            let t = Tensor::<R, f32>::empty(vec![8], &device);
            fill_(&t, 1.0);
            assert_eq!(t.to_data(), vec![1.0; 8]);
            panic!("a decode step failed");
        });
    }));
    assert!(out.is_err(), "the panic propagates");
    reset_transfer_counters();
    let base = allocation_calls();
    let t = Tensor::<R, f32>::empty(vec![8], &device);
    let u = Tensor::<R, f32>::empty(vec![8], &device);
    assert_eq!(
        allocation_calls() - base,
        2,
        "no scope, no reuse after the unwind"
    );
    assert_eq!(
        arena.stats().served,
        before_served,
        "nothing served outside the scope"
    );
    let _ = (t, u);
    // The retained pre-panic buffer still serves inside a new scope.
    with_scratch(&arena, || {
        let base = allocation_calls();
        let _v = Tensor::<R, f32>::empty(vec![8], &device);
        assert_eq!(
            allocation_calls() - base,
            0,
            "the retained buffer serves in a new scope"
        );
    });
    assert_eq!(
        arena.stats().served,
        before_served + 1,
        "the new scope served one reuse"
    );
}

/// `with_scratch` around a fallible closure: the `?`-style early return
/// below runs the scope guard on the way out.
fn fallible_work(arena: &ScratchArena, device: &Device<R>, fail: bool) -> Result<usize, String> {
    with_scratch(arena, || {
        let t = Tensor::<R, f32>::empty(vec![8], device);
        fill_(&t, 1.0);
        if fail {
            return Err("stopped early".to_string());
        }
        Ok(t.to_data().len())
    })
}

#[test]
fn early_return_clears_the_scope() {
    let _serial = serial();
    let device = device();
    let arena = ScratchArena::new(1 << 20);
    assert_eq!(fallible_work(&arena, &device, false), Ok(8));
    assert!(fallible_work(&arena, &device, true).is_err());
    let served = arena.stats().served;
    // The early return cleared the scope: outside behaves as outside.
    reset_transfer_counters();
    let base = allocation_calls();
    let t = Tensor::<R, f32>::empty(vec![8], &device);
    let u = Tensor::<R, f32>::empty(vec![8], &device);
    assert_eq!(
        allocation_calls() - base,
        2,
        "no scope, no reuse after the early return"
    );
    assert_eq!(
        arena.stats().served,
        served,
        "nothing served outside the scope"
    );
    let _ = (t, u);
    // And a new scope reuses again.
    with_scratch(&arena, || {
        let base = allocation_calls();
        let _v = Tensor::<R, f32>::empty(vec![8], &device);
        assert_eq!(
            allocation_calls() - base,
            0,
            "the retained buffer serves in a new scope"
        );
    });
}

#[test]
fn second_device_identity_falls_through() {
    let _serial = serial();
    // Two device identities over the same runtime (two `Device::default`
    // calls get different identities by construction): a buffer retained
    // for the first is never handed to the second.
    let first = Device::<R>::default();
    let second = Device::<R>::default();
    let arena = ScratchArena::new(1 << 20);
    with_scratch(&arena, || {
        let t = Tensor::<R, f32>::empty(vec![8], &first);
        fill_(&t, 1.0);
        assert_eq!(t.to_data(), vec![1.0; 8]);
        drop(t);
    });
    with_scratch(&arena, || {
        let u = Tensor::<R, f32>::empty(vec![8], &second);
        fill_(&u, 2.0);
        assert_eq!(u.to_data(), vec![2.0; 8]);
    });
    let stats = arena.stats();
    assert_eq!(
        stats.served, 0,
        "nothing is served across device identities"
    );
    assert_eq!(stats.fell_through, 2, "both allocations allocated");
    assert_eq!(
        stats.cross_stream_fallthroughs, 0,
        "same thread, so this is the device rule, not the stream rule"
    );
}

#[test]
fn per_size_retention_is_capped_at_64() {
    let _serial = serial();
    // 80 live same-size tensors: only 64 buffers are retained.
    let device = device();
    let arena = ScratchArena::new(1 << 20);
    with_scratch(&arena, || {
        let mut held = Vec::new();
        for _ in 0..80 {
            held.push(Tensor::<R, f32>::empty(vec![8], &device));
        }
        let stats = arena.stats();
        assert_eq!(stats.buffers, 64, "at most 64 buffers of one size");
        assert_eq!(stats.bytes_held, 64 * 32, "64 eight-f32 buffers");
        drop(held);
    });
}
