//! Whether operations are recorded on a tape.
//!
//! Tracking is on by default and scoped off with [`no_grad`]. Two things depend on
//! the flag:
//!
//! * [`Param::var`](crate::nn::Param::var) will start a *fresh* tape when the value
//!   it is anchored to has none. Without that, freezing the first few layers of a
//!   model (exactly what LoRA does) would leave later trainable parameters with no
//!   tape to attach to, and backward would find nothing.
//! * [`Var::record`](crate::autograd::Var::record) short-circuits entirely, so an
//!   inference pass allocates no graph even if its inputs happen to be tracked.

use std::cell::Cell;

thread_local! {
    static ENABLED: Cell<bool> = const { Cell::new(true) };
}

/// Whether new operations are being recorded on this thread.
pub fn is_enabled() -> bool {
    ENABLED.with(|flag| flag.get())
}

/// Restores the previous tracking mode when dropped.
pub struct NoGradGuard {
    previous: bool,
}

impl Drop for NoGradGuard {
    fn drop(&mut self) {
        ENABLED.with(|flag| flag.set(self.previous));
    }
}

/// Disable gradient tracking until the returned guard is dropped.
///
/// ```
/// # use mamba3::autograd::no_grad;
/// {
///     let _guard = no_grad();
///     // nothing in here is recorded
/// }
/// ```
pub fn no_grad() -> NoGradGuard {
    let previous = ENABLED.with(|flag| flag.replace(false));
    NoGradGuard { previous }
}

/// Re-enable gradient tracking until the returned guard is dropped.
pub fn enable_grad() -> NoGradGuard {
    let previous = ENABLED.with(|flag| flag.replace(true));
    NoGradGuard { previous }
}

/// Run a closure with gradient tracking disabled.
pub fn with_no_grad<T>(f: impl FnOnce() -> T) -> T {
    let _guard = no_grad();
    f()
}
