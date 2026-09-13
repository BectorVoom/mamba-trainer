//! Errors, in both directions.
//!
//! Rust failures become the exception a Python caller would actually catch: a
//! configuration or shape complaint is a `ValueError`, a missing checkpoint an
//! `OSError`, an unimplemented path a `NotImplementedError`. Nothing is flattened
//! into a bare `RuntimeError`, because `except ValueError` around a call that
//! builds a model is the code people write.
//!
//! The other direction is the harder one, and it is [`ErrorSlot`]. An environment
//! written in Python is called from inside [`mamba3::rl::Collector`], which speaks
//! `mamba3::error::Result` and has no room for a `PyErr` — so a raised exception
//! would be reduced to its `to_string()` and its traceback lost, at exactly the
//! point where a user's own code broke. The slot parks the original exception on
//! the way out and [`ErrorSlot::resolve`] re-raises it once the stack is back in
//! Python, so a bug in someone's `step()` is reported as a bug in their `step()`.

use std::cell::RefCell;

use mamba3::error::Error;
use pyo3::exceptions::{PyIOError, PyNotImplementedError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;

/// The exception type that matches a crate error.
pub fn to_py(err: Error) -> PyErr {
    let message = err.to_string();
    match err {
        Error::Config(_) | Error::Shape(_) | Error::Rank { .. } | Error::Json(_) => {
            PyValueError::new_err(message)
        }
        Error::Io(_) => PyIOError::new_err(message),
        Error::Unsupported(_) => PyNotImplementedError::new_err(message),
        Error::Autodiff(_) | Error::StateDict(_) | Error::Backend(_) => {
            PyRuntimeError::new_err(message)
        }
    }
}

/// `Result<T, mamba3::Error>` with a `?`-able Python tail.
pub trait IntoPyResult<T> {
    /// Convert into a [`PyResult`], mapping the error with [`to_py`].
    fn py(self) -> PyResult<T>;
}

impl<T> IntoPyResult<T> for Result<T, Error> {
    fn py(self) -> PyResult<T> {
        self.map_err(to_py)
    }
}

/// Where a callback implemented in Python parks the exception it raised.
///
/// One slot serves one call into Rust. It is filled at most once — the first
/// failure is the one that explains the rest — and emptied by [`Self::resolve`].
#[derive(Default)]
pub struct ErrorSlot {
    pending: RefCell<Option<PyErr>>,
}

impl ErrorSlot {
    /// Park `err` and describe it as a crate error, for the code that cannot hold
    /// a `PyErr`. The description is only ever seen if the exception is somehow
    /// lost on the way back, which [`Self::resolve`] exists to prevent.
    pub fn store(&self, context: &str, err: PyErr) -> Error {
        let message = format!("the environment's {context} raised {err}");
        let mut slot = self.pending.borrow_mut();
        if slot.is_none() {
            *slot = Some(err);
        }
        Error::config(message)
    }

    /// Re-raise whatever was parked, in preference to any error derived from it.
    pub fn resolve<T>(&self, result: Result<T, Error>) -> PyResult<T> {
        match self.pending.borrow_mut().take() {
            Some(err) => Err(err),
            None => result.py(),
        }
    }
}
