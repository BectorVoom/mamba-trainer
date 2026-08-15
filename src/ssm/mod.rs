//! The Mamba-3 state space model: configuration, the chunked parallel scan, and
//! the single-token recurrence used at decode time.
//!
//! See [`scan`] for the derivation that turns the exponential-trapezoidal
//! recurrence into the same semiseparable-matrix form Mamba-2 uses, which is what
//! lets a second-order rule run at first-order cost.

pub mod config;
pub mod scan;

pub use config::{Discretization, SsmConfig, SsmConfigBuilder, SsmMode, StateDynamics};
pub use scan::{
    ScanInputs, ScanOutput, SsmState, mamba3_scan, mamba3_step, shift_left, ssd_chunked,
};
