//! Types shared by every Mamba-CLM crate.

use serde::{Deserialize, Serialize};

/// Index of a trading day on the common calendar (CONTRACTS C1).
pub type DayIndex = usize;

/// One sample: a security and the decision day `t` (decision after the close of `t`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct SampleIndex {
    /// Security index into the panel's symbol list.
    pub s: u32,
    /// Decision day.
    pub t: u32,
}

impl SampleIndex {
    /// A sample for security `s` on day `t`.
    pub fn new(s: usize, t: DayIndex) -> Self {
        Self {
            s: s as u32,
            t: t as u32,
        }
    }

    /// The decision day as an index.
    pub fn day(&self) -> DayIndex {
        self.t as usize
    }

    /// The security as an index.
    pub fn security(&self) -> usize {
        self.s as usize
    }
}

/// Number of candidate actions (BUY, SELL, HOLD).
pub const NUM_ACTIONS: usize = 3;

/// A target exposure for the next horizon (CONTRACTS C3, C5). The discriminant is the
/// class id used by labels, logits and the backtest.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum Action {
    /// Be long over the horizon.
    Buy = 0,
    /// Be flat over the horizon.
    Sell = 1,
    /// The edge does not pay for a trade: keep the current position.
    Hold = 2,
}

impl Action {
    /// All actions in class-id order.
    pub const ALL: [Action; NUM_ACTIONS] = [Action::Buy, Action::Sell, Action::Hold];

    /// The class id.
    pub fn id(self) -> usize {
        self as usize
    }

    /// The action for a class id.
    pub fn from_id(id: usize) -> Option<Self> {
        Self::ALL.get(id).copied()
    }

    /// Short upper-case name.
    pub fn name(self) -> &'static str {
        match self {
            Action::Buy => "BUY",
            Action::Sell => "SELL",
            Action::Hold => "HOLD",
        }
    }
}
