//! Dead-end cause diagnostics under the exact-completion grammar (host only).
//!
//! [`feasibility`] checks four necessary conditions for completability of a
//! [`TraceState`](super::grammar::TraceState) prefix under
//! [`TraceState::new_exact`](super::grammar::TraceState::new_exact):
//! a failed check proves that no continuation completes the molecule.
//! [`first_doomed_step`] replays a trace and reports the first prefix that
//! fails one of them; [`classify_dead_end`] turns a dead-end trajectory into
//! a [`DeadEndCause`]. This module adds no grammar rule.
//!
//! The checks themselves live on [`TraceState::feasibility`](super::grammar::TraceState::feasibility)
//! (the v2 legality rule); the replays here apply tokens under the v1 rules
//! ([`TraceState::apply_v1`](super::grammar::TraceState::apply_v1)), so
//! prefixes the v2 rule forbids can still be built and analysed.

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

use super::chem::{Composition, HYDROGEN};
use super::grammar::{Limits, Token, TraceState};
use super::graph::MolGraph;

/// The four necessary conditions of [`feasibility`], defined in
/// [`super::grammar`]: `true` means the check passes (the prefix may still
/// be doomed for another reason). Re-exported so existing users of this
/// module keep their import path.
pub use super::grammar::Feasibility;

/// Why a prefix under the exact-completion grammar can no longer reach a
/// complete molecule: the first failing necessary condition in the fixed
/// priority [`DoomReason::HydrogenBound`], [`DoomReason::NoOpenSite`],
/// [`DoomReason::OpenValenceWithoutAtoms`], [`DoomReason::ValenceBound`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DoomReason {
    /// The remaining hydrogens fall outside the interval the remaining heavy
    /// atoms can still bring.
    HydrogenBound,
    /// Heavy atoms remain but no atom the closed-prefix rule can still point
    /// at has open valence.
    NoOpenSite,
    /// All heavy atoms are placed but the open valence cannot be closed from
    /// the newest atom.
    OpenValenceWithoutAtoms,
    /// The open valence plus any valence the remaining atoms can bring cannot
    /// pair up under the remaining closure allowance.
    ValenceBound,
}

/// A classified dead-end trajectory.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeadEndCause {
    /// Token index after which the trace first failed a necessary condition
    /// (`None` when no check explains the dead end: "unexplained").
    pub doomed_at: Option<usize>,
    /// The failing check in priority order (`None` means unexplained).
    pub reason: Option<DoomReason>,
    /// Tokens in the dead-end trace.
    pub dead_end_step: usize,
    /// Atoms in the dead-end prefix.
    pub atoms: usize,
    /// Heavy atoms still to add.
    pub remaining_heavy: u32,
    /// Hydrogens still to add (negative only for an overused budget).
    pub remaining_hydrogen: i64,
    /// Open valence of the dead-end prefix.
    pub open_valence: u32,
}

/// Check the four necessary conditions for completability of `state` under
/// the exact-completion grammar with this target composition.
///
/// The checks are [`TraceState::feasibility`](super::grammar::TraceState::feasibility)
/// (documented there with their arguments); the state's own budget and
/// limits govern, and callers pass the state's own. The budget and limits
/// arguments stay so existing callers keep compiling.
pub fn feasibility(state: &TraceState, budget: &Composition, limits: Limits) -> Feasibility {
    let _ = (budget, limits);
    state.feasibility()
}

/// Replay `trace` token by token under `new_exact` and return the first step
/// index after which [`feasibility`] has a `false` field, with the reason in
/// the fixed priority [`DoomReason::HydrogenBound`],
/// [`DoomReason::NoOpenSite`], [`DoomReason::OpenValenceWithoutAtoms`]
/// (`closable` false), [`DoomReason::ValenceBound`].
///
/// Tokens apply under the v1 rules, so prefixes the v2 feasibility lookahead
/// forbids still analyse. Returns `Ok(None)` when every prefix passes all
/// four checks (the trace may still be a dead end no check explains). Errors
/// name the first illegal token when the trace does not replay under the v1
/// exact rules.
pub fn first_doomed_step(
    trace: &[Token],
    budget: &Composition,
    limits: Limits,
) -> Result<Option<(usize, DoomReason)>> {
    let mut state = TraceState::new_exact(limits, *budget);
    for (i, token) in trace.iter().enumerate() {
        state.apply_v1(*token).map_err(|e| {
            Error::config(format!("first_doomed_step: illegal token at step {i}: {e}"))
        })?;
        let f = feasibility(&state, budget, limits);
        if !f.hydrogen {
            return Ok(Some((i, DoomReason::HydrogenBound)));
        }
        if !f.open_site {
            return Ok(Some((i, DoomReason::NoOpenSite)));
        }
        if !f.closable {
            return Ok(Some((i, DoomReason::OpenValenceWithoutAtoms)));
        }
        if !f.valence {
            return Ok(Some((i, DoomReason::ValenceBound)));
        }
    }
    Ok(None)
}

/// Classify one dead-end trace: replay it under the v1 rules (so prefixes
/// the v2 feasibility lookahead forbids still analyse), run
/// [`first_doomed_step`] for the doom point, and report the end-state
/// accounting.
///
/// `reason == None` (with `doomed_at == None`) means none of the necessary
/// conditions explains the dead end ("unexplained").
pub fn classify_dead_end(
    trace: &[Token],
    budget: &Composition,
    limits: Limits,
) -> Result<DeadEndCause> {
    let mut state = TraceState::new_exact(limits, *budget);
    for (i, token) in trace.iter().enumerate() {
        state.apply_v1(*token).map_err(|e| {
            Error::config(format!("classify_dead_end: illegal token at step {i}: {e}"))
        })?;
    }
    let doomed = first_doomed_step(trace, budget, limits)?;
    let (doomed_at, reason) = match doomed {
        Some((at, why)) => (Some(at), Some(why)),
        None => (None, None),
    };
    let mut r_total = 0u32;
    for &r in state.residual_valence() {
        r_total += u32::from(r);
    }
    let mut m = 0i64;
    for (e, &have) in state.used().iter().enumerate() {
        if e != HYDROGEN {
            m += i64::from(budget[e]) - i64::from(have);
        }
    }
    let rem_h = i64::from(budget[HYDROGEN]) - i64::from(state.used()[HYDROGEN]);
    Ok(DeadEndCause {
        doomed_at,
        reason,
        dead_end_step: trace.len(),
        atoms: state.atoms(),
        remaining_heavy: m.max(0) as u32,
        remaining_hydrogen: rem_h,
        open_valence: r_total,
    })
}

/// Cheap necessary condition for substructure containment: a pattern `P` can
/// be contained in a candidate molecule `G` only if every atom type count of
/// `G` covers `P`'s.
///
/// Atom types carry the hydrogen count, so this is stricter than element
/// counts (a methyl `C(H3)` does not cover an ethylenic `C(H2)` slot).
/// Connectivity misses pass this test; type shortfalls fail it, which is
/// what a type-level mask could prevent.
pub fn type_counts_cover(pattern: &MolGraph, molecule: &MolGraph) -> bool {
    let mut need = [0usize; 18];
    let mut have = [0usize; 18];
    for &t in pattern.atoms() {
        if (t as usize) < need.len() {
            need[t as usize] += 1;
        } else {
            return false;
        }
    }
    for &t in molecule.atoms() {
        if (t as usize) < have.len() {
            have[t as usize] += 1;
        } else {
            return false;
        }
    }
    need.iter().zip(have.iter()).all(|(n, h)| h >= n)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn type_counts_cover_distinguishes_shortfall_from_connectivity() {
        // Ethanol `[C(H3), C(H2), O(H1)]` covers its own methyl but not a
        // nitrogen pattern; a lone methyl is covered by ethanol.
        let ethanol = MolGraph::new(vec![4, 3, 9], vec![(0, 1, 1), (1, 2, 1)]).unwrap();
        let methyl = MolGraph::new(vec![4], vec![]).unwrap();
        let amine = MolGraph::new(vec![7], vec![]).unwrap();
        let methylene = MolGraph::new(vec![3], vec![]).unwrap();
        assert!(type_counts_cover(&methyl, &ethanol));
        assert!(type_counts_cover(&methyl, &methyl));
        assert!(!type_counts_cover(&amine, &ethanol));
        // Same element, different hydrogen count: `C(H2)` is not covered by a
        // molecule holding only `C(H3)`.
        let all_methyl = MolGraph::new(vec![4, 4], vec![(0, 1, 1)]).unwrap();
        assert!(!type_counts_cover(&methylene, &all_methyl));
    }
}
