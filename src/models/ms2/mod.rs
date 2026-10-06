//! Host reference of `docs/MS2_CONTRACTS.md`: the MS2 chemistry domain.
//!
//! Pure host Rust with integer masses only: no CubeCL kernels, no tensors,
//! no `Runtime` generics. The GPU pipeline is tested against these types,
//! so exactness beats speed everywhere here.
//!
//! * [`chem`] — domain tables and integer mass arithmetic (§§4.1–4.3 and 5).
//! * [`graph`] — molecules, domain classification and validity
//!   (§§4.2, 4.5 and 4.6).
//! * [`grammar`] — tokens, replay, legality masks and the canonical trace
//!   (§§4.4–4.5 and 7.4).
//! * [`targets`] — pseudo-label recipe `q-cut-v1` (§7.2).
//! * [`dataset`] — the `export_casmi.py` dataset loader and batch adapter.
//! * [`formula`] — reference formula table and peak-relation edges (§9).
//! * [`formula_evidence`] — host twin of the formula-evidence stage (§1.6).
//! * [`formula_evidence_ref`] — independent host reference for explained peaks and the
//!   formula-ranking experiment (sorted sub-vectors, binary search).
//! * [`contract`] — schemas and request validation (§§3 and 8).
//! * [`contain`] — induced-subgraph containment (§7.3).
//! * [`experiment`] — experiment datasets for the V0 experiments.
//! * [`train`] — the V0 training and evaluation driver.
//! * [`metrics`] — evaluation metrics with bootstrap intervals (§10).
//! * [`workspace`] — device capabilities and memory estimates (§§6.1 and 6.2).

pub mod chem;
pub mod contain;
pub mod contract;
pub mod dataset;
pub mod enum_cache;
pub mod experiment;
pub mod formula;
pub mod formula_enum;
pub mod formula_evidence;
pub mod formula_evidence_ref;
pub mod grammar;
pub mod graph;
pub mod identity;
pub mod ion;
pub mod metrics;
pub mod pack;
pub mod rerank;
pub mod rerank_eval;
pub mod targets;
pub mod twin;

pub mod allocate;
pub mod assign;
pub mod baselines;
pub mod batch;
pub mod decoder;
pub mod encoder;
pub mod fingerprint;
pub mod formula_head;
pub mod functional_groups;
pub mod functional_groups_eval;
pub mod generate;
pub mod ragged;
pub mod targets_batch;
pub mod train;
pub mod workspace;
pub mod calibration;

pub use chem::{
    ADDUCTS, ATOM_TYPES, Adduct, AtomType, CHEMISTRY_VERSION, Composition, ELECTRON_EXACT,
    ELECTRON_MASS, ELECTRON_RESIDUAL_NDA, ELEMENTS, HYDROGEN, Ion, MASS_SCALE, Verdict, adduct,
    atom_type, atom_type_of, composition_error_nda, composition_mass, decide, element_index, ion,
    parent_mass, parse_decimal, tolerance, tolerance_u32,
};
pub use grammar::{
    ADD_ATOM, CANONICAL_WORK_LIMIT, CLOSE_RING, Canonical, GRAMMAR_VERSION, LegalMasks, Limits,
    PAD, START, STOP, TRAVERSAL_VERSION, Token, TraceState, canonical_trace, first_illegal_step,
    replay,
};
pub use graph::{DomainReason, MAX_GRAPH_ATOMS, MolGraph, RawAtom, RawMolecule};
