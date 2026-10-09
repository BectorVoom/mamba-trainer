//! Strict versioned JSON generation API for the trained completion model
//! (`molecular-completion-generate-v1`).
//!
//! One strict request — protocol, id, provenance, `mass_role`
//! `target_molecule`, a supplied exact composition, V0-typed substructures
//! with parent hydrogen counts and a sampling config — is validated here and
//! then answered by [`CompletionModel::generate`](super::completion_model::CompletionModel::generate)
//! under the exact-completion grammar. All chemistry, sampling and ranking
//! semantics live in the model; this layer only validates, hashes and renders.
//!
//! Anything outside the model's domain (more composition heavy atoms than
//! `max_atoms`, a single pattern larger than `max_atoms` or than the
//! composition's heavy-atom count, or more ring closures than
//! `max_ring_closures`) is an `unsupported_input` response with no candidates,
//! never an exception. Malformed JSON and schema violations are errors.

use std::path::Path;

use cubecl::prelude::Runtime;
use serde_json::{Map, Value, json};

use crate::backend::Device;
use crate::error::{Error, Result};
use crate::tensor::ops::ms2::Ms2Constants;

use super::chem::{self, CHEMISTRY_VERSION, Composition};
use super::completion_data::COMPLETION_DATA_VERSION;
use super::completion_formula::{
    FormulaAllocation, FormulaPruning, MassQuery, adduct_id_by_name, formula_text,
};
use super::completion_model::{
    COMPLETION_CHECKPOINT_FORMAT, COMPLETION_MODEL_VERSION, CompletionGenerationConfig,
    CompletionRequest, CompletionTrainer, PATTERN_SLOTS, SubstructureSemantics,
};
use super::completion_request::composition_text;
use super::experiment::sha256_hex;
use super::grammar::COMPLETION_GRAMMAR_VERSION;
use super::graph::MolGraph;

/// Protocol string required at the top of every request and response.
pub const GENERATE_PROTOCOL: &str = "molecular-completion-generate-v1";

fn invalid(msg: impl Into<String>) -> Error {
    Error::config(msg.into())
}

fn require_object<'a>(v: &'a Value, what: &str) -> Result<&'a Map<String, Value>> {
    v.as_object()
        .ok_or_else(|| invalid(format!("{what} must be an object")))
}

fn require_string<'a>(v: &'a Value, what: &str) -> Result<&'a str> {
    match v.as_str() {
        Some(s) if !s.is_empty() => Ok(s),
        Some(_) => Err(invalid(format!("{what} must be a non-empty string"))),
        None => Err(invalid(format!("{what} must be a string"))),
    }
}

fn require_u32(v: &Value, what: &str) -> Result<u32> {
    match v.as_u64() {
        Some(n) if n <= u64::from(u32::MAX) => Ok(n as u32),
        Some(_) => Err(invalid(format!("{what} overflows u32"))),
        None => Err(invalid(format!("{what} must be a non-negative integer"))),
    }
}

fn require_u64(v: &Value, what: &str) -> Result<u64> {
    v.as_u64()
        .ok_or_else(|| invalid(format!("{what} must be a non-negative integer")))
}

/// Reject keys outside `allowed`; never guess semantics for extras.
fn reject_unknown_keys(obj: &Map<String, Value>, allowed: &[&str], what: &str) -> Result<()> {
    for k in obj.keys() {
        if !allowed.iter().any(|a| a == k) {
            return Err(invalid(format!(
                "{what} carries unknown field '{k}'; refused to guess semantics"
            )));
        }
    }
    Ok(())
}

/// Require `key` present (not null) and return its value.
fn take<'a>(obj: &'a Map<String, Value>, key: &str, what: &str) -> Result<&'a Value> {
    match obj.get(key) {
        Some(v) if !v.is_null() => Ok(v),
        Some(_) => Err(invalid(format!("{what}.{key} must not be null"))),
        None => Err(invalid(format!("{what} is missing required field '{key}'"))),
    }
}

/// Validated request: id, provenance, exactly one of an exact composition or
/// a target mass, patterns and sampling.
///
/// There is no learned formula ranker: mass formulas are ordered by absolute
/// mass residual with no learned prior.
struct ValidRequest {
    /// Original id string (kept for `query_id`).
    id: String,
    /// Request provenance.
    provenance: String,
    /// Exact target composition, hydrogens included (`None` for mass input).
    composition: Option<Composition>,
    /// Target mass input (`None` for composition input).
    target_mass: Option<TargetMass>,
    /// Neutralization convention (`None` for composition input).
    neutralization: Option<Neutralization>,
    /// Formula-search budgets (`None` for composition input).
    formula_search: Option<FormulaSearch>,
    /// Required substructures.
    patterns: Vec<MolGraph>,
    /// Sampling hyperparameters (`generation.trajectories` is the total
    /// budget for mass input, split evenly over the selected formulas).
    trajectories: u32,
    /// Softmax temperature.
    temperature: f32,
    /// RNG seed.
    seed: u64,
    /// Shortlist size.
    returned: u32,
    /// How the substructures constrain host acceptance.
    semantics: SubstructureSemantics,
    /// Isomer expansion count (`stereo.expand`, 0..=64).
    stereo_expand: usize,
    /// Most potential stereo elements analysed (`stereo.max_elements`).
    stereo_max_elements: usize,
    /// Optional fingerprint evidence (`None` means no fingerprint).
    fingerprint: Option<ParsedFingerprint>,
    /// Optional spectral evidence: peaks, precursor, adduct and neutral mass
    /// (`None` means none).
    spectrum: Option<super::completion_spectrum::SpectrumEvidence>,
}

/// A validated `fingerprint` object: MIST `morgan4096` probabilities with a
/// token threshold. The fingerprint must be computed or predicted outside
/// this crate; the model conditions on the supplied probabilities.
struct ParsedFingerprint {
    /// `(bit, probability)` pairs as supplied (probabilities in `(0, 1]`).
    bits: Vec<(u16, f32)>,
    /// Token threshold in `(0, 1]`: entries below it are dropped.
    threshold: f32,
}

/// A validated `target_mass` object.
struct TargetMass {
    /// Observed mass: neutral mass for `already_neutral`, precursor m/z for
    /// a precursor ion.
    value: u32,
    /// Tolerance in tenths of a ppm.
    ppm_tenths: u32,
    /// Uncertainty in micro-dalton; `None` means unknown precision.
    uncertainty: Option<u32>,
    /// Free-text source.
    source: String,
}

/// A validated `neutralization` value.
#[derive(Clone, Debug)]
enum Neutralization {
    /// The target mass is already neutral.
    AlreadyNeutral,
    /// The target mass is a precursor m/z under this adduct.
    PrecursorIon {
        /// Adduct id (one of `chem::ADDUCTS`).
        adduct: u16,
    },
}

/// Validated `formula_search` budgets.
struct FormulaSearch {
    /// Hypotheses selected (`1..=32`).
    hypotheses: u32,
    /// Enumerator node budget (`1..=50,000,000`).
    nodes_visited_max: u64,
    /// Formula-search pruning (`train_fit` by default).
    pruning: FormulaPruning,
    /// Trajectory allocation over the selected formulas (`equal` by default).
    allocation: FormulaAllocation,
}

/// Parse and validate one strict request.
///
/// The request carries exactly one of `composition` or `target_mass`: both
/// or neither is a schema error naming the rule.
fn parse_request(root: &Value) -> Result<ValidRequest> {
    let obj = require_object(root, "request")?;
    reject_unknown_keys(
        obj,
        &[
            "protocol",
            "id",
            "provenance",
            "mass_role",
            "composition",
            "target_mass",
            "neutralization",
            "formula_search",
            "substructures",
            "substructure_semantics",
            "generation",
            "stereo",
            "fingerprint",
            "spectrum",
        ],
        "request",
    )?;
    let protocol = require_string(take(obj, "protocol", "request")?, "request.protocol")?;
    if protocol != GENERATE_PROTOCOL {
        return Err(invalid(format!(
            "request.protocol must be '{GENERATE_PROTOCOL}', got '{protocol}'"
        )));
    }
    let id = require_string(take(obj, "id", "request")?, "request.id")?.to_string();
    let provenance =
        require_string(take(obj, "provenance", "request")?, "request.provenance")?.to_string();
    let mass_role = require_string(take(obj, "mass_role", "request")?, "request.mass_role")?;
    match mass_role {
        "target_molecule" => {}
        "target_fragment" | "neutral_loss" => {
            return Err(invalid(format!(
                "mass_role '{mass_role}' is rejected: precursor conventions and fragment evidence are not supported in this bounded MVP; declare mass_role 'target_molecule' or omit precursor fields"
            )));
        }
        other => {
            return Err(invalid(format!(
                "mass_role must be 'target_molecule', 'target_fragment' or 'neutral_loss'; got '{other}'"
            )));
        }
    }
    let has_composition = obj.contains_key("composition");
    let has_mass = obj.contains_key("target_mass");
    if has_composition == has_mass {
        return Err(invalid(
            "request must carry exactly one of 'composition' or 'target_mass' (both or neither is a schema error)".to_string(),
        ));
    }
    let patterns = parse_substructures(take(obj, "substructures", "request")?)?;
    let semantics = parse_semantics(obj.get("substructure_semantics"))?;
    let (trajectories, temperature, seed, returned) =
        parse_generation(take(obj, "generation", "request")?)?;
    let (stereo_expand, stereo_max_elements) = parse_stereo(obj.get("stereo"))?;
    let fingerprint = parse_fingerprint(obj.get("fingerprint"))?;
    let spectrum = parse_spectrum(obj.get("spectrum"))?;
    if has_composition {
        if obj.contains_key("neutralization") || obj.contains_key("formula_search") {
            return Err(invalid(
                "request with 'composition' must not carry 'neutralization' or 'formula_search' (those belong to 'target_mass' input)".to_string(),
            ));
        }
        let composition = parse_composition(take(obj, "composition", "request")?)?;
        return Ok(ValidRequest {
            id,
            provenance,
            composition: Some(composition),
            target_mass: None,
            neutralization: None,
            formula_search: None,
            patterns,
            trajectories,
            temperature,
            seed,
            returned,
            semantics,
            stereo_expand,
            stereo_max_elements,
            fingerprint,
            spectrum,
        });
    }
    // Mass input.
    if !obj.contains_key("neutralization") {
        return Err(invalid(
            "request with 'target_mass' is missing required field 'neutralization'".to_string(),
        ));
    }
    let target_mass = parse_target_mass(take(obj, "target_mass", "request")?)?;
    let neutralization = parse_neutralization(take(obj, "neutralization", "request")?)?;
    let formula_search = match obj.get("formula_search") {
        None => FormulaSearch {
            hypotheses: 8,
            nodes_visited_max: 2_000_000,
            pruning: FormulaPruning::default(),
            allocation: FormulaAllocation::default(),
        },
        Some(v) => parse_formula_search(v)?,
    };
    Ok(ValidRequest {
        id,
        provenance,
        composition: None,
        target_mass: Some(target_mass),
        neutralization: Some(neutralization),
        formula_search: Some(formula_search),
        patterns,
        trajectories,
        temperature,
        seed,
        returned,
        semantics,
        stereo_expand,
        stereo_max_elements,
        fingerprint,
        spectrum,
    })
}

/// Parse a `target_mass` object: micro-dalton value, ppm tolerance,
/// uncertainty (null means unknown precision) and source.
fn parse_target_mass(v: &Value) -> Result<TargetMass> {
    let obj = require_object(v, "request.target_mass")?;
    reject_unknown_keys(
        obj,
        &["units", "value", "ppm_tenths", "uncertainty_uda", "source"],
        "request.target_mass",
    )?;
    let units = require_string(
        take(obj, "units", "request.target_mass")?,
        "request.target_mass.units",
    )?;
    if units != "microdalton" {
        return Err(invalid(format!(
            "request.target_mass.units must be 'microdalton', got '{units}'"
        )));
    }
    let value_raw = take(obj, "value", "request.target_mass")?;
    let value_u64 = value_raw.as_u64().ok_or_else(|| {
        invalid("request.target_mass.value must be a non-negative integer".to_string())
    })?;
    if value_u64 > u64::from(u32::MAX) {
        return Err(invalid("request.target_mass.value overflows u32"));
    }
    let value = value_u64 as u32;
    let ppm_tenths = require_u32(
        take(obj, "ppm_tenths", "request.target_mass")?,
        "request.target_mass.ppm_tenths",
    )?;
    if ppm_tenths > 1000 {
        return Err(invalid(format!(
            "request.target_mass.ppm_tenths {ppm_tenths} exceeds the 1000 proof bound"
        )));
    }
    let uncertainty = match obj.get("uncertainty_uda") {
        None => {
            return Err(invalid(
                "request.target_mass is missing required field 'uncertainty_uda' (use null for unknown)".to_string(),
            ));
        }
        Some(Value::Null) => None,
        Some(u) => {
            let n = require_u32(u, "request.target_mass.uncertainty_uda")?;
            // The unknown-precision sentinel `u32::MAX` as an integer is a
            // schema error: unknown precision must be spelled `null` (which
            // yields `mass_evidence.status = "unavailable"` with no search).
            // Accepting the integer would silently run a search with a
            // four-billion-microdalton error bound while the enumerator
            // treats that same integer as unknown precision.
            if n == u32::MAX {
                return Err(invalid(
                    "request.target_mass.uncertainty_uda 4294967295 is the unknown-precision sentinel: use null for unknown precision".to_string(),
                ));
            }
            Some(n)
        }
    };
    let source = require_string(
        take(obj, "source", "request.target_mass")?,
        "request.target_mass.source",
    )?
    .to_string();
    Ok(TargetMass {
        value,
        ppm_tenths,
        uncertainty,
        source,
    })
}

/// Parse `neutralization`: `"already_neutral"` or
/// `{"precursor_ion": {"adduct": "[M+H]+"}}` with the adduct one of
/// `chem::ADDUCTS` by name.
fn parse_neutralization(v: &Value) -> Result<Neutralization> {
    if let Some(s) = v.as_str() {
        if s == "already_neutral" {
            return Ok(Neutralization::AlreadyNeutral);
        }
        return Err(invalid(format!(
            "request.neutralization '{s}' is unknown: expected 'already_neutral' or {{\"precursor_ion\": {{\"adduct\": name}}}}"
        )));
    }
    let obj = require_object(v, "request.neutralization")?;
    reject_unknown_keys(obj, &["precursor_ion"], "request.neutralization")?;
    let ion_v = take(obj, "precursor_ion", "request.neutralization")?;
    let ion_obj = require_object(ion_v, "request.neutralization.precursor_ion")?;
    reject_unknown_keys(ion_obj, &["adduct"], "request.neutralization.precursor_ion")?;
    let adduct_name = require_string(
        take(ion_obj, "adduct", "request.neutralization.precursor_ion")?,
        "request.neutralization.precursor_ion.adduct",
    )?;
    let Some(adduct) = adduct_id_by_name(adduct_name) else {
        return Err(invalid(format!(
            "request.neutralization.precursor_ion.adduct '{adduct_name}' is unknown: expected one of [M+H]+, [M-H]-"
        )));
    };
    Ok(Neutralization::PrecursorIon { adduct })
}

/// Parse `formula_search`: `hypotheses` in `1..=32` (default 8),
/// `nodes_visited_max` in `1..=50,000,000` (default 2,000,000), `pruning`
/// `train_fit` (default) or `chemical_only`, `allocation` `equal` (default)
/// or `train_frequency`.
fn parse_formula_search(v: &Value) -> Result<FormulaSearch> {
    let obj = require_object(v, "request.formula_search")?;
    reject_unknown_keys(
        obj,
        &["hypotheses", "nodes_visited_max", "pruning", "allocation"],
        "request.formula_search",
    )?;
    let hypotheses = match obj.get("hypotheses") {
        None => 8,
        Some(h) => {
            let n = require_u32(h, "request.formula_search.hypotheses")?;
            if !(1..=32).contains(&n) {
                return Err(invalid(format!(
                    "request.formula_search.hypotheses {n} is not in 1..=32"
                )));
            }
            n
        }
    };
    let nodes_visited_max = match obj.get("nodes_visited_max") {
        None => 2_000_000,
        Some(n) => {
            let m = require_u64(n, "request.formula_search.nodes_visited_max")?;
            if !(1..=50_000_000).contains(&m) {
                return Err(invalid(format!(
                    "request.formula_search.nodes_visited_max {m} is not in 1..=50,000,000"
                )));
            }
            m
        }
    };
    let pruning = match obj.get("pruning") {
        None => FormulaPruning::default(),
        Some(p) => {
            let text = require_string(p, "request.formula_search.pruning")?;
            FormulaPruning::parse(text).ok_or_else(|| {
                invalid(format!(
                    "request.formula_search.pruning '{text}' is unknown: expected 'train_fit' or 'chemical_only'"
                ))
            })?
        }
    };
    let allocation = match obj.get("allocation") {
        None => FormulaAllocation::default(),
        Some(a) => {
            let text = require_string(a, "request.formula_search.allocation")?;
            FormulaAllocation::parse(text).ok_or_else(|| {
                invalid(format!(
                    "request.formula_search.allocation '{text}' is unknown: expected 'equal' or 'train_frequency'"
                ))
            })?
        }
    };
    Ok(FormulaSearch {
        hypotheses,
        nodes_visited_max,
        pruning,
        allocation,
    })
}

/// Parse the supplied exact composition: keys are element symbols of
/// [`ELEMENTS`](super::chem::ELEMENTS), absent elements are 0.
fn parse_composition(v: &Value) -> Result<Composition> {
    let obj = require_object(v, "request.composition")?;
    let mut out: Composition = [0; 10];
    for (symbol, value) in obj.iter() {
        let idx = chem::element_index(symbol).ok_or_else(|| {
            invalid(format!(
                "request.composition carries unknown element '{symbol}'"
            ))
        })?;
        let count = value.as_u64().ok_or_else(|| {
            invalid(format!(
                "request.composition.{symbol} must be a non-negative integer"
            ))
        })?;
        if count > u64::from(u16::MAX) {
            return Err(invalid(format!(
                "request.composition.{symbol} count {count} does not fit u16"
            )));
        }
        out[idx] = count as u16;
    }
    let heavy: u32 = out
        .iter()
        .enumerate()
        .filter(|(e, _)| *e != chem::HYDROGEN)
        .map(|(_, &n)| u32::from(n))
        .sum();
    if heavy == 0 {
        return Err(invalid(
            "request.composition holds zero heavy atoms".to_string(),
        ));
    }
    Ok(out)
}

/// Parse the substructure list: at most 8 patterns, at most
/// [`PATTERN_SLOTS`] atoms total.
fn parse_substructures(v: &Value) -> Result<Vec<MolGraph>> {
    let list = v
        .as_array()
        .ok_or_else(|| invalid("request.substructures must be a list".to_string()))?;
    if list.len() > 8 {
        return Err(invalid(format!(
            "request.substructures holds {} patterns, past the limit of 8",
            list.len()
        )));
    }
    let mut out = Vec::with_capacity(list.len());
    let mut total_atoms = 0usize;
    for (i, s) in list.iter().enumerate() {
        let sobj = require_object(s, &format!("substructures[{i}]"))?;
        reject_unknown_keys(
            sobj,
            &[
                "atoms",
                "bonds",
                "parent_hydrogen_semantics",
                "certainty",
                "provenance",
            ],
            &format!("substructures[{i}]"),
        )?;
        let atoms_v = take(sobj, "atoms", &format!("substructures[{i}]"))?
            .as_array()
            .ok_or_else(|| invalid(format!("substructures[{i}].atoms must be a list")))?;
        if atoms_v.is_empty() {
            return Err(invalid(format!(
                "substructures[{i}].atoms must be non-empty"
            )));
        }
        let mut atoms: Vec<u8> = Vec::with_capacity(atoms_v.len());
        for a in atoms_v {
            let n = require_u64(a, &format!("substructures[{i}].atoms entry"))?;
            if n > 255 {
                return Err(invalid(format!(
                    "substructures[{i}].atoms entry {n} does not fit u8"
                )));
            }
            let id = n as u8;
            if chem::atom_type(id).is_none() {
                return Err(invalid(format!(
                    "substructures[{i}].atoms entry {n} is not a valid atom type id"
                )));
            }
            atoms.push(id);
        }
        total_atoms += atoms.len();
        if total_atoms > PATTERN_SLOTS {
            return Err(invalid(format!(
                "request.substructures holds {total_atoms} pattern atoms, past the limit of {PATTERN_SLOTS}"
            )));
        }
        let bonds_v = take(sobj, "bonds", &format!("substructures[{i}]"))?
            .as_array()
            .ok_or_else(|| invalid(format!("substructures[{i}].bonds must be a list")))?;
        let mut bonds: Vec<(usize, usize, u8)> = Vec::with_capacity(bonds_v.len());
        for b in bonds_v {
            let triple = b.as_array().ok_or_else(|| {
                invalid(format!("substructures[{i}].bonds entries must be lists"))
            })?;
            if triple.len() != 3 {
                return Err(invalid(format!(
                    "substructures[{i}].bonds entries must have 3 fields"
                )));
            }
            let a = require_u64(&triple[0], &format!("substructures[{i}].bonds endpoint"))?;
            let c = require_u64(&triple[1], &format!("substructures[{i}].bonds endpoint"))?;
            let o = require_u64(&triple[2], &format!("substructures[{i}].bonds order"))?;
            if o > 255 {
                return Err(invalid(format!(
                    "substructures[{i}].bonds order {o} does not fit u8"
                )));
            }
            // Range-check as `u64` before narrowing: a huge index must be a
            // schema error on every target, never a truncated `usize` that
            // `MolGraph::new` happens to accept.
            let n_atoms = atoms_v.len() as u64;
            if a >= n_atoms || c >= n_atoms {
                return Err(invalid(format!(
                    "substructures[{i}].bonds endpoint ({a}, {c}) names no atom of {} atoms",
                    atoms_v.len()
                )));
            }
            bonds.push((a as usize, c as usize, o as u8));
        }
        let semantics = require_string(
            take(
                sobj,
                "parent_hydrogen_semantics",
                &format!("substructures[{i}]"),
            )?,
            &format!("substructures[{i}].parent_hydrogen_semantics"),
        )?;
        if semantics != "v0_parent_hydrogen_counts" {
            return Err(invalid(format!(
                "substructures[{i}].parent_hydrogen_semantics '{semantics}' is unknown: this protocol only accepts V0 atom types carrying parent hydrogen counts ('v0_parent_hydrogen_counts')"
            )));
        }
        let certainty = require_string(
            take(sobj, "certainty", &format!("substructures[{i}]"))?,
            &format!("substructures[{i}].certainty"),
        )?;
        if certainty == "tentative" {
            return Err(invalid(format!(
                "substructures[{i}].certainty is 'tentative': tentative patterns are soft evidence and are not supported as hard constraints; confirm the pattern or remove it"
            )));
        }
        if certainty != "confirmed" {
            return Err(invalid(format!(
                "substructures[{i}].certainty must be 'confirmed'; got '{certainty}'"
            )));
        }
        require_string(
            take(sobj, "provenance", &format!("substructures[{i}]"))?,
            &format!("substructures[{i}].provenance"),
        )?;
        let graph = MolGraph::new(atoms, bonds)
            .map_err(|e| invalid(format!("substructures[{i}] does not build a molecule: {e}")))?;
        if !graph.is_connected() {
            return Err(invalid(format!("substructures[{i}] is not connected")));
        }
        out.push(graph);
    }
    Ok(out)
}

/// Parse the sampling config: trajectories, temperature, seed, returned.
#[allow(clippy::type_complexity)]
fn parse_generation(v: &Value) -> Result<(u32, f32, u64, u32)> {
    let obj = require_object(v, "request.generation")?;
    reject_unknown_keys(
        obj,
        &["trajectories", "temperature", "seed", "returned"],
        "request.generation",
    )?;
    let trajectories = require_u32(
        take(obj, "trajectories", "request.generation")?,
        "request.generation.trajectories",
    )?;
    if !(1..=1024).contains(&trajectories) {
        return Err(invalid(format!(
            "request.generation.trajectories {trajectories} is not in 1..=1024"
        )));
    }
    let temperature_v = take(obj, "temperature", "request.generation")?;
    let temperature = temperature_v
        .as_f64()
        .ok_or_else(|| invalid("request.generation.temperature must be a number".to_string()))?
        as f32;
    if !(temperature.is_finite() && temperature > 0.0) {
        return Err(invalid(format!(
            "request.generation.temperature {temperature} is not finite and positive"
        )));
    }
    let seed = require_u64(
        take(obj, "seed", "request.generation")?,
        "request.generation.seed",
    )?;
    let returned = require_u32(
        take(obj, "returned", "request.generation")?,
        "request.generation.returned",
    )?;
    if !(1..=25).contains(&returned) {
        return Err(invalid(format!(
            "request.generation.returned {returned} is not in 1..=25"
        )));
    }
    Ok((trajectories, temperature, seed, returned))
}

/// Parse the optional `substructure_semantics` value: `"contained"`
/// (default: every pattern is contained somewhere, sharing allowed),
/// `"disjoint_occurrences"` (patterns are distinct occurrences on disjoint
/// atoms) or `"complete_functional_groups"` (patterns are the molecule's
/// complete functional-group list). Absent means `"contained"`; anything
/// else is a schema error naming the field.
fn parse_semantics(v: Option<&Value>) -> Result<SubstructureSemantics> {
    let Some(v) = v else {
        return Ok(SubstructureSemantics::Contained);
    };
    let text = require_string(v, "request.substructure_semantics")?;
    SubstructureSemantics::parse(text).ok_or_else(|| {
        invalid(format!(
            "request.substructure_semantics '{text}' is unknown: expected 'contained', 'disjoint_occurrences' or 'complete_functional_groups'"
        ))
    })
}

/// Parse the optional `stereo` object: `expand` in `0..=64` (default 0,
/// how many canonical stereoisomers each candidate carries) and
/// `max_elements` in `1..=12` (default 10, most potential elements
/// analysed before the candidate reports `unresolved: too_many_elements`).
/// Absent means defaults; unknown fields are errors.
fn parse_stereo(v: Option<&Value>) -> Result<(usize, usize)> {
    let Some(v) = v else {
        return Ok((0, 10));
    };
    let obj = require_object(v, "request.stereo")?;
    reject_unknown_keys(obj, &["expand", "max_elements"], "request.stereo")?;
    let expand = match obj.get("expand") {
        None => 0,
        Some(e) => {
            let n = require_u32(e, "request.stereo.expand")?;
            if n > 64 {
                return Err(invalid(format!(
                    "request.stereo.expand {n} is not in 0..=64"
                )));
            }
            n as usize
        }
    };
    let max_elements = match obj.get("max_elements") {
        None => 10,
        Some(m) => {
            let n = require_u32(m, "request.stereo.max_elements")?;
            if !(1..=12).contains(&n) {
                return Err(invalid(format!(
                    "request.stereo.max_elements {n} is not in 1..=12"
                )));
            }
            n as usize
        }
    };
    Ok((expand, max_elements))
}

/// Most peaks a `spectrum` object may carry.
const SPECTRUM_PEAKS_MAX: usize = 4096;

/// Parse the optional `spectrum` object: `{"peaks": [[mz_uda, intensity],
/// ...], "precursor_mz_uda": 243057000, "adduct": "[M+H]+",
/// "neutral_mass_uda": 242049724}`. Absent means no spectral evidence.
/// `peaks` holds at most 4096 `[m/z in micro-dalton, intensity]` pairs (a
/// positive integer and a finite non-negative number); `adduct` is one of
/// the completion adduct names or `"unknown"`; `neutral_mass_uda` may be
/// omitted for a known adduct (it is then the precursor minus the adduct
/// shift) and is required for `"unknown"`.
fn parse_spectrum(
    v: Option<&Value>,
) -> Result<Option<super::completion_spectrum::SpectrumEvidence>> {
    use super::completion_spectrum::{
        SpectrumEvidence, completion_adduct_by_name, neutral_mass_of,
    };
    let Some(v) = v else {
        return Ok(None);
    };
    let obj = require_object(v, "request.spectrum")?;
    reject_unknown_keys(
        obj,
        &["peaks", "precursor_mz_uda", "adduct", "neutral_mass_uda"],
        "request.spectrum",
    )?;
    let positive_u32 = |value: &Value, what: &str| -> Result<u32> {
        let n = require_u64(value, what)?;
        if n == 0 || n > u64::from(u32::MAX) {
            return Err(invalid(format!("{what} {n} is not in 1..=4294967295")));
        }
        Ok(n as u32)
    };
    let peaks_v = take(obj, "peaks", "request.spectrum")?
        .as_array()
        .ok_or_else(|| invalid("request.spectrum.peaks must be a list".to_string()))?;
    if peaks_v.len() > SPECTRUM_PEAKS_MAX {
        return Err(invalid(format!(
            "request.spectrum.peaks holds {} peaks, more than {SPECTRUM_PEAKS_MAX}",
            peaks_v.len()
        )));
    }
    let mut peaks: Vec<(u32, f32)> = Vec::with_capacity(peaks_v.len());
    for (i, entry) in peaks_v.iter().enumerate() {
        let pair = entry.as_array().ok_or_else(|| {
            invalid(format!(
                "request.spectrum.peaks[{i}] must be a [mz_uda, intensity] pair"
            ))
        })?;
        if pair.len() != 2 {
            return Err(invalid(format!(
                "request.spectrum.peaks[{i}] must have 2 fields"
            )));
        }
        let mz = positive_u32(&pair[0], &format!("request.spectrum.peaks[{i}][0]"))?;
        let intensity = pair[1]
            .as_f64()
            .ok_or_else(|| invalid(format!("request.spectrum.peaks[{i}][1] must be a number")))?
            as f32;
        if !(intensity.is_finite() && intensity >= 0.0) {
            return Err(invalid(format!(
                "request.spectrum.peaks[{i}][1] intensity {intensity} is not finite and non-negative"
            )));
        }
        peaks.push((mz, intensity));
    }
    let precursor_mz = positive_u32(
        take(obj, "precursor_mz_uda", "request.spectrum")?,
        "request.spectrum.precursor_mz_uda",
    )?;
    let adduct_name = require_string(
        take(obj, "adduct", "request.spectrum")?,
        "request.spectrum.adduct",
    )?;
    let adduct = if adduct_name == "unknown" {
        0
    } else {
        completion_adduct_by_name(adduct_name)
            .ok_or_else(|| {
                invalid(format!(
                    "request.spectrum.adduct '{adduct_name}' is not a completion adduct or 'unknown'"
                ))
            })?
            .id
    };
    let neutral_mass = match obj.get("neutral_mass_uda") {
        Some(value) => positive_u32(value, "request.spectrum.neutral_mass_uda")?,
        None => neutral_mass_of(precursor_mz, adduct).ok_or_else(|| {
            invalid(format!(
                "request.spectrum.neutral_mass_uda is required: no neutral mass follows from precursor {precursor_mz} under adduct '{adduct_name}'"
            ))
        })?,
    };
    let evidence = SpectrumEvidence {
        peaks,
        precursor_mz,
        adduct,
        neutral_mass,
    };
    evidence
        .validate()
        .map_err(|e| invalid(format!("request.spectrum rejected: {e}")))?;
    Ok(Some(evidence))
}

/// Parse the optional `fingerprint` object:
/// `{"name": "morgan4096", "bits": [[index, probability], ...],
/// "threshold": 0.1}`. Absent means no fingerprint. `name` must be
/// `morgan4096` (the fingerprint must be computed or predicted outside this
/// crate); every index must be below 4096, every probability in `(0, 1]`,
/// `threshold` in `(0, 1]` (default 0.1). Anything else is a schema error
/// naming the field.
fn parse_fingerprint(v: Option<&Value>) -> Result<Option<ParsedFingerprint>> {
    let Some(v) = v else {
        return Ok(None);
    };
    let obj = require_object(v, "request.fingerprint")?;
    reject_unknown_keys(obj, &["name", "bits", "threshold"], "request.fingerprint")?;
    let name = require_string(
        take(obj, "name", "request.fingerprint")?,
        "request.fingerprint.name",
    )?;
    if name != "morgan4096" {
        return Err(invalid(format!(
            "request.fingerprint.name must be 'morgan4096', got '{name}'"
        )));
    }
    let bits_v = take(obj, "bits", "request.fingerprint")?
        .as_array()
        .ok_or_else(|| invalid("request.fingerprint.bits must be a list".to_string()))?;
    let mut bits: Vec<(u16, f32)> = Vec::with_capacity(bits_v.len());
    for (i, entry) in bits_v.iter().enumerate() {
        let pair = entry.as_array().ok_or_else(|| {
            invalid(format!(
                "request.fingerprint.bits[{i}] must be a [index, probability] pair"
            ))
        })?;
        if pair.len() != 2 {
            return Err(invalid(format!(
                "request.fingerprint.bits[{i}] must have 2 fields"
            )));
        }
        let index = require_u64(&pair[0], &format!("request.fingerprint.bits[{i}][0]"))?;
        if index >= 4096 {
            return Err(invalid(format!(
                "request.fingerprint.bits[{i}][0] index {index} is past 4096"
            )));
        }
        let prob = pair[1]
            .as_f64()
            .ok_or_else(|| invalid(format!("request.fingerprint.bits[{i}][1] must be a number")))?
            as f32;
        if !(prob.is_finite() && prob > 0.0 && prob <= 1.0) {
            return Err(invalid(format!(
                "request.fingerprint.bits[{i}][1] probability {prob} is not in (0, 1]"
            )));
        }
        bits.push((index as u16, prob));
    }
    let threshold = match obj.get("threshold") {
        None => 0.1,
        Some(t) => {
            let x = t.as_f64().ok_or_else(|| {
                invalid("request.fingerprint.threshold must be a number".to_string())
            })? as f32;
            if !(x.is_finite() && x > 0.0 && x <= 1.0) {
                return Err(invalid(format!(
                    "request.fingerprint.threshold {x} is not in (0, 1]"
                )));
            }
            x
        }
    };
    Ok(Some(ParsedFingerprint { bits, threshold }))
}

/// Reason text behind the top-level `stereochemistry` object: stereo is
/// enumerated from each candidate graph, never predicted by the model.
const STEREO_REASON: &str = "stereo elements and distinct stereoisomers are derived from each candidate graph; the model assigns no preference among them; kinds listed under a candidate's `unsupported` are not modelled";

/// The top-level `stereochemistry` object: enumeration status, never a
/// prediction. Replaces the former `"unspecified"` string.
fn stereochemistry_object() -> Value {
    json!({"status": "enumerated_not_predicted", "reason": STEREO_REASON})
}

/// One ligand as JSON: heavy atoms by index, `"H"` for hydrogen,
/// `"lone_pair"` for a nitrogen lone pair.
fn ligand_json(ligand: &super::stereo::Ligand) -> Value {
    match ligand {
        super::stereo::Ligand::Atom(i) => json!(*i),
        super::stereo::Ligand::Hydrogen => json!("H"),
        super::stereo::Ligand::LonePair => json!("lone_pair"),
    }
}

/// Render one candidate's `stereo` block from its graph (`None` when the
/// graph failed to rebuild on the mass path: unresolved, never a panic).
///
/// Element lists hold the stereogenic elements only, in assignment order;
/// `not_stereogenic` counts the dropped ones. `stereoisomers` is present
/// only when `expand > 0` (empty when unresolved).
fn stereo_block_for_graph(graph: Option<&MolGraph>, expand: usize, max_elements: usize) -> Value {
    let Some(graph) = graph else {
        let mut block = json!({
            "version": super::stereo::STEREO_VERSION,
            "assignment": "unspecified",
            "tetrahedral_centers": [],
            "double_bonds": [],
            "not_stereogenic": 0,
            "unsupported": [],
            "raw_assignments": null,
            "distinct_stereoisomers": null,
            "resolution": "unresolved: graph_rebuild",
            "molecule_wide_exact": false,
        });
        if expand > 0 {
            block["stereoisomers"] = json!([]);
            block["stereoisomers_truncated"] = json!(false);
        }
        return block;
    };
    let report = super::stereo::perceive(
        graph,
        &super::stereo::StereoLimits {
            max_elements,
            max_automorphisms: 20_000,
            work_limit: 100_000,
            max_expanded: expand,
        },
    );
    let mut tetra: Vec<Value> = Vec::new();
    let mut bonds: Vec<Value> = Vec::new();
    for indexed in &report.elements {
        match &indexed.element {
            super::stereo::StereoElement::Tetrahedral { atom, ligands } => {
                tetra.push(json!({
                    "atom": atom,
                    "ligands": ligands.iter().map(ligand_json).collect::<Vec<_>>(),
                }));
            }
            super::stereo::StereoElement::DoubleBond { a, b, ref_a, ref_b } => {
                bonds.push(json!({
                    "atoms": [a, b],
                    "reference": [ligand_json(ref_a), ligand_json(ref_b)],
                }));
            }
        }
    }
    // Isomer values aligned with the two lists above: `report.isomers`
    // runs over `report.elements` in order, so split by kind.
    let mut block = json!({
        "version": super::stereo::STEREO_VERSION,
        "assignment": "unspecified",
        "tetrahedral_centers": tetra,
        "double_bonds": bonds,
        "not_stereogenic": report.not_stereogenic,
        "unsupported": report.unsupported,
        "raw_assignments": report.raw_assignments,
        "distinct_stereoisomers": report.distinct,
        "resolution": report.resolution.text(),
        "molecule_wide_exact": report.molecule_wide_exact(),
    });
    if expand > 0 {
        let mut isomers: Vec<Value> = Vec::new();
        if report.resolution.is_resolved() {
            for assignment in &report.isomers {
                let mut tet_vals: Vec<Value> = Vec::new();
                let mut bond_vals: Vec<Value> = Vec::new();
                for (indexed, &v) in report.elements.iter().zip(assignment.iter()) {
                    match indexed.element {
                        super::stereo::StereoElement::Tetrahedral { .. } => {
                            tet_vals.push(json!(if v == 1 { "cw" } else { "ccw" }));
                        }
                        super::stereo::StereoElement::DoubleBond { .. } => {
                            bond_vals.push(json!(if v == 1 { "trans" } else { "cis" }));
                        }
                    }
                }
                isomers.push(json!({"tetrahedral": tet_vals, "double_bonds": bond_vals}));
            }
        }
        block["stereoisomers"] = Value::Array(isomers);
        block["stereoisomers_truncated"] = json!(report.isomers_truncated);
    }
    block
}
/// First 8 bytes of the SHA-256 of `id`, as a `u64` (big-endian).
fn numeric_id(id: &str) -> u64 {
    let hex = sha256_hex(id.as_bytes());
    u64::from_str_radix(&hex[..16], 16).expect("hex of a hash parses")
}

/// Canonicalize a request value for hashing: object keys sorted recursively,
/// explicitly, so `input_hash` never depends on the `serde_json`
/// `preserve_order` feature or on input whitespace. Arrays keep their order;
/// scalars pass through.
fn canonical_value(v: &Value) -> Value {
    match v {
        Value::Object(obj) => {
            let mut keys: Vec<&String> = obj.keys().collect();
            keys.sort();
            let mut out = Map::new();
            for k in keys {
                out.insert(k.clone(), canonical_value(&obj[k]));
            }
            Value::Object(out)
        }
        Value::Array(items) => Value::Array(items.iter().map(canonical_value).collect()),
        _ => v.clone(),
    }
}

/// SHA-256 over the canonical (key-sorted, whitespace-free) request JSON.
fn canonical_hash(root: &Value) -> String {
    let canon = canonical_value(root);
    let text = serde_json::to_string(&canon).expect("canonical request serializes");
    sha256_hex(text.as_bytes())
}

/// Which domain limit an `unsupported_input` response violated.
struct Unsupported {
    /// Limit name (`max_atoms`, `composition_heavy_atoms`,
    /// `max_ring_closures`).
    limit: &'static str,
    /// The limit's value.
    allowed: u64,
    /// The observed value.
    observed: u64,
}

/// Trained completion model with its sampling constants and provenance.
///
/// The model weights are `f32`, the element type of every compiled backend.
pub struct CompletionService<R: Runtime> {
    /// Trainer holding the loaded model, its configs and step count.
    trainer: CompletionTrainer<R, f32>,
    /// Resident grammar tables of the sampling kernels.
    constants: Ms2Constants<R>,
    /// Device the model lives on.
    device: Device<R>,
    /// SHA-256 (hex) of the checkpoint file bytes.
    checkpoint_sha256: String,
}

impl<R: Runtime> CompletionService<R> {
    /// The `unsupported_input` response for a request that carries spectral
    /// evidence when the model has no spectrum encoder (`None` otherwise):
    /// evidence is never silently ignored.
    fn spectrum_unsupported(
        &self,
        req: &ValidRequest,
        input_hash: &str,
        config: &super::completion_model::CompletionModelConfig,
    ) -> Option<String> {
        let evidence = req.spectrum.as_ref()?;
        if self.trainer.model().has_spectrum() {
            return None;
        }
        Some(unsupported_response(
            req,
            input_hash,
            config,
            &self.checkpoint_sha256,
            &Unsupported {
                limit: "spectrum",
                allowed: 0,
                observed: evidence.peaks.len() as u64,
            },
        ))
    }

    /// Load a checkpoint saved by
    /// [`CompletionTrainer::save`](super::completion_model::CompletionTrainer::save)
    /// onto `device`.
    ///
    /// A truncated or foreign file is an error, never a panic.
    pub fn load(checkpoint: &Path, device: &Device<R>) -> Result<Self> {
        // One read: the hash and the deserializer share these bytes, so a
        // concurrent replacement cannot make the hash describe other bytes.
        let bytes = std::fs::read(checkpoint)?;
        let checkpoint_sha256 = sha256_hex(&bytes);
        let trainer = CompletionTrainer::<R, f32>::load_bytes(&bytes, device)?;
        Ok(Self {
            trainer,
            constants: Ms2Constants::new(device),
            device: device.clone(),
            checkpoint_sha256,
        })
    }

    /// Describe the loaded model as JSON: protocol, model/grammar/data/chemistry
    /// versions, the model config, domain limits, checkpoint SHA-256 and the
    /// trained step count.
    pub fn describe(&self) -> String {
        let config = &self.trainer.model().config;
        let doc = json!({
            "protocol": GENERATE_PROTOCOL,
            "model_version": COMPLETION_MODEL_VERSION,
            "grammar_version": COMPLETION_GRAMMAR_VERSION,
            "data_version": COMPLETION_DATA_VERSION,
            "chemistry_version": CHEMISTRY_VERSION,
            "checkpoint_format": COMPLETION_CHECKPOINT_FORMAT,
            "model_config": config,
            "domain": {
                "max_atoms": config.max_atoms,
                "max_ring_closures": config.max_ring_closures,
            },
            "checkpoint_sha256": self.checkpoint_sha256,
            "trained_steps": self.trainer.step_count(),
        });
        serde_json::to_string_pretty(&doc).expect("describe JSON serializes")
    }

    /// Run one strict request and render the response JSON.
    ///
    /// Malformed JSON and schema violations are errors; a composition or
    /// substructure outside the model's domain limits is an
    /// `unsupported_input` response with no candidates, never an exception.
    /// Mass input goes through the shared mass-to-formulas-to-generation
    /// path; there is no learned formula ranker.
    pub fn generate_json(&self, request_json: &str) -> Result<String> {
        let root: Value = serde_json::from_str(request_json)
            .map_err(|e| invalid(format!("request is not valid JSON: {e}")))?;
        let req = parse_request(&root)?;
        let input_hash = canonical_hash(&root);
        if req.composition.is_some() {
            return self.generate_composition(&req, &input_hash);
        }
        self.generate_mass(&req, &input_hash)
    }

    /// Composition path: the oracle exact-composition request, unchanged.
    fn generate_composition(&self, req: &ValidRequest, input_hash: &str) -> Result<String> {
        let config = &self.trainer.model().config;
        let max_atoms = config.max_atoms as usize;
        let max_closures = config.max_ring_closures as usize;
        let composition = req.composition.expect("composition path has a composition");
        let heavy: usize = composition
            .iter()
            .enumerate()
            .filter(|(e, _)| *e != chem::HYDROGEN)
            .map(|(_, &n)| usize::from(n))
            .sum();
        if heavy > max_atoms {
            return Ok(unsupported_response(
                req,
                input_hash,
                config,
                &self.checkpoint_sha256,
                &Unsupported {
                    limit: "max_atoms",
                    allowed: u64::from(config.max_atoms),
                    observed: heavy as u64,
                },
            ));
        }
        // Patterns may overlap in the target, so their sizes never add up:
        // each single pattern must fit `max_atoms` and the composition, while
        // the total only has to fit the encoder (`MAX_PATTERNS` patterns,
        // `PATTERN_SLOTS` atoms, enforced as a schema error in
        // `parse_substructures`).
        for pattern in &req.patterns {
            let n = pattern.atoms().len();
            if n > max_atoms {
                return Ok(unsupported_response(
                    req,
                    input_hash,
                    config,
                    &self.checkpoint_sha256,
                    &Unsupported {
                        limit: "max_atoms",
                        allowed: u64::from(config.max_atoms),
                        observed: n as u64,
                    },
                ));
            }
            let pattern_heavy = pattern
                .atoms()
                .iter()
                .filter(|id| {
                    chem::atom_type(**id)
                        .map(|t| t.element != chem::HYDROGEN)
                        .unwrap_or(true)
                })
                .count();
            if pattern_heavy > heavy {
                return Ok(unsupported_response(
                    req,
                    input_hash,
                    config,
                    &self.checkpoint_sha256,
                    &Unsupported {
                        limit: "composition_heavy_atoms",
                        allowed: heavy as u64,
                        observed: pattern_heavy as u64,
                    },
                ));
            }
        }
        let pattern_closures = req
            .patterns
            .iter()
            .map(|g| g.ring_closures())
            .max()
            .unwrap_or(0);
        if pattern_closures > max_closures {
            return Ok(unsupported_response(
                req,
                input_hash,
                config,
                &self.checkpoint_sha256,
                &Unsupported {
                    limit: "max_ring_closures",
                    allowed: u64::from(config.max_ring_closures),
                    observed: pattern_closures as u64,
                },
            ));
        }
        let numeric = numeric_id(&req.id);
        let gen_config = CompletionGenerationConfig {
            trajectories: req.trajectories,
            temperature: req.temperature,
            seed: req.seed,
            returned: req.returned,
            containment_node_limit: 100_000,
            identity_work_limit: 100_000,
            condition_on_patterns: true,
            substructure_semantics: req.semantics,
        };
        // Fingerprint evidence (computed or predicted outside this crate).
        // A model without the encoder given a fingerprint is
        // `unsupported_input` naming it; a model with it given none encodes
        // an empty token set.
        let fingerprint_opt = match &req.fingerprint {
            None => None,
            Some(parsed) => {
                if self.trainer.model().fingerprint_slots() == 0 {
                    return Ok(unsupported_response(
                        req,
                        input_hash,
                        config,
                        &self.checkpoint_sha256,
                        &Unsupported {
                            limit: "fingerprint",
                            allowed: 0,
                            observed: parsed.bits.len() as u64,
                        },
                    ));
                }
                let fp = super::completion_fingerprint::SparseFingerprint::from_probabilities(
                    &parsed.bits,
                    parsed.threshold,
                )
                .map_err(|e| invalid(format!("request.fingerprint rejected: {e}")))?;
                Some(fp)
            }
        };
        if let Some(response) = self.spectrum_unsupported(req, input_hash, config) {
            return Ok(response);
        }
        let request = CompletionRequest {
            id: numeric,
            composition,
            patterns: &req.patterns,
            acceptance_patterns: None,
            fingerprint: fingerprint_opt.as_ref(),
        };
        let mut outcomes = self.trainer.model().generate_with_spectra(
            &[request],
            &[req.spectrum.as_ref()],
            &gen_config,
            &self.constants,
            &self.device,
        )?;
        let outcome = outcomes.pop().expect("one request gives one outcome");
        if let Some(reason) = &outcome.infeasible {
            return Ok(infeasible_response(
                req,
                input_hash,
                config,
                &self.checkpoint_sha256,
                reason,
            ));
        }
        Ok(supported_response(
            req,
            input_hash,
            &outcome,
            config,
            &self.checkpoint_sha256,
        ))
    }

    /// Mass path: enumerate formula hypotheses from the mass, then generate
    /// stratified over them under the fixed total budget.
    fn generate_mass(&self, req: &ValidRequest, input_hash: &str) -> Result<String> {
        let config = &self.trainer.model().config;
        let max_atoms_cfg = config.max_atoms;
        let max_closures_cfg = config.max_ring_closures;
        for pattern in &req.patterns {
            let n = pattern.atoms().len();
            if n > max_atoms_cfg as usize {
                return Ok(unsupported_response(
                    req,
                    input_hash,
                    config,
                    &self.checkpoint_sha256,
                    &Unsupported {
                        limit: "max_atoms",
                        allowed: u64::from(max_atoms_cfg),
                        observed: n as u64,
                    },
                ));
            }
        }
        let pattern_closures = req
            .patterns
            .iter()
            .map(|g| g.ring_closures())
            .max()
            .unwrap_or(0);
        if pattern_closures > max_closures_cfg as usize {
            return Ok(unsupported_response(
                req,
                input_hash,
                config,
                &self.checkpoint_sha256,
                &Unsupported {
                    limit: "max_ring_closures",
                    allowed: u64::from(max_closures_cfg),
                    observed: pattern_closures as u64,
                },
            ));
        }
        let Some(artifacts) = self.trainer.formula_artifacts() else {
            return Ok(unsupported_response(
                req,
                input_hash,
                config,
                &self.checkpoint_sha256,
                &Unsupported {
                    limit: "formula_artifacts",
                    allowed: 1,
                    observed: 0,
                },
            ));
        };
        let target = req.target_mass.as_ref().expect("mass path has a target");
        let neut = req
            .neutralization
            .as_ref()
            .expect("mass path has neutralization");
        let search = req
            .formula_search
            .as_ref()
            .expect("mass path has formula_search");
        // `source` is provenance only; read it so the field is live.
        let _ = target.source.as_str();
        let mass_query = match neut {
            Neutralization::AlreadyNeutral => MassQuery::Neutral {
                value: target.value,
                ppm_tenths: target.ppm_tenths,
                uncertainty: target.uncertainty,
            },
            Neutralization::PrecursorIon { adduct } => MassQuery::Precursor {
                value: target.value,
                adduct: *adduct,
                ppm_tenths: target.ppm_tenths,
                uncertainty: target.uncertainty,
            },
        };
        // Fingerprint evidence on the mass path (same rule as composition:
        // a model without the encoder given one is `unsupported_input`).
        let fingerprint_opt = match &req.fingerprint {
            None => None,
            Some(parsed) => {
                if self.trainer.model().fingerprint_slots() == 0 {
                    return Ok(unsupported_response(
                        req,
                        input_hash,
                        config,
                        &self.checkpoint_sha256,
                        &Unsupported {
                            limit: "fingerprint",
                            allowed: 0,
                            observed: parsed.bits.len() as u64,
                        },
                    ));
                }
                let fp = super::completion_fingerprint::SparseFingerprint::from_probabilities(
                    &parsed.bits,
                    parsed.threshold,
                )
                .map_err(|e| invalid(format!("request.fingerprint rejected: {e}")))?;
                Some(fp)
            }
        };
        if let Some(response) = self.spectrum_unsupported(req, input_hash, config) {
            return Ok(response);
        }
        let result = super::completion_formula::run_mass_completion_with_spectrum(
            self.trainer.model(),
            &self.constants,
            &self.device,
            artifacts,
            u32::from(max_atoms_cfg),
            u32::from(max_closures_cfg),
            &req.patterns,
            None,
            &mass_query,
            search.hypotheses,
            search.nodes_visited_max,
            req.trajectories,
            req.temperature,
            req.seed,
            req.returned,
            &req.id,
            true,
            search.pruning,
            search.allocation,
            req.semantics,
            fingerprint_opt.as_ref(),
            req.spectrum.as_ref(),
        )?;
        Ok(mass_response(
            req,
            input_hash,
            &result,
            config,
            &self.checkpoint_sha256,
        ))
    }
}

/// Render an `unsupported_input` response: nothing was sampled, so every
/// executed counter is 0 and `trajectories` (the executed count) is 0 while
/// `requested_trajectories` carries the request's value. `search` and
/// `ranking` are `not_evaluated`, and `unsupported` names the violated limit.
fn unsupported_response(
    req: &ValidRequest,
    input_hash: &str,
    config: &super::completion_model::CompletionModelConfig,
    checkpoint_sha256: &str,
    unsupported: &Unsupported,
) -> String {
    let doc = json!({
        "protocol": GENERATE_PROTOCOL,
        "query_id": req.id,
        "provenance": req.provenance,
        "input_hash": input_hash,
        "status": "unsupported_input",
        "substructure_semantics": semantics_echo(req),
        "model": {
            "version": COMPLETION_MODEL_VERSION,
            "grammar": COMPLETION_GRAMMAR_VERSION,
            "chemistry": CHEMISTRY_VERSION,
            "checkpoint_sha256": checkpoint_sha256,
            "max_atoms": config.max_atoms,
            "max_ring_closures": config.max_ring_closures,
        },
        "candidates": [],
        "unresolved": 0,
        "accounting": {
            "requested_trajectories": req.trajectories,
            "trajectories": 0,
            "unused_trajectories": req.trajectories,
            "finished": 0,
            "dead_end": 0,
            "truncated": 0,
            "other_status": 0,
            "rejected_replay": 0,
            "rejected_containment": 0,
            "containment_unresolved": 0,
            "identity_unresolved": 0,
            "distinct": 0,
        },
        "unsupported": {
            "limit": unsupported.limit,
            "allowed": unsupported.allowed,
            "observed": unsupported.observed,
        },
        "ranking": {
            "status": "not_evaluated",
            "reason": "no ranking was performed: the input is outside the model domain",
        },
        "mass_evidence": {
            "status": "not_evaluated",
            "reason": "composition was supplied, no mass was evaluated",
        },
        "search": {"status": "not_evaluated", "reason": "no search was performed: the input is outside the model domain"},
        "physical_verification": {
            "status": "not_evaluated",
            "reason": "no physical-verification service is connected",
        },
        "stereochemistry": stereochemistry_object(),
    });
    let doc = with_fingerprint_echo(doc, req, config);
    serde_json::to_string_pretty(&doc).expect("unsupported response serializes")
}

/// The `substructure_semantics` echo: the request's value plus its one-line
/// meaning.
fn semantics_echo(req: &ValidRequest) -> Value {
    json!({
        "value": req.semantics.as_str(),
        "meaning": req.semantics.meaning(),
    })
}

/// The `fingerprint` echo: `{"name", "tokens_used", "entries_dropped"}` when
/// the request carried a fingerprint, else `None` (the field is then absent
/// from the response, so fingerprint-less fixtures are byte-identical).
/// `tokens_used` is the selected token count at the model's slots and
/// `entries_dropped` the slot-limit loss.
fn fingerprint_echo(
    req: &ValidRequest,
    config: &super::completion_model::CompletionModelConfig,
) -> Option<Value> {
    let parsed = req.fingerprint.as_ref()?;
    let fp = super::completion_fingerprint::SparseFingerprint::from_probabilities(
        &parsed.bits,
        parsed.threshold,
    )
    .ok()?;
    let slots = config.fingerprint_slots as usize;
    let dropped = fp.dropped(slots);
    let tokens_used = fp.entries.len().saturating_sub(dropped);
    Some(json!({
        "name": "morgan4096",
        "tokens_used": tokens_used,
        "entries_dropped": dropped,
    }))
}

/// The `spectrum` echo: `{"peaks_used", "peaks_dropped", "adduct",
/// "precursor_mz_uda", "neutral_mass_uda"}` when the request carried
/// spectral evidence, else `None` (the field is then absent from the
/// response).
fn spectrum_echo(
    req: &ValidRequest,
    config: &super::completion_model::CompletionModelConfig,
) -> Option<Value> {
    let evidence = req.spectrum.as_ref()?;
    let slots = config.spectrum_slots as usize;
    Some(json!({
        "peaks_used": evidence.selected(slots).len(),
        "peaks_dropped": evidence.dropped(slots),
        "adduct": super::completion_spectrum::completion_adduct(evidence.adduct)
            .map_or("unknown", |a| a.name),
        "precursor_mz_uda": evidence.precursor_mz,
        "neutral_mass_uda": evidence.neutral_mass,
    }))
}

/// Insert the fingerprint and spectrum echoes into a response document when
/// the request carried them.
fn with_fingerprint_echo(
    mut doc: Value,
    req: &ValidRequest,
    config: &super::completion_model::CompletionModelConfig,
) -> Value {
    if let Some(echo) = fingerprint_echo(req, config) {
        if let Some(obj) = doc.as_object_mut() {
            obj.insert("fingerprint".to_string(), echo);
        }
    }
    if let Some(echo) = spectrum_echo(req, config) {
        if let Some(obj) = doc.as_object_mut() {
            obj.insert("spectrum".to_string(), echo);
        }
    }
    doc
}

/// Render an `infeasible` response: the request is well-formed but no
/// molecule with its composition can satisfy the active substructure
/// semantics (see the pre-check), so nothing was sampled. Like
/// `unsupported_input` every executed counter is 0, but the status is
/// `no_candidates` (a satisfiability verdict, not a domain rejection) with
/// the reason under `infeasible`.
fn infeasible_response(
    req: &ValidRequest,
    input_hash: &str,
    config: &super::completion_model::CompletionModelConfig,
    checkpoint_sha256: &str,
    reason: &str,
) -> String {
    let doc = json!({
        "protocol": GENERATE_PROTOCOL,
        "query_id": req.id,
        "provenance": req.provenance,
        "input_hash": input_hash,
        "status": "no_candidates",
        "substructure_semantics": semantics_echo(req),
        "model": {
            "version": COMPLETION_MODEL_VERSION,
            "grammar": COMPLETION_GRAMMAR_VERSION,
            "chemistry": CHEMISTRY_VERSION,
            "checkpoint_sha256": checkpoint_sha256,
            "max_atoms": config.max_atoms,
            "max_ring_closures": config.max_ring_closures,
        },
        "candidates": [],
        "unresolved": 0,
        "accounting": {
            "requested_trajectories": req.trajectories,
            "trajectories": 0,
            "unused_trajectories": req.trajectories,
            "finished": 0,
            "dead_end": 0,
            "truncated": 0,
            "other_status": 0,
            "rejected_replay": 0,
            "rejected_containment": 0,
            "containment_unresolved": 0,
            "identity_unresolved": 0,
            "distinct": 0,
        },
        "infeasible": {
            "reason": reason,
        },
        "ranking": {
            "status": "not_evaluated",
            "reason": "no ranking was performed: no molecule can satisfy the substructure semantics",
        },
        "mass_evidence": {
            "status": "not_evaluated",
            "reason": "composition was supplied, no mass was evaluated",
        },
        "search": {"status": "not_evaluated", "reason": "no search was performed: the composition cannot satisfy the substructure semantics"},
        "physical_verification": {
            "status": "not_evaluated",
            "reason": "no physical-verification service is connected",
        },
        "stereochemistry": stereochemistry_object(),
    });
    let doc = with_fingerprint_echo(doc, req, config);
    serde_json::to_string_pretty(&doc).expect("infeasible response serializes")
}

/// Render an `ok` / `no_candidates` response from one query outcome.
fn supported_response(
    req: &ValidRequest,
    input_hash: &str,
    outcome: &super::completion_model::QueryOutcome,
    config: &super::completion_model::CompletionModelConfig,
    checkpoint_sha256: &str,
) -> String {
    let mut candidates = Vec::with_capacity(outcome.candidates.len());
    for (i, c) in outcome.candidates.iter().enumerate() {
        let comp = c.graph.composition();
        candidates.push(json!({
            "rank": (i + 1) as u32,
            "atoms": c.graph.atoms(),
            "bonds": c.graph.bonds().iter().map(|(a, b, o)| json!([a, b, o])).collect::<Vec<_>>(),
            "composition": composition_text(&comp),
            "samples": c.samples,
            "sample_fraction": f64::from(c.samples) / f64::from(outcome.trajectories),
            "best_log_prob": c.best_log_prob,
            "stereo": stereo_block_for_graph(Some(&c.graph), req.stereo_expand, req.stereo_max_elements),
        }));
    }
    let status = if candidates.is_empty() {
        "no_candidates"
    } else {
        "ok"
    };
    let doc = json!({
        "protocol": GENERATE_PROTOCOL,
        "query_id": req.id,
        "provenance": req.provenance,
        "input_hash": input_hash,
        "status": status,
        "substructure_semantics": semantics_echo(req),
        "model": {
            "version": COMPLETION_MODEL_VERSION,
            "grammar": COMPLETION_GRAMMAR_VERSION,
            "chemistry": CHEMISTRY_VERSION,
            "checkpoint_sha256": checkpoint_sha256,
            "max_atoms": config.max_atoms,
            "max_ring_closures": config.max_ring_closures,
        },
        "candidates": candidates,
        "unresolved": outcome.unresolved.len() as u32,
        "accounting": {
            "requested_trajectories": req.trajectories,
            "trajectories": outcome.trajectories,
            "unused_trajectories": req.trajectories.saturating_sub(outcome.trajectories),
            "finished": outcome.finished,
            "dead_end": outcome.dead_end,
            "truncated": outcome.truncated,
            "other_status": outcome.other_status,
            "rejected_replay": outcome.rejected_replay,
            "rejected_containment": outcome.rejected_containment,
            "containment_unresolved": outcome.containment_unresolved,
            "identity_unresolved": outcome.identity_unresolved,
            "distinct": outcome.distinct,
        },
        "ranking": {
            "status": "sample_frequency",
            "calibrated": false,
            "reason": "ranked by sample frequency; not calibrated confidence",
        },
        "mass_evidence": {
            "status": "not_evaluated",
            "reason": "composition was supplied, no mass was evaluated",
        },
        "search": {"status": "sampled", "exhaustive": false},
        "physical_verification": {
            "status": "not_evaluated",
            "reason": "no physical-verification service is connected",
        },
        "stereochemistry": stereochemistry_object(),
    });
    let doc = with_fingerprint_echo(doc, req, config);
    serde_json::to_string_pretty(&doc).expect("supported response serializes")
}

/// Render a mass-input response from the shared pooled result.
///
/// Every candidate gains `mass` with its formula evidence; the top level
/// gains `mass_evidence` and `formula_search`. There is no learned formula
/// ranker: `formula_search.ranking` says so. `accounting` sums over the
/// formulas and keeps `requested_trajectories`.
fn mass_response(
    req: &ValidRequest,
    input_hash: &str,
    result: &super::completion_formula::MassCompletionResult,
    config: &super::completion_model::CompletionModelConfig,
    checkpoint_sha256: &str,
) -> String {
    let executed = result.accounting.trajectories;
    let mut candidates = Vec::with_capacity(result.candidates.len());
    for (i, c) in result.candidates.iter().enumerate() {
        let fraction = if executed == 0 {
            0.0
        } else {
            f64::from(c.samples) / f64::from(executed)
        };
        candidates.push(json!({
            "rank": (i + 1) as u32,
            "atoms": c.atoms,
            "bonds": c.bonds.iter().map(|(a, b, o)| json!([a, b, o])).collect::<Vec<_>>(),
            "composition": formula_text(&c.composition),
            "samples": c.samples,
            "sample_fraction": fraction,
            "best_log_prob": c.best_log_prob,
            "mass": {
                "formula": c.formula,
                "computed_uda": c.computed_uda,
                "residual_uda": c.residual_uda,
                "status": c.mass_status,
            },
            "stereo": stereo_block_for_graph(
                MolGraph::new(c.atoms.clone(), c.bonds.clone()).ok().as_ref(),
                req.stereo_expand,
                req.stereo_max_elements,
            ),
        }));
    }
    let status = if candidates.is_empty() {
        "no_candidates"
    } else {
        "ok"
    };
    let fs = &result.formula_search;
    let formulas: Vec<Value> = fs
        .formulas
        .iter()
        .map(|f| {
            let verdict = if f.ambiguous {
                "boundary_ambiguous"
            } else {
                "accepted"
            };
            let sampled = f.trajectories > 0;
            let stage = if sampled { "sampled" } else { "not_sampled" };
            json!({
                "formula": f.formula,
                "computed_uda": f.computed_uda,
                "residual_uda": f.residual_uda,
                "verdict": verdict,
                "mass_status": verdict,
                "weight": f.weight,
                "trajectories": f.trajectories,
                "finished": f.finished,
                "accepted_candidates": f.accepted_candidates,
                "sampled": sampled,
                "sampling": if sampled { "sampled" } else { "not_sampled" },
                "stage": stage,
            })
        })
        .collect();
    let en = &fs.enumerator;
    let mass_reason = match result.mass_evidence_status.as_str() {
        "accepted" => "at least one joined formula verdict is Accept",
        "boundary_ambiguous" => {
            "joined formulas verdicts are Ambiguous only; no learned formula prior"
        }
        "rejected" => "the search completed and no formula passed the mass verdict",
        "search_incomplete" => {
            "a node budget or capacity bound stopped the search, so absence proves nothing"
        }
        "mass_overflow" => {
            "precursor neutralisation left the u32 range: no search was performed and no claim is made"
        }
        "unavailable" => "unknown mass precision: no search was performed and no claim is made",
        _ => "mass evidence evaluated",
    };
    let mut mass_evidence = json!({
        "status": result.mass_evidence_status,
        "reason": mass_reason,
    });
    if let Some(terms) = &result.error_terms {
        mass_evidence["error_terms_uda"] = json!({
            "observation_uda": terms.observation,
            "composition_uda": terms.composition,
            "neutralisation_uda": terms.neutralisation,
        });
    }
    let model = json!({
        "version": COMPLETION_MODEL_VERSION,
        "grammar": COMPLETION_GRAMMAR_VERSION,
        "chemistry": CHEMISTRY_VERSION,
        "checkpoint_sha256": checkpoint_sha256,
        "max_atoms": config.max_atoms,
        "max_ring_closures": config.max_ring_closures,
    });
    let accounting = json!({
        "requested_trajectories": req.trajectories,
        "trajectories": result.accounting.trajectories,
        "unused_trajectories": req.trajectories.saturating_sub(result.accounting.trajectories),
        "finished": result.accounting.finished,
        "dead_end": result.accounting.dead_end,
        "truncated": result.accounting.truncated,
        "other_status": result.accounting.other_status,
        "rejected_replay": result.accounting.rejected_replay,
        "rejected_containment": result.accounting.rejected_containment,
        "containment_unresolved": result.accounting.containment_unresolved,
        "identity_unresolved": result.accounting.identity_unresolved,
        "distinct": result.accounting.distinct,
    });
    // F11: the ranking metadata names the rule actually used. Under
    // `equal` candidates rank by sample frequency; under `train_frequency`
    // they rank by the explicit estimate `weight * samples / trajectories`
    // of the source formula first (ties as in the equal order).
    let ranking = match fs.allocation.as_str() {
        "train_frequency" => json!({
            "status": "train_frequency_weighted_estimate",
            "calibrated": false,
            "reason": "ranked by the explicit estimate weight * samples / trajectories of the source formula (training-frequency prior times empirical trajectory hit rate); not calibrated confidence",
        }),
        _ => json!({
            "status": "sample_frequency",
            "calibrated": false,
            "reason": "ranked by sample frequency; not calibrated confidence",
        }),
    };
    let search = json!({"status": "sampled", "exhaustive": false});
    let enumerator = json!({
        "nodes_visited": en.nodes_visited,
        "hydrogen_checks": en.hydrogen_checks,
        "rows_joined": en.rows_joined,
        "rows_scored": en.rows_scored,
        "rejected_h_max": en.rejected_h_max,
        "rejected_parity": en.rejected_parity,
        "rejected_dbe": en.rejected_dbe,
        "rejected_mass": en.rejected_mass,
        "pruned_ratio_cap": en.pruned_ratio_cap,
        "pruned_rare": en.pruned_rare,
        "rejected_ratio_cap": en.rejected_ratio_cap,
        "rejected_rare": en.rejected_ratio_rare,
        "rejected_ratio_hc": en.rejected_ratio_hc,
        "rejected_ratio_nc": en.rejected_ratio_nc,
        "rejected_ratio_oc": en.rejected_ratio_oc,
        "rejected_ratio_hal": en.rejected_ratio_hal,
        "rejected_ratio_s": en.rejected_ratio_s,
        "rejected_ratio_p": en.rejected_ratio_p,
        "rejected_ratio_dbe": en.rejected_ratio_dbe,
        "exhausted": en.exhausted,
    });
    let stages: Vec<Value> = fs
        .stages
        .iter()
        .map(|s| {
            json!({
                "stage": s.stage,
                "class": s.class,
                "entering": s.entering,
                "leaving": s.leaving,
                "note": s.note,
            })
        })
        .collect();
    let formula_search = json!({
        "status": fs.status,
        "truncated": fs.truncated,
        "search_exhausted": fs.search_exhausted,
        "unsampled_reason": fs.unsampled_reason,
        "pruning": fs.pruning,
        "allocation": fs.allocation,
        "joined": fs.joined,
        "joined_chemical": fs.joined_chemical,
        "joined_chemical_reason": fs.joined_chemical_reason,
        "excluded_by_train_fit": fs.excluded_by_train_fit,
        "after_domain": fs.after_domain,
        "after_substructures": fs.after_substructures,
        "after_completability": fs.after_completability,
        "selected": fs.selected,
        "sampled": fs.sampled,
        "ranking": fs.ranking,
        "stages": stages,
        "formulas": formulas,
        "enumerator": enumerator,
    });
    let physical = json!({
        "status": "not_evaluated",
        "reason": "no physical-verification service is connected",
    });
    let mut doc = Map::new();
    doc.insert("protocol".to_string(), json!(GENERATE_PROTOCOL));
    doc.insert("query_id".to_string(), json!(req.id));
    doc.insert("provenance".to_string(), json!(req.provenance));
    doc.insert("input_hash".to_string(), json!(input_hash));
    doc.insert("status".to_string(), json!(status));
    doc.insert("substructure_semantics".to_string(), semantics_echo(req));
    doc.insert("model".to_string(), model);
    doc.insert("candidates".to_string(), Value::Array(candidates));
    doc.insert(
        "unresolved".to_string(),
        json!(result.accounting.unresolved),
    );
    doc.insert("accounting".to_string(), accounting);
    doc.insert("ranking".to_string(), ranking);
    doc.insert("mass_evidence".to_string(), mass_evidence);
    doc.insert("search".to_string(), search);
    doc.insert("formula_search".to_string(), formula_search);
    doc.insert("physical_verification".to_string(), physical);
    doc.insert("stereochemistry".to_string(), stereochemistry_object());
    if let Some(echo) = fingerprint_echo(req, config) {
        doc.insert("fingerprint".to_string(), echo);
    }
    if let Some(echo) = spectrum_echo(req, config) {
        doc.insert("spectrum".to_string(), echo);
    }
    serde_json::to_string_pretty(&Value::Object(doc)).expect("mass response serializes")
}
