//! Strict versioned JSON request layer for whole-target neutral completion
//! (`docs/MOLECULAR_COMPLETION_DESIGN.md`, bounded reference MVP).
//!
//! This layer validates one explicit request — protocol
//! `molecular-completion-request-v1`, nonempty id/provenance, a declared
//! `mass_role` of `target_molecule` under the `already_neutral`
//! convention, a microdalton target mass with explicit precision
//! metadata, a pinned `completion-bounded-v1` domain (C/N/O, 2..=6
//! heavy atoms, at most one ring closure), and V0-typed substructures
//! with provenance and confirmed certainty — then delegates all
//! chemistry, search, budgets and status semantics to the existing
//! audit adapter ([`super::completion::run_request_json`]) rather than
//! duplicating graph matching or mass arithmetic.
//!
//! Anything outside the bounded MVP is rejected with an actionable
//! error instead of being silently weakened: unknown parent hydrogen
//! semantics, tentative patterns, fragment or neutral-loss mass roles,
//! any supplied precursor mass, unknown fields and wrong types.
//! Correspondence keeps the audit's `None` = unknown-overlap versus
//! supplied `[]` = known-disjoint distinction. Ranking and physical
//! verification are always `not_evaluated`; identity ordering is the
//! deterministic baseline order, not calibrated confidence.

use serde_json::{Map, Value, json};

use crate::error::{Error, Result};

use super::chem;
use super::completion::{self, COMPLETION_VERSION, stable_hash};
use super::grammar::{self, Limits, Token};

/// Protocol string required at the top of every request.
pub const REQUEST_PROTOCOL: &str = "molecular-completion-request-v1";

fn unsupported(msg: impl Into<String>) -> Error {
    Error::Unsupported(format!("unsupported_request: {}", msg.into()))
}

fn require_object<'a>(v: &'a Value, what: &str) -> Result<&'a Map<String, Value>> {
    v.as_object()
        .ok_or_else(|| unsupported(format!("{what} must be an object")))
}

fn require_string<'a>(v: &'a Value, what: &str) -> Result<&'a str> {
    match v.as_str() {
        Some(s) if !s.is_empty() => Ok(s),
        Some(_) => Err(unsupported(format!("{what} must be a non-empty string"))),
        None => Err(unsupported(format!("{what} must be a string"))),
    }
}

fn require_u32(v: &Value, what: &str) -> Result<u32> {
    match v.as_u64() {
        Some(n) if n <= u64::from(u32::MAX) => Ok(n as u32),
        Some(_) => Err(unsupported(format!("{what} overflows u32"))),
        None => Err(unsupported(format!("{what} must be a non-negative integer"))),
    }
}

fn require_u64(v: &Value, what: &str) -> Result<u64> {
    v.as_u64()
        .ok_or_else(|| unsupported(format!("{what} must be a non-negative integer")))
}

/// Reject keys outside `allowed`; never guess semantics for extras.
fn reject_unknown_keys(obj: &Map<String, Value>, allowed: &[&str], what: &str) -> Result<()> {
    for k in obj.keys() {
        if !allowed.iter().any(|a| a == k) {
            return Err(unsupported(format!(
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
        Some(_) => Err(unsupported(format!("{what}.{key} must not be null"))),
        None => Err(unsupported(format!("{what} is missing required field '{key}'"))),
    }
}

pub(crate) fn composition_text(c: &chem::Composition) -> String {
    let suffix = |symbol: &str, count: u16| {
        if count == 1 {
            symbol.to_string()
        } else {
            format!("{symbol}{count}")
        }
    };
    let mut parts = String::new();
    if c[0] > 0 {
        parts.push_str(&suffix("C", c[0]));
    }
    parts.push_str(&suffix("H", c[chem::HYDROGEN]));
    if c[2] > 0 {
        parts.push_str(&suffix("N", c[2]));
    }
    if c[3] > 0 {
        parts.push_str(&suffix("O", c[3]));
    }
    parts
}

/// Decode one canonical identity token string through the existing
/// grammar replay into its typed graph, plus its composition text.
fn decode_identity(identity: &str, limits: Limits) -> Result<Value> {
    let mut toks: Vec<Token> = Vec::new();
    for part in identity.split(',') {
        let mut fields = part.split('/');
        let parse = |f: Option<&str>, what: &str| -> Result<u8> {
            f.and_then(|s| s.parse::<u8>().ok())
                .ok_or_else(|| unsupported(format!("identity token {what} is not a byte")))
        };
        toks.push(Token {
            kind: parse(fields.next(), "kind")?,
            atom_type: parse(fields.next(), "atom_type")?,
            bond: parse(fields.next(), "bond")?,
            pointer: parse(fields.next(), "pointer")?,
        });
    }
    let state = grammar::replay(&toks, limits, None)
        .map_err(|e| unsupported(format!("identity does not replay: {e}")))?;
    let mol = state
        .graph()
        .map_err(|e| unsupported(format!("identity does not yield a graph: {e}")))?;
    Ok(json!({
        "identity": identity,
        "atoms": mol.atoms(),
        "bonds": mol.bonds().iter().map(|&(a, b, o)| json!([a, b, o])).collect::<Vec<_>>(),
        "composition": composition_text(&mol.composition()),
    }))
}

/// Validate the strict request and translate it into the bounded audit's
/// fixture-shaped query JSON. Returns `(query_json, domain_limits)`.
fn parse_request(root: &Value) -> Result<(Value, Limits)> {
    let obj = require_object(root, "request")?;
    reject_unknown_keys(
        obj,
        &[
            "protocol",
            "id",
            "provenance",
            "mass_role",
            "neutralization",
            "target_mass",
            "domain",
            "substructures",
            "correspondence",
            "precursor",
            "budgets",
            "seed",
        ],
        "request",
    )?;
    let protocol = require_string(take(obj, "protocol", "request")?, "request.protocol")?;
    if protocol != REQUEST_PROTOCOL {
        return Err(unsupported(format!(
            "request.protocol must be '{REQUEST_PROTOCOL}', got '{protocol}'"
        )));
    }
    let id = require_string(take(obj, "id", "request")?, "request.id")?.to_string();
    let provenance =
        require_string(take(obj, "provenance", "request")?, "request.provenance")?.to_string();

    let mass_role = require_string(take(obj, "mass_role", "request")?, "request.mass_role")?;
    match mass_role {
        "target_molecule" => {}
        "target_fragment" | "neutral_loss" => {
            return Err(unsupported(format!(
                "mass_role '{mass_role}' is rejected: precursor conventions and fragment evidence are not supported in this bounded MVP; declare mass_role 'target_molecule' or omit precursor fields"
            )));
        }
        other => {
            return Err(unsupported(format!(
                "mass_role must be 'target_molecule', 'target_fragment' or 'neutral_loss'; got '{other}'"
            )));
        }
    }
    let neutralization = require_string(
        take(obj, "neutralization", "request")?,
        "request.neutralization",
    )?;
    if neutralization != "already_neutral" {
        return Err(unsupported(format!(
            "neutralization must be 'already_neutral'; got '{neutralization}'. Ion-derived masses need an explicit neutralization convention, which this bounded MVP does not implement"
        )));
    }

    // Any precursor mass at all is rejected until an explicit supported
    // relation exists; never silently ignored.
    if let Some(p) = obj.get("precursor") {
        if p.is_null() {
            // Explicit absence is fine.
        } else {
            let _ = require_object(p, "request.precursor")?;
            return Err(unsupported(
                "request.precursor is rejected: explicit precursor/target mass relations are not supported in this bounded MVP; remove precursor fields",
            ));
        }
    }

    let mass = require_object(take(obj, "target_mass", "request")?, "request.target_mass")?;
    reject_unknown_keys(
        mass,
        &["units", "value", "ppm_tenths", "uncertainty_uda", "source"],
        "request.target_mass",
    )?;
    let units = require_string(take(mass, "units", "target_mass")?, "target_mass.units")?;
    if units != "microdalton" {
        return Err(unsupported(format!(
            "target_mass.units must be 'microdalton', got '{units}'"
        )));
    }
    let observed = require_u64(take(mass, "value", "target_mass")?, "target_mass.value")?;
    if observed > u64::from(u32::MAX) {
        return Err(unsupported("target_mass.value overflows u32"));
    }
    let ppm_tenths = require_u32(take(mass, "ppm_tenths", "target_mass")?, "target_mass.ppm_tenths")?;
    if ppm_tenths > 1000 {
        return Err(unsupported(format!(
            "target_mass.ppm_tenths {ppm_tenths} exceeds the 1000 proof bound"
        )));
    }
    let uncertainty = match mass.get("uncertainty_uda") {
        None => {
            return Err(unsupported(
                "target_mass is missing required field 'uncertainty_uda' (use null for unknown)",
            ));
        }
        Some(Value::Null) => None,
        Some(v) => Some(require_u32(v, "target_mass.uncertainty_uda")?),
    };
    let source = require_string(take(mass, "source", "target_mass")?, "target_mass.source")?;

    let domain = require_object(take(obj, "domain", "request")?, "request.domain")?;
    reject_unknown_keys(
        domain,
        &["version", "elements", "min_heavy", "max_heavy", "max_ring_closures"],
        "request.domain",
    )?;
    let dversion = require_string(take(domain, "version", "domain")?, "domain.version")?;
    if dversion != COMPLETION_VERSION {
        return Err(unsupported(format!(
            "domain.version must be '{COMPLETION_VERSION}', got '{dversion}'"
        )));
    }
    let elements_v = take(domain, "elements", "domain")?
        .as_array()
        .ok_or_else(|| unsupported("domain.elements must be a list"))?;
    let mut elements = Vec::new();
    for e in elements_v {
        let s = e
            .as_str()
            .ok_or_else(|| unsupported("domain.elements entries must be strings"))?;
        let idx = chem::element_index(s).ok_or_else(|| {
            unsupported(format!(
                "domain.elements must be a subset of C/N/O; got '{s}'"
            ))
        })?;
        if !matches!(idx, 0 | 2 | 3) {
            return Err(unsupported(format!(
                "domain.elements must be a subset of C/N/O; got '{s}'"
            )));
        }
        elements.push(s.to_string());
    }
    if elements.is_empty() {
        return Err(unsupported("domain.elements must be non-empty"));
    }
    let min_heavy = require_u32(
        take(domain, "min_heavy", "domain")?,
        "domain.min_heavy",
    )? as usize;
    let max_heavy = require_u32(
        take(domain, "max_heavy", "domain")?,
        "domain.max_heavy",
    )? as usize;
    let max_ring_closures = require_u32(
        take(domain, "max_ring_closures", "domain")?,
        "domain.max_ring_closures",
    )? as usize;
    if min_heavy < 2 || min_heavy > max_heavy || max_heavy > 6 {
        return Err(unsupported(format!(
            "domain heavy-atom bounds must satisfy 2 <= min_heavy <= max_heavy <= 6; got {min_heavy}..={max_heavy}"
        )));
    }
    if max_ring_closures > 1 {
        return Err(unsupported(format!(
            "domain.max_ring_closures must be at most 1 in the bounded MVP; got {max_ring_closures}"
        )));
    }

    let subs = take(obj, "substructures", "request")?
        .as_array()
        .ok_or_else(|| unsupported("request.substructures must be a list"))?;
    let mut patterns = Vec::new();
    for (i, s) in subs.iter().enumerate() {
        let sobj = require_object(s, &format!("substructures[{i}]"))?;
        reject_unknown_keys(
            sobj,
            &["atoms", "bonds", "parent_hydrogen_semantics", "provenance", "certainty"],
            &format!("substructures[{i}]"),
        )?;
        let atoms = take(sobj, "atoms", "substructure")?
            .as_array()
            .ok_or_else(|| unsupported(format!("substructures[{i}].atoms must be a list")))?;
        let mut atoms_v = Vec::new();
        for a in atoms {
            atoms_v.push(json!(require_u64(a, "substructure.atoms entry")?));
        }
        let bonds = take(sobj, "bonds", "substructure")?
            .as_array()
            .ok_or_else(|| unsupported(format!("substructures[{i}].bonds must be a list")))?;
        let mut bonds_v = Vec::new();
        for b in bonds {
            let barray = b
                .as_array()
                .ok_or_else(|| unsupported(format!("substructures[{i}].bonds entries must be lists")))?;
            if barray.len() != 3 {
                return Err(unsupported(format!(
                    "substructures[{i}].bonds entries must have 3 fields"
                )));
            }
            bonds_v.push(json!([
                require_u64(&barray[0], "bond endpoint")?,
                require_u64(&barray[1], "bond endpoint")?,
                require_u64(&barray[2], "bond order")?
            ]));
        }
        let semantics = require_string(
            take(sobj, "parent_hydrogen_semantics", "substructure")?,
            &format!("substructures[{i}].parent_hydrogen_semantics"),
        )?;
        if semantics != "v0_parent_hydrogen_counts" {
            return Err(unsupported(format!(
                "substructures[{i}].parent_hydrogen_semantics '{semantics}' is unknown: this bounded MVP only accepts V0 atom types carrying parent hydrogen counts ('v0_parent_hydrogen_counts'); unknown parent H semantics are rejected rather than treated as hard constraints"
            )));
        }
        let sprov = require_string(
            take(sobj, "provenance", "substructure")?,
            &format!("substructures[{i}].provenance"),
        )?;
        let certainty = require_string(
            take(sobj, "certainty", "substructure")?,
            &format!("substructures[{i}].certainty"),
        )?;
        if certainty == "tentative" {
            return Err(unsupported(format!(
                "substructures[{i}].certainty is 'tentative': tentative patterns are soft evidence and are not supported as hard constraints in this bounded MVP; confirm the pattern or remove it"
            )));
        }
        if certainty != "confirmed" {
            return Err(unsupported(format!(
                "substructures[{i}].certainty must be 'confirmed'; got '{certainty}'"
            )));
        }
        let _ = sprov;
        patterns.push(json!({"atoms": atoms_v, "bonds": bonds_v}));
    }

    // Correspondence: absent => unknown overlap (None); present must be a
    // list, with [] meaning known-disjoint (Some([])).
    let correspondence = match obj.get("correspondence") {
        None => None,
        Some(Value::Null) => {
            return Err(unsupported(
                "request.correspondence must be a list of pairs or absent; null is rejected (absent already means unknown overlap)",
            ));
        }
        Some(v) => Some(
            v.as_array()
                .ok_or_else(|| unsupported("request.correspondence must be a list"))?
                .clone(),
        ),
    };

    if let Some(b) = obj.get("budgets") {
        let bobj = require_object(b, "request.budgets")?;
        reject_unknown_keys(
            bobj,
            &[
                "formula_visits",
                "graph_extensions",
                "embedding_nodes",
                "canonical_expansions",
                "retained_graphs",
                "memory_bytes",
                "watchdog_ms",
            ],
            "request.budgets",
        )?;
        for k in [
            "formula_visits",
            "graph_extensions",
            "embedding_nodes",
            "canonical_expansions",
            "retained_graphs",
            "memory_bytes",
            "watchdog_ms",
        ] {
            if let Some(v) = bobj.get(k) {
                let n = require_u64(v, &format!("budgets.{k}"))?;
                if n == 0 {
                    return Err(unsupported(format!("budgets.{k} must be positive")));
                }
            }
        }
    }
    if let Some(s) = obj.get("seed") {
        require_u64(s, "request.seed")?;
    }

    let mut query = json!({
        "name": id,
        "provenance": provenance,
        "observed_mass_uda": observed,
        "mass": {
            "ppm_tenths": ppm_tenths,
            "uncertainty_uda": match uncertainty {
                Some(u) => json!(u),
                None => Value::Null,
            },
            "source": source,
        },
        "domain": {
            "elements": elements,
            "min_heavy": min_heavy,
            "max_heavy": max_heavy,
            "max_ring_closures": max_ring_closures,
        },
        "patterns": patterns,
    });
    if let Some(c) = correspondence {
        query["correspondence"] = Value::Array(c);
    }
    let limits = Limits::new(max_heavy, max_ring_closures)?;
    Ok((query, limits))
}

/// Run one strict request end to end and render the bounded audit
/// response. See the module docs for the accepted shape.
pub fn molecular_completion_run(text: &str) -> Result<String> {
    let root: Value = serde_json::from_str(text)
        .map_err(|e| unsupported(format!("request is not valid JSON: {e}")))?;
    let (query, limits) = parse_request(&root)?;
    let mut adapter_request = query.clone();
    if let Some(b) = root.get("budgets") {
        adapter_request["budgets"] = b.clone();
    }
    if let Some(s) = root.get("seed") {
        adapter_request["seed"] = s.clone();
    }
    let report_json = completion::run_request_json(&adapter_request.to_string())?;
    let report: Value = serde_json::from_str(&report_json)
        .map_err(|e| Error::Config(format!("audit report is not JSON: {e}")))?;
    let decoded: Vec<Value> = report["accepted_identities"]
        .as_array()
        .map(|ids| {
            ids.iter()
                .filter_map(|v| v.as_str())
                .map(|id| match decode_identity(id, limits) {
                    Ok(d) => d,
                    Err(e) => json!({"identity": id, "decode_error": e.to_string()}),
                })
                .collect()
        })
        .unwrap_or_default();
    let input_hash = format!("{:016x}", stable_hash(&[text]));
    let response = json!({
        "protocol": REQUEST_PROTOCOL,
        "query_id": report["query_id"].clone(),
        "provenance": report["provenance"].clone(),
        "input_hash": input_hash,
        "ranking": {
            "status": "not_evaluated",
            "reason": "no trainable Mamba conditioning model or calibrated scoring exists in this bounded reference MVP; identity order is the deterministic baseline enumeration order, not calibrated confidence"
        },
        "physical_verification": {
            "status": "not_evaluated",
            "reason": "no physical-verification service is connected in this bounded reference MVP"
        },
        "identity_ordering": "deterministic baseline order; not calibrated confidence",
        "audit": report,
        "decoded_graphs": decoded,
    });
    Ok(serde_json::to_string_pretty(&response)?)
}
