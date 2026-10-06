//! Python surface for the bounded molecular-completion audit.
//!
//! The JSON-in/JSON-out functions are intentionally thin: all
//! semantics — formulas, budgets, statuses, counters, recovery — live in
//! `mamba3::models::ms2::completion`, and Python only marshals the
//! request/response. There is no Python-side enumerator that could drift.

use pyo3::prelude::*;

use mamba3::models::ms2::completion as comp;
use mamba3::models::ms2::completion_request;

use crate::err::to_py;

/// Protocol version of the bounded audit.
#[pyfunction]
fn completion_protocol() -> &'static str {
    comp::COMPLETION_VERSION
}

/// Run one audit request from its JSON serialization and return the report
/// as pretty-printed JSON. The request mirrors the fixture/request shape of
/// `fixtures.json` (observed_mass_uda, mass, domain, patterns,
/// correspondence, reference) plus optional `budgets` and `seed`
/// (`formula_visits`, `ggraph_extensions`, `embedding_nodes`,
/// `canonical_expansions`, `retained_graphs`, `memory_bytes`,
/// `watchdog_ms`). Invalid input JSON or a malformed query raises
/// `ValueError`; an audit status of `unsupported_input` is carried in the
/// report itself, never as an exception.
#[pyfunction]
#[pyo3(signature = (request_json))]
fn completion_run(request_json: &str) -> PyResult<String> {
    comp::run_request_json(request_json).map_err(to_py)
}

/// Run one strict `molecular-completion-request-v1` request and return
/// the bounded response JSON (nested audit report, protocol, input hash,
/// `not_evaluated` ranking/physical verification, decoded typed
/// graphs). Malformed requests raise `ValueError` with an actionable
/// message; an audit-level `unsupported_input` status is carried in the
/// nested report, never as an exception.
#[pyfunction]
#[pyo3(signature = (request_json))]
fn molecular_completion_run(request_json: &str) -> PyResult<String> {
    completion_request::molecular_completion_run(request_json).map_err(to_py)
}

/// Load a fixture-set document and return `[(name, report_json)]`.
#[pyfunction]
#[pyo3(signature = (fixture_json, seed = 1))]
fn completion_run_fixture_set(fixture_json: &str, seed: u64) -> PyResult<Vec<(String, String)>> {
    let fixtures = comp::load_fixture_set(fixture_json).map_err(to_py)?;
    let budgets = comp::CompletionBudgets::default();
    let mut out = Vec::new();
    for fixture in fixtures {
        let report = comp::run(&fixture.query, &budgets, seed).map_err(to_py)?;
        out.push((
            fixture.name.clone(),
            serde_json::to_string(&report).map_err(|e| to_py(e.into()))?,
        ));
    }
    Ok(out)
}

/// Register the completion functions on `_mamba3_rl`.
pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(completion_protocol, module)?)?;
    module.add_function(wrap_pyfunction!(completion_run, module)?)?;
    module.add_function(wrap_pyfunction!(
        molecular_completion_run,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(completion_run_fixture_set, module)?)?;
    Ok(())
}
