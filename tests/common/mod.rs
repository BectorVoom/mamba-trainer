//! Shared helpers for integration tests (task F7 Part C).
//!
//! Included by each test target with:
//!
//! ```ignore
//! #[path = "common/mod.rs"]
//! mod common;
//! ```
//!
//! This file itself is not a test target (cargo only builds `tests/*.rs`
//! directly).

use std::path::PathBuf;

/// Resolve the built example binary `name` relative to the running test
/// executable (task F7 Part C): `std::env::current_exe()` is
/// `<target>/<profile>/deps/<test>-<hash>`, so the example is
/// `<target>/<profile>/examples/<name>` — independent of the target
/// directory, profile or backend feature the suite was built with. When
/// cargo provides `CARGO_BIN_EXE_<name>` that wins; a couple of
/// manifest-relative guesses follow for hand-rolled layouts. If the binary
/// is absent, fail with a message naming the looked-for path and the cargo
/// command (with the SAME features this test was built with) to build it.
pub fn resolve_example_bin(name: &str) -> PathBuf {
    if name == "ms2_experiment"
        && let Some(p) = option_env!("CARGO_BIN_EXE_ms2_experiment")
    {
        let p = PathBuf::from(p);
        if p.exists() {
            return p;
        }
    }
    let exe = std::env::current_exe().expect("test binary path");
    // `<target>/<profile>/deps/<test>-<hash>` → `<target>/<profile>`.
    let profile_dir = exe
        .parent()
        .and_then(|d| d.parent())
        .map(|d| d.to_path_buf());
    let mut candidates: Vec<PathBuf> = Vec::new();
    if let Some(dir) = profile_dir {
        candidates.push(dir.join("examples").join(name));
    }
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let target = std::env::var("CARGO_TARGET_DIR").unwrap_or_else(|_| "target".to_string());
    candidates.push(manifest.join(&target).join("release/examples").join(name));
    candidates.push(manifest.join(&target).join("debug/examples").join(name));
    for cand in &candidates {
        if cand.exists() {
            return cand.clone();
        }
    }
    let looked = candidates
        .first()
        .cloned()
        .unwrap_or_else(|| manifest.join("examples").join(name));
    let features = if cfg!(feature = "wgpu") {
        "--features wgpu"
    } else if cfg!(feature = "cpu") {
        "--no-default-features --features cpu"
    } else {
        "--features backend"
    };
    panic!(
        "example binary not built (looked for {}): build the examples with the same target directory and features: CARGO_TARGET_DIR={} cargo build --release {} --examples",
        looked.display(),
        target,
        features
    );
}
