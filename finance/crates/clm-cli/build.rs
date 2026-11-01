//! `CLM_SOURCE_HASH`: FNV-1a over every source file this binary is built from
//! (the four study crates and the core crate's `src/`), recorded in each run's
//! signature so a resumed or reported run cannot silently mix code versions.

use std::path::{Path, PathBuf};

fn visit(dir: &Path, files: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            visit(&path, files);
        } else if path.extension().is_some_and(|e| e == "rs" || e == "toml") {
            files.push(path);
        }
    }
}

fn main() {
    let roots = [
        "../clm-data/src",
        "../clm-eval/src",
        "../clm-model/src",
        "src",
        "../../../src",
    ];
    let mut files = Vec::new();
    for r in roots {
        println!("cargo:rerun-if-changed={r}");
        visit(Path::new(r), &mut files);
    }
    files.sort();
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for file in &files {
        let mut eat = |bytes: &[u8]| {
            for byte in bytes {
                hash ^= u64::from(*byte);
                hash = hash.wrapping_mul(0x0100_0000_01b3);
            }
        };
        eat(file.to_string_lossy().as_bytes());
        eat(&std::fs::read(file).unwrap_or_default());
    }
    println!("cargo:rustc-env=CLM_SOURCE_HASH={hash:016x}");
}
