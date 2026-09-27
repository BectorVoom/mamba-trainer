//! Fingerprint the kernel sources for the compiled-kernel cache.
//!
//! CubeCL keys its on-disk kernel cache by the kernel's type and launch-time
//! (comptime) arguments, not by the kernel's body: after a kernel is edited, a
//! cache written by the old build serves the old binary. `MAMBA3_SRC_HASH` (a hash
//! of every file under `src/`) names the cache directory
//! [`backend::default_kernel_cache`] installs, so a rebuilt library starts a
//! fresh cache.

use std::path::Path;

fn visit(dir: &Path, files: &mut Vec<std::path::PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            visit(&path, files);
        } else {
            files.push(path);
        }
    }
}

fn main() {
    println!("cargo:rerun-if-changed=src");
    let mut files = Vec::new();
    visit(Path::new("src"), &mut files);
    files.sort();
    // FNV-1a over (path, contents) pairs.
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
    println!("cargo:rustc-env=MAMBA3_SRC_HASH={hash:016x}");
}
