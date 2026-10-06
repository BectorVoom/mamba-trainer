//! Real-data host/device audit of the exact-completion replay kernel.
//!
//! Usage:
//! ```text
//! cargo run --release --no-default-features --features cpu \
//!   --example molecular_completion_grammar_audit -- \
//!   --export <export.json> [--limit N] [--max-atoms 32] [--max-closures 6] [--batch 256]
//! ```
//!
//! Loads the kept canonical examples with
//! [`CompletionSet::load`](mamba3::models::ms2::completion_data::CompletionSet::load),
//! replays every canonical trace on the device with
//! [`grammar_replay`](mamba3::tensor::ops::ms2::grammar_replay) under budget
//! flag 2 and compares every word of `replay` (`[rows, T, 4 + A]`) and `atoms`
//! (`[rows, A + 1]`) against the host twin
//! ([`replay_rows`](mamba3::models::ms2::twin::replay_rows)). Prints
//! aggregates only (the data is CC BY-NC) and exits non-zero on any mismatch.

use std::path::PathBuf;
use std::time::Instant;

use mamba3::backend::{Device, check_launches};
use mamba3::backends::Auto;
use mamba3::models::ms2::completion_data::CompletionSet;
use mamba3::models::ms2::grammar::{CANONICAL_WORK_LIMIT, Limits};
use mamba3::models::ms2::twin;
use mamba3::tensor::ops::index::IdTensor;
use mamba3::tensor::ops::ms2::{self, Ms2Constants, ReplayBuffers};

type R = Auto;

fn usage() -> ! {
    eprintln!(
        "usage: molecular_completion_grammar_audit --export <export.json> [--limit N] \
         [--max-atoms 32] [--max-closures 6] [--batch 256]"
    );
    std::process::exit(2);
}

fn fail(msg: String) -> ! {
    eprintln!("molecular_completion_grammar_audit: {msg}");
    std::process::exit(1);
}

fn main() {
    let mut export: Option<PathBuf> = None;
    let mut limit: Option<usize> = None;
    let mut max_atoms = 32usize;
    let mut max_closures = 6usize;
    let mut batch = 256usize;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        let mut next = || args.next().unwrap_or_else(|| usage());
        match arg.as_str() {
            "--export" => export = Some(PathBuf::from(next())),
            "--limit" => limit = Some(next().parse().unwrap_or_else(|_| usage())),
            "--max-atoms" => max_atoms = next().parse().unwrap_or_else(|_| usage()),
            "--max-closures" => max_closures = next().parse().unwrap_or_else(|_| usage()),
            "--batch" => batch = next().parse().unwrap_or_else(|_| usage()),
            _ => usage(),
        }
    }
    let Some(export) = export else { usage() };
    if batch == 0 {
        usage();
    }
    let limits = Limits::new(max_atoms, max_closures).unwrap_or_else(|e| fail(format!("{e}")));
    let set = CompletionSet::load(&export, limits, CANONICAL_WORK_LIMIT)
        .unwrap_or_else(|e| fail(format!("{e}")));
    let keep = limit.map_or(set.examples.len(), |n| n.min(set.examples.len()));
    let examples = &set.examples[..keep];
    let device = Device::<R>::default();
    let constants = Ms2Constants::new(&device);
    let started = Instant::now();
    let mut molecules = 0usize;
    let mut tokens = 0usize;
    let mut mismatching_rows = 0usize;
    let mut illegal_count = 0usize;
    let mut first_mismatch: Option<(usize, usize, usize, u32, u32)> = None;
    let mut global_row = 0usize;
    for chunk in examples.chunks(batch.max(1)) {
        let rows = chunk.len();
        let t = chunk.iter().map(|e| e.trace.len()).max().unwrap_or(0);
        let mut flat = vec![0u32; rows * t * 4];
        let mut meta = vec![0u32; rows * 12];
        for (r, example) in chunk.iter().enumerate() {
            for (s, tok) in example.trace.iter().enumerate() {
                flat[(r * t + s) * 4] = u32::from(tok.kind);
                flat[(r * t + s) * 4 + 1] = u32::from(tok.atom_type);
                flat[(r * t + s) * 4 + 2] = u32::from(tok.bond);
                flat[(r * t + s) * 4 + 3] = u32::from(tok.pointer);
            }
            meta[r * 12] = example.trace.len() as u32;
            meta[r * 12 + 1] = 2;
            for (e, count) in example.composition.iter().enumerate() {
                meta[r * 12 + 2 + e] = u32::from(*count);
            }
        }
        let tokens_t = IdTensor::from_slice(&flat, vec![rows, t, 4], &device)
            .unwrap_or_else(|e| fail(format!("{e}")));
        let meta_t = IdTensor::from_slice(&meta, vec![rows, 12], &device)
            .unwrap_or_else(|e| fail(format!("{e}")));
        let out = ReplayBuffers::poisoned(rows, t, max_atoms, &device)
            .unwrap_or_else(|e| fail(format!("{e}")));
        ms2::grammar_replay(
            &tokens_t,
            &meta_t,
            &constants,
            max_atoms as u32,
            max_closures as u32,
            &out,
        )
        .unwrap_or_else(|e| fail(format!("{e}")));
        check_launches(&device).unwrap_or_else(|e| fail(format!("{e}")));
        let replay = out
            .replay
            .try_to_vec()
            .unwrap_or_else(|e| fail(format!("{e}")));
        let atoms = out
            .atoms
            .try_to_vec()
            .unwrap_or_else(|e| fail(format!("{e}")));
        for (r, example) in chunk.iter().enumerate() {
            let host = twin::replay_rows(&example.trace, limits, example.composition, 2, max_atoms);
            molecules += 1;
            tokens += example.trace.len();
            if host.first_illegal != u32::MAX {
                illegal_count += 1;
            }
            let length = example.trace.len();
            let mut row_bad = host.first_illegal != u32::MAX;
            for s in 0..t {
                let base = (r * t + s) * (4 + max_atoms);
                if s >= length {
                    for w in 0..4 + max_atoms {
                        let got = replay[base + w];
                        if got != 0 {
                            row_bad = true;
                            if first_mismatch.is_none() {
                                first_mismatch = Some((global_row + r, s, w, 0, got));
                            }
                        }
                    }
                    continue;
                }
                if s >= host.kinds.len() {
                    // Host stopped early at the first illegal step: the device
                    // writes zeros at and past it.
                    for w in 0..4 + max_atoms {
                        let got = replay[base + w];
                        if got != 0 {
                            row_bad = true;
                            if first_mismatch.is_none() {
                                first_mismatch = Some((global_row + r, s, w, 0, got));
                            }
                        }
                    }
                    continue;
                }
                let want_masks = [
                    host.kinds[s],
                    host.types[s],
                    host.bonds[s],
                    host.pointers[s],
                ];
                for w in 0..4 {
                    let got = replay[base + w];
                    if got != want_masks[w] {
                        row_bad = true;
                        if first_mismatch.is_none() {
                            first_mismatch = Some((global_row + r, s, w, want_masks[w], got));
                        }
                    }
                }
                for j in 0..max_atoms {
                    let got = replay[base + 4 + j];
                    if got != host.resids[s][j] {
                        row_bad = true;
                        if first_mismatch.is_none() {
                            first_mismatch =
                                Some((global_row + r, s, 4 + j, host.resids[s][j], got));
                        }
                    }
                }
            }
            for j in 0..max_atoms {
                let got = atoms[r * (max_atoms + 1) + j];
                if got != host.add_steps[j] {
                    row_bad = true;
                    if first_mismatch.is_none() {
                        first_mismatch = Some((global_row + r, t, j, host.add_steps[j], got));
                    }
                }
            }
            let got_illegal = atoms[r * (max_atoms + 1) + max_atoms];
            if got_illegal != host.first_illegal {
                row_bad = true;
                if first_mismatch.is_none() {
                    first_mismatch = Some((
                        global_row + r,
                        t,
                        max_atoms,
                        host.first_illegal,
                        got_illegal,
                    ));
                }
            }
            if row_bad {
                mismatching_rows += 1;
            }
        }
        global_row += rows;
    }
    let elapsed = started.elapsed().as_secs_f64();
    let rows_per_second = if elapsed > 0.0 {
        molecules as f64 / elapsed
    } else {
        0.0
    };
    println!("molecules   {molecules}");
    println!("tokens      {tokens}");
    println!("mismatching_rows {mismatching_rows}");
    match first_mismatch {
        Some((row, step, word, want, got)) => {
            println!("first_mismatch row {row} step {step} word {word} host {want} device {got}");
        }
        None => {
            println!("first_mismatch none");
        }
    }
    println!("illegal_non_max {illegal_count}");
    println!("wall_s      {elapsed:.3}");
    println!("rows_per_s  {rows_per_second:.1}");
    if mismatching_rows > 0 || illegal_count > 0 {
        std::process::exit(1);
    }
}
