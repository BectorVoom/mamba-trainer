//! Run [`MotifMachine`](mamba3::models::ms2::motif::MotifMachine) on the
//! text files of the Lean reference executable.
//!
//! `lean/MotifDecoder` holds the motif stack machine in Lean 4 with proofs of
//! what it guarantees, and an executable, `motifcheck`, that runs the proven
//! definitions. This example reads the same files and prints the same lines
//! (`mamba3::models::ms2::motif::text` documents the formats), so the Rust
//! machine can be compared with the proven one by `diff`:
//!
//! ```text
//! ms2_motif_check <vocab-file> <sequences-file> [<budgets-file>]
//! ```

#![cfg(feature = "backend")]

use std::io::Write;

use mamba3::error::Result;
use mamba3::models::ms2::motif::{Formula, text};

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 2 || args.len() > 3 {
        eprintln!("usage: ms2_motif_check <vocab-file> <sequences-file> [<budgets-file>]");
        std::process::exit(2);
    }
    let vocab = text::vocab(&std::fs::read_to_string(&args[0])?)?;
    let sequences = std::fs::read_to_string(&args[1])?;
    let budgets: Option<Vec<Option<Formula>>> = match args.get(2) {
        Some(path) => Some(
            std::fs::read_to_string(path)?
                .lines()
                .map(text::budget)
                .collect::<Result<_>>()?,
        ),
        None => None,
    };
    let mut out = std::io::BufWriter::new(std::io::stdout().lock());
    for (i, line) in sequences.lines().enumerate() {
        let budget = budgets.as_ref().and_then(|list| list.get(i).copied().flatten());
        writeln!(out, "{}", text::check(&vocab, budget.as_ref(), line))?;
    }
    out.flush()?;
    Ok(())
}
