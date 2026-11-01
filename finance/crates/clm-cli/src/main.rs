//! `mamba-clm`: walk-forward CV, single-fold training, inference, reports,
//! LightGBM export and kernel benchmarks for the Mamba-CLM study.
//!
//! ```text
//! mamba-clm cv      --config finance/configs/main.toml --out finance/results/runs/main
//! mamba-clm train   --config ... --out ... --model mamba_clm --year 2020
//! mamba-clm report  --config ... --run finance/results/runs/main
//! mamba-clm infer   --config ... --run ... --model mamba_clm --date 2024-03-01
//! mamba-clm export-lgbm --config ... --out finance/results/lgbm_export
//! mamba-clm folds   --config ...
//! mamba-clm bench   --out finance/results/bench
//! ```

mod config;
mod cv;
mod data;
mod report;
mod tools;

use std::path::PathBuf;

use clap::{Parser, Subcommand};

use crate::config::ExperimentConfig;

#[derive(Parser)]
#[command(name = "mamba-clm", about = "Mamba encoder + contrastive decision learning: CV, training, inference")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Run walk-forward CV for every model of a config (resumable).
    Cv {
        #[arg(long)]
        config: PathBuf,
        #[arg(long)]
        out: PathBuf,
        /// Only these models (comma-separated).
        #[arg(long, value_delimiter = ',')]
        models: Vec<String>,
        /// Only these test years (comma-separated).
        #[arg(long, value_delimiter = ',')]
        years: Vec<i32>,
    },
    /// Train one model on one fold (a filtered `cv`).
    Train {
        #[arg(long)]
        config: PathBuf,
        #[arg(long)]
        out: PathBuf,
        #[arg(long)]
        model: String,
        #[arg(long)]
        year: i32,
    },
    /// Aggregate a run into report.json / report.md.
    Report {
        #[arg(long)]
        config: PathBuf,
        #[arg(long)]
        run: PathBuf,
        /// Report even if some configured model has no finished fold.
        #[arg(long)]
        allow_partial: bool,
    },
    /// Decisions for every stock on one date.
    Infer {
        #[arg(long)]
        config: PathBuf,
        #[arg(long)]
        run: PathBuf,
        #[arg(long)]
        model: String,
        /// `yyyy-mm-dd`, a trading day.
        #[arg(long)]
        date: String,
    },
    /// Export per-fold sample tables for an external baseline (LightGBM).
    ExportLgbm {
        #[arg(long)]
        config: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
    /// Print the folds of a config.
    Folds {
        #[arg(long)]
        config: PathBuf,
    },
    /// Training throughput and inference latency of every model of a config.
    BenchModel {
        #[arg(long)]
        config: PathBuf,
        #[arg(long)]
        out: PathBuf,
        #[arg(long, default_value_t = 100)]
        steps: u64,
    },
    /// Selective-scan kernel benchmarks (T = 64 .. 1024).
    Bench {
        #[arg(long)]
        out: PathBuf,
        #[arg(long, default_value_t = 32)]
        batch: usize,
        #[arg(long, default_value_t = 256)]
        dim: usize,
        #[arg(long, default_value_t = 16)]
        state: usize,
        #[arg(long, default_value_t = 10)]
        reps: usize,
        /// Interleaved A/B of the backward's reduction interval (2 vs 8 steps).
        #[arg(long)]
        ab_flush: bool,
    },
}

fn main() {
    let cli = Cli::parse();
    let res = match cli.command {
        Command::Cv { config, out, models, years } => {
            ExperimentConfig::load(&config).and_then(|c| cv::run(&c, &out, &models, &years))
        }
        Command::Train { config, out, model, year } => {
            ExperimentConfig::load(&config).and_then(|c| cv::run(&c, &out, &[model], &[year]))
        }
        Command::Report { config, run, allow_partial } => {
            ExperimentConfig::load(&config).and_then(|c| report::run(&c, &run, allow_partial))
        }
        Command::Infer { config, run, model, date } => {
            ExperimentConfig::load(&config).and_then(|c| tools::infer(&c, &run, &model, &date))
        }
        Command::ExportLgbm { config, out } => ExperimentConfig::load(&config).and_then(|c| tools::export_lgbm(&c, &out)),
        Command::Folds { config } => ExperimentConfig::load(&config).and_then(|c| {
            let ds = data::Dataset::load(&c)?;
            for f in clm_eval::folds::make_folds(&ds.years, ds.dates.len(), &c.fold_config()) {
                println!("{}", clm_eval::folds::render_folds_line(&f));
            }
            Ok(())
        }),
        Command::BenchModel { config, out, steps } => {
            ExperimentConfig::load(&config).and_then(|c| tools::bench_model(&c, &out, steps))
        }
        Command::Bench { out, batch, dim, state, reps, ab_flush } => tools::bench(&out, batch, dim, state, reps, ab_flush),
    };
    if let Err(e) = res {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}
