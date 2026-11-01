//! Differential tests against the Lean `clmcheck` oracle (finance/lean/ClmVerify).
//!
//! Protocol: one command per stdin line; for each command the oracle prints its output
//! lines and then `end`. Commands used here:
//!
//! * `folds N scheme train_days first_test last_test val_days cal_days h e min_train_days
//!   nyears y1 c1 y2 c2 ...` → one `fold ...` line per emitted fold;
//! * `backtest n a1 .. an` → `pos ...` and `trade ...`.
//!
//! These tests are ignored by default (they need `lake build` in finance/lean/ClmVerify)
//! and fail loudly when the binary or fixtures are missing.

use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};

use clm_data::types::Action;
use clm_eval::backtest::{render_backtest, render_backtest_command};
use clm_eval::folds::{FoldConfig, Scheme, make_folds, render_folds, render_folds_command};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

fn lean_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../lean/ClmVerify")
}

fn binary() -> PathBuf {
    let p = lean_dir().join(".lake/build/bin/clmcheck");
    assert!(
        p.is_file(),
        "Lean oracle binary not found at {} — run `lake build` in finance/lean/ClmVerify",
        p.display()
    );
    p
}

/// Run all commands through the oracle in one process; returns one output group per
/// command (lines before each `end`, trailing whitespace trimmed).
fn run_oracle(commands: &[String]) -> Vec<Vec<String>> {
    let mut child = Command::new(binary())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn clmcheck");
    let mut stdin = child.stdin.take().expect("stdin");
    let input = commands.join("\n") + "\n";
    let writer = std::thread::spawn(move || {
        stdin
            .write_all(input.as_bytes())
            .expect("write to clmcheck");
        // dropping stdin closes it so the oracle sees EOF
    });
    let out = child.wait_with_output().expect("clmcheck output");
    writer.join().expect("writer thread");
    assert!(
        out.status.success(),
        "clmcheck failed: {}\n{}",
        out.status,
        String::from_utf8_lossy(&out.stderr)
    );
    groups(&String::from_utf8(out.stdout).expect("utf8"))
}

fn groups(text: &str) -> Vec<Vec<String>> {
    let mut all = Vec::new();
    let mut cur = Vec::new();
    for line in text.lines() {
        let line = line.trim_end();
        if line == "end" {
            all.push(std::mem::take(&mut cur));
        } else {
            cur.push(line.to_string());
        }
    }
    assert!(cur.is_empty(), "output ends without `end`: {cur:?}");
    all
}

/// Parse a `folds` command back into a calendar and config; `Err` if malformed (the
/// oracle must then answer with an `error` line).
fn parse_folds(cmd: &str) -> Result<(Vec<i32>, usize, FoldConfig), String> {
    let v: Vec<i64> = cmd
        .split_whitespace()
        .skip(1)
        .map(|x| x.parse().map_err(|_| format!("bad token {x}")))
        .collect::<Result<_, _>>()?;
    if v.len() < 11 {
        return Err("too few numbers".into());
    }
    let nonneg = |i: usize| -> Result<usize, String> {
        usize::try_from(v[i]).map_err(|_| format!("field {i} negative"))
    };
    let n = nonneg(0)?;
    let scheme = match v[1] {
        0 => Scheme::Expanding,
        1 => Scheme::Rolling {
            train_days: nonneg(2)?,
        },
        s => return Err(format!("bad scheme {s}")),
    };
    let cfg = FoldConfig {
        scheme,
        first_test_year: v[3] as i32,
        last_test_year: v[4] as i32,
        val_days: nonneg(5)?,
        cal_days: nonneg(6)?,
        horizon: nonneg(7)?,
        embargo: nonneg(8)?,
        min_train_days: nonneg(9)?,
    };
    let ny = nonneg(10)?;
    if v.len() != 11 + 2 * ny {
        return Err("wrong number of year runs".into());
    }
    let mut years = Vec::with_capacity(n);
    for k in 0..ny {
        let y = v[11 + 2 * k] as i32;
        if k > 0 && v[11 + 2 * k] <= v[9 + 2 * k] {
            return Err("years not increasing".into());
        }
        years.extend(std::iter::repeat_n(y, nonneg(12 + 2 * k)?));
    }
    if years.len() != n {
        return Err("year counts do not sum to N".into());
    }
    Ok((years, n, cfg))
}

fn parse_backtest(cmd: &str) -> Result<Vec<Action>, String> {
    let v: Vec<usize> = cmd
        .split_whitespace()
        .skip(1)
        .map(|x| x.parse().map_err(|_| format!("bad token {x}")))
        .collect::<Result<_, _>>()?;
    if v.is_empty() || v.len() != 1 + v[0] {
        return Err("wrong number of actions".into());
    }
    v[1..]
        .iter()
        .map(|&a| Action::from_id(a).ok_or(format!("bad action {a}")))
        .collect()
}

/// Rust's answer to a command: `None` for commands this crate does not implement,
/// `Some(Err)` for malformed ones.
fn rust_answer(cmd: &str) -> Option<Result<Vec<String>, String>> {
    match cmd.split_whitespace().next() {
        Some("folds") => {
            Some(parse_folds(cmd).map(|(years, n, cfg)| render_folds(&make_folds(&years, n, &cfg))))
        }
        Some("backtest") => Some(parse_backtest(cmd).map(|a| render_backtest(&a).to_vec())),
        _ => None,
    }
}

fn random_folds_command(rng: &mut StdRng) -> String {
    let y0 = rng.random_range(1995..2010);
    let nyears = rng.random_range(1..9);
    let mut years = Vec::new();
    for k in 0..nyears {
        let c = if rng.random_bool(0.1) {
            0
        } else {
            rng.random_range(1..40)
        };
        years.extend(std::iter::repeat_n(y0 + k, c));
    }
    let scheme = if rng.random_bool(0.5) {
        Scheme::Expanding
    } else {
        Scheme::Rolling {
            train_days: rng.random_range(0..80),
        }
    };
    let first = y0 + rng.random_range(-1..nyears + 1);
    let cfg = FoldConfig {
        scheme,
        first_test_year: first,
        last_test_year: first + rng.random_range(-1..nyears + 1),
        val_days: rng.random_range(0..15),
        cal_days: rng.random_range(0..15),
        horizon: rng.random_range(1..10),
        embargo: rng.random_range(0..6),
        min_train_days: rng.random_range(0..50),
    };
    render_folds_command(&years, years.len(), &cfg)
}

fn random_backtest_command(rng: &mut StdRng) -> String {
    let n = rng.random_range(0..30);
    let acts: Vec<Action> = (0..n)
        .map(|_| Action::from_id(rng.random_range(0..3)).unwrap())
        .collect();
    render_backtest_command(&acts)
}

fn compare(
    commands: &[String],
    lean: &[Vec<String>],
    rust: impl Fn(&str) -> Option<Result<Vec<String>, String>>,
) -> usize {
    assert_eq!(
        lean.len(),
        commands.len(),
        "oracle answered {} commands, sent {}",
        lean.len(),
        commands.len()
    );
    let mut compared = 0;
    for (cmd, got) in commands.iter().zip(lean) {
        match rust(cmd) {
            Some(Ok(want)) => assert_eq!(&want, got, "Rust and Lean disagree on `{cmd}`"),
            Some(Err(why)) => assert!(
                got.len() == 1 && got[0].starts_with("error"),
                "Rust rejects `{cmd}` ({why}) but Lean answered {got:?}"
            ),
            None => continue,
        }
        compared += 1;
    }
    compared
}

#[test]
#[ignore = "needs finance/lean/ClmVerify built (lake build)"]
fn random_commands_agree_with_lean() {
    let mut rng = StdRng::seed_from_u64(0xC1A5_0001);
    let mut cmds = Vec::new();
    for _ in 0..500 {
        cmds.push(random_folds_command(&mut rng));
    }
    for _ in 0..500 {
        cmds.push(random_backtest_command(&mut rng));
    }
    let lean = run_oracle(&cmds);
    let n = compare(&cmds, &lean, rust_answer);
    assert_eq!(n, 1000);
    // make sure the random folds are not all trivially empty
    let nonempty = lean[..500].iter().filter(|g| !g.is_empty()).count();
    assert!(
        nonempty > 100,
        "only {nonempty} folds commands emitted a fold"
    );
}

#[test]
#[ignore = "needs finance/lean/ClmVerify built (lake build)"]
fn lean_fixture_agrees_with_rust() {
    let dir = lean_dir().join("test");
    let cases_path = dir.join("cases.txt");
    let expected_path = dir.join("expected.txt");
    let cases = std::fs::read_to_string(&cases_path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", cases_path.display()));
    let expected = std::fs::read_to_string(&expected_path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", expected_path.display()));
    let cmds: Vec<String> = cases
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .map(String::from)
        .collect();
    let want = groups(&expected);
    let n = compare(&cmds, &want, rust_answer);
    assert!(n > 0, "the fixture holds no folds/backtest command");
    // and the binary reproduces its own fixture
    if lean_dir().join(".lake/build/bin/clmcheck").is_file() {
        assert_eq!(
            run_oracle(&cmds),
            want,
            "clmcheck output differs from expected.txt"
        );
    }
}

#[test]
fn command_rendering_round_trips() {
    // Not ignored: the renderer and the parser above agree with each other.
    let mut rng = StdRng::seed_from_u64(5);
    for _ in 0..200 {
        let c = random_folds_command(&mut rng);
        let (years, n, cfg) = parse_folds(&c).unwrap();
        assert_eq!(render_folds_command(&years, n, &cfg), c);
        let b = random_backtest_command(&mut rng);
        assert_eq!(render_backtest_command(&parse_backtest(&b).unwrap()), b);
    }
    assert_eq!(
        groups("fold a\nfold b\nend\nend\npos 1\ntrade 1 \nend\n"),
        vec![
            vec!["fold a".to_string(), "fold b".to_string()],
            vec![],
            vec!["pos 1".to_string(), "trade 1".to_string()]
        ]
    );
}
