//! No `#[cube]` function in the crate spells an infinity or a NaN.
//!
//! CubeCL writes a float constant into the shader as `f32(<value>)`, so a
//! non-finite one becomes `f32(inf)` or `f32(NaN)` — not WGSL. The kernel then
//! fails to compile on wgpu. `backend::check_launches` makes that failure loud
//! wherever it happens; this scan is the backstop that stops the literal being
//! written in the first place, before anyone has run the wgpu suite.
//!
//! Non-finite values a kernel genuinely needs come in as scalar arguments — see
//! `distributions::univariate::NonFinite`.
//!
//! What it cannot see: a literal handed to a macro that pastes it into a kernel
//! it generates. The generated kernels are scanned, but the literal is only in
//! the macro's invocation, which is not a `#[cube]` body.

use std::path::{Path, PathBuf};

/// Files whose `#[cube]`-looking source is compiled for the host only.
///
/// The `_host.rs` twins are generated from the shared regions with `#[cube]`
/// stripped, so they contain no attributes to find; they are listed so the
/// allowance is written down rather than implied.
const HOST_ONLY_SUFFIX: &str = "_host.rs";

/// Identifiers and calls that produce a non-finite float constant.
const FORBIDDEN_IDENTS: [&str; 3] = ["INFINITY", "NEG_INFINITY", "NAN"];
const FORBIDDEN_CALLS: [&str; 3] = ["infinity()", "neg_infinity()", "nan()"];

fn rust_files(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(dir).expect("source directory is readable") {
        let path = entry.expect("directory entry").path();
        if path.is_dir() {
            rust_files(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
}

/// The source with `//` comments blanked, so prose about infinities in a
/// kernel's comments is not mistaken for code. Line structure is kept, so
/// reported line numbers stay right.
fn strip_line_comments(source: &str) -> String {
    source
        .lines()
        .map(|line| match line.find("//") {
            Some(at) => &line[..at],
            None => line,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn is_ident_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

/// Every forbidden token in `body`, as `(byte offset, token)`.
fn forbidden_tokens(body: &str) -> Vec<(usize, &'static str)> {
    let bytes = body.as_bytes();
    let mut found = Vec::new();
    for token in FORBIDDEN_IDENTS {
        let mut from = 0;
        while let Some(pos) = body[from..].find(token) {
            let at = from + pos;
            let end = at + token.len();
            let bounded_left = at == 0 || !is_ident_byte(bytes[at - 1]);
            let bounded_right = end == bytes.len() || !is_ident_byte(bytes[end]);
            if bounded_left && bounded_right {
                found.push((at, token));
            }
            from = end;
        }
    }
    for token in FORBIDDEN_CALLS {
        let mut from = 0;
        while let Some(pos) = body[from..].find(token) {
            let at = from + pos;
            if at == 0 || !is_ident_byte(body.as_bytes()[at - 1]) {
                found.push((at, token));
            }
            from = at + token.len();
        }
    }
    found
}

/// `(line, function name, token)` for every forbidden token inside a function
/// annotated `#[cube]` or `#[cube(...)]`.
fn violations(source: &str) -> Vec<(usize, String, &'static str)> {
    let code = strip_line_comments(source);
    let mut out = Vec::new();
    for (open, name, body) in kernels(&code) {
        for (offset, token) in forbidden_tokens(body) {
            let line = code[..open + offset].matches('\n').count() + 1;
            out.push((line, name.clone(), token));
        }
    }
    out
}

/// `(byte offset of the body, function name, body)` for every function annotated
/// `#[cube]` or `#[cube(...)]` in comment-stripped `code`.
fn kernels(code: &str) -> Vec<(usize, String, &str)> {
    let mut out = Vec::new();
    let mut search = 0;
    while let Some(pos) = code[search..].find("#[cube") {
        let attr = search + pos;
        search = attr + "#[cube".len();
        // `#[cube]` or `#[cube(`, not an attribute that merely starts the same way.
        if !matches!(code.as_bytes().get(search), Some(b']') | Some(b'(')) {
            continue;
        }
        let Some(fn_rel) = code[search..].find("fn ") else {
            break;
        };
        let name_start = search + fn_rel + "fn ".len();
        let name: String = code[name_start..]
            .chars()
            .take_while(|c| c.is_alphanumeric() || *c == '_' || *c == '$')
            .collect();
        let Some(open_rel) = code[name_start..].find('{') else {
            break;
        };
        let open = name_start + open_rel;
        let mut depth = 0usize;
        let mut close = code.len();
        for (i, b) in code.as_bytes()[open..].iter().enumerate() {
            match b {
                b'{' => depth += 1,
                b'}' => {
                    depth -= 1;
                    if depth == 0 {
                        close = open + i;
                        break;
                    }
                }
                _ => {}
            }
        }
        out.push((open, name, &code[open..close]));
        search = close;
    }
    out
}

/// The kernels a training step launches besides the model's own are in the
/// scan's reach: a kernel the scan never parses is one it cannot vouch for.
#[test]
fn the_scan_reaches_the_optimizer_kernels() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/tensor/ops/fused.rs");
    let code = strip_line_comments(&std::fs::read_to_string(path).expect("fused.rs is readable"));
    let names: Vec<String> = kernels(&code).into_iter().map(|(_, n, _)| n).collect();
    for kernel in ["adamw_kernel", "ema_kernel"] {
        assert!(
            names.iter().any(|n| n == kernel),
            "{kernel} not found among {names:?}"
        );
    }
}

#[test]
fn no_kernel_spells_a_non_finite_literal() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut files = Vec::new();
    rust_files(&root, &mut files);
    files.sort();
    assert!(files.len() > 20, "found only {} source files", files.len());

    let mut report = Vec::new();
    for path in &files {
        if path.to_string_lossy().ends_with(HOST_ONLY_SUFFIX) {
            continue;
        }
        let source = std::fs::read_to_string(path).expect("source is readable");
        for (line, name, token) in violations(&source) {
            report.push(format!(
                "{}:{line}: `{token}` in #[cube] fn {name}",
                path.strip_prefix(env!("CARGO_MANIFEST_DIR"))
                    .unwrap_or(path)
                    .display()
            ));
        }
    }
    assert!(
        report.is_empty(),
        "non-finite constants in kernel code do not compile on WGSL; pass them in as \
         scalar arguments instead (see distributions::univariate::NonFinite):\n{}",
        report.join("\n")
    );
}

/// The scan itself finds what it is meant to, and only that.
#[test]
fn the_scan_finds_literals_in_kernels_and_nowhere_else() {
    let source = r#"
        #[cube]
        pub fn bad(x: f32) -> f32 {
            let mut out = x;
            if x < 0.0 {
                out = f32::NEG_INFINITY; // and a comment saying f32::NAN
            }
            out
        }

        #[cube(launch_unchecked)]
        fn also_bad<F: Float>(out: &mut Array<F>) {
            out[0] = F::nan();
        }

        #[cube]
        fn fine(x: f32, nf: NonFinite) -> f32 {
            // f32::INFINITY is only mentioned here.
            x + nf.inf + NANOSECONDS_PER_DAY
        }

        fn host_only() -> f32 {
            f32::INFINITY
        }
    "#;
    let found: Vec<(String, &str)> = violations(source)
        .into_iter()
        .map(|(_, name, token)| (name, token))
        .collect();
    assert_eq!(
        found,
        vec![
            ("bad".to_string(), "NEG_INFINITY"),
            ("also_bad".to_string(), "nan()"),
        ]
    );
}
