#!/usr/bin/env bash
# The one reproducible check for this repository.
#
#   tools/check.sh              # fmt, clippy -D warnings, CPU tests, CPU wheel + pytest
#   tools/check.sh --wgpu       # ...and the wgpu Rust suite and wgpu wheel + pytest
#   tools/check.sh --no-python  # skip the wheels and pytest
#
# Every step's exit code is checked (never a pipeline's), its full log is kept,
# and the Rust suites end with a per-group table and every skip reason the tests
# printed. The run fails on the first failing step, after printing that step's
# log tail; the wgpu suite runs with --no-fail-fast so its table is complete.
#
# Environment: MATURIN, PYTHON (see tools/build_wheel.sh), CHECK_LOG_DIR (default:
# a fresh temporary directory, printed at the start).
set -euo pipefail

wgpu=0
python_suite=1
for arg in "$@"; do
    case "$arg" in
        --wgpu) wgpu=1 ;;
        --no-python) python_suite=0 ;;
        *)
            echo "usage: $0 [--wgpu] [--no-python]" >&2
            exit 2
            ;;
    esac
done

root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root"
logs=${CHECK_LOG_DIR:-$(mktemp -d)}
mkdir -p "$logs"
echo "logs: $logs"
echo "revision: $(git rev-parse HEAD)$(git diff --quiet || echo ' (+ uncommitted changes)')"

step() {
    local name=$1
    shift
    local log="$logs/$name.log"
    local start=$SECONDS
    printf '==> %-22s' "$name"
    set +e
    "$@" >"$log" 2>&1
    local code=$?
    set -e
    if [[ $code -eq 0 ]]; then
        echo "ok ($((SECONDS - start)) s)"
    else
        echo "FAILED, exit $code ($((SECONDS - start)) s); log: $log"
        tail -n 60 "$log"
        exit "$code"
    fi
}

# Per test binary: passed / failed / ignored, then every skip reason printed.
summarise() {
    local log=$1
    awk '
        /^     Running / { name = $2; sub(/^.*\//, "", name); sub(/\.rs$/, "", name) }
        /^   Doc-tests / { name = "doc-tests" }
        /^test result:/ {
            gsub(/;/, "")
            printf "    %-26s %4d passed %3d failed %3d ignored\n", name, $4, $6, $8
            passed += $4; failed += $6; groups += 1
        }
        END { printf "    %-26s %4d passed %3d failed across %d groups\n", "total", passed, failed, groups }
    ' "$log"
    grep -hE '^(skipped|test .* \.\.\. skipped)' "$log" | sort | uniq -c | sed 's/^/    /' || true
}

step fmt cargo fmt --check
step fmt-bindings bash -c 'cd bindings/python && cargo fmt --check'
step clippy cargo clippy --no-default-features --features cpu --all-targets -- -D warnings
step clippy-bindings bash -c 'cd bindings/python && cargo clippy --no-default-features --features cpu --all-targets -- -D warnings'
step test-cpu cargo test --release --no-default-features --features cpu -- --nocapture
summarise "$logs/test-cpu.log"

if [[ $wgpu -eq 1 ]]; then
    set +e
    cargo test --release --no-default-features --features wgpu --no-fail-fast -- --nocapture \
        >"$logs/test-wgpu.log" 2>&1
    code=$?
    set -e
    echo "==> test-wgpu              exit $code"
    summarise "$logs/test-wgpu.log"
    [[ $code -eq 0 ]] || exit "$code"
fi

if [[ $python_suite -eq 1 ]]; then
    for backend in cpu $([[ $wgpu -eq 1 ]] && echo wgpu); do
        step "wheel-$backend" tools/build_wheel.sh "$backend" "$logs/wheels-$backend" --smoke
        venv=$(grep '^smoke ok: ' "$logs/wheel-$backend.log" | tail -1 | sed 's/^smoke ok: //')
        step "pytest-$backend" bash -c "'$venv/bin/pip' install --quiet pytest numpy && cd bindings/python && '$venv/bin/python' -m pytest tests -q -rs"
        grep -E '^[0-9]+ (passed|failed)|passed|SKIPPED' "$logs/pytest-$backend.log" | tail -n 20 | sed 's/^/    /'
    done
fi

echo "all checks passed"
