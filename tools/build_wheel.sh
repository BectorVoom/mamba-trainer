#!/usr/bin/env bash
# Build the mamba3_rl wheel for one backend, portable by default.
#
#   tools/build_wheel.sh cpu  [out_dir]    # --auditwheel=repair: vendors libzstd
#   tools/build_wheel.sh wgpu [out_dir]
#   tools/build_wheel.sh cpu  [out_dir] --smoke   # then import it in a fresh venv
#
# Why a script: the CPU runtime's code generator links Homebrew's libzstd, so a
# CPU wheel built without --auditwheel=repair imports on the machine that built
# it and nowhere else. The flag used to live only in the bindings README.
#
# --smoke installs the wheel into a new virtual environment and fails if the
# extension links any library outside the system and the wheel's own vendored
# directory, or if it cannot be imported with the dynamic-library search paths
# cleared. That is what turns a reintroduced external link into a failure here
# rather than on someone else's machine.
#
# Environment: MATURIN (default: maturin on PATH), PYTHON (default: python3).
set -euo pipefail

usage() {
    echo "usage: $0 cpu|wgpu [out_dir] [--smoke]" >&2
    exit 2
}

[[ $# -ge 1 ]] || usage
backend=$1
shift
out_dir=""
smoke=0
for arg in "$@"; do
    case "$arg" in
        --smoke) smoke=1 ;;
        -*) usage ;;
        *) out_dir=$arg ;;
    esac
done

root=$(cd "$(dirname "$0")/.." && pwd)
maturin=${MATURIN:-maturin}
python=${PYTHON:-python3}
out_dir=${out_dir:-"$root/target/wheels-$backend"}
mkdir -p "$out_dir"

case "$backend" in
    cpu) repair=(--auditwheel=repair) ;;
    wgpu) repair=() ;;
    *) usage ;;
esac

command -v "$maturin" >/dev/null || {
    echo "maturin not found (set MATURIN=/path/to/maturin)" >&2
    exit 1
}

# A stale wheel for another revision would be picked up by the glob below.
rm -f "$out_dir"/mamba3_rl-*.whl
(
    cd "$root/bindings/python"
    "$maturin" build --release --no-default-features --features "$backend" \
        --interpreter "$python" "${repair[@]}" -o "$out_dir"
)
wheel=$(ls "$out_dir"/mamba3_rl-*.whl)
echo "built $wheel"
shasum -a 256 "$wheel"

[[ $smoke -eq 1 ]] || exit 0

venv=$(mktemp -d)/venv
"$python" -m venv "$venv"
"$venv/bin/pip" install --quiet "$wheel"
site=$("$venv/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
extension=$(ls "$site"/mamba3_rl/_mamba3_rl*.so)

if command -v otool >/dev/null; then
    # Everything the extension links must be the system's, the Python
    # framework's, or relative to the wheel itself.
    external=$(otool -L "$extension" | tail -n +2 | awk '{print $1}' |
        grep -vE '^(/usr/lib/|/System/Library/|@rpath/|@loader_path/|@executable_path/)' || true)
    if [[ -n "$external" ]]; then
        echo "FAIL: $extension links libraries outside the wheel:" >&2
        echo "$external" >&2
        exit 1
    fi
elif command -v ldd >/dev/null; then
    missing=$(ldd "$extension" | grep "not found" || true)
    if [[ -n "$missing" ]]; then
        echo "FAIL: $extension has unresolved libraries:" >&2
        echo "$missing" >&2
        exit 1
    fi
fi

env -u DYLD_LIBRARY_PATH -u DYLD_FALLBACK_LIBRARY_PATH -u LD_LIBRARY_PATH \
    DYLD_FALLBACK_LIBRARY_PATH=/usr/lib \
    "$venv/bin/python" -c 'import mamba3_rl; print("imported mamba3_rl, backend", mamba3_rl.backend())'
echo "smoke ok: $venv"
