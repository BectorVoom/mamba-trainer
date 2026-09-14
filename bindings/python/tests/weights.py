"""Weights read back through the checkpoint writer, and identified independently.

`fingerprint` re-implements `StateDict::fingerprint` (FNV-1a over every entry's
path, shape as `u64` LE and `f32` bits LE, in path order) from the file alone, so
a test comparing it with `Policy.fingerprint()` checks the binding against the
format rather than against itself.
"""

import json
import struct

import numpy as np

FNV_OFFSET = 0xCBF29CE484222325
FNV_PRIME = 0x00000100000001B3
MASK = (1 << 64) - 1


def entries(path):
    """`{name: float32 array}` from a JSON checkpoint's weights."""
    return section(path, "state")


def section(path, key):
    """`{name: float32 array}` from one `StateDict` section of a JSON checkpoint."""
    with open(path) as f:
        data = json.load(f)
    return {
        name: np.array(entry["data"], dtype=np.float32).reshape(entry["shape"])
        for name, entry in data[key]["entries"].items()
    }


def fingerprint(weights):
    """`StateDict::fingerprint` of `{name: float32 array}`."""
    digest = FNV_OFFSET

    def feed(data):
        nonlocal digest
        for byte in data:
            digest ^= byte
            digest = (digest * FNV_PRIME) & MASK

    # `StateDict` is a `BTreeMap<String, _>`: byte order, which for these ASCII
    # paths is Python's string order.
    for name in sorted(weights, key=lambda n: n.encode()):
        value = np.asarray(weights[name], dtype=np.float32)
        feed(name.encode())
        for dim in value.shape:
            feed(struct.pack("<Q", dim))
        feed(value.astype("<f4").tobytes())
    return f"{digest:016x}"
