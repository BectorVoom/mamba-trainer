# Vendored crates

`cubecl-cpp` is crates.io `cubecl-cpp 0.10.0` with one fix, applied through
`[patch.crates-io]` in the workspace `Cargo.toml`:

- `src/hip/mma/wmma_intrinsics_compiler.rs`: the wave32 WMMA accumulator load
  and store helpers indexed the fragment row with `threadIdx.x / 16`, which is
  only the lane's half of the wave for the first wave of a workgroup. Every
  other plane of a multi-wave cube read and wrote its 16x16 tile two or three
  rows off and spilled into the neighbouring plane's scratch, so any CMMA kernel
  with more than one plane per cube produced garbage on RDNA (gfx11/gfx12,
  seen on a Radeon 860M, gfx1151, ROCm 7.1.1). The helpers now use
  `(threadIdx.x % warpSize) / 16`. Worth sending upstream.
