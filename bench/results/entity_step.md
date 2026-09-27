# Entity-step gate numbers

G0 is the vulkan baseline from `KERNEL_OPTIMIZATION_PLAN.md` §0.1 (Radeon
860M, gfx1151). The cpu rows below are substitutes measured on a Mac mini M1
(`--features cpu`, release) — same binary, different backend and machine, so
they pin launch counts, not the ms/step gates. The binding per-batch rows
must come from `bash bench/entity_step.sh` on the target machine (it refuses
a dirty tree and appends here).

| date | commit | batch | ms/step | launches/step | reads/step | top labels |
|---|---|---|---|---|---|---|
| 2026-09-27 | `91df9ea` (G0) | 8 | 310 ms | ≈ 2,300 (composed) / 2,200 (fused) | 0.1 | vulkan, Radeon 860M |
| 2026-09-27 | `91df9ea` (G0) | 32 | 1050 ms | – | – | vulkan, Radeon 860M |
| 2026-09-27 | `91df9ea` (G0) | 128 | 3320 ms | ≈ 2,200 | – | vulkan, Radeon 860M |
| 2026-09-27 | cpu substitute (M0–K11 stack) | 8 | 427.5 ms | 2008 | ~1.1 | backward:1325 mixer.scan:108 scan.intra:108 (cpu, M1) |
