# Optional physical verification (`physical-verification-v1`)

Status: implemented as a separate Python tool, 2026-10-05.
`tools/ms2/completion_physical_verify.py` screens the candidates of a
`molecular-completion-generate-v1` response with a bounded, auditable
**geometry-and-strain screen at force-field level**. It is not part of the
wheel and does not change the generator: candidates are never labelled
physically verified by the generator, and this tool never reorders,
filters or re-ranks them.

## What it does

Per molecule — one constitution, or one requested stereoisomer — in order:

1. **Structural alerts on the typed graph**, before RDKit sees it (pure
   Python over atom type ids and bonds; each alert has an id, the atom
   indices, the rule's parameters, and a one-line meaning). Alerts are
   warnings about strain or reactivity, never verdicts, and are reported
   with every status, including converged optimisations. The one exception
   is `bridgehead_double_bond`, which needs ring membership from the
   smallest set of smallest rings and reads SSSR ring info off a
   stereo-free RDKit copy (stated in its parameters); it still runs before
   any embedding or optimisation.
2. **Conversion** with `completion_stereo_check.to_rdkit` (the single
   typed-graph-to-RDKit conversion; hydrogen counts fixed, no hydrogen
   invention), applying the stereo assignment when one is given. A lost
   assignment is an error, never dropped silently.
3. **Embedding** with explicit hydrogens (ETKDGv3, fixed seed, default 10
   requested conformers, one retry with random coordinates).
4. **Optimisation** with MMFF94 when it has parameters for the whole
   molecule, else UFF, else `unsupported`. The lowest-energy converged
   conformer is kept.
5. **Geometry diagnostics** on the kept conformer, each with its numbers:
   finite coordinates; bond lengths against covalent radii; close contacts
   (only pairs with a shortest path of 4+ bonds; 1-2, 1-3 and 1-4 pairs
   are excluded); angle deviations at sp3/sp2 carbons; and re-perception
   of every assigned stereo element from the final coordinates (signed
   ligand volumes for centres, reference-ligand dihedrals for double
   bonds). `coordinates_not_finite` and `stereo_not_preserved` fail the
   molecule; bond-length, contact and angle outliers and undetermined
   stereo are recorded loudly but do not fail it.

## Status vocabulary

Exactly one per molecule: `force_field_optimization_converged` (steps
2–4 succeeded and no diagnostic failed), `calculation_failed` (reasons:
`embedding_failed`, `optimisation_not_converged`, `stereo_not_preserved`,
`timeout`, …), `unsupported` (conversion/sanitisation failure,
`no_force_field_parameters`), `error` (`stereo_assignment_lost`, …).
A candidate's block summarises its evaluated units
(`converged_without_structural_alerts` = converged with zero alerts,
`converged`, `failed`, `unsupported`, alert union, diagnostic-warnings
union); the top-level block counts statuses over candidates and over
stereoisomers, counts alerts by id, and records wall time. `converged`
means the bounded procedure ran to completion, never a stability verdict.

## Alert inventory (one line each)

- `trans_double_bond_in_small_ring`: requested trans double bond whose
  ring-substituent geometry is trans in a ring below 8 atoms (reference
  ligands are translated with parity, so a mixed ring/non-ring reference
  pair reading trans can still be ring-cis and does not fire); without
  an assignment this case is covered by the bridgehead alert below.
- `triple_bond_in_small_ring`: triple bond in a ring below 8 atoms.
- `cumulene_in_small_ring`: carbon with two double bonds whose
  double-bond axis sits in a ring below 9 atoms.
- `bridgehead_double_bond`: double bond at a bridgehead of a bridged
  bicyclic system whose largest SSSR bond-ring holding the bond is below
  8 atoms (anti-Bredt strain). Bridgeheads come from ring relationships
  (an endpoint in at least two SSSR rings with a non-adjacent partner in
  at least two rings and three disjoint paths); adjacent fusion pairs are
  fused, not bridged. Aromatic double bonds are excluded.
- `three_membered_ring_unsaturation`: double/triple bond in a 3-ring.
- `fused_small_rings`: two 3- or 4-rings sharing a bond.
- `cage_small_rings`: an atom in three or more rings of size ≤ 4.
- `peroxide`: O–O single bond. `polyoxide_chain`: O–O–O chain.
- `polynitrogen_chain`: three or more consecutive singly-bonded N atoms
  (deduplicated; aromatic N excluded via shared RDKit aromaticity
  perception on the stereo-free copy).
- `n_halogen` / `o_halogen`: N–X / O–X bonds (X = F, Cl, Br, I).
- `enol`: C=C–OH on non-aromatic carbons (aromatic C=C excluded via
  shared RDKit aromaticity perception; kekulized phenol does not fire).
  `geminal_diol`: saturated carbon with two OH groups;
  `hemiaminal`: saturated carbon with OH and amino substitution;
  `geminal_amino_alcohol`: saturated carbon bonded to both O and N,
  with a `hydroxyl` param distinguishing the amino alcohol
  (`hydroxyl: true`) from the amino ether (`hydroxyl: false`).
- `hypervalent_sulfur_or_phosphorus`: S above valence 2 or P above
  valence 3.

## The three statements (carried verbatim in every result)

- `"stability": {"status": "not_evaluated", "reason": "a converged
  force-field optimisation is not evidence of thermodynamic stability,
  kinetic persistence or synthesizability"}`
- `"electronic_structure": {"status": "not_evaluated", "reason": "no
  quantum-chemistry program is installed"}`
- `"energy_comparability": "force-field energies compare conformers of
  one molecule under one method only; they do not rank different
  molecules"`

## How to run

```sh
uv run --project /Users/ods/Documents/Enveda_CASMI python \
  tools/ms2/completion_physical_verify.py \
  --in <response.json | responses.jsonl | array file> --out <file> \
  [--conformers 10] [--max-iters 2000] [--seed 20261005] \
  [--max-candidates N] [--stereo all|first|none] [--timeout-s 60]
```

`--stereo all|first|none` selects which expanded `stereoisomers` are
evaluated per candidate (the constitution is always evaluated). Each
molecule runs in a worker process terminated and joined after
`--timeout-s` seconds (`calculation_failed`, reason `timeout`, with the
parent-computed preliminary alerts kept); a crashed worker is joined and
reported as `error` without an inline fallback. The optimiser
additionally caps each conformer at `--max-iters` iterations.
`verify_candidate(atoms,
bonds, stereo_block=None, isomer=None, config=...)` is importable and runs
one molecule in-process. `run_unit_in_worker(...)` is the timeout wrapper
main uses. Tests:
`tools/ms2/test_completion_physical_verify.py` (plain functions + `main`,
deterministic with the fixed seed).

Dependency: RDKit only (2026.03.3). No Rust, no cargo, no wheel change, no
quantum-chemistry program.
