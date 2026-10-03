# Molecular completion ambiguity: first pilot

Run date: 2026-10-03. Source: [standalone reference](../tools/ms2_completion_ambiguity.py).
Command: `python3 tools/ms2_completion_ambiguity.py`.

The pilot independently implements the integer mass decision and exhaustively
enumerates bond assignments; it does not depend on the uncommitted MS2 grammar.
It counts connected,
fully valence-satisfied, neutral C/H/O graphs with two to four heavy atoms and
at most one ring closure. Target mass is the computed neutral monoisotopic mass,
with 10 ppm tolerance and 50 microdaltons observation uncertainty. Substructures
retain parent hydrogen counts. Counts are of distinct canonical typed graphs,
without stereochemical distinctions. No precursor constraint or physical check
was used.

| Synthetic query | Accepted formulas | Compatible graphs | Status | Edge assignments |
|---|---:|---:|---|---:|
| C2H6O, mass only | 1 | 2 | Complete | 21 |
| C2H6O, CH3 atom | 1 | 2 | Complete | 21 |
| C2H6O, CH2–OH bond | 1 | 1 | Complete | 21 |
| C2H6O, overlapping CH3–CH2 and CH2–OH bonds | 1 | 1 | Complete | 21 |
| C3H8O, mass only | 1 | 3 | Complete | 175 |

The C2H6O count corresponds to ethanol and dimethyl ether in this domain.
The C3H8O count corresponds to 1-propanol, 2-propanol and methoxyethane. These
well-known alternatives provide a check independent of the search output. The
CH2–OH typed edge removes dimethyl ether. An isolated CH3 atom does not.

Each query visited 204 formula vectors. The C2H6O mass-only query tested 12
canonical permutations; the C3H8O query tested 120. All five searches
finished under the declared work bounds. The experiment binary reports the
other counters, including embedding-match work, in CSV format. Its tests assert
the independently known mass-only and hydroxyl counts.

This is a small synthetic demonstration that substructure evidence can resolve
isomers missed by mass alone. It does not measure corpus ambiguity, top-1 model
accuracy, precursor fragmentation evidence or physical stability. Larger domains,
N-containing structures, uncertain mass inputs, incompatible or symmetric
substructures, and explicit search-budget failures remain to be measured before
the general design gate in the experiment specification is met.

The pilot is an independent Python reference with no external dependencies.
Its three unit tests passed. No device kernel or Rust API is part of this commit;
GPU and Rust/Python parity checks for the planned integration remain pending.
