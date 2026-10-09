# Codex review: functional-group detector ms2-fg-v4 after FG6 and FG7 (existential stable witness)

Reviewer: codex exec (read-only), 2026-10-06. Verdict: the existential rule is sound and the fragment verdicts are numbering-invariant (no counterexample; 10,870 fragment permutations checked with its own translation); the FG6 detector items are addressed. Remaining: the reference tool's export mode bypasses graph validation, its `--graphs` mode accepts fractional and boolean bond fields, no population test permutes dense open fragments, and two defensive branches do not assert on an invalid matching. Fix task: FG8.

The new existential rule appears sound and numbering-invariant. I found two reference-validation defects and one missing regression population. No counterexample made a decided-delocalised fragment bond fixed in its parent.

1. **Medium — export mode still bypasses graph validation.**  
   [functional_groups_ref.py:1219](tools/ms2/functional_groups_ref.py:1219)

   Concrete input: export atom IDs `[8,4,4,4]`, meaning `O(H0,v2)` and three `C(H3,v4)`, with singles `(0,1),(0,2),(0,3)`.

   **Expected:** reject oxygen’s incident order sum three against valence two.  
   **Actual:** `export_molecule_record` succeeds and reports **three ethers**. I reproduced this with the current reference. A lone `C(H3,v4)` export graph also succeeds despite unsatisfied valence.

   **Minimal fix:** call `validate_graph` after converting export atom IDs and before constructing/counting the RDKit molecule. Preserve the export record format.

2. **Medium — `--graphs` accepts fractional endpoints and bond orders.**  
   [functional_groups_ref.py:1091](tools/ms2/functional_groups_ref.py:1091), with reconstruction at [line 649](tools/ms2/functional_groups_ref.py:649)

   Concrete graph: two `C(H3,v4)` atoms, bond `[0.9,1,1.9]`.

   **Expected:** malformed-input error and nonzero exit; endpoints and orders must be integers.  
   **Actual:** validation truncates this to `(0,1,1)`, counting succeeds, and the CLI exits zero while storing the original fractional bond. Boolean values such as `[false,true,true]` also pass.

   **Minimal fix:** require bond triples containing actual integers, excluding booleans, before range/order checks. Validate integer hydrogen and valence fields similarly; use the validated values consistently.

   A non-record entry such as `{"graphs":[null]}` does exit unsuccessfully, but crashes at `g.get(...)` outside the exception handler. Validate the container and record shapes so these receive normal graph-error diagnostics.

3. **Low — no dense random-fragment numbering-invariance population test.**  
   [ms2_functional_groups.rs:4673](tests/ms2_functional_groups.rs:4673), alongside [line 4557](tests/ms2_functional_groups.rs:4557)

   Concrete fragment: types `[16,15,1,2,2]`, bonds  
   `[(0,1,2),(0,2,1),(1,3,1),(1,4,2),(2,3,2),(2,4,1)]`.  
   Embed it in the existing seven-atom parent by adding `O(H0,v2)` with `(0,5,2)` and `C(H3,v4)` with `(1,6,1)`.

   **Expected:** population coverage exercises open fragments like this under permutation, including unstable atoms and stable witnesses.  
   **Actual:** the trigger covers this one fragment exhaustively; the dense numbering test permutes only **closed parents**. The dense fragment-soundness test checks fragments against their parents without permuting those fragments. The older fragment-permutation test uses a fixed collection of conventional ring parents.

   **Minimal fix:** permute sampled connected fragments from the dense-parent generator. Compare mapped-back `decided_bonds`, determined counts/anchors, and `undetermined`; assert that the population includes unstable atoms with decided-delocalised bonds.

The constrained-search construction itself checks out:

- Removing every unstable-owned vertex also removes its locally matched port/core partner. Stored port–port doubles crossing to stable atoms have **both endpoints removed and their pair forced**. Thus the restricted seed covers every remaining vertex except the deliberately freed ports or cores.
- A perfect remainder matching lifts to a valid full matching by restoring the forced pairs and unchanged removed pairs. Every unstable vertex retains its stored mate.
- Consequently, the tested symmetric-difference component avoids unstable atoms. Its projected changes preserve inside double degree at every atom, so applying those changes with outside bonds unchanged gives a valid alternative assignment in every order-preserving completion.
- Conversely, any stable witness cycle avoids the stable endpoints of forced crossing doubles too: using such an endpoint would require following its stored matching edge into an unstable vertex. Flipping that cycle supplies a perfect matching of exactly the searched remainder. Stable atoms may participate in both forced and free portions; that does not invalidate this argument.

Therefore the restricted search does not miss a stable witness merely because forced and free parts share an original atom, assuming the matcher correctly decides perfect-matching existence.

The decided-fixed side is also order-independent. [fixed_decided:1819](src/models/ms2/functional_groups.rs:1819) searches all reachable `(atom, expected-order)` states. Its early return means “some boundary exit exists”; it does not classify according to the first unsuccessful path. I found no corresponding numbering defect in the determined/undetermined detection built on these statuses.

My bounded Python translation checked **200 closed parents, 2,174 connected fragments, 10,870 fragment permutations, and 5,269 decided fragment/parent comparisons**, with zero mismatches. Eight fragments contained unstable atoms while also reporting stable delocalisation. The assignment oracle streamed results, capped each case at 100,000 nodes, and used at most 65 nodes in this sample. These are translation checks, not Rust-test execution.

With Rust’s endpoint normalization and bond sorting included, reverting to the first-witness rule fails `fg7_trigger_all_numberings` in **60/120 numberings**. In the identity numbering, `(1,3)` and `(2,3)` become `Unknown` instead of the asserted `Some(true)`. The regression is effective.

The FG7 tests are memory-bounded for their stated input sizes: assignment results are streamed, matching enumeration is replaced by existence search, and the matching memo is node-capped. However, the blanket comment that every case aborts after 2,000,000 nodes is inaccurate: `fg7_oracle_capped`, `fg7_for_each_assignment`, and the dense graph generator have no search-node counter. Their assignment caps and vertex bounds limit storage, not total search work.

FG6’s original `SearchOutcome` branches now distinguish invalid from valid nonperfect matchings correctly. One defensive qualification remains: the new constrained-search guards silently return false on invalid matchings, and `confirm_no_perfect` treats an invalid fresh matching as agreement. These do not universally implement the documentation’s promised loud invalid-matching assertion. I observed no invalid matcher result.

All **174 fixture molecules / 806 stored forms** pass closure validation. The variable-`d=2`, contraction-counter, and every-stored-form assertions are present. Closed V0 graphs admit neutral S(v6)/P(v5); charged atoms are outside V0. Rejecting open fragments in `process_graph` is consistent with its explicit whole-molecule contract.

**Verdict:**

- **(a) Soundness:** the new existential rule is sound; no fragment/parent counterexample found.
- **(b) Numbering invariance:** the stable-witness and sealed-end predicates are invariant; bounded fragment checks passed. Dense-fragment population coverage remains missing.
- **(c) FG6:** the original hand-built counterexamples, core outcome split, scope restrictions, fixtures, and requested assertions are addressed. Export validation, strict CLI input validation, and universal loud invalid-result handling remain incomplete.

Read-only throughout; no Cargo, file modifications, or process allowed above 768 MiB.