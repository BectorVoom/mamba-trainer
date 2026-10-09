import MotifDecoder.Mask
import MotifDecoder.Connectivity
import MotifDecoder.Grammar
import MotifDecoder.Bridge
/-! `#print axioms` for every main theorem; the output is recorded in REPORT.md. -/
open MotifDecoder

-- the model
#print axioms step_iff
#print axioms reachBy_iff_run
#print axioms reachable_iff
#print axioms check_ok_iff
#print axioms check_error
-- item 1
#print axioms ReachBy.wf
#print axioms ReachBy.inv
-- item 2
#print axioms nth_attach_src
#print axioms nth_attach_dst
#print axioms nth_attach_other
#print axioms ReachBy.valence
#print axioms ReachBy.orderSum_bonds
#print axioms ReachBy.valence_total
#print axioms ReachBy.free_le_orig
-- item 3
#print axioms ReachBy.natoms_eq
#print axioms ReachBy.nbonds_eq
#print axioms ReachBy.nbonds_eq'
#print axioms ReachBy.cyclomatic_int
#print axioms ReachBy.cyclomatic
-- item 4
#print axioms ReachBy.count_eq
#print axioms ReachBy.fits_iff
#print axioms finished_some_iff
#print axioms AcceptedAs.budget
#print axioms AcceptedAs.budget_motifs
-- item 5
#print axioms end_allowed_in_body
#print axioms entry_exists
#print axioms bond_one_allowed
#print axioms ReachBy.bond_one_allowed
#print axioms AcceptedAs.length_eq
#print axioms Accepted.length_eq
#print axioms AcceptedAs.added_eq
#print axioms AcceptedAs.attachOrder_eq
#print axioms ReachBy.progress
#print axioms done_nothing_allowed
-- item 6
#print axioms ReachBy.connected
#print axioms ReachBy.connected_pair
#print axioms ReachBy.natoms_pos
-- item 7
#print axioms parse_serialize
#print axioms serialize_injective
#print axioms serialize_of_parse
#print axioms parse_eq_some_iff
-- beyond the list: the machine only accepts sentences of the grammar
#print axioms Accepted.exists_tree
#print axioms Accepted.parses
