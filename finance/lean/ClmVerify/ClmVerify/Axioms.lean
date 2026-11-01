import ClmVerify.ScanProofs
import ClmVerify.WindowProofs
import ClmVerify.FoldProofs
import ClmVerify.BacktestProofs
/-! `#print axioms` for every main theorem; the output is recorded in REPORT.md. -/
open ClmVerify

-- C7 scan
#print axioms Aff.comb_assoc
#print axioms Aff.one_comb
#print axioms Aff.comb_one
#print axioms Aff.apply_comb
#print axioms seqScan_getElem?
#print axioms final_eq_comb_fold
#print axioms summarize_eq_fold
#print axioms chunked_eq_seq_of_split
#print axioms chunksOf_flatten
#print axioms chunksOf_shape
#print axioms chunkedScan_eq_seqScan
#print axioms seqScan_causal
#print axioms ys_causal
#print axioms chunkedScan_causal
#print axioms revScan_eq_reverse
#print axioms chunkedRevScan_eq_revScan
#print axioms adjoint_getElem?
#print axioms adjoint_identity
#print axioms chunked_adjoint
#print axioms adjoint_indep_u
-- C2, C3
#print axioms mem_featureReads
#print axioms featureReads_le
#print axioms mem_labelDays
#print axioms labelPriceReads_sub
#print axioms labelDays_bounds
#print axioms feature_lt_label
#print axioms theta_nonneg
#print axioms not_buy_and_sell
#print axioms classify_buy_iff
#print axioms classify_sell_iff
#print axioms classify_hold_iff
#print axioms classify_exactly_one
-- C4
#print axioms mkFold_some
#print axioms fold_emitted
#print axioms fold_layout
#print axioms fold_chain
#print axioms seg_stop_le_start
#print axioms seg_disjoint
#print axioms seg_ordered
#print axioms seg_inside
#print axioms mem_days
#print axioms test_eq_year
#print axioms rolling_train_len
#print axioms kept_eq_range
#print axioms mem_kept
#print axioms mem_kept_iff_range
#print axioms not_kept_of_violation
#print axioms not_kept_of_label_reaches_next
#print axioms not_kept_test_of_label_past_end
#print axioms purge
#print axioms kept_test_label_inside
#print axioms folds_test_increasing
#print axioms folds_test_disjoint
#print axioms expanding_train_mono
-- C5
#print axioms posTrades_mem
#print axioms hold_no_trade
#print axioms hold_keeps_first
#print axioms hold_keeps
#print axioms trades_le_nonhold
#print axioms execDay_gt
#print axioms pnlReads_ge
#print axioms runDay_skip
#print axioms runDay_reads_inside
#print axioms runDay_traded
#print axioms runDay_missing
#print axioms runDay_hold
#print axioms runDay_net
#print axioms runBacktest_mem
#print axioms accounting
#print axioms runBacktest_trades_le
#print axioms runBacktest_eq_posTrades
