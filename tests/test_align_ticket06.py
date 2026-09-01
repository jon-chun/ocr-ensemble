"""A5: alignment, Consensus Entropy (with restored confidence bands),
and deterministic equal-vote fusion (ticket 06).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from ocr_ensemble.align import (
    GAP,
    ConsensusEntropyBandsConfig,
    align_hypotheses,
    align_pair,
    alignment_artifact_payload,
    compute_consensus_entropy,
    fuse_hypotheses,
    normalize_text,
    select_center,
    write_alignment_artifact,
)


def _row(matrix, hyp_idx):
    return tuple(
        None if c.symbols.get(hyp_idx, GAP) is GAP else c.symbols[hyp_idx]
        for c in matrix.columns
    )


# ---------------------------------------------------------------------------
# 8.1 text policy
# ---------------------------------------------------------------------------


def test_text_policy_normalizes_crlf_and_cr_to_lf():
    assert normalize_text("a\r\nb\rc") == "a\nb\nc"


def test_text_policy_nfc_equivalence():
    # "é" as combining sequence (e + U+0301) vs precomposed U+00E9
    decomposed = "é"
    precomposed = "é"
    assert normalize_text(decomposed) == normalize_text(precomposed) == precomposed


def test_text_policy_does_not_fold_case_or_punctuation():
    assert normalize_text("Cat, Dog.") == "Cat, Dog."


# ---------------------------------------------------------------------------
# 8.1 pairwise alignment fixtures
# ---------------------------------------------------------------------------


def test_identical_two_member_alignment_has_no_gaps():
    m = align_hypotheses(("cat", "cat"))
    assert _row(m, 0) == _row(m, 1) == ("c", "a", "t")


def test_substitution_aligns_center_and_symbol_in_same_column():
    m = align_hypotheses(("cat", "cot"))
    assert _row(m, 0) == ("c", "a", "t")
    assert _row(m, 1) == ("c", "o", "t")


def test_insertion_emits_trailing_gap_on_center_row():
    # center chosen must be "cat" (shorter, still valid center by tie rule
    # since there's only one other hypothesis) -- assert on rows directly
    # rather than assuming which side is center.
    m = align_hypotheses(("cat", "cats"))
    cat_row = _row(m, m.center_index)
    other_row = _row(m, 1 - m.center_index)
    assert cat_row == ("c", "a", "t", None)
    assert other_row == ("c", "a", "t", "s")


def test_deletion_emits_gap_on_the_shorter_side():
    m = align_hypotheses(("cats", "cat"))
    long_idx = 0 if len(_row(m, 0)) >= len(_row(m, 1)) else 1
    short_idx = 1 - long_idx
    assert _row(m, long_idx) == ("c", "a", "t", "s")
    assert _row(m, short_idx) == ("c", "a", "t", None)


def test_marker_handled_as_one_atomic_symbol():
    m = align_hypotheses(("the <ILLEGIBLE> cat", "the big cat"))
    illegible_row = _row(m, 0)
    assert "<ILLEGIBLE>" in illegible_row
    # the marker occupies exactly one column, aligned against a single
    # symbol on the other side (not exploded into '<', 'I', 'L', ...)
    marker_col_index = illegible_row.index("<ILLEGIBLE>")
    other_row = _row(m, 1)
    assert other_row[marker_col_index] is not None


def test_center_selection_ties_by_earliest_manifest_position():
    # three hypotheses where two are equally good centers by summed
    # distance -- the earlier one in manifest order must win.
    selection = select_center((("a", "b", "c"), ("a", "b", "d"), ("x", "y", "z")))
    assert selection.center_index == 0


def test_unequal_competing_insertion_runs_in_adjacent_slots():
    m = align_hypotheses(("ac", "abc", "aXXc"))
    # must not raise, and center-character columns ('a' then 'c') must
    # remain present and never move relative to one another regardless of
    # how wide the competing insertion runs are.
    center_row = _row(m, m.center_index)
    non_gap = [s for s in center_row if s is not None]
    assert non_gap == ["a", "c"]


def test_unequal_competing_insertion_runs_in_the_same_slot():
    # two non-center hypotheses both insert at the same gap position
    # (between 'a' and 'c') with different run lengths -- the master slot
    # width must be the longest run, left-aligned, with shorter rows padded
    # on the right by internal gaps, and center columns must not move.
    m = align_hypotheses(("ac", "aXYZc", "aWc"))
    center_row = _row(m, m.center_index)
    non_gap = [s for s in center_row if s is not None]
    assert non_gap == ["a", "c"]

    long_row = _row(m, [i for i in range(3) if _row(m, i) == ("a", "X", "Y", "Z", "c")][0])
    short_row = _row(m, [i for i in range(3) if _row(m, i) == ("a", "W", None, None, "c")][0])
    assert long_row == ("a", "X", "Y", "Z", "c")
    assert short_row == ("a", "W", None, None, "c")


def test_pairwise_dp_matches_known_edit_distance():
    result = align_pair(("k", "i", "t", "t", "e", "n"), ("s", "i", "t", "t", "i", "n", "g"))
    assert result.distance == 3


# ---------------------------------------------------------------------------
# 8.2 Consensus Entropy
# ---------------------------------------------------------------------------


def test_ce_is_zero_for_identical_hypotheses():
    m = align_hypotheses(("cat", "cat", "cat"))
    assert compute_consensus_entropy(m) == 0.0


def test_ce_not_available_below_quorum():
    m = align_hypotheses(("solo",))
    assert compute_consensus_entropy(m) == "not_available"


def test_ce_not_available_when_l_is_zero():
    m = align_hypotheses(("", ""))
    assert compute_consensus_entropy(m) == "not_available"


def test_ce_in_open_interval_for_partial_disagreement():
    m = align_hypotheses(("cat", "cot"))
    ce = compute_consensus_entropy(m)
    assert isinstance(ce, float)
    assert 0.0 < ce < 1.0
    # column 2 (index 1) is a 2-way tie: H = ln(2)/ln(2) = 1; columns 0
    # and 2 are unanimous: H = 0. Average over 3 columns = 1/3.
    assert math.isclose(ce, 1 / 3, rel_tol=1e-9)


def test_ce_is_one_for_full_three_way_disagreement_column():
    m = align_hypotheses(("cat", "cot", "cut"))
    ce = compute_consensus_entropy(m)
    assert isinstance(ce, float)
    # columns 0,2 unanimous (H=0); column 1 is 3-way all-different
    # (H = ln(3)/ln(3) = 1). Average = 1/3.
    assert math.isclose(ce, 1 / 3, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# Restored CE confidence bands
# ---------------------------------------------------------------------------


def test_band_boundary_values_are_inclusive_to_named_band():
    cfg = ConsensusEntropyBandsConfig(ce_low_uncertainty_max=0.2, ce_high_uncertainty_min=0.6)
    assert cfg.band_for(0.2) == "high_confidence"
    assert cfg.band_for(0.6) == "low_confidence"


def test_band_strictly_between_thresholds_is_medium():
    cfg = ConsensusEntropyBandsConfig(ce_low_uncertainty_max=0.2, ce_high_uncertainty_min=0.6)
    assert cfg.band_for(0.4) == "medium_confidence"


def test_band_config_rejects_non_strict_ordering():
    with pytest.raises(ValueError):
        ConsensusEntropyBandsConfig(ce_low_uncertainty_max=0.6, ce_high_uncertainty_min=0.6)
    with pytest.raises(ValueError):
        ConsensusEntropyBandsConfig(ce_low_uncertainty_max=0.7, ce_high_uncertainty_min=0.6)


def test_band_config_rejects_out_of_unit_interval():
    with pytest.raises(ValueError):
        ConsensusEntropyBandsConfig(ce_low_uncertainty_max=-0.1, ce_high_uncertainty_min=0.6)
    with pytest.raises(ValueError):
        ConsensusEntropyBandsConfig(ce_low_uncertainty_max=0.2, ce_high_uncertainty_min=1.1)


# ---------------------------------------------------------------------------
# 8.3 deterministic equal-vote fusion
# ---------------------------------------------------------------------------


def test_fusion_of_identical_hypotheses_reproduces_the_text():
    m = align_hypotheses(("cat", "cat"))
    fused = fuse_hypotheses(m)
    assert fused.text == "cat"
    assert all(not c.tie for c in fused.columns)


def test_fusion_tie_resolved_by_earliest_manifest_configuration():
    m = align_hypotheses(("cat", "cot", "cut"))
    fused = fuse_hypotheses(m)
    assert fused.text == "cat"
    tie_columns = [c for c in fused.columns if c.tie]
    assert len(tie_columns) == 1
    assert tie_columns[0].winning_symbol == "a"
    assert 0 in tie_columns[0].supporting_result_indices[
        tie_columns[0].candidate_symbols.index("a")
    ]


def test_fusion_winning_gap_suppresses_the_column():
    m = align_hypotheses(("cat", "cat", "cats"))
    fused = fuse_hypotheses(m)
    # majority of hypotheses lack the trailing 's' -> gap wins -> suppressed
    assert fused.text == "cat"
    trailing = fused.columns[-1]
    assert trailing.winning_symbol is GAP
    assert trailing.emitted is False


def test_fusion_persists_full_column_detail():
    m = align_hypotheses(("cat", "cot"))
    fused = fuse_hypotheses(m)
    tie_col = fused.columns[1]
    assert set(tie_col.candidate_symbols) == {"a", "o"}
    assert sum(tie_col.candidate_counts) == 2
    assert len(tie_col.supporting_result_indices) == len(tie_col.candidate_symbols)


# ---------------------------------------------------------------------------
# content-addressed alignment artifact
# ---------------------------------------------------------------------------


def test_alignment_artifact_payload_rejects_wrong_contributor_count():
    m = align_hypotheses(("cat", "cot"))
    with pytest.raises(ValueError):
        alignment_artifact_payload(m, ("only-one-id",))


def test_alignment_artifact_payload_represents_gap_as_null_and_round_trips():
    m = align_hypotheses(("cat", "cats"))
    payload = alignment_artifact_payload(m, ("result-a", "result-b"))
    body = json.dumps(payload)
    reloaded = json.loads(body)
    assert reloaded["contributor_result_ids"] == ["result-a", "result-b"]
    assert reloaded["hypothesis_count"] == 2
    last_column_symbols = reloaded["columns"][-1]["symbols"]
    assert None in last_column_symbols.values()


def test_write_alignment_artifact_is_content_addressed_and_deterministic(tmp_path: Path):
    m = align_hypotheses(("cat", "cot"))
    path_a = write_alignment_artifact(tmp_path, m, ("result-a", "result-b"))
    path_b = write_alignment_artifact(tmp_path, m, ("result-a", "result-b"))
    assert path_a == path_b
    assert path_a.exists()

    m_different = align_hypotheses(("cat", "cut"))
    path_c = write_alignment_artifact(tmp_path, m_different, ("result-a", "result-c"))
    assert path_c != path_a
