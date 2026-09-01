"""A5: character alignment, Consensus Entropy, and deterministic equal-vote
fusion (ticket 06).

Scope: the shared alignment substrate and its two consumers (CE, fusion) for
the single- and two-hypothesis cases the stub-model slice produces. Full
N-member alignment robustness (three-plus-way progressive merge edge cases
beyond what tickets 06's own fixtures cover) is exercised further in
ticket 12; the merge algorithm itself is written generally, not special-cased
to two members.
"""

from __future__ import annotations

import dataclasses
import math
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from ocr_ensemble.identity import canonical_json_bytes, sha256_hex
from ocr_ensemble.records import ConsensusResult
from ocr_ensemble.records import FusedHypothesis as SealedFusedHypothesis
from ocr_ensemble.records import seal as _seal
from ocr_ensemble.storage import write_jsonl_atomic

TEXT_POLICY_ID = "text_policy_v1"
ALIGNMENT_POLICY_ID = "alignment_v1"
FUSION_POLICY_ID = "fusion_v1"

# A typed sentinel for the internal alignment gap. A single reserved object
# (not a string) so it can never collide with any Unicode text or the
# <ILLEGIBLE> marker, which is itself just an ordinary string token.
GAP = object()

ILLEGIBLE_MARKER = "<ILLEGIBLE>"


def normalize_text(text: str) -> str:
    """``text_policy_v1``: CRLF/CR -> LF, then NFC. No case,
    diacritic, punctuation, historical-spelling, or whitespace folding.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return unicodedata.normalize("NFC", text)


def _tokenize(text: str) -> tuple[str, ...]:
    """Split into alignment symbols, with ``<ILLEGIBLE>`` tokenized as one
    atomic symbol rather than its constituent characters.
    """
    tokens: list[str] = []
    i = 0
    n = len(text)
    marker_len = len(ILLEGIBLE_MARKER)
    while i < n:
        if text[i : i + marker_len] == ILLEGIBLE_MARKER:
            tokens.append(ILLEGIBLE_MARKER)
            i += marker_len
        else:
            tokens.append(text[i])
            i += 1
    return tuple(tokens)


@dataclass(frozen=True)
class PairwiseAlignment:
    """One pairwise alignment result: parallel symbol sequences of equal
    length, ``GAP`` standing in wherever one side has no symbol.
    """

    a_symbols: tuple[object, ...]
    b_symbols: tuple[object, ...]
    distance: int


def align_pair(a_tokens: tuple[str, ...], b_tokens: tuple[str, ...]) -> PairwiseAlignment:
    """Unit-cost character Levenshtein DP with the exact traceback
    tie-break priority: diagonal, then deletion from ``b``,
    then insertion into ``b``.

    "Deletion" and "insertion" are named from ``a``'s perspective as the
    reference/center row: a deletion consumes an ``a`` symbol and emits a gap
    for ``b``; an insertion emits a gap for ``a`` and consumes a ``b`` symbol.
    """
    n, m = len(a_tokens), len(b_tokens)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
    for j in range(1, m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sub_cost = 0 if a_tokens[i - 1] == b_tokens[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j - 1] + sub_cost,
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
            )

    a_out: list[object] = []
    b_out: list[object] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            sub_cost = 0 if a_tokens[i - 1] == b_tokens[j - 1] else 1
            if dp[i][j] == dp[i - 1][j - 1] + sub_cost:
                a_out.append(a_tokens[i - 1])
                b_out.append(b_tokens[j - 1])
                i -= 1
                j -= 1
                continue
        if i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            a_out.append(a_tokens[i - 1])
            b_out.append(GAP)
            i -= 1
            continue
        # j > 0 guaranteed here
        a_out.append(GAP)
        b_out.append(b_tokens[j - 1])
        j -= 1

    a_out.reverse()
    b_out.reverse()
    return PairwiseAlignment(
        a_symbols=tuple(a_out), b_symbols=tuple(b_out), distance=dp[n][m]
    )


@dataclass(frozen=True)
class CenterSelection:
    center_index: int
    pairwise_distances: tuple[tuple[int, int, int], ...]  # (i, j, distance)


def select_center(hypotheses: tuple[tuple[str, ...], ...]) -> CenterSelection:
    """Choose the hypothesis with minimum summed pairwise distance to every
    other hypothesis; tie by earliest manifest position.
    """
    k = len(hypotheses)
    pairwise: dict[tuple[int, int], int] = {}
    distances_recorded: list[tuple[int, int, int]] = []
    for i in range(k):
        for j in range(i + 1, k):
            d = align_pair(hypotheses[i], hypotheses[j]).distance
            pairwise[(i, j)] = d
            distances_recorded.append((i, j, d))

    def summed_distance(idx: int) -> int:
        total = 0
        for other in range(k):
            if other == idx:
                continue
            key = (idx, other) if idx < other else (other, idx)
            total += pairwise[key]
        return total

    best_index = min(range(k), key=lambda idx: (summed_distance(idx), idx))
    return CenterSelection(
        center_index=best_index, pairwise_distances=tuple(distances_recorded)
    )


def _split_pairwise(
    center_alignment: tuple[object, ...], other_alignment: tuple[object, ...]
) -> tuple[list[list[object]], list[object]]:
    """Split one pairwise alignment into ``len(center_tokens)+1`` insertion
    slots -- the run of ``other`` symbols associated with the gap immediately
    before each center character, plus a trailing slot after the last one
    (``len(center)+1`` insertion slots) -- and, separately,
    the ``other`` symbol aligned directly against each center character
    (``GAP`` for a deletion, the matched or substituted symbol otherwise).
    """
    runs: list[list[object]] = [[]]
    center_char_symbols: list[object] = []
    for center_sym, other_sym in zip(center_alignment, other_alignment):
        if center_sym is GAP:
            runs[-1].append(other_sym)
        else:
            center_char_symbols.append(other_sym)
            runs.append([])
    return runs, center_char_symbols


@dataclass(frozen=True)
class AlignmentColumn:
    index: int
    symbols: dict[int, object]  # hypothesis_index -> symbol (GAP allowed)


@dataclass(frozen=True)
class AlignmentMatrix:
    hypothesis_count: int
    columns: tuple[AlignmentColumn, ...]
    center_index: int


def align_hypotheses(hypotheses_text: tuple[str, ...]) -> AlignmentMatrix:
    """Progressive multi-way alignment: tokenize and
    normalize every hypothesis, pick a center, align every other hypothesis
    to the center, then merge by the deterministic slot-widening rule so
    center-character columns never move relative to one another.
    """
    tokenized = tuple(_tokenize(normalize_text(t)) for t in hypotheses_text)
    k = len(tokenized)
    if k == 1:
        single_columns = tuple(
            AlignmentColumn(index=i, symbols={0: sym})
            for i, sym in enumerate(tokenized[0])
        )
        return AlignmentMatrix(hypothesis_count=1, columns=single_columns, center_index=0)

    selection = select_center(tokenized)
    center_index = selection.center_index
    center_tokens = tokenized[center_index]
    num_slots = len(center_tokens) + 1

    # slots[slot_index][hypothesis_index] -> list of symbols contributed to
    # that slot by that hypothesis (its own insertion run, left-aligned).
    slots: list[dict[int, list[object]]] = [dict() for _ in range(num_slots)]
    # center_char_symbols[other_idx][center_char_position] -> the symbol that
    # hypothesis aligned directly against that center character (GAP for a
    # deletion, the matched/substituted symbol otherwise).
    center_char_symbols: dict[int, list[object]] = {}

    other_indices = [idx for idx in range(k) if idx != center_index]
    for other_idx in other_indices:
        pairwise = align_pair(center_tokens, tokenized[other_idx])
        runs, char_syms = _split_pairwise(pairwise.a_symbols, pairwise.b_symbols)
        assert len(runs) == num_slots
        assert len(char_syms) == len(center_tokens)
        for slot_idx, run in enumerate(runs):
            slots[slot_idx][other_idx] = list(run)
        center_char_symbols[other_idx] = char_syms

    columns: list[AlignmentColumn] = []
    for slot_idx in range(num_slots):
        contributions = slots[slot_idx]
        width = max((len(run) for run in contributions.values()), default=0)
        for pos in range(width):
            symbols: dict[int, object] = {}
            for other_idx in other_indices:
                run = contributions.get(other_idx, [])
                symbols[other_idx] = run[pos] if pos < len(run) else GAP
            symbols[center_index] = GAP
            columns.append(AlignmentColumn(index=len(columns), symbols=symbols))
        if slot_idx < len(center_tokens):
            symbols = {
                other_idx: center_char_symbols[other_idx][slot_idx]
                for other_idx in other_indices
            }
            symbols[center_index] = center_tokens[slot_idx]
            columns.append(AlignmentColumn(index=len(columns), symbols=symbols))

    reindexed = tuple(
        dataclasses.replace(col, index=i) for i, col in enumerate(columns)
    )
    return AlignmentMatrix(hypothesis_count=k, columns=reindexed, center_index=center_index)


ConfidenceBand = Literal["high_confidence", "medium_confidence", "low_confidence"]


@dataclass(frozen=True)
class ConsensusEntropyBandsConfig:
    """Versioned config keys for confidence-band thresholds. Boundary values are
    inclusive to their named band, never both.
    """

    ce_low_uncertainty_max: float
    ce_high_uncertainty_min: float

    def __post_init__(self) -> None:
        for name, value in (
            ("ce_low_uncertainty_max", self.ce_low_uncertainty_max),
            ("ce_high_uncertainty_min", self.ce_high_uncertainty_min),
        ):
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"{name} must be in [0,1], got {value!r}")
        if self.ce_low_uncertainty_max >= self.ce_high_uncertainty_min:
            raise ValueError(
                "ce_low_uncertainty_max must be < ce_high_uncertainty_min "
                f"(got {self.ce_low_uncertainty_max!r} >= "
                f"{self.ce_high_uncertainty_min!r})"
            )

    def band_for(self, ce: float) -> ConfidenceBand:
        if ce <= self.ce_low_uncertainty_max:
            return "high_confidence"
        if ce >= self.ce_high_uncertainty_min:
            return "low_confidence"
        return "medium_confidence"


QUORUM_MINIMUM = 2  # max(2, floor(N/2)+1) collapses to 2 for any N>=2


def compute_consensus_entropy(
    matrix: AlignmentMatrix,
) -> float | Literal["not_available"]:
    """Consensus Entropy: equal-vote symbol frequency per aligned column
    (including the internal gap), natural-log Shannon entropy normalized by
    ``ln(k)``, averaged over ``L`` columns. Below quorum or ``L=0`` yields
    ``not_available``.
    """
    k = matrix.hypothesis_count
    if k < QUORUM_MINIMUM:
        return "not_available"
    L = len(matrix.columns)
    if L == 0:
        return "not_available"

    total_h = 0.0
    for column in matrix.columns:
        counts: dict[object, int] = {}
        for hyp_idx in range(k):
            sym = column.symbols.get(hyp_idx, GAP)
            key = id(sym) if sym is GAP else sym
            counts[key] = counts.get(key, 0) + 1
        h_c = 0.0
        for count in counts.values():
            p = count / k
            h_c -= p * math.log(p)
        total_h += h_c / math.log(k)

    return total_h / L


@dataclass(frozen=True)
class FusionColumn:
    index: int
    candidate_symbols: tuple[object, ...]
    candidate_counts: tuple[int, ...]
    supporting_result_indices: tuple[tuple[int, ...], ...]
    winning_symbol: object
    tie: bool
    emitted: bool


@dataclass(frozen=True)
class FusedHypothesis:
    columns: tuple[FusionColumn, ...]
    text: str


def fuse_hypotheses(matrix: AlignmentMatrix) -> FusedHypothesis:
    """Deterministic equal-vote fusion: one vote per
    eligible configuration, modal symbol wins per column (winning gap
    suppresses the column), ties resolved by earliest configuration in
    manifest order.
    """
    columns: list[FusionColumn] = []
    emitted_symbols: list[str] = []

    for column in matrix.columns:
        by_symbol: dict[object, list[int]] = {}
        first_seen_order: list[object] = []
        for hyp_idx in range(matrix.hypothesis_count):
            sym = column.symbols.get(hyp_idx, GAP)
            key = sym  # GAP is a single sentinel object, safe as a dict key
            if key not in by_symbol:
                by_symbol[key] = []
                first_seen_order.append(key)
            by_symbol[key].append(hyp_idx)

        max_count = max(len(v) for v in by_symbol.values())
        tied_symbols = [s for s in first_seen_order if len(by_symbol[s]) == max_count]
        tie = len(tied_symbols) > 1
        # "earliest configuration in manifest order" -- among tied symbols,
        # pick the one whose supporting set contains the smallest hypothesis
        # index.
        winning_symbol = min(
            tied_symbols, key=lambda s: min(by_symbol[s])
        )

        emitted = winning_symbol is not GAP
        if emitted:
            emitted_symbols.append(winning_symbol)  # type: ignore[arg-type]

        candidate_symbols = tuple(first_seen_order)
        candidate_counts = tuple(len(by_symbol[s]) for s in first_seen_order)
        supporting = tuple(tuple(by_symbol[s]) for s in first_seen_order)

        columns.append(
            FusionColumn(
                index=column.index,
                candidate_symbols=candidate_symbols,
                candidate_counts=candidate_counts,
                supporting_result_indices=supporting,
                winning_symbol=winning_symbol,
                tie=tie,
                emitted=emitted,
            )
        )

    return FusedHypothesis(columns=tuple(columns), text="".join(emitted_symbols))


def _symbol_to_json(sym: object) -> str | None:
    """``GAP`` -> JSON ``null``; every real alignment symbol is a non-null
    string, so this is an unambiguous round trip.
    """
    return None if sym is GAP else sym


def alignment_artifact_payload(
    matrix: AlignmentMatrix, contributor_result_ids: tuple[str, ...]
) -> dict:
    """The complete column matrix plus contributor result IDs, in the shape
    persisted as a content-addressed alignment artifact (the final step of
    alignment). ``contributor_result_ids[hyp_idx]`` must be the
    ``PageModelResult.page_model_result_id`` (or Fused Hypothesis ID, for a
    future re-alignment) that produced hypothesis ``hyp_idx`` -- alignment
    itself is pure text-in, so the caller supplies this mapping.
    """
    if len(contributor_result_ids) != matrix.hypothesis_count:
        raise ValueError(
            f"expected {matrix.hypothesis_count} contributor_result_ids, "
            f"got {len(contributor_result_ids)}"
        )
    return {
        "text_policy_id": TEXT_POLICY_ID,
        "hypothesis_count": matrix.hypothesis_count,
        "center_index": matrix.center_index,
        "contributor_result_ids": list(contributor_result_ids),
        "columns": [
            {
                "index": column.index,
                "symbols": {
                    str(hyp_idx): _symbol_to_json(column.symbols.get(hyp_idx, GAP))
                    for hyp_idx in range(matrix.hypothesis_count)
                },
            }
            for column in matrix.columns
        ],
    }


def write_alignment_artifact(
    output_root: Path, matrix: AlignmentMatrix, contributor_result_ids: tuple[str, ...]
) -> Path:
    """Persist the alignment artifact as a content-addressed blob under
    ``output_root/artifacts/alignment/`` and return its path.
    """
    payload = alignment_artifact_payload(matrix, contributor_result_ids)
    body = canonical_json_bytes(payload)
    digest = sha256_hex(body)
    blob_dir = output_root / "artifacts" / "alignment"
    blob_dir.mkdir(parents=True, exist_ok=True)
    blob_path = blob_dir / f"alignment-{digest}.json"
    blob_path.write_bytes(body)
    return blob_path


def seal_consensus_and_fusion(
    *,
    run_manifest_id: str,
    evaluation_unit_id: str,
    eligible_page_model_result_ids: tuple[str, ...],
    matrix: AlignmentMatrix,
    alignment_artifact_path: Path,
    consensus_entropy: float | Literal["not_available"],
    fused_text: str,
    created_at: str,
) -> tuple[ConsensusResult, SealedFusedHypothesis]:
    """Seal the canonical ``ConsensusResult`` and ``FusedHypothesis`` records
    for one Evaluation Unit's alignment (see the docstrings on both classes in
    ``records.py`` for their field definitions). Both reference the same
    persisted alignment artifact so postprocess can reproduce them from one
    source, satisfying reproducibility checks.
    """
    artifact_bytes = alignment_artifact_path.read_bytes()
    artifact_sha256 = sha256_hex(artifact_bytes)
    artifact_uri = alignment_artifact_path.resolve().as_uri()

    consensus_result = _seal(
        ConsensusResult(
            consensus_result_id="placeholder",
            run_manifest_id=run_manifest_id,
            evaluation_unit_id=evaluation_unit_id,
            eligible_page_model_result_ids=tuple(sorted(eligible_page_model_result_ids)),
            quorum_size=QUORUM_MINIMUM,
            eligible_hypothesis_count=matrix.hypothesis_count,
            consensus_entropy=(
                None if consensus_entropy == "not_available" else consensus_entropy
            ),
            alignment_artifact_sha256=artifact_sha256,
            alignment_artifact_uri=artifact_uri,
            text_policy_id=TEXT_POLICY_ID,
            alignment_policy_id=ALIGNMENT_POLICY_ID,
            created_at=created_at,
        )
    )

    fused_hypothesis = _seal(
        SealedFusedHypothesis(
            fused_hypothesis_id="placeholder",
            run_manifest_id=run_manifest_id,
            evaluation_unit_id=evaluation_unit_id,
            consensus_result_id=consensus_result.consensus_result_id,
            text=fused_text,
            fusion_policy_id=FUSION_POLICY_ID,
            alignment_artifact_sha256=artifact_sha256,
            alignment_artifact_uri=artifact_uri,
            created_at=created_at,
        )
    )

    return consensus_result, fused_hypothesis


def write_consensus_results(output_root: Path, results: list[ConsensusResult]) -> None:
    write_jsonl_atomic(
        output_root / "consensus_results.jsonl",
        (dataclasses.asdict(r) for r in results),
    )


def write_fused_hypotheses(output_root: Path, hypotheses: list[SealedFusedHypothesis]) -> None:
    write_jsonl_atomic(
        output_root / "fused_hypotheses.jsonl",
        (dataclasses.asdict(h) for h in hypotheses),
    )
