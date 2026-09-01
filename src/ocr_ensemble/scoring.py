"""A7: score materialization -- the sole owner of Benchmark Observation
creation (ticket 07). No other stage in this ticket set
computes or re-derives a canonical score.

Ticket 07 scope: one fixture page, one roster member's Page-Model Result,
one Effective Ground-Truth Snapshot, plus (when quorum allows) the fused
observation from ticket 06's alignment/fusion output. Full N-member breadth
and imported-baseline observations are later tickets' scope; this module's
``materialize_scores`` signature is written to a general
shape so those later tickets extend it rather than replace it.
"""

from __future__ import annotations

import dataclasses
import re
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Literal

from ocr_ensemble.align import (
    ILLEGIBLE_MARKER,
    AlignmentMatrix,
    normalize_text,
)
from ocr_ensemble.identity import sha256_hex
from ocr_ensemble.records import (
    BenchmarkObservation,
    DispatchIntent,
    EffectiveGroundTruthSnapshot,
    EvaluationUnit,
    GroundTruthAssertion,
    PageInputVariant,
    PageModelResult,
    RosterMemberConfiguration,
    RunManifest,
    SourcePage,
)
from ocr_ensemble.records import seal as _seal
from ocr_ensemble.storage import write_jsonl_atomic

METRIC_POLICY_ID = "metric_v1"
TEXT_POLICY_ID = "ocr_exact_text_v1"

# Unicode 15.1 White_Space=Yes code points, pinned as explicit codepoints
# rather than inherited from a changing runtime default, to keep character
# tokenization stable across Python/Unicode upgrades. 29 codepoints total.
_WHITE_SPACE_CODEPOINTS = (
    0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x1C, 0x1D, 0x1E, 0x1F, 0x20,
    0x85, 0xA0, 0x1680,
    0x2000, 0x2001, 0x2002, 0x2003, 0x2004, 0x2005, 0x2006, 0x2007,
    0x2008, 0x2009, 0x200A, 0x2028, 0x2029, 0x202F, 0x205F, 0x3000,
)
_WHITE_SPACE_CHARS = "".join(chr(cp) for cp in _WHITE_SPACE_CODEPOINTS)
_WORD_SPLIT_RE = re.compile("[" + re.escape(_WHITE_SPACE_CHARS) + "]+")


@dataclass(frozen=True)
class GtReference:
    """The resolved ground-truth reference for one evaluation unit, or the
    reason none is usable (blank/fully_illegible/transcribable
    semantics; the ``gt_availability`` states).
    """

    availability: Literal["available", "unavailable", "conflicted", "excluded_fully_illegible"]
    text: str | None
    assertion_id: str | None


def resolve_gt_reference(
    *,
    evaluation_unit_id: str,
    snapshot: EffectiveGroundTruthSnapshot,
    assertions: dict[str, GroundTruthAssertion],
) -> GtReference:
    """Project the snapshot's conflict/unavailable sets plus its
    ``assertion_ids`` into exactly one reference decision for one unit.
    ``fully_illegible`` is a distinct, deliberate non-error state,
    never conflated with "no ground truth."
    """
    if evaluation_unit_id in snapshot.conflicted_evaluation_unit_ids:
        return GtReference(availability="conflicted", text=None, assertion_id=None)
    if evaluation_unit_id in snapshot.unavailable_evaluation_unit_ids:
        return GtReference(availability="unavailable", text=None, assertion_id=None)

    matches = [
        assertions[aid]
        for aid in snapshot.assertion_ids
        if aid in assertions and assertions[aid].evaluation_unit_id == evaluation_unit_id
    ]
    if not matches:
        return GtReference(availability="unavailable", text=None, assertion_id=None)
    # exactly one active head per target is the snapshot's own invariant;
    # conflicted/unavailable are already handled above.
    assertion = matches[0]
    if assertion.target_state == "fully_illegible":
        return GtReference(
            availability="excluded_fully_illegible", text=None, assertion_id=assertion.assertion_id
        )
    return GtReference(
        availability="available", text=assertion.text or "", assertion_id=assertion.assertion_id
    )


@dataclass(frozen=True)
class EditCounts:
    substitutions: int
    insertions: int
    deletions: int


def _char_tokens(text: str) -> tuple[str, ...]:
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


def _word_tokens(text: str) -> tuple[str, ...]:
    stripped = _WORD_SPLIT_RE.sub(" ", text).strip(" ")
    if not stripped:
        return ()
    return tuple(stripped.split(" "))


def _edit_counts(reference: tuple[str, ...], hypothesis: tuple[str, ...]) -> EditCounts:
    """Standard unit-cost Levenshtein edit-count DP with a traceback that
    classifies each edit as substitution/insertion/deletion (reference is
    the "true" side: an insertion is a hypothesis-only token, a deletion is
    a reference-only token that the hypothesis is missing).
    """
    n, m = len(reference), len(hypothesis)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
    for j in range(1, m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if reference[i - 1] == hypothesis[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j - 1] + cost, dp[i - 1][j] + 1, dp[i][j - 1] + 1)

    subs = ins = dels = 0
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            cost = 0 if reference[i - 1] == hypothesis[j - 1] else 1
            if dp[i][j] == dp[i - 1][j - 1] + cost:
                if cost:
                    subs += 1
                i -= 1
                j -= 1
                continue
        if i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            dels += 1
            i -= 1
            continue
        ins += 1
        j -= 1

    return EditCounts(substitutions=subs, insertions=ins, deletions=dels)


@dataclass(frozen=True)
class RateResult:
    rate: float | None
    edits: EditCounts | None
    reference_length: int | None


def _compute_rate(reference_tokens: tuple[str, ...], hypothesis_tokens: tuple[str, ...]) -> RateResult:
    """Blank-reference convention: empty ref + empty hyp -> 0;
    empty ref + nonempty hyp -> insertion count (never divide by zero);
    nonempty ref + empty hyp -> 1.0.
    """
    ref_len = len(reference_tokens)
    if ref_len == 0:
        if len(hypothesis_tokens) == 0:
            return RateResult(rate=0.0, edits=EditCounts(0, 0, 0), reference_length=0)
        edits = EditCounts(substitutions=0, insertions=len(hypothesis_tokens), deletions=0)
        return RateResult(rate=float(len(hypothesis_tokens)), edits=edits, reference_length=0)

    edits = _edit_counts(reference_tokens, hypothesis_tokens)
    rate = (edits.substitutions + edits.insertions + edits.deletions) / ref_len
    return RateResult(rate=rate, edits=edits, reference_length=ref_len)


@dataclass(frozen=True)
class ScoreComputation:
    cer: float | None
    wer: float | None
    reference_length_chars: int | None
    reference_length_words: int | None
    char_substitutions: int | None
    char_insertions: int | None
    char_deletions: int | None
    word_substitutions: int | None
    word_insertions: int | None
    word_deletions: int | None
    hypothesis_text_sha256: str | None


def compute_scores(gt: GtReference, hypothesis_text: str | None) -> ScoreComputation:
    """CER/WER honoring blank/fully_illegible/
    transcribable semantics. ``fully_illegible`` and no-GT/conflicted-GT
    states get null metrics, never zero: no-GT or conflicted-GT
    fields are null, never zero.
    """
    if gt.availability != "available":
        return ScoreComputation(
            cer=None,
            wer=None,
            reference_length_chars=None,
            reference_length_words=None,
            char_substitutions=None,
            char_insertions=None,
            char_deletions=None,
            word_substitutions=None,
            word_insertions=None,
            word_deletions=None,
            hypothesis_text_sha256=(
                sha256_hex(normalize_text(hypothesis_text).encode("utf-8"))
                if hypothesis_text is not None
                else None
            ),
        )

    reference_norm = normalize_text(gt.text or "")
    hypothesis_norm = normalize_text(hypothesis_text) if hypothesis_text is not None else ""
    hypothesis_sha = (
        sha256_hex(hypothesis_norm.encode("utf-8")) if hypothesis_text is not None else None
    )

    char_result = _compute_rate(_char_tokens(reference_norm), _char_tokens(hypothesis_norm))
    word_result = _compute_rate(_word_tokens(reference_norm), _word_tokens(hypothesis_norm))

    return ScoreComputation(
        cer=char_result.rate,
        wer=word_result.rate,
        reference_length_chars=char_result.reference_length,
        reference_length_words=word_result.reference_length,
        char_substitutions=char_result.edits.substitutions if char_result.edits else None,
        char_insertions=char_result.edits.insertions if char_result.edits else None,
        char_deletions=char_result.edits.deletions if char_result.edits else None,
        word_substitutions=word_result.edits.substitutions if word_result.edits else None,
        word_insertions=word_result.edits.insertions if word_result.edits else None,
        word_deletions=word_result.edits.deletions if word_result.edits else None,
        hypothesis_text_sha256=hypothesis_sha,
    )


def materialize_member_observation(
    *,
    manifest: RunManifest,
    source_page: SourcePage,
    evaluation_unit: EvaluationUnit,
    page_input_variant: PageInputVariant,
    roster_member: RosterMemberConfiguration,
    intent: DispatchIntent,
    result: PageModelResult,
    gt_snapshot: EffectiveGroundTruthSnapshot,
    gt_reference: GtReference,
    split: Literal["calibration", "locked_evaluation", "diagnostic"],
    created_at: str,
) -> BenchmarkObservation:
    """One raw-member observation for a single Page-Model Result.
    Attributed cost/latency are that result's totals across every
    attempt (``resource_attribution="member_all_attempts"``).
    """
    fully_illegible = gt_reference.availability == "excluded_fully_illegible"
    scores = compute_scores(gt_reference, result.parsed_text)

    return _seal(
        BenchmarkObservation(
            benchmark_observation_id="placeholder",
            run_manifest_id=manifest.run_manifest_id,
            dataset_id=manifest.dataset_id,
            dataset_version=manifest.dataset_version,
            corpus_evidence_kind=manifest.corpus_evidence_kind,
            split=split,
            evaluation_unit_id=evaluation_unit.evaluation_unit_id,
            source_page_id=source_page.source_page_id,
            source_sha256=source_page.source_sha256,
            page_input_variant_id=page_input_variant.page_input_variant_id,
            input_variant_sha256=page_input_variant.artifact_sha256,
            subject_kind="page_model_result",
            subject_id=result.page_model_result_id,
            roster_member_config_id=roster_member.roster_member_config_id,
            exact_model_id=roster_member.exact_model_id,
            provider=roster_member.provider,
            router=roster_member.router,
            prompt_sha256=roster_member.prompt_sha256,
            hyperparameters_sha256=(
                sha256_hex(repr(sorted(roster_member.hyperparameters.items())).encode("utf-8"))
                if roster_member.hyperparameters
                else None
            ),
            adapter_version=roster_member.adapter_version,
            challenge_seed_family=None,
            failure_condition_codes=(),
            evaluation_scope_status=evaluation_unit.evaluation_scope_status,
            layout_complexity=evaluation_unit.layout_complexity,
            language=evaluation_unit.language,
            mixed_language=evaluation_unit.mixed_language,
            gt_snapshot_id=gt_snapshot.effective_gt_snapshot_id,
            gt_assertion_id=gt_reference.assertion_id,
            gt_availability=(
                "excluded_fully_illegible" if fully_illegible else gt_reference.availability
            ),
            metric_policy_id=METRIC_POLICY_ID,
            hypothesis_text_sha256=scores.hypothesis_text_sha256,
            reference_length_chars=scores.reference_length_chars,
            reference_length_words=scores.reference_length_words,
            char_substitutions=scores.char_substitutions,
            char_insertions=scores.char_insertions,
            char_deletions=scores.char_deletions,
            word_substitutions=scores.word_substitutions,
            word_insertions=scores.word_insertions,
            word_deletions=scores.word_deletions,
            cer=scores.cer,
            wer=scores.wer,
            consensus_entropy=None,
            quorum_size=None,
            eligible_hypothesis_count=None,
            outcome=result.terminal_outcome,
            attributed_cost_usd=result.total_cost_usd,
            attributed_latency_ms=result.total_measured_duration_ms,
            resource_attribution="member_all_attempts",
            created_at=created_at,
        )
    )


def materialize_fused_observation(
    *,
    manifest: RunManifest,
    source_page: SourcePage,
    evaluation_unit: EvaluationUnit,
    page_input_variant: PageInputVariant,
    fused_hypothesis_id: str,
    fused_text: str,
    member_results: tuple[PageModelResult, ...],
    alignment_matrix: AlignmentMatrix,
    consensus_entropy: float | Literal["not_available"],
    quorum_size: int,
    gt_snapshot: EffectiveGroundTruthSnapshot,
    gt_reference: GtReference,
    split: Literal["calibration", "locked_evaluation", "diagnostic"],
    created_at: str,
) -> BenchmarkObservation:
    """The fused observation: ``ensemble_acquisition`` cost
    sums every member Page-Model Result's cost (including failed/ineligible
    paid attempts) for the unit; latency is wall-clock earliest-start to
    latest-finish, which this ticket's stub single-attempt fixtures cannot
    exercise meaningfully -- callers pass the already-computed wall-clock
    span; ticket 06/07's own fixture uses the summed ``total_measured_duration_ms``
    as an honest stand-in until multi-member timing is real (ticket 09+).
    """
    fully_illegible = gt_reference.availability == "excluded_fully_illegible"
    scores = compute_scores(gt_reference, fused_text)
    total_cost = sum((r.total_cost_usd for r in member_results), Decimal("0"))
    total_latency = sum(r.total_measured_duration_ms for r in member_results)

    return _seal(
        BenchmarkObservation(
            benchmark_observation_id="placeholder",
            run_manifest_id=manifest.run_manifest_id,
            dataset_id=manifest.dataset_id,
            dataset_version=manifest.dataset_version,
            corpus_evidence_kind=manifest.corpus_evidence_kind,
            split=split,
            evaluation_unit_id=evaluation_unit.evaluation_unit_id,
            source_page_id=source_page.source_page_id,
            source_sha256=source_page.source_sha256,
            page_input_variant_id=page_input_variant.page_input_variant_id,
            input_variant_sha256=page_input_variant.artifact_sha256,
            subject_kind="fused_hypothesis",
            subject_id=fused_hypothesis_id,
            roster_member_config_id=None,
            exact_model_id=None,
            provider=None,
            router=None,
            prompt_sha256=None,
            hyperparameters_sha256=None,
            adapter_version=None,
            challenge_seed_family=None,
            failure_condition_codes=(),
            evaluation_scope_status=evaluation_unit.evaluation_scope_status,
            layout_complexity=evaluation_unit.layout_complexity,
            language=evaluation_unit.language,
            mixed_language=evaluation_unit.mixed_language,
            gt_snapshot_id=gt_snapshot.effective_gt_snapshot_id,
            gt_assertion_id=gt_reference.assertion_id,
            gt_availability=(
                "excluded_fully_illegible" if fully_illegible else gt_reference.availability
            ),
            metric_policy_id=METRIC_POLICY_ID,
            hypothesis_text_sha256=scores.hypothesis_text_sha256,
            reference_length_chars=scores.reference_length_chars,
            reference_length_words=scores.reference_length_words,
            char_substitutions=scores.char_substitutions,
            char_insertions=scores.char_insertions,
            char_deletions=scores.char_deletions,
            word_substitutions=scores.word_substitutions,
            word_insertions=scores.word_insertions,
            word_deletions=scores.word_deletions,
            cer=scores.cer,
            wer=scores.wer,
            consensus_entropy=(
                None if consensus_entropy == "not_available" else consensus_entropy
            ),
            quorum_size=quorum_size,
            eligible_hypothesis_count=len(member_results),
            outcome="success",
            attributed_cost_usd=total_cost,
            attributed_latency_ms=total_latency,
            resource_attribution="ensemble_acquisition",
            created_at=created_at,
        )
    )


def write_benchmark_observations(output_root: Path, observations: list[BenchmarkObservation]) -> None:
    write_jsonl_atomic(
        output_root / "benchmark_observations.jsonl",
        (dataclasses.asdict(o) for o in observations),
    )
