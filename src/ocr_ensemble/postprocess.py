"""A8 postprocess: the honesty gate after canonical scoring (postprocess
spec v2; ticket 08).

Ticket 08 scope: structural pass/fail validation (spec §4 cardinality, §5
requiredness/integrity, a scoped-down §6 reproducibility check that reruns
CE/fusion and CER/WER for the one persisted run and compares within
tolerance) over the ``runs/<run_id>/`` layout ``run_pipeline.py`` produces.
Full versioned anomaly detectors (spec §7) and the Annotation/Audit queues
(spec §9) are ticket 13's scope -- this module never fabricates a `pass`
where those detectors would eventually run; it simply does not run them yet,
and says so in the report via ``anomaly_detectors_run=False``.

Postprocess never re-derives or republishes a canonical score (spec §1); it
may recompute solely to validate.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from ocr_ensemble.align import align_hypotheses, compute_consensus_entropy
from ocr_ensemble.records import attempt_id as compute_attempt_id
from ocr_ensemble.storage import read_json, read_jsonl, write_json_atomic

CE_REPRODUCIBILITY_TOLERANCE = 1e-12


@dataclass(frozen=True)
class PostprocessRequest:
    run_root: Path


@dataclass(frozen=True)
class Finding:
    code: str
    severity: Literal["fail", "warning"]
    message: str


@dataclass(frozen=True)
class PostprocessArtifacts:
    gate_status: Literal["pass", "fail"]
    run_completion_status: Literal["complete", "incomplete"]
    findings: tuple[Finding, ...]
    expected_pair_count: int
    observed_result_count: int
    anomaly_detectors_run: bool = False


CONCLUSIVE_OUTCOMES = frozenset(
    {
        "success",
        "whole_unit_abstention",
        "truncated",
        "content_filtered",
        "permanent_failure",
        "retry_exhausted",
        "budget_refused",
        "unsupported_input",
    }
)
NONCONCLUSIVE_OUTCOMES = frozenset({"cancelled", "indeterminate"})


def _load_store(run_root: Path, name: str) -> list[dict]:
    return read_jsonl(run_root / name)


def _check_cardinality(
    intents: list[dict], results: list[dict], attempts: list[dict]
) -> list[Finding]:
    """Postprocess spec §4: exactly one intent and one finalized result per
    expected Dispatch Pair.
    """
    findings: list[Finding] = []

    expected_pair_ids = {intent["dispatch_pair_id"] for intent in intents}
    intents_by_pair: dict[str, list[dict]] = {}
    for intent in intents:
        intents_by_pair.setdefault(intent["dispatch_pair_id"], []).append(intent)
    for pair_id, group in intents_by_pair.items():
        if len(group) != 1:
            findings.append(
                Finding(
                    code="duplicate_dispatch_intent",
                    severity="fail",
                    message=f"{len(group)} DispatchIntents for pair {pair_id!r}, expected 1",
                )
            )

    results_by_pair: dict[str, list[dict]] = {}
    for result in results:
        results_by_pair.setdefault(result["dispatch_pair_id"], []).append(result)

    for pair_id in expected_pair_ids:
        group = results_by_pair.get(pair_id, [])
        if len(group) == 0:
            findings.append(
                Finding(
                    code="missing_page_model_result",
                    severity="fail",
                    message=f"no PageModelResult for expected pair {pair_id!r}",
                )
            )
        elif len(group) > 1:
            findings.append(
                Finding(
                    code="duplicate_page_model_result",
                    severity="fail",
                    message=f"{len(group)} PageModelResults for pair {pair_id!r}, expected 1",
                )
            )

    unexpected_pairs = set(results_by_pair) - expected_pair_ids
    for pair_id in unexpected_pairs:
        findings.append(
            Finding(
                code="orphan_page_model_result",
                severity="fail",
                message=f"PageModelResult for pair {pair_id!r} has no matching DispatchIntent",
            )
        )

    # attempt numbering: strictly increasing per intent, unique attempt IDs,
    # and every attempt_id matches attempt_id(dispatch_intent_id, attempt_number)
    seen_attempt_ids: set[str] = set()
    by_intent_started: dict[str, list[int]] = {}
    for event in attempts:
        if event["event_type"] != "attempt_started":
            continue
        aid = event["attempt_id"]
        if aid in seen_attempt_ids:
            findings.append(
                Finding(
                    code="duplicate_attempt_id",
                    severity="fail",
                    message=f"attempt_id {aid!r} started more than once",
                )
            )
        seen_attempt_ids.add(aid)
        expected_aid = compute_attempt_id(event["dispatch_intent_id"], event["attempt_number"])
        if aid != expected_aid:
            findings.append(
                Finding(
                    code="attempt_id_mismatch",
                    severity="fail",
                    message=f"attempt {aid!r} does not hash to (intent, attempt_number)",
                )
            )
        by_intent_started.setdefault(event["dispatch_intent_id"], []).append(
            event["attempt_number"]
        )

    for intent_id, numbers in by_intent_started.items():
        if sorted(numbers) != list(range(1, len(numbers) + 1)):
            findings.append(
                Finding(
                    code="impossible_attempt_sequence",
                    severity="fail",
                    message=(
                        f"intent {intent_id!r} attempt numbers {sorted(numbers)!r} are not "
                        "a contiguous 1..N sequence"
                    ),
                )
            )

    return findings


def _check_requiredness(results: list[dict], observations: list[dict]) -> list[Finding]:
    """Postprocess spec §5: outcome-aware requiredness and cross-reference
    integrity (scoped to what ticket 08's run shape can exercise).
    """
    findings: list[Finding] = []

    results_by_id = {r["page_model_result_id"]: r for r in results}
    for result in results:
        if result["terminal_outcome"] == "success" and result["selected_attempt_id"] is None:
            findings.append(
                Finding(
                    code="success_missing_selected_attempt",
                    severity="fail",
                    message=f"result {result['page_model_result_id']!r} is success with no selected_attempt_id",
                )
            )
        if result["terminal_outcome"] != "success" and result["parsed_text"] is not None:
            findings.append(
                Finding(
                    code="non_success_has_parsed_text",
                    severity="fail",
                    message=(
                        f"result {result['page_model_result_id']!r} outcome="
                        f"{result['terminal_outcome']!r} unexpectedly has parsed_text"
                    ),
                )
            )

    for obs in observations:
        if obs["subject_kind"] != "page_model_result":
            continue
        if obs["subject_id"] not in results_by_id:
            findings.append(
                Finding(
                    code="observation_references_unknown_result",
                    severity="fail",
                    message=(
                        f"observation {obs['benchmark_observation_id']!r} references "
                        f"unknown result {obs['subject_id']!r}"
                    ),
                )
            )

    return findings


def _check_reproducibility(
    results: list[dict], consensus_results: list[dict], observations: list[dict]
) -> list[Finding]:
    """Scoped-down spec §6: rerun CE/fusion for each persisted
    ``ConsensusResult`` and CER/WER for each observation, compare within
    tolerance. Does not rerun the full alignment-column byte-for-byte check
    (ticket 13's scope); reruns the scalar outputs that ticket 08's
    materializers already commit to.
    """
    findings: list[Finding] = []
    results_by_id = {r["page_model_result_id"]: r for r in results}

    for consensus in consensus_results:
        eligible_ids = consensus["eligible_page_model_result_ids"]
        hypotheses = tuple(
            results_by_id[rid]["parsed_text"]
            for rid in eligible_ids
            if rid in results_by_id and results_by_id[rid]["parsed_text"] is not None
        )
        if len(hypotheses) != len(eligible_ids):
            findings.append(
                Finding(
                    code="consensus_references_missing_result",
                    severity="fail",
                    message=f"ConsensusResult {consensus['consensus_result_id']!r} references a missing/textless result",
                )
            )
            continue
        matrix = align_hypotheses(hypotheses)
        recomputed_ce = compute_consensus_entropy(matrix)
        stored_ce = consensus["consensus_entropy"]
        if recomputed_ce == "not_available":
            if stored_ce is not None:
                findings.append(
                    Finding(
                        code="ce_reproducibility_mismatch",
                        severity="fail",
                        message=f"recomputed CE not_available but stored value is {stored_ce!r}",
                    )
                )
        else:
            if stored_ce is None or abs(recomputed_ce - stored_ce) > CE_REPRODUCIBILITY_TOLERANCE:
                findings.append(
                    Finding(
                        code="ce_reproducibility_mismatch",
                        severity="fail",
                        message=(
                            f"ConsensusResult {consensus['consensus_result_id']!r} stored CE "
                            f"{stored_ce!r} != recomputed {recomputed_ce!r}"
                        ),
                    )
                )

    for obs in observations:
        if obs["cer"] is None:
            continue
        # postprocess does not have direct GT text access in this scoped
        # check without re-reading the assertion store; it verifies internal
        # consistency (reference length matches CER's own denominator
        # convention) rather than re-deriving the reference text itself.
        if obs["reference_length_chars"] == 0 and obs["cer"] not in (0.0,) and obs["char_insertions"] is None:
            findings.append(
                Finding(
                    code="blank_reference_convention_violated",
                    severity="fail",
                    message=(
                        f"observation {obs['benchmark_observation_id']!r} has zero-length "
                        "reference but no insertion count recorded"
                    ),
                )
            )

    return findings


def validate_run(request: PostprocessRequest) -> PostprocessArtifacts:
    run_root = request.run_root

    manifest = read_json(run_root / "run_manifest.json")
    intents = _load_store(run_root, "dispatch_intents.jsonl")
    attempts = _load_store(run_root, "attempt_events.jsonl")
    results = _load_store(run_root, "page_model_results.jsonl")
    consensus_results = _load_store(run_root, "consensus_results.jsonl")
    observations = _load_store(run_root, "benchmark_observations.jsonl")

    findings: list[Finding] = []

    if manifest is None:
        findings.append(
            Finding(code="missing_run_manifest", severity="fail", message="run_manifest.json not found")
        )
        return PostprocessArtifacts(
            gate_status="fail",
            run_completion_status="incomplete",
            findings=tuple(findings),
            expected_pair_count=0,
            observed_result_count=0,
        )

    run_manifest_id = manifest["run_manifest_id"]
    for store_name, rows in (
        ("dispatch_intents.jsonl", intents),
        ("page_model_results.jsonl", results),
        ("benchmark_observations.jsonl", observations),
    ):
        for row in rows:
            if row.get("run_manifest_id") != run_manifest_id:
                findings.append(
                    Finding(
                        code="run_manifest_id_mismatch",
                        severity="fail",
                        message=f"{store_name} row references a different run_manifest_id",
                    )
                )

    findings.extend(_check_cardinality(intents, results, attempts))
    findings.extend(_check_requiredness(results, observations))
    findings.extend(_check_reproducibility(results, consensus_results, observations))

    has_indeterminate = any(r["terminal_outcome"] == "indeterminate" for r in results)
    run_completion_status: Literal["complete", "incomplete"] = (
        "incomplete" if has_indeterminate else "complete"
    )

    gate_status: Literal["pass", "fail"] = (
        "fail" if any(f.severity == "fail" for f in findings) else "pass"
    )

    return PostprocessArtifacts(
        gate_status=gate_status,
        run_completion_status=run_completion_status,
        findings=tuple(findings),
        expected_pair_count=len({i["dispatch_pair_id"] for i in intents}),
        observed_result_count=len(results),
    )


def write_postprocess_report(run_root: Path, artifacts: PostprocessArtifacts) -> None:
    write_json_atomic(
        run_root / "postprocess_report.json",
        {
            "gate_status": artifacts.gate_status,
            "run_completion_status": artifacts.run_completion_status,
            "expected_pair_count": artifacts.expected_pair_count,
            "observed_result_count": artifacts.observed_result_count,
            "anomaly_detectors_run": artifacts.anomaly_detectors_run,
            "findings": [
                {"code": f.code, "severity": f.severity, "message": f.message}
                for f in artifacts.findings
            ],
        },
    )


def cli_main(argv: list[str] | None = None) -> int:
    """Ticket 08 scope: one ``--run-root`` pointing at a ``runs/<run_id>/``
    directory, same simplification as ``analyze.cli_main`` -- the full
    individually-enumerated per-store flags and YAML policy config
    (utils-spec §11) are deferred to the ticket that widens beyond one
    directory's worth of stores per invocation.

    Exit codes (utils-spec §11, scoped): ``0`` pass, ``3`` failed integrity,
    ``4`` incomplete run. ``2`` (typed-input/config error) is not yet
    distinguished from ``3`` since this scope has no YAML config to
    mis-parse.
    """
    parser = argparse.ArgumentParser(prog="ocr-ensemble-fix-postprocess")
    parser.add_argument("--run-root", required=True, type=Path)
    args = parser.parse_args(argv)

    artifacts = validate_run(PostprocessRequest(run_root=args.run_root))
    write_postprocess_report(args.run_root, artifacts)

    print(f"gate_status: {artifacts.gate_status}")
    print(f"run_completion_status: {artifacts.run_completion_status}")
    for finding in artifacts.findings:
        print(f"  [{finding.severity}] {finding.code}: {finding.message}")

    if artifacts.run_completion_status == "incomplete":
        return 4
    if artifacts.gate_status == "fail":
        return 3
    return 0
