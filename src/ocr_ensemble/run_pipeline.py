"""Run orchestrator: drives A0-A7 end to end for the one-fixture-page stub
slice into a ``runs/<run_id>/`` directory tree (the required
stores list), so postprocess (A8) and analyze (A9) have a real on-disk run
to validate and report on rather than an in-process fixture-only pipeline.

Ticket 08 scope: two stub roster members (so quorum-2 fusion/CE are
genuinely exercised, not degenerate) against the ticket-02 pinned fixture,
one Evaluation Unit, one Dataset Split row (``diagnostic``). Real provider
breadth (ticket 09), full corpus/whole-dataset preprocessing, and locked/
calibration run modes remain later tickets' scope -- this module's
``run_one_page_pipeline`` is intentionally narrow, not a general run
orchestrator for every future pipeline shape.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

from ocr_ensemble.adapters.havi import HaviFailureModeAdapter
from ocr_ensemble.align import (
    align_hypotheses,
    compute_consensus_entropy,
    fuse_hypotheses,
    seal_consensus_and_fusion,
    write_alignment_artifact,
    write_consensus_results,
    write_fused_hypotheses,
)
from ocr_ensemble.budget import BudgetLedgerStore, Ledger
from ocr_ensemble.dispatch import StubBehavior, dispatch_pair
from ocr_ensemble.ground_truth import resolve_effective_snapshot, write_effective_gt_snapshot
from ocr_ensemble.manifest import (
    seal_dispatch_intents,
    seal_run_manifest,
    seal_stub_dataset_split,
    seal_stub_roster_member,
    write_dispatch_intents,
    write_run_manifest,
)
from ocr_ensemble.preprocess import PreprocessRequest, preprocess_dataset
from ocr_ensemble.records import DispatchIntent, PageModelResult
from ocr_ensemble.results import derive_page_model_result, write_page_model_results
from ocr_ensemble.scoring import (
    materialize_fused_observation,
    materialize_member_observation,
    resolve_gt_reference,
    write_benchmark_observations,
)
from ocr_ensemble.storage import read_jsonl

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_ROOT = REPO_ROOT / "aiai-ocr-dataset"
FIXTURE_RELATIVE_PATH = "D-COMP/ocr_d-comp_low-1_20260706.png"

AUTHOR_ACTOR_ID = "havi-annotator-1"
APPROVER_ACTOR_ID = "havi-annotator-2"


@dataclass(frozen=True)
class RunPipelineResult:
    run_id: str
    output_root: Path


def run_one_page_pipeline(
    *,
    output_root: Path,
    dataset_root: Path = DEFAULT_DATASET_ROOT,
    run_created_at: str = "2026-08-31T00:00:00Z",
    member_texts: tuple[str, str] = ("the quick brown fox", "the quick brown fax"),
) -> RunPipelineResult:
    """Run the full A0-A7 pipeline for the pinned ticket-02 fixture and
    persist every required store under ``output_root`` (the
    ``runs/<run_id>/`` layout). Returns the run's manifest ID (``run_id``)
    and the root it was written to -- callers pass that same root as every
    postprocess/analyze CLI input path.
    """
    artifacts = preprocess_dataset(
        PreprocessRequest(
            dataset_root=dataset_root,
            output_root=output_root,
            fixture_relative_paths=(FIXTURE_RELATIVE_PATH,),
        )
    )
    source_page = artifacts.source_pages[0]
    evaluation_unit = artifacts.evaluation_units[0]
    page_input_variant = artifacts.input_variants[0]

    roster = (
        seal_stub_roster_member(position=0),
        seal_stub_roster_member(prompt_template_id="ocr_transcription_v1_b", position=1),
    )
    dataset_split = seal_stub_dataset_split(
        evaluation_unit_id=evaluation_unit.evaluation_unit_id,
        dataset_id="havi_failure_mode_v1",
        dataset_version="aiai-ocr-dataset-2026-08-31",
    )
    manifest = seal_run_manifest(
        evaluation_unit=evaluation_unit,
        page_input_variant=page_input_variant,
        roster=roster,
        dataset_split=dataset_split,
        created_at=run_created_at,
        budget_policy={"run_ceiling_usd": "1.00"},
        dataset_id="havi_failure_mode_v1",
        dataset_version="aiai-ocr-dataset-2026-08-31",
    )
    intents = seal_dispatch_intents(
        run_manifest=manifest,
        page_input_variant=page_input_variant,
        evaluation_unit=evaluation_unit,
        roster=roster,
    )
    write_run_manifest(output_root, manifest)
    write_dispatch_intents(output_root, intents)

    store = BudgetLedgerStore({"run": Ledger(name="run", ceiling_usd=Decimal("1.00"))})
    journal_path = output_root / "attempt_events.jsonl"
    results: list[PageModelResult] = []
    for intent, text in zip(intents, member_texts):
        dispatch_pair(
            intent=intent,
            ledger_store=store,
            ledger_names=("run",),
            output_root=output_root,
            journal_path=journal_path,
            behavior=StubBehavior(fixed_text=text),
        )
        result = derive_page_model_result(
            intent=intent, journal_path=journal_path, created_at=run_created_at
        )
        results.append(result)
    write_page_model_results(output_root, results)

    eligible_results = [r for r in results if r.terminal_outcome == "success"]
    hypotheses = tuple(r.parsed_text for r in eligible_results if r.parsed_text is not None)
    matrix = align_hypotheses(hypotheses)
    ce = compute_consensus_entropy(matrix)
    fused = fuse_hypotheses(matrix)
    alignment_path = write_alignment_artifact(
        output_root, matrix, tuple(r.page_model_result_id for r in eligible_results)
    )
    consensus_result, fused_hypothesis = seal_consensus_and_fusion(
        run_manifest_id=manifest.run_manifest_id,
        evaluation_unit_id=evaluation_unit.evaluation_unit_id,
        eligible_page_model_result_ids=tuple(r.page_model_result_id for r in eligible_results),
        matrix=matrix,
        alignment_artifact_path=alignment_path,
        consensus_entropy=ce,
        fused_text=fused.text,
        created_at=run_created_at,
    )
    write_consensus_results(output_root, [consensus_result])
    write_fused_hypotheses(output_root, [fused_hypothesis])

    adapter = HaviFailureModeAdapter()
    imported = adapter.import_ground_truth(
        dataset_root,
        evaluation_unit_ids_by_dataset_item_id={
            FIXTURE_RELATIVE_PATH: evaluation_unit.evaluation_unit_id
        },
        author_actor_id=AUTHOR_ACTOR_ID,
        approver_actor_id=APPROVER_ACTOR_ID,
        author_created_at=run_created_at,
        approver_created_at=run_created_at,
    )[0]
    snapshot = resolve_effective_snapshot(
        assertions=[imported.assertion], events=list(imported.initial_events)
    )
    write_effective_gt_snapshot(output_root, snapshot)

    gt_reference = resolve_gt_reference(
        evaluation_unit_id=evaluation_unit.evaluation_unit_id,
        snapshot=snapshot,
        assertions={imported.assertion.assertion_id: imported.assertion},
    )

    observations = []
    for intent, member, result in zip(intents, roster, results):
        observations.append(
            materialize_member_observation(
                manifest=manifest,
                source_page=source_page,
                evaluation_unit=evaluation_unit,
                page_input_variant=page_input_variant,
                roster_member=member,
                intent=intent,
                result=result,
                gt_snapshot=snapshot,
                gt_reference=gt_reference,
                split="diagnostic",
                created_at=run_created_at,
            )
        )
    if eligible_results:
        observations.append(
            materialize_fused_observation(
                manifest=manifest,
                source_page=source_page,
                evaluation_unit=evaluation_unit,
                page_input_variant=page_input_variant,
                fused_hypothesis_id=fused_hypothesis.fused_hypothesis_id,
                fused_text=fused.text,
                member_results=tuple(eligible_results),
                alignment_matrix=matrix,
                consensus_entropy=ce,
                quorum_size=consensus_result.quorum_size,
                gt_snapshot=snapshot,
                gt_reference=gt_reference,
                split="diagnostic",
                created_at=run_created_at,
            )
        )
    write_benchmark_observations(output_root, observations)

    return RunPipelineResult(run_id=manifest.run_manifest_id, output_root=output_root)


def cli_main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="ocr-ensemble-fix-run-pipeline")
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    args = parser.parse_args(argv)

    result = run_one_page_pipeline(output_root=args.output_root, dataset_root=args.dataset_root)
    print(f"run_id: {result.run_id}")
    print(f"output_root: {result.output_root}")
    return 0
