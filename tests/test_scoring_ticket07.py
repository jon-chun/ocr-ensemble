"""A7: score materialization for the one-page run (ticket 07).
Exercises the real ticket-02/03/05/06 pipeline end to end:
preprocess -> manifest/dispatch -> A4 result -> A6 GT snapshot -> A7
Benchmark Observation.
"""

from __future__ import annotations

import dataclasses
from decimal import Decimal
from pathlib import Path

import pytest

from ocr_ensemble.adapters.havi import HaviFailureModeAdapter
from ocr_ensemble.align import align_hypotheses, compute_consensus_entropy, fuse_hypotheses
from ocr_ensemble.budget import BudgetLedgerStore, Ledger
from ocr_ensemble.dispatch import StubBehavior, dispatch_pair
from ocr_ensemble.ground_truth import resolve_effective_snapshot
from ocr_ensemble.manifest import (
    seal_dispatch_intents,
    seal_run_manifest,
    seal_stub_dataset_split,
    seal_stub_roster_member,
)
from ocr_ensemble.preprocess import PreprocessRequest, preprocess_dataset
from ocr_ensemble.records import seal as _seal
from ocr_ensemble.results import derive_page_model_result
from ocr_ensemble.scoring import (
    GtReference,
    compute_scores,
    materialize_fused_observation,
    materialize_member_observation,
    resolve_gt_reference,
    write_benchmark_observations,
)
from ocr_ensemble.storage import read_jsonl

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = REPO_ROOT / "aiai-ocr-dataset"
FIXTURE_RELATIVE_PATH = "D-COMP/ocr_d-comp_low-1_20260706.png"

AUTHOR_ACTOR_ID = "havi-annotator-1"
APPROVER_ACTOR_ID = "havi-annotator-2"
AUTHOR_CREATED_AT = "2026-08-31T10:00:00Z"
APPROVER_CREATED_AT = "2026-08-31T11:00:00Z"


@pytest.fixture()
def pipeline(tmp_path: Path):
    output_root = tmp_path / "preprocess-out"
    artifacts = preprocess_dataset(
        PreprocessRequest(
            dataset_root=DATASET_ROOT,
            output_root=output_root,
            fixture_relative_paths=(FIXTURE_RELATIVE_PATH,),
        )
    )
    source_page = artifacts.source_pages[0]
    evaluation_unit = artifacts.evaluation_units[0]
    page_input_variant = artifacts.input_variants[0]

    roster_member = seal_stub_roster_member()
    dataset_split = seal_stub_dataset_split(
        evaluation_unit_id=evaluation_unit.evaluation_unit_id,
        dataset_id="havi_failure_mode_v1",
        dataset_version="aiai-ocr-dataset-2026-08-31",
    )
    manifest = seal_run_manifest(
        evaluation_unit=evaluation_unit,
        page_input_variant=page_input_variant,
        roster=(roster_member,),
        dataset_split=dataset_split,
        created_at="2026-08-31T00:00:00Z",
        budget_policy={"run_ceiling_usd": "1.00"},
        dataset_id="havi_failure_mode_v1",
        dataset_version="aiai-ocr-dataset-2026-08-31",
    )
    intents = seal_dispatch_intents(
        run_manifest=manifest,
        page_input_variant=page_input_variant,
        evaluation_unit=evaluation_unit,
        roster=(roster_member,),
    )

    store = BudgetLedgerStore({"run": Ledger(name="run", ceiling_usd=Decimal("1.00"))})
    journal_path = output_root / "attempt_events.jsonl"
    dispatch_pair(
        intent=intents[0],
        ledger_store=store,
        ledger_names=("run",),
        output_root=output_root,
        journal_path=journal_path,
        behavior=StubBehavior(fixed_text="the quick brown fox"),
    )
    result = derive_page_model_result(
        intent=intents[0], journal_path=journal_path, created_at="2026-08-31T00:01:00Z"
    )

    adapter = HaviFailureModeAdapter()
    imported = adapter.import_ground_truth(
        DATASET_ROOT,
        evaluation_unit_ids_by_dataset_item_id={
            FIXTURE_RELATIVE_PATH: evaluation_unit.evaluation_unit_id
        },
        author_actor_id=AUTHOR_ACTOR_ID,
        approver_actor_id=APPROVER_ACTOR_ID,
        author_created_at=AUTHOR_CREATED_AT,
        approver_created_at=APPROVER_CREATED_AT,
    )[0]
    snapshot = resolve_effective_snapshot(
        assertions=[imported.assertion], events=list(imported.initial_events)
    )

    return {
        "output_root": output_root,
        "manifest": manifest,
        "source_page": source_page,
        "evaluation_unit": evaluation_unit,
        "page_input_variant": page_input_variant,
        "roster_member": roster_member,
        "intent": intents[0],
        "result": result,
        "snapshot": snapshot,
        "assertions": {imported.assertion.assertion_id: imported.assertion},
        "gt_text": imported.assertion.text,
    }


def test_member_observation_references_every_upstream_stage(pipeline):
    gt_reference = resolve_gt_reference(
        evaluation_unit_id=pipeline["evaluation_unit"].evaluation_unit_id,
        snapshot=pipeline["snapshot"],
        assertions=pipeline["assertions"],
    )
    assert gt_reference.availability == "available"

    observation = materialize_member_observation(
        manifest=pipeline["manifest"],
        source_page=pipeline["source_page"],
        evaluation_unit=pipeline["evaluation_unit"],
        page_input_variant=pipeline["page_input_variant"],
        roster_member=pipeline["roster_member"],
        intent=pipeline["intent"],
        result=pipeline["result"],
        gt_snapshot=pipeline["snapshot"],
        gt_reference=gt_reference,
        split="diagnostic",
        created_at="2026-08-31T00:02:00Z",
    )

    assert observation.run_manifest_id == pipeline["manifest"].run_manifest_id
    assert observation.page_input_variant_id == pipeline["page_input_variant"].page_input_variant_id
    assert observation.subject_kind == "page_model_result"
    assert observation.subject_id == pipeline["result"].page_model_result_id
    assert observation.gt_snapshot_id == pipeline["snapshot"].effective_gt_snapshot_id
    assert observation.gt_availability == "available"
    assert observation.metric_policy_id == "metric_v1"
    assert observation.resource_attribution == "member_all_attempts"
    assert observation.attributed_cost_usd == pipeline["result"].total_cost_usd
    assert observation.outcome == "success"
    # the stub hypothesis text certainly does not match the real gold text
    assert observation.cer is not None and observation.cer > 0


def test_score_materializer_is_deterministic(pipeline):
    gt_reference = resolve_gt_reference(
        evaluation_unit_id=pipeline["evaluation_unit"].evaluation_unit_id,
        snapshot=pipeline["snapshot"],
        assertions=pipeline["assertions"],
    )
    kwargs = dict(
        manifest=pipeline["manifest"],
        source_page=pipeline["source_page"],
        evaluation_unit=pipeline["evaluation_unit"],
        page_input_variant=pipeline["page_input_variant"],
        roster_member=pipeline["roster_member"],
        intent=pipeline["intent"],
        result=pipeline["result"],
        gt_snapshot=pipeline["snapshot"],
        gt_reference=gt_reference,
        split="diagnostic",
        created_at="2026-08-31T00:02:00Z",
    )
    first = materialize_member_observation(**kwargs)
    second = materialize_member_observation(**kwargs)
    assert first.benchmark_observation_id == second.benchmark_observation_id
    assert first.record_sha256() == second.record_sha256()


def test_write_benchmark_observations_round_trips_via_atomic_jsonl(pipeline):
    gt_reference = resolve_gt_reference(
        evaluation_unit_id=pipeline["evaluation_unit"].evaluation_unit_id,
        snapshot=pipeline["snapshot"],
        assertions=pipeline["assertions"],
    )
    observation = materialize_member_observation(
        manifest=pipeline["manifest"],
        source_page=pipeline["source_page"],
        evaluation_unit=pipeline["evaluation_unit"],
        page_input_variant=pipeline["page_input_variant"],
        roster_member=pipeline["roster_member"],
        intent=pipeline["intent"],
        result=pipeline["result"],
        gt_snapshot=pipeline["snapshot"],
        gt_reference=gt_reference,
        split="diagnostic",
        created_at="2026-08-31T00:02:00Z",
    )
    write_benchmark_observations(pipeline["output_root"], [observation])
    rows = read_jsonl(pipeline["output_root"] / "benchmark_observations.jsonl")
    assert len(rows) == 1
    assert rows[0]["benchmark_observation_id"] == observation.benchmark_observation_id
    assert rows[0]["attributed_cost_usd"] == str(observation.attributed_cost_usd)


def test_fused_observation_references_alignment_and_ce(pipeline):
    # ticket 03's stub slice produces one roster member; synthesize a second
    # eligible hypothesis to exercise the quorum-2 fusion/CE path exactly as
    # ticket 06's own fixtures do, since real multi-member breadth is
    # ticket 09's scope.
    second_result = dataclasses.replace(
        pipeline["result"],
        page_model_result_id="placeholder",
        roster_member_config_id="roster_member_config_sha256:" + "9" * 64,
        parsed_text="the quick brown fax",
    )
    second_result = _seal(second_result)

    hypotheses = (pipeline["result"].parsed_text, second_result.parsed_text)
    matrix = align_hypotheses(hypotheses)
    ce = compute_consensus_entropy(matrix)
    fused = fuse_hypotheses(matrix)

    gt_reference = resolve_gt_reference(
        evaluation_unit_id=pipeline["evaluation_unit"].evaluation_unit_id,
        snapshot=pipeline["snapshot"],
        assertions=pipeline["assertions"],
    )

    observation = materialize_fused_observation(
        manifest=pipeline["manifest"],
        source_page=pipeline["source_page"],
        evaluation_unit=pipeline["evaluation_unit"],
        page_input_variant=pipeline["page_input_variant"],
        fused_hypothesis_id="fused_hypothesis_sha256:" + "d" * 64,
        fused_text=fused.text,
        member_results=(pipeline["result"], second_result),
        alignment_matrix=matrix,
        consensus_entropy=ce,
        quorum_size=2,
        gt_snapshot=pipeline["snapshot"],
        gt_reference=gt_reference,
        split="diagnostic",
        created_at="2026-08-31T00:02:00Z",
    )

    assert observation.subject_kind == "fused_hypothesis"
    assert observation.subject_id == "fused_hypothesis_sha256:" + "d" * 64
    assert observation.consensus_entropy == ce
    assert observation.quorum_size == 2
    assert observation.eligible_hypothesis_count == 2
    assert observation.resource_attribution == "ensemble_acquisition"
    assert observation.attributed_cost_usd == (
        pipeline["result"].total_cost_usd + second_result.total_cost_usd
    )
    assert observation.roster_member_config_id is None
    assert observation.exact_model_id is None


# ---------------------------------------------------------------------------
# blank / fully_illegible / no-GT semantics
# ---------------------------------------------------------------------------


def test_blank_reference_empty_hypothesis_yields_zero():
    gt = GtReference(availability="available", text="", assertion_id="a1")
    scores = compute_scores(gt, "")
    assert scores.cer == 0.0
    assert scores.wer == 0.0


def test_blank_reference_nonempty_hypothesis_is_insertion_count_not_division_by_zero():
    gt = GtReference(availability="available", text="", assertion_id="a1")
    scores = compute_scores(gt, "xyz")
    assert scores.cer == 3.0
    assert scores.char_insertions == 3


def test_fully_illegible_excludes_cer_wer_but_is_not_conflated_with_no_gt():
    gt = GtReference(availability="excluded_fully_illegible", text=None, assertion_id="a2")
    scores = compute_scores(gt, "some hallucinated text")
    assert scores.cer is None
    assert scores.wer is None
    # the hypothesis is still hashed for downstream abstention-correctness
    # scoring even though CER/WER are null
    assert scores.hypothesis_text_sha256 is not None


def test_no_gt_and_conflicted_gt_are_null_never_zero():
    for availability in ("unavailable", "conflicted"):
        gt = GtReference(availability=availability, text=None, assertion_id=None)
        scores = compute_scores(gt, "anything")
        assert scores.cer is None
        assert scores.wer is None
