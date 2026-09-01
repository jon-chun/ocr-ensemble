"""A4: derive Page-Model Result from the attempt journal (ticket 04)."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest

from ocr_ensemble.budget import BudgetLedgerStore, Ledger
from ocr_ensemble.dispatch import BudgetRefused, StubBehavior, dispatch_pair
from ocr_ensemble.manifest import (
    seal_dispatch_intents,
    seal_run_manifest,
    seal_stub_dataset_split,
    seal_stub_roster_member,
)
from ocr_ensemble.preprocess import PreprocessRequest, preprocess_dataset
from ocr_ensemble.results import derive_page_model_result, write_page_model_results
from ocr_ensemble.storage import read_jsonl

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = REPO_ROOT / "aiai-ocr-dataset"
FIXTURE_RELATIVE_PATH = "D-COMP/ocr_d-comp_low-1_20260706.png"


@pytest.fixture()
def sealed_fixture(tmp_path: Path):
    output_root = tmp_path / "preprocess-out"
    artifacts = preprocess_dataset(
        PreprocessRequest(
            dataset_root=DATASET_ROOT,
            output_root=output_root,
            fixture_relative_paths=(FIXTURE_RELATIVE_PATH,),
        )
    )
    return output_root, artifacts.evaluation_units[0], artifacts.input_variants[0]


@pytest.fixture()
def manifest_and_intent(sealed_fixture):
    output_root, evaluation_unit, page_input_variant = sealed_fixture
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
    return output_root, manifest, intents


def test_success_on_first_attempt_is_selected(manifest_and_intent):
    output_root, _manifest, intents = manifest_and_intent
    store = BudgetLedgerStore({"run": Ledger(name="run", ceiling_usd=Decimal("1.00"))})
    journal_path = output_root / "attempt_events.jsonl"

    dispatch_pair(
        intent=intents[0],
        ledger_store=store,
        ledger_names=("run",),
        output_root=output_root,
        journal_path=journal_path,
        behavior=StubBehavior(),
    )

    result = derive_page_model_result(
        intent=intents[0], journal_path=journal_path, created_at="2026-08-31T00:01:00Z"
    )

    assert result.terminal_outcome == "success"
    assert result.eligibility == "eligible"
    assert result.ineligibility_reasons == ()
    assert result.parsed_text == "the quick brown fox"
    assert len(result.attempt_ids) == 1
    assert result.selected_attempt_id == result.attempt_ids[0]
    assert result.total_cost_usd == Decimal("0.005")
    assert result.dispatch_pair_id == intents[0].dispatch_pair_id
    assert result.run_manifest_id == intents[0].run_manifest_id


def test_retry_then_success_selects_the_successful_attempt_only(manifest_and_intent):
    output_root, _manifest, intents = manifest_and_intent
    store = BudgetLedgerStore({"run": Ledger(name="run", ceiling_usd=Decimal("1.00"))})
    journal_path = output_root / "attempt_events.jsonl"

    dispatch_pair(
        intent=intents[0],
        ledger_store=store,
        ledger_names=("run",),
        output_root=output_root,
        journal_path=journal_path,
        behavior=StubBehavior(fail_attempts=2),
    )

    result = derive_page_model_result(
        intent=intents[0], journal_path=journal_path, created_at="2026-08-31T00:01:00Z"
    )

    assert result.terminal_outcome == "success"
    assert result.eligibility == "eligible"
    assert len(result.attempt_ids) == 3
    assert result.selected_attempt_id == result.attempt_ids[-1]
    assert result.parsed_text == "the quick brown fox"
    # cost/duration must sum every charged attempt, not only the selected one
    assert result.total_cost_usd == Decimal("0.005")


def test_retry_then_exhausted_derives_retry_exhausted(manifest_and_intent):
    output_root, _manifest, intents = manifest_and_intent
    store = BudgetLedgerStore({"run": Ledger(name="run", ceiling_usd=Decimal("1.00"))})
    journal_path = output_root / "attempt_events.jsonl"

    dispatch_pair(
        intent=intents[0],
        ledger_store=store,
        ledger_names=("run",),
        output_root=output_root,
        journal_path=journal_path,
        behavior=StubBehavior(fail_attempts=4),
    )

    result = derive_page_model_result(
        intent=intents[0], journal_path=journal_path, created_at="2026-08-31T00:01:00Z"
    )

    assert result.terminal_outcome == "retry_exhausted"
    assert result.eligibility == "ineligible"
    assert result.ineligibility_reasons == ("retry_exhausted",)
    assert result.selected_attempt_id is None
    assert result.parsed_text is None
    assert len(result.attempt_ids) == 4
    assert result.total_cost_usd == Decimal("0")


def test_crash_leaves_result_indeterminate_not_masked_by_earlier_failures(manifest_and_intent):
    output_root, _manifest, intents = manifest_and_intent
    store = BudgetLedgerStore({"run": Ledger(name="run", ceiling_usd=Decimal("1.00"))})
    journal_path = output_root / "attempt_events.jsonl"

    dispatch_pair(
        intent=intents[0],
        ledger_store=store,
        ledger_names=("run",),
        output_root=output_root,
        journal_path=journal_path,
        behavior=StubBehavior(fail_attempts=1, crash_before_finish=True),
    )

    result = derive_page_model_result(
        intent=intents[0], journal_path=journal_path, created_at="2026-08-31T00:01:00Z"
    )

    assert result.terminal_outcome == "indeterminate"
    assert result.eligibility == "ineligible"
    assert result.selected_attempt_id is None
    # the started-never-finished attempt is still counted in attempt_ids
    assert len(result.attempt_ids) == 2


def test_budget_refused_with_zero_attempts_derives_budget_refused(manifest_and_intent):
    output_root, _manifest, intents = manifest_and_intent
    store = BudgetLedgerStore({"run": Ledger(name="run", ceiling_usd=Decimal("0.001"))})
    journal_path = output_root / "attempt_events.jsonl"

    with pytest.raises(BudgetRefused):
        dispatch_pair(
            intent=intents[0],
            ledger_store=store,
            ledger_names=("run",),
            output_root=output_root,
            journal_path=journal_path,
            behavior=StubBehavior(),
        )

    result = derive_page_model_result(
        intent=intents[0], journal_path=journal_path, created_at="2026-08-31T00:01:00Z"
    )

    assert result.terminal_outcome == "budget_refused"
    assert result.eligibility == "ineligible"
    assert result.attempt_ids == ()
    assert result.selected_attempt_id is None
    assert result.total_cost_usd == Decimal("0")


def test_no_journal_activity_at_all_also_derives_budget_refused(manifest_and_intent):
    """No attempt was ever started and no refusal was ever recorded either
    (e.g. a Dispatch Pair the batch never reached) -- must never collapse to
    a fabricated 'success' or an empty/placeholder result silently treated as
    eligible.
    """
    output_root, _manifest, intents = manifest_and_intent
    journal_path = output_root / "attempt_events.jsonl"
    journal_path.write_text("")

    result = derive_page_model_result(
        intent=intents[0], journal_path=journal_path, created_at="2026-08-31T00:01:00Z"
    )

    assert result.terminal_outcome == "budget_refused"
    assert result.eligibility == "ineligible"
    assert result.attempt_ids == ()


def test_write_page_model_results_round_trips_via_atomic_jsonl(manifest_and_intent):
    output_root, _manifest, intents = manifest_and_intent
    store = BudgetLedgerStore({"run": Ledger(name="run", ceiling_usd=Decimal("1.00"))})
    journal_path = output_root / "attempt_events.jsonl"

    dispatch_pair(
        intent=intents[0],
        ledger_store=store,
        ledger_names=("run",),
        output_root=output_root,
        journal_path=journal_path,
        behavior=StubBehavior(),
    )
    result = derive_page_model_result(
        intent=intents[0], journal_path=journal_path, created_at="2026-08-31T00:01:00Z"
    )

    write_page_model_results(output_root, [result])
    rows = read_jsonl(output_root / "page_model_results.jsonl")

    assert len(rows) == 1
    assert rows[0]["page_model_result_id"] == result.page_model_result_id
    assert rows[0]["terminal_outcome"] == "success"
    assert rows[0]["total_cost_usd"] == "0.005"


def test_derived_result_is_deterministic_across_repeated_derivation(manifest_and_intent):
    output_root, _manifest, intents = manifest_and_intent
    store = BudgetLedgerStore({"run": Ledger(name="run", ceiling_usd=Decimal("1.00"))})
    journal_path = output_root / "attempt_events.jsonl"

    dispatch_pair(
        intent=intents[0],
        ledger_store=store,
        ledger_names=("run",),
        output_root=output_root,
        journal_path=journal_path,
        behavior=StubBehavior(fail_attempts=1),
    )

    first = derive_page_model_result(
        intent=intents[0], journal_path=journal_path, created_at="2026-08-31T00:01:00Z"
    )
    second = derive_page_model_result(
        intent=intents[0], journal_path=journal_path, created_at="2026-08-31T00:01:00Z"
    )

    assert first.page_model_result_id == second.page_model_result_id
    assert first.record_sha256() == second.record_sha256()
