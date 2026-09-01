"""A2/A3 manifest sealing and stub-model dispatch lifecycle (ticket 03)."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest

from ocr_ensemble.budget import BudgetLedgerStore, Ledger
from ocr_ensemble.dispatch import (
    MAX_ATTEMPTS,
    BudgetRefused,
    IntentIndeterminate,
    StubBehavior,
    dispatch_pair,
)
from ocr_ensemble.manifest import (
    seal_dispatch_intents,
    seal_run_manifest,
    seal_stub_dataset_split,
    seal_stub_roster_member,
)
from ocr_ensemble.preprocess import PreprocessRequest, preprocess_dataset
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


# ---------------------------------------------------------------------------
# A2: manifest and intent sealing
# ---------------------------------------------------------------------------


def test_seals_exactly_one_dispatch_intent_for_one_roster_member(manifest_and_intent):
    _output_root, manifest, intents = manifest_and_intent
    assert len(intents) == 1
    assert len(manifest.roster) == 1

    intent = intents[0]
    assert intent.run_manifest_id == manifest.run_manifest_id
    assert intent.roster_member_config_id == manifest.roster[0].roster_member_config_id
    assert intent.ordinal == 0


def test_manifest_and_intent_ids_are_deterministic(sealed_fixture):
    _output_root, evaluation_unit, page_input_variant = sealed_fixture
    roster_member = seal_stub_roster_member()
    dataset_split = seal_stub_dataset_split(
        evaluation_unit_id=evaluation_unit.evaluation_unit_id,
        dataset_id="havi_failure_mode_v1",
        dataset_version="aiai-ocr-dataset-2026-08-31",
    )

    def _seal_all():
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
        return manifest, intents

    manifest1, intents1 = _seal_all()
    manifest2, intents2 = _seal_all()
    assert manifest1.run_manifest_id == manifest2.run_manifest_id
    assert intents1[0].dispatch_intent_id == intents2[0].dispatch_intent_id


# ---------------------------------------------------------------------------
# A3: reserve-then-attempt-then-log happy path
# ---------------------------------------------------------------------------


def test_dispatch_reserves_exposure_before_any_attempt_event(manifest_and_intent):
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

    events = read_jsonl(journal_path)
    assert len(events) == 2
    assert events[0]["event_type"] == "attempt_started"
    assert events[1]["event_type"] == "attempt_finished"
    assert events[1]["outcome"] == "response_received"
    assert events[1]["parsed_output"]["text"] == "the quick brown fox"

    ledger = store.ledger("run")
    assert ledger.reserved_usd == Decimal("0")  # reconciled, nothing left reserved
    assert ledger.settled_usd == Decimal("0.005")


def test_dispatch_events_are_fsynced_and_readable_as_valid_json(manifest_and_intent):
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

    import json

    for line in journal_path.read_text(encoding="utf-8").splitlines():
        json.loads(line)  # must not raise


# ---------------------------------------------------------------------------
# A3: budget refusal
# ---------------------------------------------------------------------------


def test_dispatch_refused_when_reservation_cannot_be_admitted(manifest_and_intent):
    output_root, _manifest, intents = manifest_and_intent
    # ceiling smaller than STUB_MAXIMUM_EXPOSURE_USD -- reservation must fail
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

    events = read_jsonl(journal_path)
    assert len(events) == 1
    assert events[0]["event_type"] == "dispatch_refused"
    assert events[0]["dispatch_intent_id"] == intents[0].dispatch_intent_id
    assert events[0]["dispatch_pair_id"] == intents[0].dispatch_pair_id
    assert store.ledger("run").reserved_usd == Decimal("0")
    assert store.ledger("run").settled_usd == Decimal("0")


# ---------------------------------------------------------------------------
# A3: retry ceiling
# ---------------------------------------------------------------------------


def test_retries_after_rate_limit_then_succeeds(manifest_and_intent):
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

    events = read_jsonl(journal_path)
    finishes = [e for e in events if e["event_type"] == "attempt_finished"]
    assert [f["outcome"] for f in finishes] == ["rate_limited", "rate_limited", "response_received"]
    # rate-limited attempts must not be charged
    assert store.ledger("run").settled_usd == Decimal("0.005")


def test_retry_ceiling_stops_at_four_total_contacts(manifest_and_intent):
    # Three automatic redispatches after the initial attempt,
    # four contacts total, even under sustained rate-limiting.
    output_root, _manifest, intents = manifest_and_intent
    store = BudgetLedgerStore({"run": Ledger(name="run", ceiling_usd=Decimal("1.00"))})
    journal_path = output_root / "attempt_events.jsonl"

    dispatch_pair(
        intent=intents[0],
        ledger_store=store,
        ledger_names=("run",),
        output_root=output_root,
        journal_path=journal_path,
        behavior=StubBehavior(fail_attempts=MAX_ATTEMPTS),
    )

    events = read_jsonl(journal_path)
    starts = [e for e in events if e["event_type"] == "attempt_started"]
    finishes = [e for e in events if e["event_type"] == "attempt_finished"]
    assert len(starts) == MAX_ATTEMPTS
    assert len(finishes) == MAX_ATTEMPTS
    assert all(f["outcome"] == "rate_limited" for f in finishes)
    assert [s["attempt_number"] for s in starts] == list(range(1, MAX_ATTEMPTS + 1))


def test_resume_after_ceiling_exhaustion_does_not_attempt_a_fifth_contact(manifest_and_intent):
    output_root, _manifest, intents = manifest_and_intent
    store = BudgetLedgerStore({"run": Ledger(name="run", ceiling_usd=Decimal("1.00"))})
    journal_path = output_root / "attempt_events.jsonl"

    dispatch_pair(
        intent=intents[0],
        ledger_store=store,
        ledger_names=("run",),
        output_root=output_root,
        journal_path=journal_path,
        behavior=StubBehavior(fail_attempts=MAX_ATTEMPTS),
    )
    events_before = read_jsonl(journal_path)

    dispatch_pair(
        intent=intents[0],
        ledger_store=store,
        ledger_names=("run",),
        output_root=output_root,
        journal_path=journal_path,
        behavior=StubBehavior(fail_attempts=MAX_ATTEMPTS),
    )
    events_after = read_jsonl(journal_path)

    assert len(events_after) == len(events_before)


# ---------------------------------------------------------------------------
# A3: crash / resume / indeterminate
# ---------------------------------------------------------------------------


def test_crash_before_finish_leaves_reservation_held_and_journal_indeterminate(
    manifest_and_intent,
):
    output_root, _manifest, intents = manifest_and_intent
    store = BudgetLedgerStore({"run": Ledger(name="run", ceiling_usd=Decimal("1.00"))})
    journal_path = output_root / "attempt_events.jsonl"

    dispatch_pair(
        intent=intents[0],
        ledger_store=store,
        ledger_names=("run",),
        output_root=output_root,
        journal_path=journal_path,
        behavior=StubBehavior(crash_before_finish=True),
    )

    events = read_jsonl(journal_path)
    assert len(events) == 1
    assert events[0]["event_type"] == "attempt_started"
    # An attempt_started with no matching finish event is
    # indeterminate. Its reservations remain held.
    assert store.ledger("run").reserved_usd > Decimal("0")
    assert store.ledger("run").settled_usd == Decimal("0")


def test_resume_on_indeterminate_attempt_raises_and_never_retransmits(manifest_and_intent):
    output_root, _manifest, intents = manifest_and_intent
    store = BudgetLedgerStore({"run": Ledger(name="run", ceiling_usd=Decimal("1.00"))})
    journal_path = output_root / "attempt_events.jsonl"

    dispatch_pair(
        intent=intents[0],
        ledger_store=store,
        ledger_names=("run",),
        output_root=output_root,
        journal_path=journal_path,
        behavior=StubBehavior(crash_before_finish=True),
    )

    with pytest.raises(IntentIndeterminate):
        dispatch_pair(
            intent=intents[0],
            ledger_store=store,
            ledger_names=("run",),
            output_root=output_root,
            journal_path=journal_path,
            behavior=StubBehavior(),  # would succeed if it were (wrongly) retried
        )

    # no new event was appended by the refused resume attempt
    events = read_jsonl(journal_path)
    assert len(events) == 1


def test_resume_on_terminal_success_is_a_no_op(manifest_and_intent):
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
    events_after_first = read_jsonl(journal_path)
    settled_after_first = store.ledger("run").settled_usd

    dispatch_pair(
        intent=intents[0],
        ledger_store=store,
        ledger_names=("run",),
        output_root=output_root,
        journal_path=journal_path,
        behavior=StubBehavior(),
    )
    events_after_second = read_jsonl(journal_path)

    assert events_after_second == events_after_first
    assert store.ledger("run").settled_usd == settled_after_first  # never double-charged
