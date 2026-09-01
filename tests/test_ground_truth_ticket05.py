"""Ticket 05: A6 ground-truth assertion/event journal for one fixture.

Exercises the real ticket-02 fixture end to end: preprocess -> HAVI
``import_ground_truth`` -> A6 resolver -> ``EffectiveGroundTruthSnapshot``.
Also proves the conflict and supersession rules using
synthetic assertions/events (isolated from the real fixture identity).
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from ocr_ensemble.adapters.havi import (
    HAVI_GOLD_GUIDELINE_ID,
    HaviFailureModeAdapter,
)
from ocr_ensemble.ground_truth import (
    append_event,
    author_and_approve_gold_full,
    resolve_effective_snapshot,
)
from ocr_ensemble.preprocess import PreprocessRequest, preprocess_dataset
from ocr_ensemble.records import GroundTruthAssertion
from ocr_ensemble.storage import append_jsonl_atomic, read_jsonl

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = REPO_ROOT / "aiai-ocr-dataset"
FIXTURE_RELATIVE_PATH = "D-COMP/ocr_d-comp_low-1_20260706.png"

AUTHOR_ACTOR_ID = "havi-annotator-1"
APPROVER_ACTOR_ID = "havi-annotator-2"
AUTHOR_CREATED_AT = "2026-08-31T10:00:00Z"
APPROVER_CREATED_AT = "2026-08-31T11:00:00Z"


@pytest.fixture()
def sealed_evaluation_unit_id(tmp_path: Path) -> str:
    request = PreprocessRequest(
        dataset_root=DATASET_ROOT,
        output_root=tmp_path / "preprocess-out",
        fixture_relative_paths=(FIXTURE_RELATIVE_PATH,),
    )
    artifacts = preprocess_dataset(request)
    return artifacts.evaluation_units[0].evaluation_unit_id


@pytest.fixture()
def imported_ground_truth(sealed_evaluation_unit_id: str):
    adapter = HaviFailureModeAdapter()
    imported = adapter.import_ground_truth(
        DATASET_ROOT,
        evaluation_unit_ids_by_dataset_item_id={
            FIXTURE_RELATIVE_PATH: sealed_evaluation_unit_id
        },
        author_actor_id=AUTHOR_ACTOR_ID,
        approver_actor_id=APPROVER_ACTOR_ID,
        author_created_at=AUTHOR_CREATED_AT,
        approver_created_at=APPROVER_CREATED_AT,
    )
    assert len(imported) == 1
    return imported[0]


# ---------------------------------------------------------------------------
# HAVI authoring: one gold_full assertion + submit/approve events
# ---------------------------------------------------------------------------


def test_havi_import_ground_truth_produces_one_gold_full_assertion(
    imported_ground_truth, sealed_evaluation_unit_id: str
):
    assertion = imported_ground_truth.assertion
    assert assertion.evaluation_unit_id == sealed_evaluation_unit_id
    assert assertion.role == "gold_full"
    assert assertion.target_state == "transcribable"
    assert assertion.guideline_id == HAVI_GOLD_GUIDELINE_ID
    assert assertion.text is not None
    assert "Breakfast for School Children" in assertion.text
    assert assertion.author_actor_id == AUTHOR_ACTOR_ID
    assert assertion.assertion_id.startswith("ground_truth_assertion_sha256:")


def test_havi_import_ground_truth_events_are_submit_then_approve_by_independent_actor(
    imported_ground_truth,
):
    submit_event, approve_event = imported_ground_truth.initial_events
    assert submit_event.action == "submit"
    assert submit_event.actor_id == AUTHOR_ACTOR_ID
    assert approve_event.action == "approve"
    assert approve_event.actor_id == APPROVER_ACTOR_ID
    # independent-actor rule: the approver must differ from the author
    assert approve_event.actor_id != submit_event.actor_id


def test_havi_import_ground_truth_hash_chains_events_correctly(imported_ground_truth):
    submit_event, approve_event = imported_ground_truth.initial_events
    assert submit_event.sequence == 0
    assert submit_event.prior_event_hash is None
    assert approve_event.sequence == 1
    assert approve_event.prior_event_hash == submit_event.record_sha256()


def test_havi_import_ground_truth_approval_authorization_id_is_none_for_human_workflow(
    imported_ground_truth,
):
    # approval_authorization_id is required only for dataset_import-sourced
    # approvals; this HAVI case is human_workflow.
    _submit_event, approve_event = imported_ground_truth.initial_events
    assert approve_event.event_source == "human_workflow"
    assert approve_event.approval_authorization_id is None
    assert imported_ground_truth.approval_authorization is None


def test_author_and_approve_gold_full_rejects_same_actor_as_author_and_approver():
    with pytest.raises(ValueError, match="independent"):
        author_and_approve_gold_full(
            evaluation_unit_id="evaluation_unit_sha256:" + "a" * 64,
            text="some text",
            guideline_id="g1",
            source="havi_human_authored",
            author_actor_id="same-actor",
            approver_actor_id="same-actor",
            author_created_at=AUTHOR_CREATED_AT,
            approver_created_at=APPROVER_CREATED_AT,
        )


def test_append_event_rejects_authorization_id_outside_dataset_import_approve():
    with pytest.raises(ValueError, match="forbidden"):
        append_event(
            None,
            assertion_id="ground_truth_assertion_sha256:" + "a" * 64,
            action="submit",
            actor_id="actor-1",
            event_source="human_workflow",
            created_at=AUTHOR_CREATED_AT,
            approval_authorization_id="approval_authorization_sha256:" + "b" * 64,
        )


def test_append_event_requires_authorization_id_for_dataset_import_approve():
    with pytest.raises(ValueError, match="require"):
        append_event(
            None,
            assertion_id="ground_truth_assertion_sha256:" + "a" * 64,
            action="approve",
            actor_id="actor-1",
            event_source="dataset_import",
            created_at=AUTHOR_CREATED_AT,
        )


# ---------------------------------------------------------------------------
# A6 resolver: the fixture round-trips to zero conflicts, zero unavailable
# ---------------------------------------------------------------------------


def test_resolver_produces_clean_snapshot_for_the_one_fixture_assertion(
    imported_ground_truth, sealed_evaluation_unit_id: str
):
    snapshot = resolve_effective_snapshot(
        [imported_ground_truth.assertion], list(imported_ground_truth.initial_events)
    )
    assert snapshot.assertion_ids == (imported_ground_truth.assertion.assertion_id,)
    assert snapshot.event_ids == tuple(
        sorted(e.event_id for e in imported_ground_truth.initial_events)
    )
    assert snapshot.authorization_ids == ()
    assert snapshot.conflicted_evaluation_unit_ids == ()
    assert snapshot.unavailable_evaluation_unit_ids == ()
    assert snapshot.effective_gt_snapshot_id.startswith("effective_gt_snapshot_sha256:")


def test_resolver_snapshot_id_is_deterministic_given_same_inputs(imported_ground_truth):
    assertions = [imported_ground_truth.assertion]
    events = list(imported_ground_truth.initial_events)
    snapshot1 = resolve_effective_snapshot(assertions, events)
    snapshot2 = resolve_effective_snapshot(assertions, events)
    assert snapshot1.effective_gt_snapshot_id == snapshot2.effective_gt_snapshot_id


# ---------------------------------------------------------------------------
# Conflict rule: never resolved by timestamp
# ---------------------------------------------------------------------------


def _gold_full(evaluation_unit_id: str, *, text: str, author: str, approver: str, created_at: str):
    return author_and_approve_gold_full(
        evaluation_unit_id=evaluation_unit_id,
        text=text,
        guideline_id="shared_guideline_v1",
        source="havi_human_authored",
        author_actor_id=author,
        approver_actor_id=approver,
        author_created_at=created_at,
        approver_created_at=created_at,
    )


def test_two_concurrently_active_approved_assertions_are_conflicted_never_by_timestamp():
    eu_id = "evaluation_unit_sha256:" + "c" * 64

    older_assertion, (older_submit, older_approve) = _gold_full(
        eu_id,
        text="the old approved version",
        author="author-old",
        approver="approver-old",
        created_at="2020-01-01T00:00:00Z",
    )
    newer_assertion, (newer_submit, newer_approve) = _gold_full(
        eu_id,
        text="the much newer approved version",
        author="author-new",
        approver="approver-new",
        created_at="2026-08-31T00:00:00Z",
    )

    snapshot = resolve_effective_snapshot(
        [older_assertion, newer_assertion],
        [older_submit, older_approve, newer_submit, newer_approve],
    )

    assert eu_id in snapshot.conflicted_evaluation_unit_ids
    assert eu_id not in snapshot.unavailable_evaluation_unit_ids
    # both active heads are retained in the snapshot preimage -- the resolver
    # must not silently drop or prefer either one, and specifically must not
    # have picked the assertion with the later created_at.
    assert older_assertion.assertion_id in snapshot.assertion_ids
    assert newer_assertion.assertion_id in snapshot.assertion_ids
    assert len(snapshot.assertion_ids) == 2


def test_conflict_is_symmetric_regardless_of_which_assertion_is_newer():
    # proves the resolver isn't accidentally keying off list/dict order, which
    # could otherwise mask a timestamp-based bug in either direction.
    eu_id = "evaluation_unit_sha256:" + "d" * 64

    a_new, (a_new_s, a_new_a) = _gold_full(
        eu_id, text="new one first in list", author="author-x", approver="approver-x",
        created_at="2026-08-31T00:00:00Z",
    )
    a_old, (a_old_s, a_old_a) = _gold_full(
        eu_id, text="old one second in list", author="author-y", approver="approver-y",
        created_at="2019-01-01T00:00:00Z",
    )

    snapshot = resolve_effective_snapshot(
        [a_new, a_old], [a_new_s, a_new_a, a_old_s, a_old_a]
    )
    assert eu_id in snapshot.conflicted_evaluation_unit_ids
    assert a_new.assertion_id in snapshot.assertion_ids
    assert a_old.assertion_id in snapshot.assertion_ids


def test_zero_approved_assertions_is_unavailable_not_conflicted():
    eu_id = "evaluation_unit_sha256:" + "e" * 64
    assertion = GroundTruthAssertion(
        assertion_id="placeholder",
        evaluation_unit_id=eu_id,
        role="gold_full",
        target_state="transcribable",
        text="never approved",
        guideline_id="g1",
        source="havi_human_authored",
        author_actor_id="author-1",
        created_at="2026-08-31T00:00:00Z",
        source_artifact_sha256=None,
    )
    assertion = dataclasses.replace(assertion, assertion_id=assertion.semantic_id())
    submit_only = append_event(
        None,
        assertion_id=assertion.assertion_id,
        action="submit",
        actor_id="author-1",
        event_source="human_workflow",
        created_at="2026-08-31T00:00:00Z",
    )

    snapshot = resolve_effective_snapshot([assertion], [submit_only])
    assert eu_id in snapshot.unavailable_evaluation_unit_ids
    assert eu_id not in snapshot.conflicted_evaluation_unit_ids
    assert snapshot.assertion_ids == ()


# ---------------------------------------------------------------------------
# Supersession: retires the prior head without mutating the prior assertion
# ---------------------------------------------------------------------------


def test_supersede_retires_prior_head_without_mutating_prior_assertion_record():
    eu_id = "evaluation_unit_sha256:" + "f" * 64

    original_assertion, (orig_submit, orig_approve) = _gold_full(
        eu_id, text="original approved text", author="author-1", approver="approver-1",
        created_at="2026-01-01T00:00:00Z",
    )
    replacement_assertion, (repl_submit, repl_approve) = _gold_full(
        eu_id, text="replacement approved text", author="author-2", approver="approver-2",
        created_at="2026-06-01T00:00:00Z",
    )

    original_hash_before = original_assertion.record_sha256()
    original_fields_before = original_assertion

    supersede_event = append_event(
        orig_approve,
        assertion_id=original_assertion.assertion_id,
        action="supersede",
        actor_id="approver-1",
        event_source="human_workflow",
        created_at="2026-06-01T00:05:00Z",
        superseded_by_assertion_id=replacement_assertion.assertion_id,
    )

    snapshot = resolve_effective_snapshot(
        [original_assertion, replacement_assertion],
        [orig_submit, orig_approve, supersede_event, repl_submit, repl_approve],
    )

    assert snapshot.conflicted_evaluation_unit_ids == ()
    assert snapshot.unavailable_evaluation_unit_ids == ()
    assert snapshot.assertion_ids == (replacement_assertion.assertion_id,)

    # the prior GroundTruthAssertion record itself must be provably unchanged
    # by supersession -- no mutation, only a new event appended.
    assert original_assertion.record_sha256() == original_hash_before
    assert original_assertion == original_fields_before
    assert original_assertion.text == "original approved text"


def test_supersede_event_chains_onto_the_approve_event_it_retires():
    eu_id = "evaluation_unit_sha256:" + "1" * 64
    assertion, (submit_event, approve_event) = _gold_full(
        eu_id, text="text", author="author-1", approver="approver-1",
        created_at="2026-01-01T00:00:00Z",
    )
    supersede_event = append_event(
        approve_event,
        assertion_id=assertion.assertion_id,
        action="supersede",
        actor_id="approver-1",
        event_source="human_workflow",
        created_at="2026-01-02T00:00:00Z",
        superseded_by_assertion_id="ground_truth_assertion_sha256:" + "2" * 64,
    )
    assert supersede_event.sequence == approve_event.sequence + 1
    assert supersede_event.prior_event_hash == approve_event.record_sha256()


# ---------------------------------------------------------------------------
# Journal persistence: append-only, fsync-then-atomic-rename discipline (§2)
# ---------------------------------------------------------------------------


def test_journal_events_append_atomically_and_round_trip(
    tmp_path: Path, imported_ground_truth
):
    assertions_path = tmp_path / "ground_truth_assertions.jsonl"
    events_path = tmp_path / "ground_truth_events.jsonl"

    append_jsonl_atomic(assertions_path, dataclasses.asdict(imported_ground_truth.assertion))
    for event in imported_ground_truth.initial_events:
        append_jsonl_atomic(events_path, dataclasses.asdict(event))

    stored_assertions = read_jsonl(assertions_path)
    stored_events = read_jsonl(events_path)

    assert len(stored_assertions) == 1
    assert stored_assertions[0]["assertion_id"] == imported_ground_truth.assertion.assertion_id
    assert len(stored_events) == 2
    assert [e["action"] for e in stored_events] == ["submit", "approve"]
    # append never rewrites a prior line's content
    assert stored_events[0]["prior_event_hash"] is None
    assert stored_events[1]["prior_event_hash"] == imported_ground_truth.initial_events[
        0
    ].record_sha256()
