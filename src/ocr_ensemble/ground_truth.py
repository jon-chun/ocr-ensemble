"""A6: append-only ground-truth journal and the deterministic snapshot resolver.

Two independent responsibilities live here:

- journal construction: turning one authored ``GroundTruthAssertion`` into a
  hash-chained sequence of ``GroundTruthEvent`` records (``submit``, ``approve``,
  ``reject``, ``supersede``) with correctly wired ``sequence``/``prior_event_hash``;
  and
- the A6 resolver: a pure function from an accumulated set of assertions and
  events to one ``EffectiveGroundTruthSnapshot``.

Neither ever mutates a prior ``GroundTruthAssertion`` or ``GroundTruthEvent``.
Every "action" is instead a new immutable event appended to the journal; a
projection over that append-only history determines what is currently
scoreable. This module contains no dataset-specific parsing;
callers such as ``adapters/havi.py`` build the assertion text and hand it here.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Literal, Sequence

from ocr_ensemble.records import (
    EffectiveGroundTruthSnapshot,
    GroundTruthAssertion,
    GroundTruthEvent,
)
from ocr_ensemble.storage import write_json_atomic

RESOLVER_POLICY_VERSION = "gt_resolver_v1"


def _seal_event(event: GroundTruthEvent) -> GroundTruthEvent:
    return dataclasses.replace(event, event_id=event.semantic_id())


def append_event(
    prior_event: GroundTruthEvent | None,
    *,
    assertion_id: str,
    action: Literal["submit", "approve", "reject", "supersede"],
    actor_id: str,
    event_source: Literal["human_workflow", "dataset_import"],
    created_at: str,
    approval_authorization_id: str | None = None,
    reason: str | None = None,
    superseded_by_assertion_id: str | None = None,
) -> GroundTruthEvent:
    """Build the next event in a hash-chained journal.

    ``prior_event_hash`` references the previous event's ``record_sha256()``
    (each event's ``prior_event_hash`` references the previous event's content
    hash), never its semantic ID -- the chain
    must break if any field of the prior event, including its timestamp, is
    altered after the fact. The first event in a chain passes
    ``prior_event=None`` and gets ``prior_event_hash=None``.
    """
    if action == "approve" and event_source == "dataset_import":
        if approval_authorization_id is None:
            raise ValueError(
                "approve events with event_source='dataset_import' require "
                "approval_authorization_id"
            )
    elif approval_authorization_id is not None:
        raise ValueError(
            "approval_authorization_id is forbidden for submit/reject/supersede "
            "events and for human_workflow approve events"
        )

    sequence = 0 if prior_event is None else prior_event.sequence + 1
    prior_event_hash = None if prior_event is None else prior_event.record_sha256()

    event = GroundTruthEvent(
        event_id="placeholder",
        sequence=sequence,
        prior_event_hash=prior_event_hash,
        assertion_id=assertion_id,
        action=action,
        actor_id=actor_id,
        event_source=event_source,
        approval_authorization_id=approval_authorization_id,
        reason=reason,
        superseded_by_assertion_id=superseded_by_assertion_id,
        created_at=created_at,
    )
    return _seal_event(event)


def author_and_approve_gold_full(
    *,
    evaluation_unit_id: str,
    text: str,
    guideline_id: str,
    source: str,
    author_actor_id: str,
    approver_actor_id: str,
    author_created_at: str,
    approver_created_at: str,
    target_state: Literal["transcribable", "blank", "fully_illegible"] = "transcribable",
    source_artifact_sha256: str | None = None,
) -> tuple[GroundTruthAssertion, tuple[GroundTruthEvent, GroundTruthEvent]]:
    """Author one ``gold_full`` assertion plus its ``submit``/``approve`` events
    under the HAVI two-pass authoring rule: the author
    transcribes without model-output exposure, and a *different* actor approves
    after independently verifying under the same guideline.

    This is dataset-shape-agnostic on purpose -- ``adapters/havi.py`` is the
    only place that decides *what* text and actors to pass in; this function
    only enforces the independent-approval invariant and the event-chaining
    mechanics shared by any human-workflow gold authoring.
    """
    if author_actor_id == approver_actor_id:
        raise ValueError(
            "HAVI gold requires an independent approving actor: "
            "approval must be by an independent actor"
        )

    assertion = GroundTruthAssertion(
        assertion_id="placeholder",
        evaluation_unit_id=evaluation_unit_id,
        role="gold_full",
        target_state=target_state,
        text=text,
        guideline_id=guideline_id,
        source=source,
        author_actor_id=author_actor_id,
        created_at=author_created_at,
        source_artifact_sha256=source_artifact_sha256,
    )
    assertion = dataclasses.replace(assertion, assertion_id=assertion.semantic_id())

    submit_event = append_event(
        None,
        assertion_id=assertion.assertion_id,
        action="submit",
        actor_id=author_actor_id,
        event_source="human_workflow",
        created_at=author_created_at,
    )
    approve_event = append_event(
        submit_event,
        assertion_id=assertion.assertion_id,
        action="approve",
        actor_id=approver_actor_id,
        event_source="human_workflow",
        created_at=approver_created_at,
    )
    return assertion, (submit_event, approve_event)


# ---------------------------------------------------------------------------
# A6 resolver: append-only assertions/events -> Effective Ground-Truth Snapshot
# ---------------------------------------------------------------------------


def resolve_effective_snapshot(
    assertions: Sequence[GroundTruthAssertion],
    events: Sequence[GroundTruthEvent],
    *,
    resolver_policy_version: str = RESOLVER_POLICY_VERSION,
) -> EffectiveGroundTruthSnapshot:
    """Project an append-only assertion/event journal into one deterministic
    ``EffectiveGroundTruthSnapshot``.

    For each ``(evaluation_unit_id, role, guideline_id)`` target: an assertion
    is an active approved head if its latest-by-``sequence`` event is
    ``approve`` and no later ``reject``/``supersede`` event for that same
    assertion exists. Exactly one active head per target is usable; zero is
    unavailable, more than one is conflicted. Conflicts are resolved by
    counting active heads only -- ``created_at`` and event ``sequence`` never
    break a tie: conflicts are fatal for scoring and
    are never resolved by timestamp.

    Pure function: never mutates any input record.
    """
    assertions_by_id = {a.assertion_id: a for a in assertions}

    events_by_assertion: dict[str, list[GroundTruthEvent]] = {}
    for event in events:
        events_by_assertion.setdefault(event.assertion_id, []).append(event)
    for assertion_events in events_by_assertion.values():
        assertion_events.sort(key=lambda e: e.sequence)

    active_heads: dict[str, list[str]] = {}
    used_assertion_ids: set[str] = set()
    used_event_ids: set[str] = set()

    for assertion_id, assertion_events in events_by_assertion.items():
        assertion = assertions_by_id.get(assertion_id)
        if assertion is None:
            continue
        latest_action = assertion_events[-1].action
        if latest_action != "approve":
            continue

        used_assertion_ids.add(assertion_id)
        used_event_ids.update(e.event_id for e in assertion_events)

        target = (assertion.evaluation_unit_id, assertion.role, assertion.guideline_id)
        active_heads.setdefault(target, []).append(assertion_id)

    conflicted_unit_ids: set[str] = set()
    unavailable_unit_ids: set[str] = set()
    all_target_evaluation_unit_ids = {a.evaluation_unit_id for a in assertions}

    for target, heads in active_heads.items():
        evaluation_unit_id = target[0]
        if len(heads) > 1:
            conflicted_unit_ids.add(evaluation_unit_id)

    for evaluation_unit_id in all_target_evaluation_unit_ids:
        has_any_active_head = any(
            target[0] == evaluation_unit_id for target, heads in active_heads.items() if heads
        )
        if not has_any_active_head:
            unavailable_unit_ids.add(evaluation_unit_id)

    snapshot = EffectiveGroundTruthSnapshot(
        effective_gt_snapshot_id="placeholder",
        assertion_ids=tuple(sorted(used_assertion_ids)),
        event_ids=tuple(sorted(used_event_ids)),
        authorization_ids=(),
        conflicted_evaluation_unit_ids=tuple(sorted(conflicted_unit_ids)),
        unavailable_evaluation_unit_ids=tuple(sorted(unavailable_unit_ids)),
        resolver_policy_version=resolver_policy_version,
    )
    return dataclasses.replace(snapshot, effective_gt_snapshot_id=snapshot.semantic_id())


def write_effective_gt_snapshot(output_root: Path, snapshot: EffectiveGroundTruthSnapshot) -> None:
    write_json_atomic(
        output_root / "effective_gt_snapshot.json", dataclasses.asdict(snapshot)
    )


__all__ = [
    "RESOLVER_POLICY_VERSION",
    "append_event",
    "author_and_approve_gold_full",
    "resolve_effective_snapshot",
    "write_effective_gt_snapshot",
]
