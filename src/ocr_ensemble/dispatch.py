"""A3 dispatch: reserve-then-attempt-then-log lifecycle against a stub model
(ticket 03).

A stub "model" never makes a network call. It exists to exercise the
reservation/attempt-journal/retry/crash-resume mechanics honestly -- the same
code path a real provider adapter (ticket 09) will run through -- without
spending money or needing credentials.

Ticket 03 scope is single-member dispatch: one Dispatch Pair, reserved and
either reconciled (rate-limited or success) or left indeterminate by a
simulated crash. Every reservation here is eventually contacted, so
``BudgetLedgerStore.release`` (a reservation abandoned with zero contact,
e.g. a sibling roster member in the same breadth-first batch losing
admission while this one already reserved) is exercised by ``budget.py``'s
own unit tests but not by this module -- multi-member breadth-first batching
is a later ticket's scope, not invented here.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Literal

from ocr_ensemble.budget import BudgetLedgerStore
from ocr_ensemble.identity import sha256_hex
from ocr_ensemble.records import (
    ApiCallAttemptFinished,
    ApiCallAttemptStarted,
    DispatchIntent,
    DispatchRefused,
    FieldValue,
    ParsedOcrOutput,
    RawEnvelopeRef,
    attempt_event_id,
    attempt_id,
    dispatch_refused_event_id,
)
from ocr_ensemble.storage import append_jsonl_atomic, read_jsonl

# Automatic redispatch is capped at three contacts after the
# initial attempt, across all resumes -- four contacts total.
MAX_ATTEMPTS = 4

STUB_MAXIMUM_EXPOSURE_USD = Decimal("0.01")
STUB_ACTUAL_COST_USD = Decimal("0.005")
STUB_SANITIZATION_POLICY_VERSION = "sanitize_v1"
STUB_PARSER_VERSION = "stub_parser_v1"


class BudgetRefused(Exception):
    """The batch reservation for this Dispatch Pair could not be admitted;
    the pair derives ``budget_refused`` rather than being
    silently skipped.
    """


@dataclass(frozen=True)
class StubBehavior:
    """What the stub model should simulate for one intent's attempts.

    ``fail_attempts`` is the number of leading attempts that come back
    ``rate_limited`` before a final attempt succeeds (or exhausts the
    retry ceiling if ``fail_attempts >= MAX_ATTEMPTS``). ``crash_before_finish``
    makes the *next* attempt after those failures start and never finish,
    simulating a process crash for the resume test.
    """

    fixed_text: str = "the quick brown fox"
    fail_attempts: int = 0
    crash_before_finish: bool = False
    latency_ms: float = 5.0


def _outcome_for_attempt(behavior: StubBehavior, attempt_number: int) -> Literal["success", "rate_limited"]:
    if attempt_number <= behavior.fail_attempts:
        return "rate_limited"
    return "success"


def _stub_envelope(payload: dict, output_root: Path, event_label: str) -> RawEnvelopeRef:
    body = json.dumps(payload, sort_keys=True).encode("utf-8")
    digest = sha256_hex(body)
    blob_dir = output_root / "artifacts" / "raw-provider-envelope"
    blob_dir.mkdir(parents=True, exist_ok=True)
    blob_path = blob_dir / f"{event_label}-{digest[:16]}.json"
    blob_path.write_bytes(body)
    preview = body[:200].decode("utf-8", errors="replace")
    return RawEnvelopeRef(
        envelope_sha256=digest,
        cas_uri=blob_path.resolve().as_uri(),
        compressed_byte_size=len(body),
        uncompressed_byte_size=len(body),
        media_type="application/json",
        sanitization_policy_version=STUB_SANITIZATION_POLICY_VERSION,
        preview_utf8=preview,
    )


@dataclass(frozen=True)
class IntentJournalState:
    """What the attempt journal says about one Dispatch Intent so far."""

    highest_attempt_number: int
    last_attempt_outcome: Literal["rate_limited", "success"] | None
    indeterminate: bool  # a started event with no matching finish event


def _intent_journal_state(
    existing_events: list[dict], dispatch_intent_id: str
) -> IntentJournalState:
    """Project the raw event journal into per-intent resume state.

    Every ``attempt_started`` must have a matching ``attempt_finished`` for
    the *same* ``attempt_id`` to be terminal; a started event with no match is
    the intent's current indeterminate attempt regardless of
    how many earlier attempts on this same intent already finished.
    """
    # ApiCallAttemptFinished carries no dispatch_intent_id --
    # only attempt_id. So finish events for this intent must be matched by
    # attempt_id membership in this intent's own started set, in a second
    # pass, not filtered by a field that doesn't exist on that event type.
    started_numbers: dict[int, str] = {}
    highest_attempt_number = 0
    for event in existing_events:
        if event.get("dispatch_intent_id") != dispatch_intent_id:
            continue
        if event["event_type"] == "attempt_started":
            n = event["attempt_number"]
            started_numbers[n] = event["attempt_id"]
            highest_attempt_number = max(highest_attempt_number, n)

    this_intent_attempt_ids = set(started_numbers.values())
    finished_attempt_ids: set[str] = set()
    last_finished_outcome: str | None = None
    for event in existing_events:
        if event["event_type"] != "attempt_finished":
            continue
        if event["attempt_id"] not in this_intent_attempt_ids:
            continue
        finished_attempt_ids.add(event["attempt_id"])
        if event["outcome"] == "response_received":
            last_finished_outcome = "success"
        elif event["outcome"] == "rate_limited":
            last_finished_outcome = "rate_limited"

    indeterminate = any(
        attempt_id_ not in finished_attempt_ids for attempt_id_ in started_numbers.values()
    )

    return IntentJournalState(
        highest_attempt_number=highest_attempt_number,
        last_attempt_outcome=last_finished_outcome,
        indeterminate=indeterminate,
    )


class IntentIndeterminate(Exception):
    """The intent's most recent attempt started but never finished:
    resume must not automatically retransmit. A human must
    record an explicit reconciliation decision first.
    """


def dispatch_pair(
    *,
    intent: DispatchIntent,
    ledger_store: BudgetLedgerStore,
    ledger_names: tuple[str, ...],
    output_root: Path,
    journal_path: Path,
    behavior: StubBehavior,
    started_at_fn=lambda attempt_no: f"2026-08-31T00:00:{attempt_no:02d}Z",
    finished_at_fn=lambda attempt_no: f"2026-08-31T00:00:{attempt_no:02d}.500Z",
) -> None:
    """Run one Dispatch Pair through reservation and the stub call to a
    terminal or indeterminate outcome, appending every event to
    ``journal_path`` as it happens (never batched at the end -- a crash mid-run
    must still leave a truthful partial journal).

    Raises ``BudgetRefused`` if the reservation cannot be admitted; does not
    write any attempt event in that case -- a refused batch
    dispatches no member. Raises ``IntentIndeterminate`` on resume if the
    journal's latest attempt for this intent started but never finished --
    this function never automatically retransmits past that state.
    """
    existing_events = read_jsonl(journal_path)
    state = _intent_journal_state(existing_events, intent.dispatch_intent_id)

    if state.indeterminate:
        raise IntentIndeterminate(
            f"{intent.dispatch_intent_id!r} has an indeterminate attempt "
            f"(attempt {state.highest_attempt_number} started, never finished); "
            "requires explicit human reconciliation before any further dispatch"
        )

    if state.last_attempt_outcome == "success":
        return  # already terminal-success; never re-dispatch

    attempt_number = state.highest_attempt_number + 1
    if attempt_number > MAX_ATTEMPTS:
        return  # already exhausted the retry ceiling on a prior run

    while attempt_number <= MAX_ATTEMPTS:
        this_attempt_id = attempt_id(intent.dispatch_intent_id, attempt_number)

        reservations = ledger_store.try_reserve_batch(
            [("stub-call", STUB_MAXIMUM_EXPOSURE_USD, ledger_names)]
        )
        if reservations is None:
            reason = (
                f"could not reserve {STUB_MAXIMUM_EXPOSURE_USD} for "
                f"{intent.dispatch_intent_id!r} attempt {attempt_number}"
            )
            refused_event = DispatchRefused(
                event_type="dispatch_refused",
                event_id=dispatch_refused_event_id(intent.dispatch_intent_id, reason),
                dispatch_intent_id=intent.dispatch_intent_id,
                dispatch_pair_id=intent.dispatch_pair_id,
                reason=reason,
                refused_at=started_at_fn(attempt_number),
            )
            append_jsonl_atomic(journal_path, _event_to_dict(refused_event))
            raise BudgetRefused(reason)
        [reservation] = reservations

        request_payload = {"dispatch_intent_id": intent.dispatch_intent_id, "attempt_number": attempt_number}
        request_envelope = _stub_envelope(request_payload, output_root, f"{this_attempt_id}-request")
        request_sha256 = request_envelope.envelope_sha256

        start_event = ApiCallAttemptStarted(
            event_type="attempt_started",
            event_id="placeholder",
            attempt_id=this_attempt_id,
            dispatch_intent_id=intent.dispatch_intent_id,
            attempt_number=attempt_number,
            provider_idempotency_key=None,  # stub provider: no idempotency support
            reservation_ids=(reservation.reservation_id,),
            maximum_exposure_usd=STUB_MAXIMUM_EXPOSURE_USD,
            request_sha256=request_sha256,
            request_envelope=request_envelope,
            started_at=started_at_fn(attempt_number),
        )
        start_event = dataclasses.replace(
            start_event,
            event_id=attempt_event_id(
                this_attempt_id, "attempt_started", {"attempt_number": attempt_number}
            ),
        )
        append_jsonl_atomic(journal_path, _event_to_dict(start_event))

        if behavior.crash_before_finish and attempt_number > behavior.fail_attempts:
            # simulate a process crash: reservation stays held, no finish event
            # is ever written. Caller (a test) inspects the journal directly.
            return

        outcome = _outcome_for_attempt(behavior, attempt_number)

        if outcome == "rate_limited":
            ledger_store.reconcile(reservation.reservation_id, Decimal("0"))
            finish_event = ApiCallAttemptFinished(
                event_type="attempt_finished",
                event_id="placeholder",
                attempt_id=this_attempt_id,
                finished_at=finished_at_fn(attempt_number),
                outcome="rate_limited",
                http_status=FieldValue(availability="reported", value=429),
                provider_finish_reason=None,
                error_code="rate_limited",
                error_message_sanitized="stub: simulated rate limit",
                tokens_input=FieldValue(availability="not_applicable", value=None),
                tokens_output=FieldValue(availability="not_applicable", value=None),
                tokens_thinking=FieldValue(availability="not_applicable", value=None),
                provider_duration_ms=FieldValue(availability="not_applicable", value=None),
                measured_duration_ms=behavior.latency_ms,
                actual_cost_usd=FieldValue(availability="reported", value=Decimal("0")),
                pricing_snapshot_id="pricing_v1",
                raw_envelope=None,
                parsed_output=None,
            )
            finish_event = dataclasses.replace(
                finish_event,
                event_id=attempt_event_id(
                    this_attempt_id, "attempt_finished", {"outcome": "rate_limited"}
                ),
            )
            append_jsonl_atomic(journal_path, _event_to_dict(finish_event))
            attempt_number += 1
            continue

        # success
        response_payload = {"kind": "transcription", "text": behavior.fixed_text}
        response_envelope = _stub_envelope(response_payload, output_root, f"{this_attempt_id}-response")
        ledger_store.reconcile(reservation.reservation_id, STUB_ACTUAL_COST_USD)

        parsed = ParsedOcrOutput(
            kind="transcription",
            text=behavior.fixed_text,
            parser_version=STUB_PARSER_VERSION,
            complete=True,
        )
        finish_event = ApiCallAttemptFinished(
            event_type="attempt_finished",
            event_id="placeholder",
            attempt_id=this_attempt_id,
            finished_at=finished_at_fn(attempt_number),
            outcome="response_received",
            http_status=FieldValue(availability="reported", value=200),
            provider_finish_reason="stop",
            error_code=None,
            error_message_sanitized=None,
            tokens_input=FieldValue(availability="reported", value=42),
            tokens_output=FieldValue(availability="reported", value=8),
            tokens_thinking=FieldValue(availability="not_applicable", value=None),
            provider_duration_ms=FieldValue(availability="reported", value=behavior.latency_ms),
            measured_duration_ms=behavior.latency_ms,
            actual_cost_usd=FieldValue(availability="reported", value=STUB_ACTUAL_COST_USD),
            pricing_snapshot_id="pricing_v1",
            raw_envelope=response_envelope,
            parsed_output=parsed,
        )
        finish_event = dataclasses.replace(
            finish_event,
            event_id=attempt_event_id(
                this_attempt_id, "attempt_finished", {"outcome": "response_received"}
            ),
        )
        append_jsonl_atomic(journal_path, _event_to_dict(finish_event))
        return

    # fell out of the loop: every attempt up to MAX_ATTEMPTS was rate-limited
    return


def _event_to_dict(event) -> dict:
    d = dataclasses.asdict(event)
    return d
