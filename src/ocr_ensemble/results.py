"""A4: derive one Page-Model Result per Dispatch Pair from the attempt journal
(ticket 04).

Selection is deterministic: the first attempt (by ``attempt_number``) whose
finish event is a complete, successful transcription wins outright. Failing
that, the design calls for a "versioned terminal-precedence table" that
picks exactly one non-success ``terminal_outcome`` from everything the
journal recorded for the intent -- but that table is never actually
published anywhere in the docs (a real spec gap, not an oversight on this
module's part). ``TERMINAL_OUTCOME_PRECEDENCE_V1`` below is this module's own
first version of that table, named and versioned so a future doc patch can
replace it without call sites changing. Flagged for the user; not silently
invented and forgotten.

Precedence (highest first) when there is no successful attempt:
1. ``indeterminate``       -- a started attempt with no matching finish; the
                              journal cannot honestly claim anything more
                              specific, and this must never be masked by an
                              earlier finished attempt's outcome.
2. ``budget_refused``      -- the pair was never contacted at all because a
                              reservation could not be admitted.
3. ``content_filtered``    -- the provider actively refused to answer.
4. ``cancelled``           -- contact was cancelled before completion.
5. ``unsupported_input``   -- provider rejected the request as unsupported
                              (mapped from ``provider_rejected``).
6. ``truncated``           -- a transcription was returned but
                              ``parsed_output.complete`` is ``False``.
7. ``whole_unit_abstention`` -- the model explicitly declined to transcribe.
8. ``retry_exhausted``     -- every attempt up to the ceiling came back
                              ``rate_limited``/``transport_error``/``timeout``
                              with nothing more specific recorded.
9. ``permanent_failure``   -- fallback: attempts finished but none of the
                              above applied (parse failures, unmapped
                              provider errors).

An ``indeterminate`` or ``budget_refused`` outcome always wins over any
finished attempt's outcome on the same intent, because those two describe
the *intent's overall state*, not a single attempt's result -- e.g. an
intent can have two finished ``rate_limited`` attempts followed by one
started-never-finished attempt; the honest terminal_outcome is
``indeterminate``, not ``retry_exhausted``, because dispatch has not
actually given up on it yet.
"""

from __future__ import annotations

import dataclasses
from decimal import Decimal
from pathlib import Path

from ocr_ensemble.records import DispatchIntent, PageModelResult
from ocr_ensemble.records import seal as _seal
from ocr_ensemble.storage import read_jsonl, write_jsonl_atomic

SCHEMA_VERSION = "ocr-ensemble-schema/v2"
TERMINAL_OUTCOME_PRECEDENCE_V1 = "page_model_result_terminal_outcome_precedence_v1"

_RETRIABLE_FINISH_OUTCOMES = {"rate_limited", "transport_error", "timeout"}


@dataclasses.dataclass(frozen=True)
class _AttemptProjection:
    attempt_id: str
    attempt_number: int
    started: bool
    finish_outcome: str | None
    parsed_output: dict | None
    actual_cost_usd: Decimal
    measured_duration_ms: float


def _project_attempts(
    events: list[dict], dispatch_intent_id: str
) -> tuple[list[_AttemptProjection], bool, str | None]:
    """Mirror of ``dispatch._intent_journal_state``'s two-pass matching, but
    retaining full per-attempt detail (not just resume state) plus whether a
    ``dispatch_refused`` event exists for this intent.
    """
    started_by_number: dict[int, str] = {}
    for event in events:
        if event["event_type"] != "attempt_started":
            continue
        if event.get("dispatch_intent_id") != dispatch_intent_id:
            continue
        started_by_number[event["attempt_number"]] = event["attempt_id"]

    this_intent_attempt_ids = set(started_by_number.values())
    finished_by_attempt_id: dict[str, dict] = {}
    for event in events:
        if event["event_type"] != "attempt_finished":
            continue
        if event["attempt_id"] not in this_intent_attempt_ids:
            continue
        finished_by_attempt_id[event["attempt_id"]] = event

    refused = any(
        event["event_type"] == "dispatch_refused"
        and event.get("dispatch_intent_id") == dispatch_intent_id
        for event in events
    )

    attempts: list[_AttemptProjection] = []
    indeterminate = False
    for number in sorted(started_by_number):
        aid = started_by_number[number]
        finish = finished_by_attempt_id.get(aid)
        if finish is None:
            indeterminate = True
            attempts.append(
                _AttemptProjection(
                    attempt_id=aid,
                    attempt_number=number,
                    started=True,
                    finish_outcome=None,
                    parsed_output=None,
                    actual_cost_usd=Decimal("0"),
                    measured_duration_ms=0.0,
                )
            )
            continue
        actual_cost = finish["actual_cost_usd"]
        cost_value = (
            Decimal(actual_cost["value"])
            if actual_cost.get("availability") == "reported"
            else Decimal("0")
        )
        attempts.append(
            _AttemptProjection(
                attempt_id=aid,
                attempt_number=number,
                started=True,
                finish_outcome=finish["outcome"],
                parsed_output=finish["parsed_output"],
                actual_cost_usd=cost_value,
                measured_duration_ms=finish["measured_duration_ms"],
            )
        )

    return attempts, indeterminate, ("budget_refused" if refused else None)


def _derive_outcome(
    attempts: list[_AttemptProjection], indeterminate: bool, refusal: str | None
) -> tuple[str, str | None, str | None]:
    """Return ``(terminal_outcome, selected_attempt_id, parsed_text)``."""
    for attempt in attempts:
        if attempt.finish_outcome != "response_received":
            continue
        parsed = attempt.parsed_output
        if parsed is not None and parsed["kind"] == "transcription" and parsed["complete"]:
            return "success", attempt.attempt_id, parsed["text"]

    if indeterminate:
        return "indeterminate", None, None
    if refusal is not None:
        return "budget_refused", None, None
    if not attempts:
        return "budget_refused", None, None

    for attempt in attempts:
        if attempt.finish_outcome == "content_filtered":
            return "content_filtered", None, None
    for attempt in attempts:
        if attempt.finish_outcome == "cancelled":
            return "cancelled", None, None
    for attempt in attempts:
        if attempt.finish_outcome == "provider_rejected":
            return "unsupported_input", None, None
    for attempt in attempts:
        if attempt.finish_outcome == "response_received":
            parsed = attempt.parsed_output
            if parsed is not None and parsed["kind"] == "transcription" and not parsed["complete"]:
                return "truncated", attempt.attempt_id, parsed["text"]
            if parsed is not None and parsed["kind"] == "whole_unit_abstention":
                return "whole_unit_abstention", None, None

    if all(a.finish_outcome in _RETRIABLE_FINISH_OUTCOMES for a in attempts):
        return "retry_exhausted", None, None

    return "permanent_failure", None, None


_INELIGIBLE_OUTCOMES = {
    "indeterminate",
    "budget_refused",
    "content_filtered",
    "cancelled",
    "unsupported_input",
    "retry_exhausted",
    "permanent_failure",
}


def derive_page_model_result(
    *, intent: DispatchIntent, journal_path: Path, created_at: str
) -> PageModelResult:
    """Read the attempt journal and derive exactly one ``PageModelResult`` for
    ``intent``'s Dispatch Pair. Pure with respect to the
    journal contents at call time -- callers decide when "the journal is done
    changing for this pair" (e.g. after ``dispatch_pair`` returns or raises).
    """
    events = read_jsonl(journal_path)
    attempts, indeterminate, refusal = _project_attempts(events, intent.dispatch_intent_id)
    terminal_outcome, selected_attempt_id, parsed_text = _derive_outcome(
        attempts, indeterminate, refusal
    )

    total_cost_usd = sum((a.actual_cost_usd for a in attempts), Decimal("0"))
    total_measured_duration_ms = sum(a.measured_duration_ms for a in attempts)

    eligibility = "ineligible" if terminal_outcome in _INELIGIBLE_OUTCOMES else "eligible"
    ineligibility_reasons: tuple[str, ...] = (
        (terminal_outcome,) if eligibility == "ineligible" else ()
    )

    return _seal(
        PageModelResult(
            page_model_result_id="placeholder",
            dispatch_pair_id=intent.dispatch_pair_id,
            dispatch_intent_id=intent.dispatch_intent_id,
            run_manifest_id=intent.run_manifest_id,
            evaluation_unit_id=intent.evaluation_unit_id,
            page_input_variant_id=intent.page_input_variant_id,
            roster_member_config_id=intent.roster_member_config_id,
            selected_attempt_id=selected_attempt_id,
            attempt_ids=tuple(a.attempt_id for a in attempts),
            terminal_outcome=terminal_outcome,
            parsed_text=parsed_text,
            eligibility=eligibility,
            ineligibility_reasons=ineligibility_reasons,
            total_cost_usd=total_cost_usd,
            total_measured_duration_ms=total_measured_duration_ms,
            created_at=created_at,
        )
    )


def write_page_model_results(output_root: Path, results: list[PageModelResult]) -> None:
    write_jsonl_atomic(
        output_root / "page_model_results.jsonl",
        (dataclasses.asdict(r) for r in results),
    )
