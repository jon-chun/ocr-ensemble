"""Budget ledger admission and reconciliation (ticket 03)."""

from __future__ import annotations

import threading
from decimal import Decimal

import pytest

from ocr_ensemble.budget import (
    BudgetIntegrityError,
    BudgetLedgerStore,
    Ledger,
    maximum_exposure_usd,
)


def _store(**ceilings: str | None) -> BudgetLedgerStore:
    ledgers = {
        name: Ledger(name=name, ceiling_usd=Decimal(value) if value is not None else None)
        for name, value in ceilings.items()
    }
    return BudgetLedgerStore(ledgers)


def test_maximum_exposure_sums_every_upper_bound_term():
    exposure = maximum_exposure_usd(
        max_input_units=1000,
        input_unit_price_usd=Decimal("0.0001"),
        max_output_units=500,
        output_unit_price_usd=Decimal("0.0002"),
        fixed_charges_usd=Decimal("0.01"),
    )
    assert exposure == Decimal("0.21")


def test_reserve_succeeds_within_ceiling():
    store = _store(run="10.00")
    reservations = store.try_reserve_batch([("call-1", Decimal("2.50"), ("run",))])
    assert reservations is not None
    assert store.ledger("run").reserved_usd == Decimal("2.50")


def test_reserve_refused_when_batch_exceeds_a_single_ledger_ceiling():
    store = _store(run="1.00")
    reservations = store.try_reserve_batch([("call-1", Decimal("2.00"), ("run",))])
    assert reservations is None
    assert store.ledger("run").reserved_usd == Decimal("0")


def test_batch_reservation_is_all_or_nothing_across_ledgers():
    # run has headroom but model does not -- neither ledger should be touched.
    store = _store(run="10.00", model="1.00")
    reservations = store.try_reserve_batch(
        [
            ("call-1", Decimal("2.00"), ("run", "model")),
        ]
    )
    assert reservations is None
    assert store.ledger("run").reserved_usd == Decimal("0")
    assert store.ledger("model").reserved_usd == Decimal("0")


def test_breadth_first_batch_reserves_every_member_or_none():
    # Atomically admits and reserves the entire remaining
    # roster batch for one Evaluation Unit before issuing any member request.
    store = _store(run="5.00")
    requests = [
        ("model-a", Decimal("2.00"), ("run",)),
        ("model-b", Decimal("2.00"), ("run",)),
        ("model-c", Decimal("2.00"), ("run",)),  # pushes total to 6.00 > 5.00
    ]
    reservations = store.try_reserve_batch(requests)
    assert reservations is None
    assert store.ledger("run").reserved_usd == Decimal("0")

    smaller_requests = requests[:2]
    reservations = store.try_reserve_batch(smaller_requests)
    assert reservations is not None
    assert len(reservations) == 2
    assert store.ledger("run").reserved_usd == Decimal("4.00")


def test_reconcile_moves_reserved_to_settled_at_actual_cost():
    store = _store(run="10.00")
    [reservation] = store.try_reserve_batch([("call-1", Decimal("2.00"), ("run",))])
    store.reconcile(reservation.reservation_id, Decimal("1.75"))

    ledger = store.ledger("run")
    assert ledger.reserved_usd == Decimal("0")
    assert ledger.settled_usd == Decimal("1.75")
    assert ledger.committed_usd() == Decimal("1.75")


def test_reconcile_above_reserved_amount_raises_and_freezes():
    store = _store(run="10.00")
    [reservation] = store.try_reserve_batch([("call-1", Decimal("2.00"), ("run",))])
    with pytest.raises(BudgetIntegrityError):
        store.reconcile(reservation.reservation_id, Decimal("2.01"))

    # the reservation is retained (not silently dropped) so the freeze state
    # is inspectable rather than the exposure quietly vanishing.
    ledger = store.ledger("run")
    assert ledger.reserved_usd == Decimal("2.00")
    assert ledger.settled_usd == Decimal("0")


def test_release_returns_reservation_without_any_settled_charge():
    store = _store(run="10.00")
    [reservation] = store.try_reserve_batch([("call-1", Decimal("2.00"), ("run",))])
    store.release(reservation.reservation_id)

    ledger = store.ledger("run")
    assert ledger.reserved_usd == Decimal("0")
    assert ledger.settled_usd == Decimal("0")


def test_reservation_after_release_cannot_be_reconciled_or_released_again():
    store = _store(run="10.00")
    [reservation] = store.try_reserve_batch([("call-1", Decimal("2.00"), ("run",))])
    store.release(reservation.reservation_id)
    with pytest.raises(KeyError):
        store.release(reservation.reservation_id)
    with pytest.raises(KeyError):
        store.reconcile(reservation.reservation_id, Decimal("0"))


def test_no_ceiling_configured_means_unbounded_headroom():
    store = _store(run=None)
    reservations = store.try_reserve_batch([("call-1", Decimal("1000000.00"), ("run",))])
    assert reservations is not None


def test_conjunctive_ledgers_all_must_have_headroom():
    # run ceiling generous, model ceiling tight -- model should gate admission.
    store = _store(run="1000.00", model="0.50")
    reservations = store.try_reserve_batch([("call-1", Decimal("1.00"), ("run", "model"))])
    assert reservations is None
    assert store.ledger("run").reserved_usd == Decimal("0")


def test_concurrent_reservations_never_overshoot_ceiling():
    # 20 threads each try to reserve 1.00 against a 10.00 ceiling; exactly 10
    # should succeed and the ledger must never exceed its ceiling regardless
    # of interleaving.
    store = _store(run="10.00")
    results: list[bool] = []
    lock = threading.Lock()

    def attempt() -> None:
        reservations = store.try_reserve_batch([("call", Decimal("1.00"), ("run",))])
        with lock:
            results.append(reservations is not None)

    threads = [threading.Thread(target=attempt) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert sum(results) == 10
    assert store.ledger("run").reserved_usd == Decimal("10.00")
    assert store.ledger("run").committed_usd() <= Decimal("10.00")
