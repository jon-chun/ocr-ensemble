"""A3 budget admission: literal hard-cap ledgers.

Every paid attempt is governed by conjunctive ledgers: mandatory run ceiling,
mandatory account-period ceiling, optional account sub-cap, optional model
cap. A reservation must succeed against every applicable ledger before any
network contact; the same amount is later reconciled to actual cost. An
actual charge above the reserved bound is a budget-integrity failure, not a
warning.

This module implements the ledger arithmetic and atomicity in-process (one
run control store, one shared account-period store) rather than the
cross-process/cross-run durable store the full contract describes -- ticket
03 is a single stub-model run proving the reserve/reconcile mechanics, not
the account-period-store's cross-run durability. That narrowing is called
out here rather than silently assumed permanent.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from decimal import Decimal


class BudgetIntegrityError(RuntimeError):
    """An actual charge exceeded its reserved bound: dispatch
    for that Billing Account freezes and postprocess fails the run. Raising
    this is the freeze -- callers must stop dispatching on this Billing
    Account when it's seen, not retry past it.
    """


@dataclass
class Ledger:
    """One named ceiling (run, account-period, account-subcap, or model) with
    a reserved-but-not-yet-settled balance and a settled-spend balance. A
    reservation is provisional exposure; reconciliation replaces it with the
    real charge, which may be lower (never higher without raising
    ``BudgetIntegrityError``).
    """

    name: str
    ceiling_usd: Decimal | None  # None = no ceiling configured for this ledger
    reserved_usd: Decimal = field(default_factory=lambda: Decimal("0"))
    settled_usd: Decimal = field(default_factory=lambda: Decimal("0"))

    def committed_usd(self) -> Decimal:
        return self.reserved_usd + self.settled_usd

    def headroom_usd(self) -> Decimal | None:
        if self.ceiling_usd is None:
            return None
        return self.ceiling_usd - self.committed_usd()


@dataclass
class Reservation:
    reservation_id: str
    amount_usd: Decimal
    ledger_names: tuple[str, ...]


class BudgetLedgerStore:
    """Thread-safe multi-ledger reserve/release/reconcile bookkeeping.

    A single "reserve" call is atomic across every applicable ledger: either
    every ledger has headroom and all are debited together, or none are
    touched and the whole batch is refused. It atomically
    admits and reserves the entire remaining roster batch before issuing
    any member request; if the batch cannot be reserved, no member in that
    batch is dispatched.
    """

    def __init__(self, ledgers: dict[str, Ledger]) -> None:
        self._ledgers = ledgers
        self._lock = threading.Lock()
        self._reservations: dict[str, Reservation] = {}
        self._next_id = 0

    def _fresh_reservation_id(self) -> str:
        self._next_id += 1
        return f"reservation-{self._next_id}"

    def try_reserve_batch(
        self, requests: list[tuple[str, Decimal, tuple[str, ...]]]
    ) -> list[Reservation] | None:
        """Attempt to reserve every ``(label, amount_usd, ledger_names)`` in
        ``requests`` atomically. Returns the list of ``Reservation``s on
        success, or ``None`` if any single ledger lacks headroom for the
        summed exposure of the requests referencing it -- in which case
        nothing is reserved.
        """
        with self._lock:
            summed_by_ledger: dict[str, Decimal] = {}
            for _label, amount, ledger_names in requests:
                for name in ledger_names:
                    summed_by_ledger[name] = summed_by_ledger.get(name, Decimal("0")) + amount

            for name, amount in summed_by_ledger.items():
                ledger = self._ledgers.get(name)
                if ledger is None:
                    continue
                headroom = ledger.headroom_usd()
                if headroom is not None and amount > headroom:
                    return None

            for name, amount in summed_by_ledger.items():
                ledger = self._ledgers.get(name)
                if ledger is not None:
                    ledger.reserved_usd += amount

            reservations: list[Reservation] = []
            for _label, amount, ledger_names in requests:
                reservation_id = self._fresh_reservation_id()
                reservation = Reservation(
                    reservation_id=reservation_id,
                    amount_usd=amount,
                    ledger_names=ledger_names,
                )
                self._reservations[reservation_id] = reservation
                reservations.append(reservation)
            return reservations

    def reconcile(self, reservation_id: str, actual_cost_usd: Decimal) -> None:
        """Replace a reservation's provisional exposure with the real charge.

        Raises ``BudgetIntegrityError`` if the actual charge
        exceeds the amount that was reserved for it -- the reservation was
        supposed to be a defensible upper bound, so this indicates either a
        pricing-registry error or a provider billing anomaly, not something
        to silently absorb.
        """
        with self._lock:
            reservation = self._reservations.pop(reservation_id, None)
            if reservation is None:
                raise KeyError(f"no such reservation: {reservation_id!r}")
            if actual_cost_usd > reservation.amount_usd:
                # restore the reservation so the freeze is inspectable, then raise
                self._reservations[reservation_id] = reservation
                raise BudgetIntegrityError(
                    f"actual_cost_usd={actual_cost_usd} exceeds reserved "
                    f"amount_usd={reservation.amount_usd} for {reservation_id!r}: "
                    "an actual charge above the reserved bound "
                    "is a budget-integrity failure"
                )
            for name in reservation.ledger_names:
                ledger = self._ledgers.get(name)
                if ledger is not None:
                    ledger.reserved_usd -= reservation.amount_usd
                    ledger.settled_usd += actual_cost_usd

    def release(self, reservation_id: str) -> None:
        """Release a reservation without any charge (e.g. the call was never
        attempted, or was cancelled before contact). Distinct from
        ``reconcile(..., Decimal(0))``: a zero-cost reconciliation still
        implies contact was attempted.
        """
        with self._lock:
            reservation = self._reservations.pop(reservation_id, None)
            if reservation is None:
                raise KeyError(f"no such reservation: {reservation_id!r}")
            for name in reservation.ledger_names:
                ledger = self._ledgers.get(name)
                if ledger is not None:
                    ledger.reserved_usd -= reservation.amount_usd

    def ledger(self, name: str) -> Ledger:
        return self._ledgers[name]


def maximum_exposure_usd(
    *,
    max_input_units: int,
    input_unit_price_usd: Decimal,
    max_output_units: int,
    output_unit_price_usd: Decimal,
    fixed_charges_usd: Decimal,
) -> Decimal:
    """``maximum_exposure_usd`` formula: the sum of every
    upper-bound billing term this pricing snapshot declares. Not a typical
    estimate -- every term must be an upper bound, or the caller has no
    business calling this function with it.
    """
    return (
        Decimal(max_input_units) * input_unit_price_usd
        + Decimal(max_output_units) * output_unit_price_usd
        + fixed_charges_usd
    )
