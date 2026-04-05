"""
Core execution types: order lifecycle, intents, and fill events.

Standard library only. Used by the execution engine (not implemented here).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(str, Enum):
    NEW = "NEW"
    QUEUED = "QUEUED"
    ACTIVE = "ACTIVE"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    EXPIRED = "EXPIRED"


_TERMINAL = frozenset(
    {OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.EXPIRED}
)


@dataclass
class SimOrder:
    """
    Simulated limit order with explicit lifecycle and fill tracking.

    Typical flow: NEW -> QUEUED (optional) -> ACTIVE (via activate()) ->
    PARTIALLY_FILLED* -> FILLED, or CANCELLED / EXPIRED.
    """

    order_id: str
    symbol: str
    side: OrderSide
    limit_price: float
    original_quantity: float

    status: OrderStatus
    created_time: float
    activated_time: float | None
    activate_at: float
    expire_at: float

    filled_quantity: float = 0.0
    remaining_quantity: float = field(init=False)

    latency_ms: float = 0.0
    queue_position_factor: float = 0.2

    def __post_init__(self) -> None:
        if self.original_quantity <= 0:
            raise ValueError("original_quantity must be positive")
        if self.expire_at <= self.created_time:
            raise ValueError("expire_at must be after created_time")
        if self.activate_at < self.created_time:
            raise ValueError("activate_at must not be before created_time")
        if not (0.0 < self.queue_position_factor <= 1.0):
            raise ValueError("queue_position_factor must be in (0, 1]")
        object.__setattr__(self, "remaining_quantity", float(self.original_quantity))

    def is_active(self, now: float) -> bool:
        """True if the order rests in the (simulated) book and may still trade."""
        if self.status not in (OrderStatus.ACTIVE, OrderStatus.PARTIALLY_FILLED):
            return False
        if now >= self.expire_at:
            return False
        return now >= self.activate_at

    def is_expired(self, now: float) -> bool:
        """True if wall-clock expiry has passed while the order could still be transitioned to EXPIRED."""
        if self.status in _TERMINAL:
            return False
        return now >= self.expire_at

    def is_terminal(self) -> bool:
        """True if FILLED, CANCELLED, or EXPIRED."""
        return self.status in _TERMINAL

    def active_time(self, now: float) -> float:
        """Seconds since activation; 0.0 if never activated."""
        if self.activated_time is None:
            return 0.0
        return now - self.activated_time

    def activate(self, now: float) -> None:
        """
        Transition from NEW or QUEUED to ACTIVE after simulated latency.
        Caller should invoke when now >= activate_at.
        """
        if self.status not in (OrderStatus.NEW, OrderStatus.QUEUED):
            raise RuntimeError(f"Invalid activation from state {self.status}")
        if now < self.activate_at:
            return
        self.status = OrderStatus.ACTIVE
        self.activated_time = now

    def apply_fill(self, quantity: float) -> None:
        """Apply a partial or full fill; updates status to PARTIALLY_FILLED or FILLED."""
        if quantity <= 0:
            raise ValueError("fill quantity must be positive")
        if self.status in _TERMINAL:
            raise ValueError(f"cannot fill order in status {self.status}")
        if self.status not in (OrderStatus.ACTIVE, OrderStatus.PARTIALLY_FILLED):
            raise ValueError(f"cannot fill order in status {self.status}")

        if quantity > self.remaining_quantity and not math.isclose(
            quantity, self.remaining_quantity, rel_tol=0.0, abs_tol=1e-9
        ):
            raise ValueError(
                f"fill quantity {quantity} exceeds remaining {self.remaining_quantity}"
            )
        q = min(quantity, self.remaining_quantity)
        self.filled_quantity += q
        self.remaining_quantity -= q

        if math.isclose(self.remaining_quantity, 0.0, abs_tol=1e-9):
            self.remaining_quantity = 0.0
            self.status = OrderStatus.FILLED
        else:
            self.status = OrderStatus.PARTIALLY_FILLED

    def cancel(self) -> None:
        """Mark the order cancelled if it is not already in a terminal state."""
        if self.status in _TERMINAL:
            return
        self.status = OrderStatus.CANCELLED
        # Remaining quantity is left as-is for audit; engine may clear or ignore.

    def expire(self) -> None:
        """Mark the order as expired if not already terminal."""
        if self.status in _TERMINAL:
            return
        self.status = OrderStatus.EXPIRED


@dataclass(frozen=True, slots=True)
class FillEvent:
    """One execution print against an order."""

    order_id: str
    side: OrderSide
    price: float
    quantity: float
    timestamp: float
    is_maker: bool


@dataclass(frozen=True, slots=True)
class OrderIntent:
    """Strategy output: desired limit order parameters (no lifecycle)."""

    side: OrderSide
    price: float
    quantity: float
    symbol: str
