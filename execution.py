"""
Simulated limit-order execution: latency activation, timeout, partial fills vs visible liquidity.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional

from config import Config
from logger import get_logger, log_event
from orderbook import LocalOrderBook
from pnl import PnLEngine
from strategy import Signal

log = get_logger()


class OrderStatus(str, Enum):
    PENDING = "PENDING"
    ACTIVE = "ACTIVE"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    EXPIRED = "EXPIRED"


@dataclass
class SimOrder:
    id: str
    side: str  # 'buy' | 'sell'
    limit_price: float
    size: float
    remaining: float
    created_mono: float
    activates_mono: float
    expires_mono: float
    status: OrderStatus = OrderStatus.PENDING
    filled: float = 0.0


class ExecutionSimulator:
    def __init__(self, cfg: Config, pnl: PnLEngine) -> None:
        self.cfg = cfg
        self.pnl = pnl
        self.orders: Dict[str, SimOrder] = {}
        self._spread_fn: Optional[Callable[[], float]] = None

    def set_spread_sampler(self, fn: Callable[[], float]) -> None:
        self._spread_fn = fn

    def cancel_all_active(self, reason: str) -> None:
        for o in list(self.orders.values()):
            if o.status in (OrderStatus.PENDING, OrderStatus.ACTIVE):
                o.status = OrderStatus.CANCELLED
                log_event("order_cancel", order_id=o.id, reason=reason)

    def submit_limit(self, side: str, limit_price: float, size: float) -> str:
        now = time.monotonic()
        oid = uuid.uuid4().hex[:12]
        lat = self.cfg.latency_ms / 1000.0
        order = SimOrder(
            id=oid,
            side=side,
            limit_price=limit_price,
            size=size,
            remaining=size,
            created_mono=now,
            activates_mono=now + lat,
            expires_mono=now + self.cfg.order_timeout_sec,
            status=OrderStatus.PENDING,
        )
        self.orders[oid] = order
        log_event(
            "order_submit",
            order_id=oid,
            side=side,
            price=limit_price,
            size=size,
            latency_ms=self.cfg.latency_ms,
        )
        return oid

    def _cap_level(self, level_size: float) -> float:
        return max(0.0, level_size * self.cfg.max_fill_fraction_of_visible_level)

    def _aggressive_buy(self, book: LocalOrderBook, order: SimOrder) -> float:
        """Return fill qty from crossing bid through asks."""
        if order.remaining <= 0:
            return 0.0
        if not book.best_ask() or order.limit_price < book.best_ask().price:
            return 0.0
        need = order.remaining
        filled = 0.0
        for ap, sz in book.liquidity_buy(order.limit_price):
            cap = self._cap_level(sz)
            take = min(need, cap)
            if take <= 0:
                break
            liq_cap = cap
            self.pnl.apply_buy(ap, take, liq_cap)
            filled += take
            need -= take
            if need <= 1e-12:
                break
        return filled

    def _aggressive_sell(self, book: LocalOrderBook, order: SimOrder) -> float:
        if order.remaining <= 0:
            return 0.0
        if not book.best_bid() or order.limit_price > book.best_bid().price:
            return 0.0
        need = order.remaining
        filled = 0.0
        for bp, sz in book.liquidity_sell(order.limit_price):
            cap = self._cap_level(sz)
            take = min(need, cap)
            if take <= 0:
                break
            liq_cap = cap
            self.pnl.apply_sell(bp, take, liq_cap)
            filled += take
            need -= take
            if need <= 1e-12:
                break
        return filled

    def _passive_buy(self, book: LocalOrderBook, order: SimOrder, spread_z: float) -> float:
        if order.remaining <= 0:
            return 0.0
        bb = book.best_bid()
        if not bb:
            return 0.0
        if order.limit_price < bb.price - 1e-9:
            return 0.0
        if spread_z > self.cfg.max_spread_z_for_passive:
            return 0.0
        at = book.bid_size_at(bb.price)
        cap = self._cap_level(at) * self.cfg.passive_fill_touch_fraction
        take = min(order.remaining, cap)
        if take <= 1e-12:
            return 0.0
        self.pnl.apply_buy(bb.price, take, cap)
        return take

    def _passive_sell(self, book: LocalOrderBook, order: SimOrder, spread_z: float) -> float:
        if order.remaining <= 0:
            return 0.0
        ba = book.best_ask()
        if not ba:
            return 0.0
        if order.limit_price > ba.price + 1e-9:
            return 0.0
        if spread_z > self.cfg.max_spread_z_for_passive:
            return 0.0
        at = book.ask_size_at(ba.price)
        cap = self._cap_level(at) * self.cfg.passive_fill_touch_fraction
        take = min(order.remaining, cap)
        if take <= 1e-12:
            return 0.0
        self.pnl.apply_sell(ba.price, take, cap)
        return take

    def on_book_update(self, book: LocalOrderBook, spread_z: float) -> None:
        now = time.monotonic()
        for order in list(self.orders.values()):
            if order.status == OrderStatus.PENDING and now >= order.activates_mono:
                order.status = OrderStatus.ACTIVE
                log_event("order_active", order_id=order.id, side=order.side)

            if order.status != OrderStatus.ACTIVE:
                continue

            if now >= order.expires_mono and order.remaining > 1e-9:
                order.status = OrderStatus.EXPIRED
                log_event("order_expired", order_id=order.id, remaining=order.remaining)
                continue

            filled = 0.0
            if order.side == "buy":
                filled += self._aggressive_buy(book, order)
                if order.remaining > 1e-9:
                    filled += self._passive_buy(book, order, spread_z)
            else:
                filled += self._aggressive_sell(book, order)
                if order.remaining > 1e-9:
                    filled += self._passive_sell(book, order, spread_z)

            if filled > 0:
                order.remaining -= filled
                order.filled += filled
                log_event(
                    "order_partial",
                    order_id=order.id,
                    filled_this_tick=filled,
                    remaining=order.remaining,
                )

            if order.remaining <= 1e-9:
                order.status = OrderStatus.FILLED
                log_event("order_filled", order_id=order.id)

    def active_order_count(self) -> int:
        return sum(1 for o in self.orders.values() if o.status in (OrderStatus.PENDING, OrderStatus.ACTIVE))


def plan_orders_for_signal(
    signal: Signal,
    book: LocalOrderBook,
    cfg: Config,
) -> List[tuple]:
    """
    Return list of (side, price, size) limit intents — join-touch style inside spread.
    """
    bb, ba = book.best_bid(), book.best_ask()
    if not bb or not ba:
        return []
    mid = (bb.price + ba.price) / 2.0
    step = max((ba.price - bb.price) / 4, 0.25)
    size = cfg.order_size_contracts
    intents: List[tuple] = []

    if signal == Signal.BUY:
        # Passive buy near bid; may still cross if book moves
        px = min(bb.price + step * 0.5, ba.price - 1e-6)
        intents.append(("buy", px, size))
    elif signal == Signal.SELL:
        px = max(ba.price - step * 0.5, bb.price + 1e-6)
        intents.append(("sell", px, size))
    elif signal == Signal.BOTH:
        intents.append(("buy", bb.price + 1e-3, size))
        intents.append(("sell", ba.price - 1e-3, size))
    return intents
