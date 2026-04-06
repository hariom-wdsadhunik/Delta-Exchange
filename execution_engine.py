"""
Simulated execution: order lifecycle, matching, and fill events (no PnL / strategy).
"""

from __future__ import annotations

import random
import uuid
from dataclasses import dataclass
from typing import Dict, List

from execution_types import FillEvent, OrderIntent, OrderSide, OrderStatus, SimOrder
from orderbook import LocalOrderBook


@dataclass
class SubmitResult:
    accepted: bool
    order_id: str | None = None
    reason: str | None = None


class ExecutionEngine:
    """
    Tracks simulated orders, activates / expires them, and produces FillEvents.
    Position limits are enforced on submit and per fill (clipped).
    """

    def __init__(
        self,
        max_position: float,
        *,
        order_timeout_sec: float = 2.5,
        slippage_factor: float = 0.15,
        passive_base_fraction: float = 0.08,
        time_priority_half_life_sec: float = 0.5,
        latency_ms_range: tuple[float, float] = (30.0, 120.0),
        queue_factor_range: tuple[float, float] = (0.1, 0.3),
        stale_quote_threshold: float = 1.0,
        market_move_cancel_threshold: float | None = None,
        min_spread_for_passive: float = 0.5,
    ) -> None:
        if max_position <= 0:
            raise ValueError("max_position must be positive")
        self._max_position = max_position
        self._order_timeout_sec = order_timeout_sec
        self._slippage_factor = slippage_factor
        self._passive_base_fraction = passive_base_fraction
        self._time_half_life = time_priority_half_life_sec
        self._lat_min, self._lat_max = latency_ms_range
        self._qpf_min, self._qpf_max = queue_factor_range
        self._stale_quote_threshold = stale_quote_threshold
        self._market_move_cancel_threshold = market_move_cancel_threshold
        self._min_spread_for_passive = min_spread_for_passive

        self.active_orders: List[SimOrder] = []
        # Reference mid captured on first tick the order is live in the book (for drift cancel)
        self._mid_at_submit: Dict[str, float] = {}

    # --- public API ---

    def submit_order(self, intent: OrderIntent, now: float, position: float) -> SubmitResult:
        """Risk-check, then create a QUEUED SimOrder with sampled latency and queue factor."""
        if intent.quantity <= 0:
            return SubmitResult(False, reason="non_positive_quantity")

        if intent.side == OrderSide.BUY:
            if position + intent.quantity > self._max_position + 1e-12:
                return SubmitResult(False, reason="max_long_exceeded")
        else:
            if position - intent.quantity < -self._max_position - 1e-12:
                return SubmitResult(False, reason="max_short_exceeded")

        lat_ms = random.uniform(self._lat_min, self._lat_max)
        qpf = random.uniform(self._qpf_min, self._qpf_max)
        oid = uuid.uuid4().hex[:16]
        activate_at = now + lat_ms / 1000.0
        expire_at = now + self._order_timeout_sec

        order = SimOrder(
            order_id=oid,
            symbol=intent.symbol,
            side=intent.side,
            limit_price=intent.price,
            original_quantity=intent.quantity,
            status=OrderStatus.QUEUED,
            created_time=now,
            activated_time=None,
            activate_at=activate_at,
            expire_at=expire_at,
            latency_ms=lat_ms,
            queue_position_factor=qpf,
        )
        self.active_orders.append(order)
        return SubmitResult(True, order_id=oid)

    def process_tick(self, book: LocalOrderBook, now: float, position: float) -> List[FillEvent]:
        """Expire, activate, match, prune terminal orders; returns all fill events this tick."""
        for o in list(self.active_orders):
            if o.is_terminal():
                continue
            if o.is_expired(now):
                o.expire()

        for o in list(self.active_orders):
            if o.is_terminal():
                continue
            if o.status == OrderStatus.QUEUED and now >= o.activate_at:
                o.activate(now)

        self._cancel_stale_or_moved(book)

        fills = self.match_orders(book, now, position)
        self._prune_terminal()
        return fills

    def match_orders(self, book: LocalOrderBook, now: float, position: float) -> List[FillEvent]:
        """Simulate aggressive then passive fills; updates orders in place."""
        events: List[FillEvent] = []
        working = [position]

        for order in list(self.active_orders):
            if not order.is_active(now):
                continue
            if order.side == OrderSide.BUY:
                events.extend(self._match_buy(order, book, now, working))
            else:
                events.extend(self._match_sell(order, book, now, working))

        return events

    # --- internals ---

    def _prune_terminal(self) -> None:
        kept: List[SimOrder] = []
        for o in list(self.active_orders):
            if o.is_terminal():
                self._mid_at_submit.pop(o.order_id, None)
            else:
                kept.append(o)
        self.active_orders = kept

    def _snapshot_reference_mid(self, order: SimOrder, book: LocalOrderBook) -> None:
        if order.order_id in self._mid_at_submit:
            return
        m = book.mid()
        if m is not None:
            self._mid_at_submit[order.order_id] = m

    def _cancel_stale_or_moved(self, book: LocalOrderBook) -> None:
        """
        Cancel working orders left behind by the market:
        - BUY: limit far below best bid
        - SELL: limit far above best ask
        - Optional: mid moved vs reference mid by more than market_move_cancel_threshold
        """
        bb = book.best_bid()
        ba = book.best_ask()
        mid = book.mid()

        for order in list(self.active_orders):
            if order.is_terminal():
                continue
            self._snapshot_reference_mid(order, book)

            if self._market_move_cancel_threshold is not None and mid is not None:
                ref = self._mid_at_submit.get(order.order_id)
                if ref is not None:
                    if abs(mid - ref) > self._market_move_cancel_threshold:
                        order.cancel()
                        continue

            if order.side == OrderSide.BUY:
                if bb is not None and order.limit_price < bb.price - self._stale_quote_threshold:
                    order.cancel()
            else:
                if ba is not None and order.limit_price > ba.price + self._stale_quote_threshold:
                    order.cancel()

    def _max_buy_additional(self, pos: float) -> float:
        return max(0.0, self._max_position - pos)

    def _max_sell_additional(self, pos: float) -> float:
        return max(0.0, pos + self._max_position)

    @staticmethod
    def _decay_queue_factor(order: SimOrder) -> None:
        """Worsen effective queue priority after each fill."""
        order.queue_position_factor *= 0.9

    def _time_priority_scale(self, order: SimOrder, now: float) -> float:
        """Older (active longer) orders get larger scale in (0, 1)."""
        t = order.active_time(now)
        return t / (t + self._time_half_life) if t > 0 else 0.0

    def _spread(self, book: LocalOrderBook) -> float:
        s = book.spread()
        return float(s) if s is not None else 0.0

    def _passive_spread_allowed(self, book: LocalOrderBook) -> bool:
        """Passive fills need room in the spread; skip when book is too tight."""
        s = book.spread()
        if s is None:
            return False
        return s >= self._min_spread_for_passive

    def _match_buy(
        self,
        order: SimOrder,
        book: LocalOrderBook,
        now: float,
        working: List[float],
    ) -> List[FillEvent]:
        events: List[FillEvent] = []
        ba = book.best_ask()
        if ba is None:
            return events

        # Aggressive: limit crosses the ask
        if order.limit_price >= ba.price:
            base_slip = self._spread(book) * self._slippage_factor
            for level_index, (ap, sz) in enumerate(
                book.liquidity_buy(order.limit_price), start=1
            ):
                if order.remaining_quantity <= 1e-12:
                    break
                avail = sz * order.queue_position_factor
                cap = self._max_buy_additional(working[0])
                take = min(order.remaining_quantity, avail, cap)
                if take <= 1e-12:
                    continue
                slip = base_slip * level_index
                raw_px = ap + slip
                exec_px = min(order.limit_price, raw_px)
                if exec_px < ap - 1e-12:
                    continue
                order.apply_fill(take)
                self._decay_queue_factor(order)
                working[0] += take
                events.append(
                    FillEvent(
                        order_id=order.order_id,
                        side=OrderSide.BUY,
                        price=exec_px,
                        quantity=take,
                        timestamp=now,
                        is_maker=False,
                    )
                )
            return events

        # Passive: not crossing — queue at best bid only (conservative)
        bb = book.best_bid()
        if bb is None:
            return events
        if order.limit_price < bb.price - 1e-12:
            return events
        if not self._passive_spread_allowed(book):
            return events

        tscale = self._time_priority_scale(order, now)
        if tscale <= 1e-12:
            return events

        visible = book.bid_size_at(bb.price)
        avail = visible * order.queue_position_factor
        cap = self._max_buy_additional(working[0])
        fill_try = avail * self._passive_base_fraction * tscale
        fill_try = min(fill_try, visible * 0.1)
        take = min(order.remaining_quantity, fill_try, cap)
        if take <= 1e-12:
            return events

        exec_px = min(order.limit_price, bb.price)
        order.apply_fill(take)
        self._decay_queue_factor(order)
        working[0] += take
        events.append(
            FillEvent(
                order_id=order.order_id,
                side=OrderSide.BUY,
                price=exec_px,
                quantity=take,
                timestamp=now,
                is_maker=True,
            )
        )
        return events

    def _match_sell(
        self,
        order: SimOrder,
        book: LocalOrderBook,
        now: float,
        working: List[float],
    ) -> List[FillEvent]:
        events: List[FillEvent] = []
        bb = book.best_bid()
        if bb is None:
            return events

        if order.limit_price <= bb.price:
            base_slip = self._spread(book) * self._slippage_factor
            for level_index, (bp, sz) in enumerate(
                book.liquidity_sell(order.limit_price), start=1
            ):
                if order.remaining_quantity <= 1e-12:
                    break
                avail = sz * order.queue_position_factor
                cap = self._max_sell_additional(working[0])
                take = min(order.remaining_quantity, avail, cap)
                if take <= 1e-12:
                    continue
                slip = base_slip * level_index
                raw_px = bp - slip
                if raw_px < order.limit_price - 1e-12:
                    continue
                exec_px = raw_px
                order.apply_fill(take)
                self._decay_queue_factor(order)
                working[0] -= take
                events.append(
                    FillEvent(
                        order_id=order.order_id,
                        side=OrderSide.SELL,
                        price=exec_px,
                        quantity=take,
                        timestamp=now,
                        is_maker=False,
                    )
                )
            return events

        ba = book.best_ask()
        if ba is None:
            return events
        if order.limit_price > ba.price + 1e-12:
            return events
        if not self._passive_spread_allowed(book):
            return events

        tscale = self._time_priority_scale(order, now)
        if tscale <= 1e-12:
            return events

        visible = book.ask_size_at(ba.price)
        avail = visible * order.queue_position_factor
        cap = self._max_sell_additional(working[0])
        fill_try = avail * self._passive_base_fraction * tscale
        fill_try = min(fill_try, visible * 0.1)
        take = min(order.remaining_quantity, fill_try, cap)
        if take <= 1e-12:
            return events

        exec_px = max(order.limit_price, ba.price)
        order.apply_fill(take)
        self._decay_queue_factor(order)
        working[0] -= take
        events.append(
            FillEvent(
                order_id=order.order_id,
                side=OrderSide.SELL,
                price=exec_px,
                quantity=take,
                timestamp=now,
                is_maker=True,
            )
        )
        return events
