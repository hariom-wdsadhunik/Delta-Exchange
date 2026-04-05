"""
PnL accounting: balance, position, fees, realized and mark-to-market unrealized PnL.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from config import Config
from logger import log_event


@dataclass
class PnLState:
    balance: float
    position: float = 0.0
    avg_entry: float = 0.0
    realized_pnl: float = 0.0
    fees_paid: float = 0.0
    trade_count: int = 0
    fill_qty_total: float = 0.0
    fill_liquidity_caps: List[float] = field(default_factory=list)


class PnLEngine:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.state = PnLState(balance=cfg.starting_balance_quote)

    def _fee(self, notional: float) -> float:
        return abs(notional) * self.cfg.fee_rate_per_side

    def apply_buy(self, price: float, qty: float, liquidity_cap: float) -> None:
        if qty <= 0:
            return
        fill_qty = qty
        fee = self._fee(price * fill_qty)
        self.state.fees_paid += fee
        self.state.trade_count += 1
        self.state.fill_qty_total += fill_qty
        self.state.fill_liquidity_caps.append(liquidity_cap)

        self.state.balance -= price * fill_qty + fee

        pos = self.state.position
        avg = self.state.avg_entry
        q = fill_qty

        if pos < 0:
            cover = min(q, -pos)
            self.state.realized_pnl += cover * (avg - price)
            pos += cover
            q -= cover
            if pos == 0:
                avg = 0.0
            self.state.position = pos
            self.state.avg_entry = avg

        if q > 0:
            pos = self.state.position
            avg = self.state.avg_entry
            if pos >= 0:
                new_pos = pos + q
                avg = (avg * pos + price * q) / new_pos if pos > 0 else price
                self.state.position = new_pos
                self.state.avg_entry = avg

        log_event(
            "fill_buy",
            price=price,
            qty=fill_qty,
            position=self.state.position,
            fee=fee,
            liquidity_cap=liquidity_cap,
        )

    def apply_sell(self, price: float, qty: float, liquidity_cap: float) -> None:
        if qty <= 0:
            return
        fill_qty = qty
        fee = self._fee(price * fill_qty)
        self.state.fees_paid += fee
        self.state.trade_count += 1
        self.state.fill_qty_total += fill_qty
        self.state.fill_liquidity_caps.append(liquidity_cap)

        self.state.balance += price * fill_qty - fee

        pos = self.state.position
        avg = self.state.avg_entry
        q = fill_qty

        if pos > 0:
            close = min(q, pos)
            self.state.realized_pnl += close * (price - avg)
            pos -= close
            q -= close
            if pos == 0:
                avg = 0.0
            self.state.position = pos
            self.state.avg_entry = avg

        if q > 0:
            pos = self.state.position
            avg = self.state.avg_entry
            if pos <= 0:
                short_sz = -pos
                new_short = short_sz + q
                avg = (avg * short_sz + price * q) / new_short if short_sz > 0 else price
                self.state.position = -new_short
                self.state.avg_entry = avg

        log_event(
            "fill_sell",
            price=price,
            qty=fill_qty,
            position=self.state.position,
            fee=fee,
            liquidity_cap=liquidity_cap,
        )

    def unrealized(self, mark: float) -> float:
        p = self.state.position
        if p == 0:
            return 0.0
        if p > 0:
            return p * (mark - self.state.avg_entry)
        return (-p) * (self.state.avg_entry - mark)

    def equity(self, mark: float) -> float:
        return self.state.balance + self.unrealized(mark)
