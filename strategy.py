"""
Signal generation: spread filter, order-book imbalance, optional inventory skew & volatility filter.
"""

from __future__ import annotations

from collections import deque
from enum import Enum
from typing import Deque, Optional, Tuple

from config import Config
from orderbook import LocalOrderBook


class Signal(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    BOTH = "BOTH"
    NONE = "NONE"


def _imbalance(book: LocalOrderBook, levels: int) -> Optional[float]:
    bv = book.cumulative_bid_size(levels)
    av = book.cumulative_ask_size(levels)
    if bv + av <= 0:
        return None
    return (bv - av) / (bv + av)


class StrategyEngine:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self._spread_hist: Deque[float] = deque(maxlen=cfg.volatility_spread_lookback)

    def volatility_ok(self, spread: float) -> bool:
        if not self.cfg.volatility_filter_enabled:
            return True
        if len(self._spread_hist) < 5:
            return True
        sorted_s = sorted(self._spread_hist)
        med = sorted_s[len(sorted_s) // 2]
        if med <= 0:
            return True
        return spread <= med * self.cfg.max_spread_vs_median_ratio

    def spread_zscore(self, spread: float) -> float:
        if len(self._spread_hist) < 5:
            return 0.0
        xs = list(self._spread_hist)
        m = sum(xs) / len(xs)
        var = sum((x - m) ** 2 for x in xs) / max(1, len(xs) - 1)
        std = var**0.5
        if std < 1e-12:
            return 0.0
        return (spread - m) / std

    def evaluate(self, book: LocalOrderBook, position: float) -> Tuple[Signal, dict]:
        meta: dict = {}
        sp = book.spread()
        mid = book.mid()
        if sp is None or mid is None:
            return Signal.NONE, {"reason": "incomplete_book"}

        self._spread_hist.append(sp)

        if sp < self.cfg.min_spread_abs:
            return Signal.NONE, {"reason": "spread_too_tight", "spread": sp}

        if not self.volatility_ok(sp):
            return Signal.NONE, {"reason": "volatility_filter", "spread": sp}

        imb = _imbalance(book, self.cfg.imbalance_levels)
        if imb is None:
            return Signal.NONE, {"reason": "no_imbalance"}

        thr = self.cfg.imbalance_threshold
        if self.cfg.inventory_skew_enabled:
            thr += self.cfg.inventory_skew_per_contract * abs(position)
            # Lean against inventory: harder to add to same side
            if position > 0 and imb > 0:
                thr *= 1.0 + min(2.0, position / max(self.cfg.max_position_contracts, 1e-9))
            if position < 0 and imb < 0:
                thr *= 1.0 + min(2.0, abs(position) / max(self.cfg.max_position_contracts, 1e-9))

        meta.update({"spread": sp, "imbalance": imb, "threshold": thr, "mid": mid})

        wide = sp >= self.cfg.spread_wide_for_both
        if wide:
            return Signal.BOTH, meta

        if imb > thr:
            return Signal.BUY, meta
        if imb < -thr:
            return Signal.SELL, meta
        return Signal.NONE, meta
