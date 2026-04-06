"""
Strategy layer: book features → Signal + OrderIntent list (no execution).

Effective imbalance for signals / aggression is **level + momentum**:
  ``imb_eff = clip(imb + imbalance_momentum_weight * Δimb, [-1, 1])``
where ``Δimb`` is the change vs the last stored imbalance sample.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Deque, List, Optional

from execution_types import OrderIntent, OrderSide
from orderbook import LocalOrderBook, Quote


class Signal(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    BOTH = "BOTH"
    NONE = "NONE"


@dataclass
class StrategyOutput:
    signal: Signal
    intents: List[OrderIntent]
    # Populated when a book imbalance was computed this tick:
    imbalance: Optional[float] = None
    imbalance_delta: Optional[float] = None
    effective_imbalance: Optional[float] = None


class StrategyEngine:
    """
    Spread and imbalance filters with optional inventory skew; emits limit OrderIntents.
    Directional/BOTH decisions use **imbalance + weighted change in imbalance** (see module doc).
    """

    def __init__(
        self,
        *,
        symbol: str,
        base_order_size: float = 1.0,
        min_spread_abs: float = 0.5,
        imbalance_levels: int = 5,
        imbalance_level_decay: float = 0.72,
        imbalance_threshold: float = 0.15,
        spread_wide_for_both: float = 2.0,
        inventory_skew_enabled: bool = True,
        inventory_skew_per_contract: float = 0.04,
        max_position: float = 5.0,
        inventory_size_scale: float = 0.85,
        cooldown_sec: float = 0.25,
        # Placement: passive by default; max aggression only if |imb_eff| clears thr by strong_signal_margin
        buy_aggression_in_spread: float = 0.2,
        sell_aggression_in_spread: float = 0.2,
        aggression_passive_in_spread: float = 0.05,
        strong_signal_margin: float = 0.1,
        spread_median_spike_threshold: float | None = 2.5,
        spread_history_maxlen: int = 20,
        spread_median_min_samples: int = 5,
        imbalance_flip_block_enabled: bool = True,
        imbalance_flip_window_sec: float = 0.6,
        imbalance_flip_min_abs: float = 0.08,
        imbalance_flip_history_maxlen: int = 48,
        imbalance_momentum_weight: float = 1.0,
    ) -> None:
        if base_order_size <= 0:
            raise ValueError("base_order_size must be positive")
        if max_position <= 0:
            raise ValueError("max_position must be positive")
        if not 0.0 < imbalance_level_decay <= 1.0:
            raise ValueError("imbalance_level_decay must be in (0, 1]")
        self._symbol = symbol
        self._base_order_size = base_order_size
        self._min_spread_abs = min_spread_abs
        self._imbalance_levels = imbalance_levels
        self._imbalance_decay = imbalance_level_decay
        self._imbalance_thr = imbalance_threshold
        self._spread_wide_for_both = spread_wide_for_both
        self._inventory_skew_enabled = inventory_skew_enabled
        self._inventory_skew_per_contract = inventory_skew_per_contract
        self._max_position = max_position
        self._inventory_size_scale = inventory_size_scale
        self._cooldown_sec = cooldown_sec
        self._buy_agg = buy_aggression_in_spread
        self._sell_agg = sell_aggression_in_spread
        self._passive_agg = aggression_passive_in_spread
        self._strong_margin = strong_signal_margin
        self._spread_spike_thr = spread_median_spike_threshold
        self._spread_hist_max = spread_history_maxlen
        self._spread_median_min = spread_median_min_samples

        self._last_emit_time: float = -1e18
        self._spread_hist: Deque[float] = deque(maxlen=spread_history_maxlen)
        self._flip_block_enabled = imbalance_flip_block_enabled
        self._flip_window = imbalance_flip_window_sec
        self._flip_min_abs = imbalance_flip_min_abs
        self._imb_hist: Deque[tuple[float, float]] = deque(
            maxlen=imbalance_flip_history_maxlen
        )
        self._imb_momentum_w = imbalance_momentum_weight

    @staticmethod
    def _pack(
        signal: Signal,
        intents: List[OrderIntent],
        *,
        imb: Optional[float] = None,
        delta: Optional[float] = None,
        imb_eff: Optional[float] = None,
    ) -> StrategyOutput:
        return StrategyOutput(
            signal,
            intents,
            imbalance=imb,
            imbalance_delta=delta,
            effective_imbalance=imb_eff,
        )

    def evaluate(
        self,
        book: LocalOrderBook,
        position: float,
        now: float,
        symbol: Optional[str] = None,
    ) -> StrategyOutput:
        """
        Produce a signal and zero or more OrderIntents. Respects cooldown when emitting.

        Uses ``imb_eff = imb + w*Δimb`` (clamped) vs threshold for BUY/SELL/BOTH; raw ``imb``
        is still used for fast-flip detection. ``StrategyOutput`` carries imb / Δ / imb_eff
        when imbalance was computed.
        """
        sym = symbol or self._symbol
        bb = book.best_bid()
        ba = book.best_ask()
        if bb is None or ba is None:
            return StrategyOutput(Signal.NONE, [])

        spread = book.spread()
        if spread is None or spread < self._min_spread_abs:
            return StrategyOutput(Signal.NONE, [])

        if self._spread_spike_blocks(spread):
            self._record_spread(spread)
            return StrategyOutput(Signal.NONE, [])
        self._record_spread(spread)

        imb = self._imbalance(book)
        if imb is None:
            return StrategyOutput(Signal.NONE, [])

        flip_risk = self._imbalance_fast_flip_blocks(now, imb)
        delta = self._imbalance_change_since_last(imb)
        imb_eff = self._effective_imbalance_with_momentum(imb, delta)
        self._record_imbalance_sample(now, imb)
        if flip_risk:
            return self._pack(Signal.NONE, [], imb=imb, delta=delta, imb_eff=imb_eff)

        thr = self._effective_imbalance_threshold(position)
        signal = self._raw_signal(spread, imb_eff, thr)

        if signal == Signal.NONE:
            return self._pack(Signal.NONE, [], imb=imb, delta=delta, imb_eff=imb_eff)

        intents = self._build_intents(
            signal, bb, ba, spread, position, sym, imb_eff, thr
        )

        if not intents:
            return self._pack(Signal.NONE, [], imb=imb, delta=delta, imb_eff=imb_eff)

        if now - self._last_emit_time < self._cooldown_sec:
            return self._pack(Signal.NONE, [], imb=imb, delta=delta, imb_eff=imb_eff)

        self._last_emit_time = now
        return self._pack(signal, intents, imb=imb, delta=delta, imb_eff=imb_eff)

    # --- book features ---

    def _record_spread(self, spread: float) -> None:
        self._spread_hist.append(spread)

    def _median_spread(self) -> Optional[float]:
        if len(self._spread_hist) < self._spread_median_min:
            return None
        s = sorted(self._spread_hist)
        n = len(s)
        mid = n // 2
        if n % 2:
            return float(s[mid])
        return (s[mid - 1] + s[mid]) / 2.0

    def _spread_spike_blocks(self, spread: float) -> bool:
        """True if current spread vs rolling median is too high — skip trading this tick."""
        if self._spread_spike_thr is None:
            return False
        med = self._median_spread()
        if med is None or med <= 1e-12:
            return False
        return spread / med > self._spread_spike_thr

    def _record_imbalance_sample(self, now: float, imb: float) -> None:
        self._imb_hist.append((now, imb))
        cutoff = now - self._flip_window - 2.0
        while self._imb_hist and self._imb_hist[0][0] < cutoff:
            self._imb_hist.popleft()

    def _imbalance_change_since_last(self, imb: float) -> float:
        """Δimb vs the most recent stored sample (before this tick is appended)."""
        if not self._imb_hist:
            return 0.0
        return imb - self._imb_hist[-1][1]

    def _effective_imbalance_with_momentum(self, imb: float, delta: float) -> float:
        """imbalance + (weight × change in imbalance); clamped to [-1, 1]."""
        raw = imb + self._imb_momentum_w * delta
        return max(-1.0, min(1.0, raw))

    def _imbalance_fast_flip_blocks(self, now: float, imb: float) -> bool:
        """
        Skip trading if imbalance sign flipped vs a recent sample within the window
        (unstable book / fast regime change).
        """
        if not self._flip_block_enabled:
            return False
        eps = self._flip_min_abs
        if abs(imb) < eps:
            return False
        cutoff = now - self._flip_window
        for t, prev in reversed(self._imb_hist):
            if t < cutoff:
                break
            if abs(prev) < eps:
                continue
            if prev * imb < 0:
                return True
        return False

    def _weighted_side_size(self, book: LocalOrderBook, *, bids: bool) -> float:
        """Depth near touch counts more: weight level i by decay^i."""
        quotes = (
            book.top_bids()[: self._imbalance_levels]
            if bids
            else book.top_asks()[: self._imbalance_levels]
        )
        d = self._imbalance_decay
        total = 0.0
        w = 1.0
        for q in quotes:
            total += q.size * w
            w *= d
        return total

    def _imbalance(self, book: LocalOrderBook) -> Optional[float]:
        bv = self._weighted_side_size(book, bids=True)
        av = self._weighted_side_size(book, bids=False)
        if bv + av <= 0:
            return None
        return (bv - av) / (bv + av)

    def _effective_imbalance_threshold(self, position: float) -> float:
        thr = self._imbalance_thr
        if not self._inventory_skew_enabled:
            return thr
        thr += self._inventory_skew_per_contract * abs(position)
        if position > 0 and thr > 0:
            thr *= 1.0 + min(2.0, position / self._max_position)
        if position < 0 and thr > 0:
            thr *= 1.0 + min(2.0, abs(position) / self._max_position)
        return thr

    def _raw_signal(self, spread: float, effective_imb: float, thr: float) -> Signal:
        # Uses level + momentum (imb_eff); wide spread + flat eff → BOTH.
        if spread >= self._spread_wide_for_both and abs(effective_imb) <= thr:
            return Signal.BOTH
        if effective_imb > thr:
            return Signal.BUY
        if effective_imb < -thr:
            return Signal.SELL
        return Signal.NONE

    # --- sizing (inventory-aware) ---

    def _buy_size(self, position: float) -> float:
        if position <= 0:
            return self._base_order_size
        # Long: scale down BUY size; floor at 25% of base
        room = max(0.0, self._max_position - position)
        cap = min(self._base_order_size, room)
        scale = max(
            0.25,
            1.0 - self._inventory_size_scale * (position / self._max_position),
        )
        return max(0.0, min(cap, self._base_order_size * scale))

    def _sell_size(self, position: float) -> float:
        if position >= 0:
            return self._base_order_size
        scale = max(
            0.25,
            1.0 - self._inventory_size_scale * (abs(position) / self._max_position),
        )
        room = max(0.0, position + self._max_position)
        cap = min(self._base_order_size, room)
        return max(0.0, min(cap, self._base_order_size * scale))

    # --- placement ---

    def _strong_bullish_signal(self, imb_eff: float, thr: float) -> bool:
        """Strong BUY: effective imbalance meaningfully above threshold."""
        return imb_eff >= thr + self._strong_margin

    def _strong_bearish_signal(self, imb_eff: float, thr: float) -> bool:
        """Strong SELL: effective imbalance meaningfully below -threshold."""
        return imb_eff <= -thr - self._strong_margin

    def _buy_aggression_effective(self, signal: Signal, imb_eff: float, thr: float) -> float:
        p, mx = self._passive_agg, self._buy_agg
        if signal == Signal.BOTH:
            return p
        if signal != Signal.BUY:
            return p
        return mx if self._strong_bullish_signal(imb_eff, thr) else p

    def _sell_aggression_effective(self, signal: Signal, imb_eff: float, thr: float) -> float:
        p, mx = self._passive_agg, self._sell_agg
        if signal == Signal.BOTH:
            return p
        if signal != Signal.SELL:
            return p
        return mx if self._strong_bearish_signal(imb_eff, thr) else p

    def _price_buy(
        self, bb: float, ba: float, spread: float, aggression: float
    ) -> float:
        step = spread * aggression
        return min(bb + step, ba - 1e-9)

    def _price_sell(
        self, bb: float, ba: float, spread: float, aggression: float
    ) -> float:
        step = spread * aggression
        return max(ba - step, bb + 1e-9)

    def _build_intents(
        self,
        signal: Signal,
        bb: Quote,
        ba: Quote,
        spread: float,
        position: float,
        symbol: str,
        imb: float,
        thr: float,
    ) -> List[OrderIntent]:
        intents: List[OrderIntent] = []
        bbp, bap = bb.price, ba.price
        agg_b = self._buy_aggression_effective(signal, imb, thr)
        agg_s = self._sell_aggression_effective(signal, imb, thr)

        if signal == Signal.BUY:
            qty = self._buy_size(position)
            if qty <= 1e-12:
                return intents
            px = self._price_buy(bbp, bap, spread, agg_b)
            intents.append(OrderIntent(OrderSide.BUY, px, qty, symbol))

        elif signal == Signal.SELL:
            qty = self._sell_size(position)
            if qty <= 1e-12:
                return intents
            px = self._price_sell(bbp, bap, spread, agg_s)
            intents.append(OrderIntent(OrderSide.SELL, px, qty, symbol))

        elif signal == Signal.BOTH:
            qb = self._buy_size(position)
            qs = self._sell_size(position)
            if qb > 1e-12:
                intents.append(
                    OrderIntent(
                        OrderSide.BUY,
                        self._price_buy(bbp, bap, spread, agg_b),
                        qb,
                        symbol,
                    )
                )
            if qs > 1e-12:
                intents.append(
                    OrderIntent(
                        OrderSide.SELL,
                        self._price_sell(bbp, bap, spread, agg_s),
                        qs,
                        symbol,
                    )
                )

        return intents
