"""
Pre-trade risk: position limits and trade rate limiting.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Deque, Optional, Tuple

from config import Config


class RiskManager:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self._trade_times: Deque[float] = deque()

    def _prune(self, now: float) -> None:
        cutoff = now - 60.0
        while self._trade_times and self._trade_times[0] < cutoff:
            self._trade_times.popleft()

    def record_trade(self, now: Optional[float] = None) -> None:
        self._trade_times.append(now if now is not None else time.monotonic())

    def can_open(
        self,
        side: str,
        size: float,
        position: float,
        now: Optional[float] = None,
    ) -> Tuple[bool, str]:
        """
        side: 'buy' or 'sell' (intent to increase long or decrease / go short).
        """
        t = now if now is not None else time.monotonic()
        self._prune(t)
        if len(self._trade_times) >= self.cfg.max_trades_per_minute:
            return False, "max_trades_per_minute"

        if side == "buy":
            new_pos = position + size
            if new_pos > self.cfg.max_position_contracts + 1e-9:
                return False, "max_long_position"
        elif side == "sell":
            new_pos = position - size
            if new_pos < -self.cfg.max_position_contracts - 1e-9:
                return False, "max_short_position"
        else:
            return False, "unknown_side"
        return True, "ok"
