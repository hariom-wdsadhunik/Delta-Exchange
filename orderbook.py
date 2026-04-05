"""
Local L2 order book: maintain top-N bids/asks with efficient updates and best-quote access.
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple


def _f(x: str | float | int) -> float:
    return float(x)


@dataclass
class Quote:
    price: float
    size: float


class LocalOrderBook:
    """
    Bids: sorted descending by price. Asks: sorted ascending by price.
    Internal storage as sorted price lists + price->size map for O(log n) updates.
    """

    def __init__(self, depth: int = 10) -> None:
        self.depth = depth
        self._bid_prices: List[float] = []  # descending
        self._ask_prices: List[float] = []  # ascending
        self._bid_sz: dict[float, float] = {}
        self._ask_sz: dict[float, float] = {}
        self.symbol: str = ""
        self.last_ts: Optional[int] = None

    def clear(self) -> None:
        self._bid_prices.clear()
        self._ask_prices.clear()
        self._bid_sz.clear()
        self._ask_sz.clear()

    def set_symbol(self, symbol: str) -> None:
        self.symbol = symbol

    def apply_snapshot_pairs(
        self,
        bids: Iterable[Tuple[str | float, str | float | int]],
        asks: Iterable[Tuple[str | float, str | float | int]],
        ts: Optional[int] = None,
    ) -> None:
        """Replace book from L2 snapshot lists [[price, size], ...]."""
        self.clear()
        for p, s in bids:
            self._set_bid(_f(p), max(0.0, float(s)))
        for p, s in asks:
            self._set_ask(_f(p), max(0.0, float(s)))
        self._trim()
        if ts is not None:
            self.last_ts = ts

    def _set_bid(self, price: float, size: float) -> None:
        if size <= 0:
            self._remove_bid(price)
            return
        if price not in self._bid_sz:
            idx = bisect_left(self._bid_prices, -price)
            self._bid_prices.insert(idx, -price)
        self._bid_sz[price] = size

    def _set_ask(self, price: float, size: float) -> None:
        if size <= 0:
            self._remove_ask(price)
            return
        if price not in self._ask_sz:
            idx = bisect_left(self._ask_prices, price)
            self._ask_prices.insert(idx, price)
        self._ask_sz[price] = size

    def _remove_bid(self, price: float) -> None:
        if price not in self._bid_sz:
            return
        del self._bid_sz[price]
        ip = bisect_left(self._bid_prices, -price)
        if ip < len(self._bid_prices) and self._bid_prices[ip] == -price:
            del self._bid_prices[ip]

    def _remove_ask(self, price: float) -> None:
        if price not in self._ask_sz:
            return
        del self._ask_sz[price]
        ip = bisect_left(self._ask_prices, price)
        if ip < len(self._ask_prices) and self._ask_prices[ip] == price:
            del self._ask_prices[ip]

    def _trim(self) -> None:
        while len(self._bid_prices) > self.depth:
            worst = -self._bid_prices[-1]
            self._remove_bid(worst)
        while len(self._ask_prices) > self.depth:
            worst = self._ask_prices[-1]
            self._remove_ask(worst)

    def top_bids(self) -> List[Quote]:
        out: List[Quote] = []
        for neg_p in self._bid_prices[: self.depth]:
            p = -neg_p
            out.append(Quote(p, self._bid_sz[p]))
        return out

    def top_asks(self) -> List[Quote]:
        out: List[Quote] = []
        for p in self._ask_prices[: self.depth]:
            out.append(Quote(p, self._ask_sz[p]))
        return out

    def best_bid(self) -> Optional[Quote]:
        b = self.top_bids()
        return b[0] if b else None

    def best_ask(self) -> Optional[Quote]:
        a = self.top_asks()
        return a[0] if a else None

    def mid(self) -> Optional[float]:
        bb, ba = self.best_bid(), self.best_ask()
        if bb and ba:
            return (bb.price + ba.price) / 2.0
        return None

    def spread(self) -> Optional[float]:
        bb, ba = self.best_bid(), self.best_ask()
        if bb and ba:
            return ba.price - bb.price
        return None

    def bid_size_at(self, price: float) -> float:
        return self._bid_sz.get(price, 0.0)

    def ask_size_at(self, price: float) -> float:
        return self._ask_sz.get(price, 0.0)

    def cumulative_bid_size(self, levels: int) -> float:
        return sum(q.size for q in self.top_bids()[:levels])

    def cumulative_ask_size(self, levels: int) -> float:
        return sum(q.size for q in self.top_asks()[:levels])

    def liquidity_buy(self, max_price: float) -> List[Tuple[float, float]]:
        """Asks at or below max_price, ascending (for aggressive buy fill)."""
        out: List[Tuple[float, float]] = []
        for q in self.top_asks():
            if q.price <= max_price:
                out.append((q.price, q.size))
        return out

    def liquidity_sell(self, min_price: float) -> List[Tuple[float, float]]:
        """Bids at or above min_price, descending (for aggressive sell fill)."""
        out: List[Tuple[float, float]] = []
        for q in self.top_bids():
            if q.price >= min_price:
                out.append((q.price, q.size))
        return out
