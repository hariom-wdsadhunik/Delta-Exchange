"""
Live execution adapter for Delta Exchange REST API (optional mode).

This module is intentionally separate from simulation logic. It exposes:
- DeltaRestClient for signed REST requests
- LiveExecutionEngine with submit/process methods compatible with main loop usage
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List
from urllib import error, parse, request

from execution_engine import SubmitResult
from execution_types import FillEvent, OrderIntent, OrderSide
from logger import log_event


class DeltaApiError(RuntimeError):
    """Raised for API transport/auth/server errors."""


@dataclass(slots=True)
class _TrackedOrder:
    order_id: str
    side: OrderSide
    price: float
    size: float
    submitted_mono: float


class DeltaRestClient:
    """Minimal signed REST client for Delta Exchange."""

    PROD_BASE_URL = "https://api.india.delta.exchange"
    TESTNET_BASE_URL = "https://cdn-ind.testnet.deltaex.org"

    def __init__(
        self,
        *,
        symbol: str,
        use_testnet: bool = True,
        api_key: str | None = None,
        api_secret: str | None = None,
        timeout_sec: float = 10.0,
    ) -> None:
        self.symbol = symbol.upper()
        self.use_testnet = use_testnet
        self.base_url = self.TESTNET_BASE_URL if use_testnet else self.PROD_BASE_URL
        self.api_key = (api_key or os.getenv("API_KEY") or os.getenv("DELTA_API_KEY") or "").strip()
        self.api_secret = (
            api_secret or os.getenv("API_SECRET") or os.getenv("DELTA_API_SECRET") or ""
        ).strip()
        self.timeout_sec = timeout_sec
        self._product_id: int | None = None

        if not self.api_key or not self.api_secret:
            raise ValueError(
                "Missing API credentials. Set API_KEY/API_SECRET (or DELTA_API_KEY/DELTA_API_SECRET)."
            )

    # --- public API required by task ---

    def place_limit_order(self, side: str, price: float, size: float) -> Dict[str, Any]:
        """
        Place a live limit order.

        side: "buy" or "sell"
        """
        side_l = side.lower().strip()
        if side_l not in {"buy", "sell"}:
            raise ValueError(f"Unsupported side: {side}")
        if size <= 0:
            raise ValueError("size must be positive")
        if price <= 0:
            raise ValueError("price must be positive")

        payload = {
            "product_id": self._get_product_id(),
            "product_symbol": self.symbol,
            "order_type": "limit_order",
            "side": side_l,
            "size": size,
            "limit_price": str(price),
            "time_in_force": "gtc",
        }
        return self._request("POST", "/v2/orders", payload=payload)

    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        if not order_id:
            raise ValueError("order_id is required")
        payload = {"id": order_id}
        return self._request("DELETE", "/v2/orders", payload=payload)

    def get_open_orders(self) -> List[Dict[str, Any]]:
        params = {
            "product_ids": str(self._get_product_id()),
            "state": "open",
            "page_size": 50,
        }
        resp = self._request("GET", "/v2/orders", params=params)
        result = resp.get("result", [])
        if isinstance(result, list):
            return result
        return []

    # --- extra live polling helper ---

    def get_recent_fills(self, *, page_size: int = 50) -> List[Dict[str, Any]]:
        params = {
            "product_ids": str(self._get_product_id()),
            "page_size": max(1, min(50, page_size)),
        }
        resp = self._request("GET", "/v2/fills", params=params)
        result = resp.get("result", [])
        if isinstance(result, list):
            return result
        return []

    # --- internals ---

    def _get_product_id(self) -> int:
        if self._product_id is not None:
            return self._product_id
        resp = self._request(
            "GET",
            f"/v2/products/{parse.quote(self.symbol)}",
            auth_required=False,
        )
        result = resp.get("result", {})
        pid = result.get("id") if isinstance(result, dict) else None
        if pid is None:
            raise DeltaApiError(f"Could not resolve product id for symbol={self.symbol}")
        self._product_id = int(pid)
        return self._product_id

    def _headers(
        self,
        *,
        method: str,
        path: str,
        query_string: str,
        payload: str,
    ) -> Dict[str, str]:
        ts = str(int(time.time()))
        message = f"{method}{ts}{path}{query_string}{payload}"
        signature = hmac.new(
            self.api_secret.encode("utf-8"),
            message.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "delta-live-execution/1.0",
            "api-key": self.api_key,
            "timestamp": ts,
            "signature": signature,
        }

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: Dict[str, Any] | None = None,
        payload: Dict[str, Any] | None = None,
        auth_required: bool = True,
    ) -> Dict[str, Any]:
        query = parse.urlencode(params or {}, doseq=True)
        query_string = f"?{query}" if query else ""
        payload_json = json.dumps(payload, separators=(",", ":"), ensure_ascii=True) if payload else ""
        url = f"{self.base_url}{path}{query_string}"

        headers: Dict[str, str]
        if auth_required:
            headers = self._headers(
                method=method,
                path=path,
                query_string=query_string,
                payload=payload_json,
            )
        else:
            headers = {
                "Accept": "application/json",
                "Content-Type": "application/json",
                "User-Agent": "delta-live-execution/1.0",
            }

        req = request.Request(
            url=url,
            data=payload_json.encode("utf-8") if payload_json else None,
            headers=headers,
            method=method,
        )

        try:
            with request.urlopen(req, timeout=self.timeout_sec) as resp:
                body = resp.read().decode("utf-8")
                data = json.loads(body) if body else {}
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
            raise DeltaApiError(
                f"HTTP {exc.code} for {method} {path}: {body or exc.reason}"
            ) from exc
        except error.URLError as exc:
            raise DeltaApiError(f"Network error for {method} {path}: {exc.reason}") from exc
        except json.JSONDecodeError as exc:
            raise DeltaApiError(f"Invalid JSON from {method} {path}: {exc}") from exc

        if isinstance(data, dict) and data.get("success") is False:
            msg = data.get("error") or data.get("message") or data
            raise DeltaApiError(f"API rejected {method} {path}: {msg}")
        if not isinstance(data, dict):
            raise DeltaApiError(f"Unexpected response format for {method} {path}")
        return data


class LiveExecutionEngine:
    """
    Poll-based live execution adapter.

    - submit_order: sends live order
    - process_tick: polls fills and returns new FillEvent objects
    """

    def __init__(
        self,
        *,
        symbol: str,
        use_testnet: bool,
        poll_interval_sec: float = 0.5,
    ) -> None:
        self._client = DeltaRestClient(symbol=symbol, use_testnet=use_testnet)
        self._poll_interval_sec = max(0.2, poll_interval_sec)
        self._next_poll_mono = 0.0

        self._tracked_orders: Dict[str, _TrackedOrder] = {}
        self._seen_fill_ids: set[str] = set()
        self._last_open_order_refresh = 0.0

    def place_limit_order(self, side: str, price: float, size: float) -> Dict[str, Any]:
        return self._client.place_limit_order(side, price, size)

    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        return self._client.cancel_order(order_id)

    def get_open_orders(self) -> List[Dict[str, Any]]:
        return self._client.get_open_orders()

    def submit_order(self, intent: OrderIntent, now: float, position: float) -> SubmitResult:
        del position  # risk is handled by caller before submit
        side = "buy" if intent.side == OrderSide.BUY else "sell"
        try:
            resp = self.place_limit_order(side, intent.price, intent.quantity)
            result = resp.get("result", {}) if isinstance(resp, dict) else {}
            order_id = _extract_order_id(result)
            if not order_id:
                return SubmitResult(False, reason="missing_order_id_in_response")
            self._tracked_orders[order_id] = _TrackedOrder(
                order_id=order_id,
                side=intent.side,
                price=float(intent.price),
                size=float(intent.quantity),
                submitted_mono=now,
            )
            return SubmitResult(True, order_id=order_id)
        except Exception as exc:
            log_event("live_submit_error", reason=str(exc), side=side)
            return SubmitResult(False, reason=str(exc))

    def process_tick(self, book: Any, now: float, position: float) -> List[FillEvent]:
        del book, position  # live fills come from exchange, not local matching
        if now < self._next_poll_mono:
            return []
        self._next_poll_mono = now + self._poll_interval_sec

        events: List[FillEvent] = []
        try:
            fills = self._client.get_recent_fills(page_size=50)
            for fill in fills:
                event = self._fill_to_event(fill, now)
                if event is not None:
                    events.append(event)
            self._refresh_open_orders_if_due(now)
        except Exception as exc:
            log_event("live_poll_error", reason=str(exc))
        return events

    def _fill_to_event(self, fill: Dict[str, Any], now: float) -> FillEvent | None:
        fill_id = str(fill.get("id", "")).strip()
        order_id = str(fill.get("order_id", "")).strip()
        if not fill_id or not order_id:
            return None
        if fill_id in self._seen_fill_ids:
            return None
        if order_id not in self._tracked_orders:
            return None

        side_raw = str(fill.get("side", "")).lower().strip()
        if side_raw not in {"buy", "sell"}:
            return None

        try:
            price = float(fill.get("price"))
            qty = float(fill.get("size"))
        except (TypeError, ValueError):
            return None
        if qty <= 0:
            return None

        self._seen_fill_ids.add(fill_id)
        side = OrderSide.BUY if side_raw == "buy" else OrderSide.SELL
        is_maker = str(fill.get("role", "")).lower() == "maker"
        return FillEvent(
            order_id=order_id,
            side=side,
            price=price,
            quantity=qty,
            timestamp=now,
            is_maker=is_maker,
        )

    def _refresh_open_orders_if_due(self, now: float) -> None:
        if now - self._last_open_order_refresh < 2.0:
            return
        self._last_open_order_refresh = now
        try:
            open_orders = self.get_open_orders()
        except Exception as exc:
            log_event("live_open_orders_error", reason=str(exc))
            return

        open_ids = {str(_extract_order_id(o)) for o in open_orders if _extract_order_id(o)}
        for order_id in list(self._tracked_orders.keys()):
            if order_id not in open_ids:
                self._tracked_orders.pop(order_id, None)


def _extract_order_id(obj: Dict[str, Any]) -> str | None:
    # Delta order response fields vary across endpoints; accept common keys.
    for key in ("id", "order_id"):
        val = obj.get(key)
        if val is not None and str(val).strip():
            return str(val).strip()
    return None
