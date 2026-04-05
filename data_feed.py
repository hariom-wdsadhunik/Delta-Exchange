"""
WebSocket market data: Delta Exchange public stream (DEMO testnet) + optional mock feed.
"""

from __future__ import annotations

import json
import random
import threading
import time
from queue import Queue
from typing import Any, Dict, List, Optional

import websocket

from config import Config
from logger import get_logger
from orderbook import LocalOrderBook

log = get_logger()


class DeltaPublicFeed:
    """
    Connects to Delta public WebSocket, subscribes to ob_l2 (or l2_orderbook), enqueues snapshots.
    """

    def __init__(self, cfg: Config, out_queue: "Queue[dict]") -> None:
        self.cfg = cfg
        self.out_queue = out_queue
        self._ws: Optional[websocket.WebSocketApp] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self.connected = threading.Event()

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="delta-ws", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._ws:
            try:
                self._ws.close()
            except Exception:
                pass
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)

    def _subscribe_payload(self) -> dict:
        return {
            "type": "subscribe",
            "payload": {
                "channels": [
                    {
                        "name": self.cfg.orderbook_channel,
                        "symbols": [self.cfg.symbol],
                    }
                ]
            },
        }

    def _parse_message(self, raw: str) -> None:
        try:
            msg: Dict[str, Any] = json.loads(raw)
        except json.JSONDecodeError:
            return
        mtype = msg.get("type")
        if mtype == "subscriptions":
            log.info("subscriptions ack: %s", msg)
            return
        if mtype == self.cfg.orderbook_channel:
            self._handle_ob_l2(msg)
            return
        # Legacy snapshot shape
        if mtype == "l2_orderbook":
            self._handle_l2_legacy(msg)

    def _handle_ob_l2(self, msg: Dict[str, Any]) -> None:
        self.out_queue.put(dict(msg))

    def _handle_l2_legacy(self, msg: Dict[str, Any]) -> None:
        # Normalize so downstream sees uniform shape
        sym = msg.get("symbol", self.cfg.symbol)
        bids_raw = msg.get("buy") or []
        asks_raw = msg.get("sell") or []
        bids = [[str(x["limit_price"]), str(x["size"])] for x in bids_raw]
        asks = [[str(x["limit_price"]), str(x["size"])] for x in asks_raw]
        ts = msg.get("timestamp")
        self.out_queue.put(
            {
                "type": self.cfg.orderbook_channel,
                "sy": sym,
                "b": bids,
                "a": asks,
                "ts": int(ts) if ts is not None else None,
            }
        )

    def _on_open(self, ws: websocket.WebSocketApp) -> None:
        self.connected.set()
        ws.send(json.dumps(self._subscribe_payload()))
        log.info("WebSocket open; subscribed %s %s", self.cfg.orderbook_channel, self.cfg.symbol)

    def _run(self) -> None:
        while not self._stop.is_set():
            self.connected.clear()

            def on_message(_ws: Any, message: str) -> None:
                self._parse_message(message)

            def on_error(_ws: Any, err: Any) -> None:
                log.warning("WebSocket error: %s", err)

            def on_close(_ws: Any, *_a: Any) -> None:
                log.info("WebSocket closed")
                self.connected.clear()

            self._ws = websocket.WebSocketApp(
                self.cfg.ws_public_url,
                on_open=self._on_open,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close,
            )
            try:
                self._ws.run_forever(ping_interval=25, ping_timeout=10)
            except Exception as e:
                log.warning("run_forever exception: %s", e)
            if not self._stop.is_set():
                time.sleep(2.0)


class MockOrderBookFeed:
    """
    Generates synthetic ob_l2-style updates for offline testing (no API).
    """

    def __init__(self, cfg: Config, out_queue: "Queue[dict]") -> None:
        self.cfg = cfg
        self.out_queue = out_queue
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._mid = 95_000.0 + random.random() * 500.0

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="mock-feed", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        tick = 0
        while not self._stop.is_set():
            # Random walk mid; build fictitious ladder
            self._mid += random.gauss(0, 1.5)
            spread = max(0.5, random.gauss(2.0, 0.4))
            bb = self._mid - spread / 2
            ba = self._mid + spread / 2
            step = 0.5
            bids: List[List[str]] = []
            asks: List[List[str]] = []
            for i in range(15):
                bp = bb - i * step
                ap = ba + i * step
                bids.append([f"{bp:.1f}", str(int(50 + random.random() * 200))])
                asks.append([f"{ap:.1f}", str(int(50 + random.random() * 200))])
            # Occasionally skew one side for imbalance signals
            if tick % 40 == 0:
                for i in range(3):
                    bids[i][1] = str(int(bids[i][1]) + 800)
            if tick % 40 == 20:
                for i in range(3):
                    asks[i][1] = str(int(asks[i][1]) + 800)
            tick += 1
            ts = int(time.time() * 1_000_000)
            self.out_queue.put(
                {
                    "type": self.cfg.orderbook_channel,
                    "sy": self.cfg.symbol,
                    "b": bids,
                    "a": asks,
                    "ts": ts,
                }
            )
            time.sleep(self.cfg.mock_tick_interval_sec)


def apply_queue_item_to_book(cfg: Config, item: dict, book: LocalOrderBook) -> bool:
    """Apply one WebSocket / mock message to the local book. Returns True if applied."""
    if not isinstance(item, dict):
        return False
    if item.get("type") != cfg.orderbook_channel:
        return False
    bids = item.get("b") or []
    asks = item.get("a") or []
    ts = item.get("ts")
    book.set_symbol(str(item.get("sy", cfg.symbol)))
    book.apply_snapshot_pairs(bids, asks, int(ts) if ts is not None else None)
    return True
