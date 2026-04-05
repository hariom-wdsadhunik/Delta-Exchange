"""
Delta Exchange DEMO — retail HFT-style limit-order simulator (market data + simulated execution).
"""

from __future__ import annotations

import argparse
import time
from queue import Empty, Queue

from config import load_config
from data_feed import DeltaPublicFeed, MockOrderBookFeed, apply_queue_item_to_book
from execution import ExecutionSimulator, plan_orders_for_signal
from logger import get_logger, log_event, setup_logging
from orderbook import LocalOrderBook
from pnl import PnLEngine
from risk import RiskManager
from strategy import Signal, StrategyEngine


def _drain_queue(q: Queue, cfg, book: LocalOrderBook, max_burst: int = 500) -> int:
    n = 0
    for _ in range(max_burst):
        try:
            item = q.get_nowait()
        except Empty:
            break
        if apply_queue_item_to_book(cfg, item, book):
            n += 1
    return n


def _sanity_checks(cfg, book: LocalOrderBook, pnl: PnLEngine, log) -> None:
    if abs(pnl.state.position) > cfg.max_position_contracts + 1e-6:
        log.error(
            "SANITY: position exceeds max |pos|=%s max=%s",
            pnl.state.position,
            cfg.max_position_contracts,
        )
    for i, cap in enumerate(pnl.state.fill_liquidity_caps[-50:]):
        if cap <= 0 and pnl.state.trade_count > 0:
            log.warning("SANITY: non-positive liquidity cap recorded")
    sp = book.spread()
    if sp is not None and sp < 0:
        log.error("SANITY: negative spread %s", sp)


def run(mock: bool) -> None:
    setup_logging()
    log = get_logger()
    cfg = load_config()

    if not mock and not cfg.api_key:
        log.warning(
            "DELTA_API_KEY not set — public market data does not require it; "
            "keys are loaded from .env for consistency and future private channels."
        )

    book = LocalOrderBook(depth=cfg.orderbook_depth)
    pnl = PnLEngine(cfg)
    risk = RiskManager(cfg)
    strategy = StrategyEngine(cfg)
    execution = ExecutionSimulator(cfg, pnl)

    q: Queue = Queue()

    if mock:
        feed = MockOrderBookFeed(cfg, q)
        feed.start()
        log.info("Mock data feed running (no WebSocket).")
    else:
        ws = DeltaPublicFeed(cfg, q)
        ws.start()
        log.info("Live WebSocket feed: %s symbol=%s", cfg.ws_public_url, cfg.symbol)

    last_signal_time = 0.0
    orders_submitted = 0
    last_stats = time.monotonic()
    updates_seen = 0
    last_logged_signal: Signal | None = None

    try:
        while True:
            _drain_queue(q, cfg, book)
            sp = book.spread() or 0.0
            sz = strategy.spread_zscore(sp)

            tc0 = pnl.state.trade_count
            execution.on_book_update(book, sz)
            d_trades = pnl.state.trade_count - tc0
            for _ in range(d_trades):
                risk.record_trade()

            mid = book.mid()
            if mid is None:
                time.sleep(0.05)
                continue

            sig, meta = strategy.evaluate(book, pnl.state.position)
            if sig != Signal.NONE and sig != last_logged_signal:
                log_event(
                    "signal",
                    signal=sig.value,
                    **{k: meta[k] for k in meta if k != "reason"},
                )
                last_logged_signal = sig
            elif sig == Signal.NONE:
                last_logged_signal = None

            now = time.monotonic()
            if (
                sig != Signal.NONE
                and now - last_signal_time >= cfg.strategy_cooldown_sec
                and execution.active_order_count() < 4
            ):
                for side, px, sz_ in plan_orders_for_signal(sig, book, cfg):
                    ok, reason = risk.can_open(side, sz_, pnl.state.position)
                    if not ok:
                        log_event("risk_block", side=side, reason=reason)
                        continue
                    execution.submit_limit(side, px, sz_)
                    orders_submitted += 1
                last_signal_time = now

            updates_seen += 1
            if now - last_stats >= cfg.stats_interval_sec:
                last_stats = now
                _sanity_checks(cfg, book, pnl, log)
                filled_orders = sum(
                    1 for o in execution.orders.values() if o.status.name == "FILLED"
                )
                fill_rate = (pnl.state.trade_count / orders_submitted) if orders_submitted else 0.0
                log.info(
                    "STATS | mid=%.4f spread=%.4f | pos=%.4f | trades=%s | "
                    "orders_submitted=%s fill_rate_trades_per_order=%.3f | "
                    "realized=%.4f unrealized=%.4f fees=%.4f balance=%.4f | "
                    "active_orders=%s book_updates=%s",
                    mid,
                    sp,
                    pnl.state.position,
                    pnl.state.trade_count,
                    orders_submitted,
                    fill_rate,
                    pnl.state.realized_pnl,
                    pnl.unrealized(mid),
                    pnl.state.fees_paid,
                    pnl.state.balance,
                    execution.active_order_count(),
                    updates_seen,
                )
                print(
                    f"[LIVE] pos={pnl.state.position:.4f} trades={pnl.state.trade_count} "
                    f"fill_rate={fill_rate:.3f} realized={pnl.state.realized_pnl:.4f} "
                    f"unrealized={pnl.unrealized(mid):.4f} fees={pnl.state.fees_paid:.4f}",
                    flush=True,
                )

            time.sleep(0.02)
    except KeyboardInterrupt:
        log.info("Interrupted — shutting down.")
    finally:
        if mock:
            feed.stop()
        else:
            ws.stop()


def main() -> None:
    p = argparse.ArgumentParser(description="Delta DEMO HFT-style simulator")
    p.add_argument(
        "--mock",
        action="store_true",
        help="Run with synthetic order book (no API connection).",
    )
    args = p.parse_args()
    run(mock=args.mock)


if __name__ == "__main__":
    main()
