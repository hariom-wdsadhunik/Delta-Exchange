"""
Simulator main loop: market data → book → strategy → execution → PnL / risk → stats.
"""

from __future__ import annotations

import argparse
import time
from queue import Empty, Queue

from config import load_config
from data_feed import (
    DeltaPublicFeed,
    MockOrderBookFeed,
    apply_queue_item_to_book,
)
from execution_engine import ExecutionEngine
from execution_types import OrderSide
from logger import get_logger, log_event, setup_logging
from orderbook import LocalOrderBook
from pnl import PnLEngine
from risk import RiskManager
from strategy_engine import Signal, StrategyEngine


def _drain_book_queue(q: Queue, cfg, book: LocalOrderBook, max_burst: int = 200) -> int:
    n = 0
    for _ in range(max_burst):
        try:
            item = q.get_nowait()
        except Empty:
            break
        if apply_queue_item_to_book(cfg, item, book):
            n += 1
    return n


def _apply_fills(
    fills,
    pnl: PnLEngine,
    risk: RiskManager,
    now_mono: float,
) -> None:
    for fill in fills:
        cap = max(fill.quantity, 1e-12)
        if fill.side == OrderSide.BUY:
            pnl.apply_buy(fill.price, fill.quantity, cap)
        else:
            pnl.apply_sell(fill.price, fill.quantity, cap)
        risk.record_trade(now_mono)


def _submit_intents(
    intents,
    execution: ExecutionEngine,
    risk: RiskManager,
    position: float,
    now_mono: float,
) -> int:
    submitted = 0
    for intent in intents:
        side = "buy" if intent.side == OrderSide.BUY else "sell"
        ok, reason = risk.can_open(side, intent.quantity, position, now_mono)
        if not ok:
            log_event("risk_block_submit", side=side, reason=reason)
            continue
        res = execution.submit_order(intent, now_mono, position)
        if res.accepted:
            submitted += 1
            log_event("order_submitted", order_id=res.order_id, side=side)
        else:
            log_event("execution_reject", reason=res.reason, side=side)
    return submitted


def run(*, mock: bool) -> None:
    setup_logging()
    log = get_logger()
    cfg = load_config()

    book = LocalOrderBook(depth=cfg.orderbook_depth)
    pnl = PnLEngine(cfg)
    risk = RiskManager(cfg)
    execution = ExecutionEngine(
        cfg.max_position_contracts,
        order_timeout_sec=cfg.order_timeout_sec,
    )
    strategy = StrategyEngine(
        symbol=cfg.symbol,
        base_order_size=cfg.order_size_contracts,
        min_spread_abs=cfg.min_spread_abs,
        imbalance_levels=cfg.imbalance_levels,
        imbalance_threshold=cfg.imbalance_threshold,
        spread_wide_for_both=cfg.spread_wide_for_both,
        inventory_skew_enabled=cfg.inventory_skew_enabled,
        inventory_skew_per_contract=cfg.inventory_skew_per_contract,
        max_position=cfg.max_position_contracts,
        cooldown_sec=cfg.strategy_cooldown_sec,
        spread_median_spike_threshold=(
            cfg.max_spread_vs_median_ratio if cfg.volatility_filter_enabled else None
        ),
    )

    q: Queue = Queue()
    feed = MockOrderBookFeed(cfg, q) if mock else DeltaPublicFeed(cfg, q)
    feed.start()
    log.info("Feed started (mock=%s) symbol=%s", mock, cfg.symbol)

    last_stats_mono = time.monotonic()
    loop_count = 0

    try:
        while True:
            t_mono = time.monotonic()
            _drain_book_queue(q, cfg, book)

            mid = book.mid()
            pos = pnl.state.position

            out = strategy.evaluate(book, pos, t_mono, cfg.symbol)

            if out.intents and out.signal != Signal.NONE:
                log_event(
                    "strategy_emit",
                    signal=out.signal.value,
                    n_intents=len(out.intents),
                    imb=out.imbalance,
                    imb_delta=out.imbalance_delta,
                    imb_eff=out.effective_imbalance,
                )
                _submit_intents(out.intents, execution, risk, pos, t_mono)

            fills = execution.process_tick(book, t_mono, pos)
            if fills:
                log_event("fills_tick", n=len(fills))
            _apply_fills(fills, pnl, risk, t_mono)

            loop_count += 1
            if t_mono - last_stats_mono >= cfg.stats_interval_sec:
                last_stats_mono = t_mono
                mark = mid if mid is not None else 0.0
                ur = pnl.unrealized(mark) if mid is not None else 0.0
                log.info(
                    "STATS | balance=%.4f position=%.4f trades=%s | "
                    "realized_pnl=%.4f unrealized_pnl=%.4f fees=%.4f | mid=%s loops=%s",
                    pnl.state.balance,
                    pnl.state.position,
                    pnl.state.trade_count,
                    pnl.state.realized_pnl,
                    ur,
                    pnl.state.fees_paid,
                    f"{mid:.4f}" if mid is not None else "n/a",
                    loop_count,
                )
                print(
                    f"[STATS] balance={pnl.state.balance:.2f} pos={pnl.state.position:.4f} "
                    f"trades={pnl.state.trade_count} "
                    f"realized={pnl.state.realized_pnl:.4f} unrealized={ur:.4f} "
                    f"fees={pnl.state.fees_paid:.4f}",
                    flush=True,
                )

            time.sleep(0.02)
    except KeyboardInterrupt:
        log.info("Stopped by user.")
    finally:
        feed.stop()


def main() -> None:
    p = argparse.ArgumentParser(description="Delta demo trading simulator loop")
    p.add_argument(
        "--mock",
        action="store_true",
        help="Use synthetic order book feed (no WebSocket).",
    )
    args = p.parse_args()
    run(mock=args.mock)


if __name__ == "__main__":
    main()
