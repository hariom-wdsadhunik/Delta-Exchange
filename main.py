"""
Simulator main loop: market data -> book -> strategy -> execution -> PnL/risk -> stats.
"""

from __future__ import annotations

import argparse
import time
from queue import Empty, Queue
from typing import Optional

from config import load_config
from data_feed import (
    DeltaPublicFeed,
    MockOrderBookFeed,
    apply_queue_item_to_book,
)
from execution_engine import ExecutionEngine
from execution_types import OrderSide
from live_execution import LiveExecutionEngine
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


def _apply_shadow_fills(fills, shadow_pnl: PnLEngine) -> None:
    for fill in fills:
        cap = max(fill.quantity, 1e-12)
        if fill.side == OrderSide.BUY:
            shadow_pnl.apply_buy(fill.price, fill.quantity, cap)
        else:
            shadow_pnl.apply_sell(fill.price, fill.quantity, cap)


def _submit_intents(
    intents,
    execution,
    risk: RiskManager,
    position: float,
    now_mono: float,
    *,
    mode: str,
) -> int:
    submitted = 0
    for intent in intents:
        side = "buy" if intent.side == OrderSide.BUY else "sell"
        ok, reason = risk.can_open(side, intent.quantity, position, now_mono)
        if not ok:
            log_event("risk_block_submit", side=side, reason=reason, mode=mode)
            continue
        res = execution.submit_order(intent, now_mono, position)
        if res.accepted:
            submitted += 1
            log_event("order_submitted", order_id=res.order_id, side=side, mode=mode)
        else:
            log_event("execution_reject", reason=res.reason, side=side, mode=mode)
    return submitted


def _submit_intents_live(
    intents,
    *,
    live_execution: LiveExecutionEngine,
    shadow_execution: ExecutionEngine,
    risk: RiskManager,
    live_position: float,
    shadow_position: float,
    now_mono: float,
) -> tuple[int, int]:
    submitted_live = 0
    submitted_shadow = 0
    for intent in intents:
        side = "buy" if intent.side == OrderSide.BUY else "sell"
        ok, reason = risk.can_open(side, intent.quantity, live_position, now_mono)
        if not ok:
            log_event("risk_block_submit", side=side, reason=reason, mode="live")
            continue

        live_res = live_execution.submit_order(intent, now_mono, live_position)
        if not live_res.accepted:
            log_event("execution_reject", reason=live_res.reason, side=side, mode="live")
            continue

        submitted_live += 1
        log_event("order_submitted", order_id=live_res.order_id, side=side, mode="live")

        shadow_res = shadow_execution.submit_order(intent, now_mono, shadow_position)
        if shadow_res.accepted:
            submitted_shadow += 1
            log_event(
                "order_submitted",
                order_id=shadow_res.order_id,
                side=side,
                mode="shadow_sim",
                live_order_id=live_res.order_id,
            )
        else:
            log_event(
                "execution_reject",
                reason=shadow_res.reason,
                side=side,
                mode="shadow_sim",
                live_order_id=live_res.order_id,
            )
    return submitted_live, submitted_shadow


def _log_fill_comparison(real_fills, sim_fills) -> None:
    if not real_fills and not sim_fills:
        return

    real_qty = sum(f.quantity for f in real_fills)
    sim_qty = sum(f.quantity for f in sim_fills)
    real_notional = sum(f.quantity * f.price for f in real_fills)
    sim_notional = sum(f.quantity * f.price for f in sim_fills)
    real_vwap = (real_notional / real_qty) if real_qty > 1e-12 else 0.0
    sim_vwap = (sim_notional / sim_qty) if sim_qty > 1e-12 else 0.0

    log_event(
        "fill_compare",
        real_n=len(real_fills),
        sim_n=len(sim_fills),
        real_qty=real_qty,
        sim_qty=sim_qty,
        qty_delta=(real_qty - sim_qty),
        real_vwap=real_vwap,
        sim_vwap=sim_vwap,
        vwap_delta=(real_vwap - sim_vwap),
    )


def run(*, mock: bool, enable_live_trading: bool) -> None:
    setup_logging()
    log = get_logger()
    cfg = load_config()

    live_mode = bool(cfg.live_trading and enable_live_trading)
    if cfg.live_trading and not enable_live_trading:
        log.warning(
            "LIVE_TRADING is true in config, but --enable-live-trading was not provided. "
            "Running in SIMULATION MODE."
        )
    if enable_live_trading and not cfg.live_trading:
        log.warning(
            "--enable-live-trading provided, but LIVE_TRADING is false in config/env. "
            "Running in SIMULATION MODE."
        )

    book = LocalOrderBook(depth=cfg.orderbook_depth)
    pnl = PnLEngine(cfg)
    shadow_pnl: Optional[PnLEngine] = None
    risk = RiskManager(cfg)

    sim_execution = ExecutionEngine(
        cfg.max_position_contracts,
        order_timeout_sec=cfg.order_timeout_sec,
    )
    live_execution: Optional[LiveExecutionEngine] = None
    shadow_execution: Optional[ExecutionEngine] = None

    if live_mode:
        try:
            live_execution = LiveExecutionEngine(
                symbol=cfg.symbol,
                use_testnet=cfg.use_testnet,
            )
            shadow_execution = ExecutionEngine(
                cfg.max_position_contracts,
                order_timeout_sec=cfg.order_timeout_sec,
            )
            shadow_pnl = PnLEngine(cfg)
        except Exception as exc:
            log.error(
                "Live mode initialization failed (%s). Falling back to SIMULATION MODE.",
                exc,
            )
            live_mode = False

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

    if live_mode:
        log.warning("LIVE MODE")
        log.warning("Live REST orders enabled (testnet=%s).", cfg.use_testnet)
    else:
        log.info("SIMULATION MODE")
    log.info("Feed started (mock=%s) symbol=%s", mock, cfg.symbol)

    last_stats_mono = time.monotonic()
    loop_count = 0

    try:
        while True:
            t_mono = time.monotonic()
            _drain_book_queue(q, cfg, book)

            mid = book.mid()
            pos = pnl.state.position  # refresh
            shadow_pos = shadow_pnl.state.position if shadow_pnl is not None else pos

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
                if live_mode and live_execution is not None and shadow_execution is not None:
                    _submit_intents_live(
                        out.intents,
                        live_execution=live_execution,
                        shadow_execution=shadow_execution,
                        risk=risk,
                        live_position=pos,
                        shadow_position=shadow_pos,
                        now_mono=t_mono,
                    )
                else:
                    _submit_intents(
                        out.intents,
                        sim_execution,
                        risk,
                        pos,
                        t_mono,
                        mode="sim",
                    )

            if live_mode and live_execution is not None and shadow_execution is not None:
                real_fills = live_execution.process_tick(book, t_mono, pos)
                sim_fills = shadow_execution.process_tick(book, t_mono, shadow_pos)
                if real_fills:
                    log_event("fills_tick", n=len(real_fills), mode="live")
                if sim_fills:
                    log_event("fills_tick", n=len(sim_fills), mode="shadow_sim")
                _apply_fills(real_fills, pnl, risk, t_mono)
                if shadow_pnl is not None:
                    _apply_shadow_fills(sim_fills, shadow_pnl)
                _log_fill_comparison(real_fills, sim_fills)
            else:
                fills = sim_execution.process_tick(book, t_mono, pos)
                if fills:
                    log_event("fills_tick", n=len(fills), mode="sim")
                _apply_fills(fills, pnl, risk, t_mono)

            loop_count += 1
            if t_mono - last_stats_mono >= cfg.stats_interval_sec:
                last_stats_mono = t_mono
                mark = mid if mid is not None else 0.0
                ur = pnl.unrealized(mark) if mid is not None else 0.0
                mode_tag = "LIVE" if live_mode else "SIM"

                if live_mode and shadow_pnl is not None:
                    sur = shadow_pnl.unrealized(mark) if mid is not None else 0.0
                    log.info(
                        "STATS [%s] | real_balance=%.4f real_pos=%.4f real_trades=%s | "
                        "realized_pnl=%.4f unrealized_pnl=%.4f fees=%.4f | "
                        "sim_balance=%.4f sim_pos=%.4f sim_trades=%s | "
                        "sim_realized=%.4f sim_unrealized=%.4f | mid=%s loops=%s",
                        mode_tag,
                        pnl.state.balance,
                        pnl.state.position,
                        pnl.state.trade_count,
                        pnl.state.realized_pnl,
                        ur,
                        pnl.state.fees_paid,
                        shadow_pnl.state.balance,
                        shadow_pnl.state.position,
                        shadow_pnl.state.trade_count,
                        shadow_pnl.state.realized_pnl,
                        sur,
                        f"{mid:.4f}" if mid is not None else "n/a",
                        loop_count,
                    )
                    print(
                        f"[STATS {mode_tag}] real_bal={pnl.state.balance:.2f} "
                        f"real_pos={pnl.state.position:.4f} real_trades={pnl.state.trade_count} "
                        f"realized={pnl.state.realized_pnl:.4f} unrealized={ur:.4f} "
                        f"sim_bal={shadow_pnl.state.balance:.2f} "
                        f"sim_pos={shadow_pnl.state.position:.4f} sim_trades={shadow_pnl.state.trade_count} "
                        f"sim_realized={shadow_pnl.state.realized_pnl:.4f} sim_unrealized={sur:.4f} "
                        f"fees={pnl.state.fees_paid:.4f}",
                        flush=True,
                    )
                else:
                    log.info(
                        "STATS [%s] | balance=%.4f position=%.4f trades=%s | "
                        "realized_pnl=%.4f unrealized_pnl=%.4f fees=%.4f | mid=%s loops=%s",
                        mode_tag,
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
                        f"[STATS {mode_tag}] balance={pnl.state.balance:.2f} pos={pnl.state.position:.4f} "
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
    p.add_argument(
        "--enable-live-trading",
        action="store_true",
        help=(
            "Explicitly enable LIVE mode. LIVE_TRADING must also be true "
            "in config/env for live orders to be sent."
        ),
    )
    args = p.parse_args()
    run(mock=args.mock, enable_live_trading=args.enable_live_trading)


if __name__ == "__main__":
    main()
