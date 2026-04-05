"""
Central configuration: trading / simulation parameters and environment-backed secrets.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    # Connectivity (DEMO testnet defaults per Delta docs)
    ws_public_url: str = "wss://socket-ind-pub.testnet.deltaex.org"
    symbol: str = "BTCUSD"
    orderbook_channel: str = "ob_l2"  # top ~15 levels; use l2_orderbook on legacy if needed

    # API credentials (required for future private channels; loaded from env only)
    api_key: str = field(default_factory=lambda: os.getenv("DELTA_API_KEY", ""))
    api_secret: str = field(default_factory=lambda: os.getenv("DELTA_API_SECRET", ""))

    # Order book depth kept locally
    orderbook_depth: int = 10

    # Strategy — spread & imbalance
    min_spread_abs: float = 0.5  # minimum absolute spread (same units as price)
    imbalance_levels: int = 5  # levels per side for volume imbalance
    imbalance_threshold: float = 0.15  # |imbalance| must exceed this for directional signal
    spread_wide_for_both: float = 2.0  # if spread >= this, allow BOTH (quote both sides)

    # Execution simulation
    order_size_contracts: float = 1.0
    latency_ms: float = 50.0
    order_timeout_sec: float = 2.5
    # When bid/ask touch our resting price, fraction of visible size that may fill per tick
    passive_fill_touch_fraction: float = 0.08
    # Optional: scale passive fills down when volatility (spread change) is high
    volatility_spread_lookback: int = 20
    max_spread_z_for_passive: float = 3.0  # above this z-score, skip passive fills

    # Fees (simple per-notional model; tune to your product)
    fee_rate_per_side: float = 0.0005  # 5 bps per fill side (maker+taker blended approx)

    # Risk
    max_position_contracts: float = 5.0
    max_trades_per_minute: int = 30

    # PnL
    starting_balance_quote: float = 100_000.0  # notional accounting currency units

    # Main loop
    stats_interval_sec: float = 5.0
    strategy_cooldown_sec: float = 0.25  # min time between new order submissions

    # Phase 5 — inventory skew (widens threshold if carrying inventory)
    inventory_skew_enabled: bool = True
    inventory_skew_per_contract: float = 0.04  # added to imbalance threshold per |position|

    # Phase 5 — volatility filter (skip new orders if spread spikes)
    volatility_filter_enabled: bool = True
    max_spread_vs_median_ratio: float = 2.5

    # Phase 5 — fill probability (soft cap on aggressive consumption)
    max_fill_fraction_of_visible_level: float = 0.95

    # Testing / mock
    mock_tick_interval_sec: float = 0.15


def load_config() -> Config:
    """Build config from defaults + environment overrides."""
    c = Config()
    if v := os.getenv("DELTA_WS_PUBLIC_URL"):
        c.ws_public_url = v.strip()
    if v := os.getenv("DELTA_SYMBOL"):
        c.symbol = v.strip().upper()
    if v := os.getenv("DELTA_ORDERBOOK_CHANNEL"):
        c.orderbook_channel = v.strip()
    if v := os.getenv("MIN_SPREAD_ABS"):
        c.min_spread_abs = float(v)
    if v := os.getenv("IMBALANCE_THRESHOLD"):
        c.imbalance_threshold = float(v)
    if v := os.getenv("ORDER_SIZE_CONTRACTS"):
        c.order_size_contracts = float(v)
    if v := os.getenv("LATENCY_MS"):
        c.latency_ms = float(v)
    if v := os.getenv("ORDER_TIMEOUT_SEC"):
        c.order_timeout_sec = float(v)
    if v := os.getenv("FEE_RATE_PER_SIDE"):
        c.fee_rate_per_side = float(v)
    if v := os.getenv("MAX_POSITION_CONTRACTS"):
        c.max_position_contracts = float(v)
    if v := os.getenv("MAX_TRADES_PER_MINUTE"):
        c.max_trades_per_minute = int(v)
    return c
