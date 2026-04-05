"""
Structured logging for signals, orders, fills, and PnL.
"""

from __future__ import annotations

import logging
import sys
from typing import Any, Mapping

_LOG: logging.Logger | None = None


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    global _LOG
    if _LOG is not None:
        return _LOG
    log = logging.getLogger("delta_hft_sim")
    log.setLevel(level)
    log.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    log.addHandler(sh)

    _LOG = log
    return log


def get_logger() -> logging.Logger:
    if _LOG is None:
        return setup_logging()
    return _LOG


def log_event(event: str, **fields: Any) -> None:
    """Single-line structured-ish log without external deps."""
    parts = [f"{k}={_fmt_val(v)}" for k, v in sorted(fields.items())]
    get_logger().info("%s | %s", event, " ".join(parts))


def _fmt_val(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.8g}"
    if isinstance(v, Mapping):
        return str(dict(v))
    return str(v)
