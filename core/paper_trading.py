"""Append-only paper trading using the same risk path as future execution."""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import math
import uuid
from typing import Any

import pandas as pd

from core.config import PAPER_TRADING_ROOT
from core.storage import get_json, list_keys, put_json
from core.trading_controls import (
    PortfolioState, RiskPolicy, TradeCandidate, evaluate_trade,
)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _stamp(value: datetime) -> str:
    return value.strftime("%Y%m%dT%H%M%S%fZ")


def create_paper_signal(candidate: TradeCandidate, state: PortfolioState,
                        policy: RiskPolicy = RiskPolicy(),
                        model_id: str | None = None,
                        experiment_id: str | None = None) -> dict[str, Any]:
    created = utcnow()
    decision = evaluate_trade(candidate, state, policy)
    identifier = f"paper-{created:%Y%m%d}-{uuid.uuid4().hex[:16]}"
    return {
        "schema_version": "paper-v1", "signal_id": identifier,
        "created_at_utc": created.isoformat(), "mode": "paper",
        "model_id": model_id, "experiment_id": experiment_id,
        "candidate": candidate.__dict__, "portfolio_state": state.__dict__,
        "risk_policy": policy.__dict__, "risk_decision": decision.as_dict(),
        "execution_status": "OPEN" if decision.allowed else "REJECTED",
        "live_capital_used": False,
    }


def save_paper_signal(signal: dict[str, Any]) -> str:
    created = pd.to_datetime(signal.get("created_at_utc"), utc=True, errors="coerce")
    if pd.isna(created):
        raise ValueError("Paper-signaal mist een geldige created_at_utc.")
    signal_id = str(signal.get("signal_id") or "").strip()
    if not signal_id:
        raise ValueError("Paper-signaal mist signal_id.")
    key = (
        f"{PAPER_TRADING_ROOT}/signals/{created:%Y}/{created:%m}/{created:%d}/"
        f"{_stamp(created.to_pydatetime())}_{signal_id}.json"
    )
    put_json(key, signal)
    return key


def _find_signal(signal_id: str) -> tuple[str | None, dict[str, Any] | None]:
    for key in list_keys(f"{PAPER_TRADING_ROOT}/signals/"):
        if key.endswith(".json") and signal_id in key:
            value = get_json(key)
            if isinstance(value, dict) and value.get("signal_id") == signal_id:
                return key, value
    return None, None


def settle_paper_signal(signal_id: str, exit_price: float,
                        actual_cost: float = 0.0,
                        settled_at: datetime | None = None) -> tuple[str, dict[str, Any]]:
    source_key, signal = _find_signal(signal_id)
    if not signal:
        raise ValueError(f"Onbekend paper-signaal: {signal_id}")
    decision = signal.get("risk_decision") or {}
    if not decision.get("allowed"):
        raise ValueError("Een afgewezen NO_TRADE-signaal kan niet als trade worden afgerekend.")
    candidate = signal.get("candidate") or {}
    entry = float(candidate.get("option_price") or 0)
    if entry <= 0 or not math.isfinite(float(exit_price)) or float(exit_price) < 0:
        raise ValueError("Ongeldige entry- of exitprijs.")
    contracts = int(decision.get("contracts", 0) or 0)
    multiplier = int(candidate.get("contract_multiplier", 100) or 100)
    gross_pnl = (float(exit_price) - entry) * contracts * multiplier
    net_pnl = gross_pnl - max(0.0, float(actual_cost))
    actual_return = (float(exit_price) - entry) / entry
    actual_cost_return = max(0.0, float(actual_cost)) / max(entry * contracts * multiplier, 1e-12)
    settled = settled_at or utcnow()
    record = {
        "schema_version": "paper-settlement-v1", "signal_id": signal_id,
        "signal_key": source_key, "settled_at_utc": settled.isoformat(),
        "entry_price": entry, "exit_price": float(exit_price),
        "contracts": contracts, "contract_multiplier": multiplier,
        "gross_pnl": gross_pnl, "actual_cost": float(actual_cost), "net_pnl": net_pnl,
        "expected_net_return": float(decision.get("expected_edge_return", 0) or 0),
        "actual_net_return": actual_return - actual_cost_return,
        "actual_cost_return": actual_cost_return,
        "positive_outcome": bool(net_pnl > 0), "live_capital_used": False,
    }
    digest = sha256(f"{signal_id}|{settled.isoformat()}".encode()).hexdigest()[:12]
    key = (
        f"{PAPER_TRADING_ROOT}/settlements/{settled:%Y}/{settled:%m}/{settled:%d}/"
        f"{_stamp(settled)}_{signal_id}_{digest}.json"
    )
    put_json(key, record)
    return key, record


def paper_trading_status(limit: int = 1000) -> dict[str, Any]:
    signal_keys = sorted(key for key in list_keys(f"{PAPER_TRADING_ROOT}/signals/") if key.endswith(".json"))
    settlement_keys = sorted(key for key in list_keys(f"{PAPER_TRADING_ROOT}/settlements/") if key.endswith(".json"))
    settled = []
    errors = []
    for key in settlement_keys[-max(1, limit):]:
        try:
            value = get_json(key)
            if isinstance(value, dict):
                settled.append(value)
        except Exception as exc:
            errors.append(f"{key}: {type(exc).__name__}: {exc}")
    pnl = pd.to_numeric(pd.Series([x.get("net_pnl") for x in settled]), errors="coerce").dropna()
    ret = pd.to_numeric(pd.Series([x.get("actual_net_return") for x in settled]), errors="coerce").dropna()
    wins = pnl[pnl > 0]
    losses = pnl[pnl <= 0]
    return {
        "signals": len(signal_keys), "settled_trades": len(settlement_keys),
        "loaded_settlements": len(settled),
        "net_pnl": float(pnl.sum()) if len(pnl) else 0.0,
        "expected_value_per_trade": float(pnl.mean()) if len(pnl) else None,
        "win_rate": float((pnl > 0).mean()) if len(pnl) else None,
        "profit_factor": (
            float(wins.sum() / abs(losses.sum())) if len(losses) and abs(losses.sum()) > 0 else None
        ),
        "mean_net_return": float(ret.mean()) if len(ret) else None,
        "last_signal_key": signal_keys[-1] if signal_keys else None,
        "last_settlement_key": settlement_keys[-1] if settlement_keys else None,
        "errors": errors[-20:], "live_capital_used": False,
    }
