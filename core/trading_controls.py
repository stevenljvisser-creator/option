"""Independent decision, risk and kill-switch controls.

Signal generation never decides position size or execution.  This module is
shared by historical evaluation, paper trading and any future broker adapter so
that a strong forecast cannot bypass portfolio or data-quality limits.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import math
from typing import Any


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
        return number if math.isfinite(number) else default
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class RiskPolicy:
    max_portfolio_exposure_pct: float = 0.30
    max_ticker_exposure_pct: float = 0.07
    max_sector_exposure_pct: float = 0.18
    risk_per_trade_pct: float = 0.005
    max_daily_loss_pct: float = 0.02
    max_drawdown_pct: float = 0.12
    max_relative_spread: float = 0.12
    min_open_interest: int = 50
    min_daily_contract_volume: int = 20
    max_model_uncertainty: float = 0.35
    max_pairwise_correlation: float = 0.85
    stale_data_minutes: float = 20.0
    safety_margin_return: float = 0.005


@dataclass
class PortfolioState:
    equity: float
    gross_exposure: float = 0.0
    ticker_exposure: float = 0.0
    sector_exposure: float = 0.0
    daily_pnl: float = 0.0
    drawdown_pct: float = 0.0
    correlated_exposure_pct: float = 0.0


@dataclass
class TradeCandidate:
    ticker: str
    side: str
    expected_return: float
    expected_cost_return: float
    uncertainty: float
    option_price: float
    contract_multiplier: int = 100
    relative_spread: float | None = None
    open_interest: int | None = None
    daily_volume: int | None = None
    data_age_minutes: float = 0.0
    feature_complete: bool = True
    leakage_valid: bool = True
    model_valid: bool = True
    system_healthy: bool = True
    drift_alert: bool = False
    event_id: str | None = None


@dataclass
class RiskDecision:
    decision: str
    allowed: bool
    contracts: int
    expected_edge_return: float
    required_edge_return: float
    reasons: list[str] = field(default_factory=list)
    kill_switch: bool = False
    evaluated_at_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def kill_switch_reasons(candidate: TradeCandidate, state: PortfolioState,
                        policy: RiskPolicy = RiskPolicy()) -> list[str]:
    reasons = []
    if not candidate.system_healthy:
        reasons.append("systeemfout")
    if not candidate.feature_complete:
        reasons.append("onvolledige_features")
    if not candidate.leakage_valid:
        reasons.append("leakage_audit_mislukt")
    if not candidate.model_valid:
        reasons.append("model_ongeldig")
    if candidate.drift_alert:
        reasons.append("model_of_feature_drift")
    if _finite(state.daily_pnl) <= -abs(_finite(state.equity)) * policy.max_daily_loss_pct:
        reasons.append("maximaal_dagverlies")
    if _finite(state.drawdown_pct) >= policy.max_drawdown_pct:
        reasons.append("maximale_drawdown")
    if _finite(candidate.data_age_minutes) > policy.stale_data_minutes:
        reasons.append("verouderde_data")
    return reasons


def evaluate_trade(candidate: TradeCandidate, state: PortfolioState,
                   policy: RiskPolicy = RiskPolicy()) -> RiskDecision:
    """Return TRADE or NO_TRADE and a separately constrained position size."""
    hard = kill_switch_reasons(candidate, state, policy)
    edge = _finite(candidate.expected_return) - max(0.0, _finite(candidate.expected_cost_return))
    uncertainty_penalty = max(0.0, _finite(candidate.uncertainty)) * 0.25
    required = policy.safety_margin_return + uncertainty_penalty
    reasons = list(hard)

    spread = candidate.relative_spread
    if spread is None:
        reasons.append("spread_onbekend")
    elif _finite(spread, math.inf) > policy.max_relative_spread:
        reasons.append("spread_te_hoog")
    if candidate.open_interest is None or candidate.open_interest < policy.min_open_interest:
        reasons.append("onvoldoende_open_interest")
    if candidate.daily_volume is None or candidate.daily_volume < policy.min_daily_contract_volume:
        reasons.append("onvoldoende_liquiditeit")
    if _finite(candidate.uncertainty, math.inf) > policy.max_model_uncertainty:
        reasons.append("modelonzekerheid_te_hoog")
    if edge <= required:
        reasons.append("edge_niet_boven_kosten_en_marge")
    if state.equity <= 0:
        reasons.append("ongeldig_portfoliovermogen")
    if state.gross_exposure >= state.equity * policy.max_portfolio_exposure_pct:
        reasons.append("maximale_portefeuille_exposure")
    if state.ticker_exposure >= state.equity * policy.max_ticker_exposure_pct:
        reasons.append("tickerconcentratie")
    if state.sector_exposure >= state.equity * policy.max_sector_exposure_pct:
        reasons.append("sectorconcentratie")
    if state.correlated_exposure_pct >= policy.max_pairwise_correlation:
        reasons.append("correlatieconcentratie")

    if reasons:
        return RiskDecision(
            decision="NO_TRADE", allowed=False, contracts=0,
            expected_edge_return=edge, required_edge_return=required,
            reasons=sorted(set(reasons)), kill_switch=bool(hard),
        )

    price = max(0.01, _finite(candidate.option_price, 0.01))
    contract_notional = price * max(1, int(candidate.contract_multiplier))
    risk_budget = state.equity * policy.risk_per_trade_pct
    room = min(
        state.equity * policy.max_portfolio_exposure_pct - state.gross_exposure,
        state.equity * policy.max_ticker_exposure_pct - state.ticker_exposure,
        state.equity * policy.max_sector_exposure_pct - state.sector_exposure,
    )
    contracts = int(max(0.0, min(risk_budget, room)) // contract_notional)
    if contracts < 1:
        return RiskDecision(
            decision="NO_TRADE", allowed=False, contracts=0,
            expected_edge_return=edge, required_edge_return=required,
            reasons=["positie_binnen_risicolimieten_te_klein"],
        )
    return RiskDecision(
        decision="TRADE", allowed=True, contracts=contracts,
        expected_edge_return=edge, required_edge_return=required,
    )


def live_promotion_gate(experiment: dict[str, Any] | None,
                        paper: dict[str, Any] | None,
                        data_healthy: bool,
                        broker_adapter_configured: bool = False) -> dict[str, Any]:
    """Hard, fail-closed gate.  OptionEdge ships with live execution disabled."""
    exp = experiment or {}
    metrics = exp.get("holdout_metrics") or exp.get("metrics") or {}
    paper = paper or {}
    checks = {
        "leakage_audit_valid": bool(exp.get("leakage_valid")),
        "positive_out_of_sample_ev": _finite(metrics.get("expected_value_per_trade"), -math.inf) > 0,
        "positive_after_costs": _finite(metrics.get("net_return"), -math.inf) > 0,
        "acceptable_drawdown": _finite(metrics.get("max_drawdown"), math.inf) <= 0.15,
        "sufficient_trades": int(metrics.get("number_of_trades", 0) or 0) >= 100,
        "multiple_periods": int(exp.get("profitable_test_periods", 0) or 0) >= 3,
        "paper_trading_positive": _finite(paper.get("expected_value_per_trade"), -math.inf) > 0,
        "paper_minimum_trades": int(paper.get("settled_trades", 0) or 0) >= 50,
        "data_stream_healthy": bool(data_healthy),
        "costs_not_provisional": not bool(metrics.get("costs_provisional", True)),
        "broker_adapter_configured": bool(broker_adapter_configured),
    }
    passed = all(checks.values())
    return {
        "eligible": passed,
        "live_execution_enabled": False,
        "checks": checks,
        "blockers": [name for name, ok in checks.items() if not ok],
        "note": (
            "Zelfs na alle onderzoeks- en paper-gates blijft live uitvoering uit totdat een "
            "afzonderlijke brokeradapter, expliciete kapitaallimieten en handmatige autorisatie bestaan."
        ),
    }


def default_safety_status() -> dict[str, Any]:
    gate = live_promotion_gate(None, None, data_healthy=False, broker_adapter_configured=False)
    return {
        "architecture": "data -> features -> model -> expected edge -> risk manager -> execution",
        "kill_switch_fail_closed": True,
        "paper_trading_available": True,
        "live": gate,
        "policy": asdict(RiskPolicy()),
    }
