"""Alpha-first, cost-aware and leakage-audited research engine.

The engine deliberately starts with a historical-mean baseline and Ridge.  It
uses expanding purged walk-forward tests, reserves an untouched final holdout,
stores NO_TRADE decisions, and evaluates economic outcomes after costs.  More
complex estimators can be added behind the same contract only when they improve
out-of-sample economic performance.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from hashlib import sha256
import math
import re
from typing import Any, Iterable
import uuid

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from core.config import EXPERIMENT_ROOT
from core.leakage import (
    audit_feature_columns, audit_point_in_time, audit_purged_split,
    merge_audits, require_valid,
)
from core.storage import get_json, list_keys, put_df, put_json


@dataclass(frozen=True)
class WalkForwardConfig:
    min_train_sessions: int = 80
    test_sessions: int = 20
    purge_sessions: int = 5
    embargo_sessions: int = 1
    n_splits: int = 5
    holdout_fraction: float = 0.15
    minimum_holdout_sessions: int = 20


@dataclass(frozen=True)
class CostModel:
    commission_per_contract_side: float = 0.65
    regulatory_fee_per_contract_side: float = 0.03
    slippage_bps_per_side: float = 5.0
    market_impact_bps_per_side: float = 2.0
    adverse_fill_bps_per_side: float = 2.0
    fallback_roundtrip_spread_pct: float = 0.08
    minimum_tick: float = 0.01
    contract_multiplier: int = 100
    safety_margin_return: float = 0.005


SECTOR_MAP={
    "NVDA":"Technology","AMD":"Technology","AVGO":"Technology","AAPL":"Technology",
    "MSFT":"Technology","GOOGL":"Communication","META":"Communication",
    "AMZN":"Consumer Discretionary","TSLA":"Consumer Discretionary",
    "JPM":"Financials","BAC":"Financials","GS":"Financials","LLY":"Health Care",
    "UNH":"Health Care","XOM":"Energy","CVX":"Energy","CAT":"Industrials",
    "BA":"Industrials","WMT":"Consumer Staples","COST":"Consumer Staples",
    "SPY":"Broad Market","QQQ":"Broad Market",
}


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def prepare_trade_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Select one liquid near-ATM contract per ticker-day to avoid pseudo-replication."""
    if frame is None or frame.empty:
        return pd.DataFrame()
    data = frame.copy()
    required = {"source_ticker", "source_day", "minute", "strike", "close_stock"}
    missing = sorted(required - set(data))
    if missing:
        raise ValueError("Trade research mist kolommen: " + ", ".join(missing))
    data["minute"] = pd.to_datetime(data["minute"], utc=True, errors="coerce")
    data["target_minute"] = pd.to_datetime(data.get("target_minute"), utc=True, errors="coerce")
    strike = pd.to_numeric(data["strike"], errors="coerce")
    spot = pd.to_numeric(data["close_stock"], errors="coerce")
    data["_moneyness_distance"] = (strike / spot.replace(0, np.nan) - 1).abs()
    if "dte" in data:
        data["_dte_rank"] = pd.to_numeric(data["dte"], errors="coerce").clip(lower=0)
    else:
        expiry = pd.to_datetime(data.get("expiry"), utc=True, errors="coerce")
        data["_dte_rank"] = (expiry - data["minute"]).dt.total_seconds() / 86400
    data = data.dropna(subset=["minute", "_moneyness_distance", "_dte_rank"])
    data = data[data["_dte_rank"] > 0]
    sort_cols = ["source_ticker", "source_day", "_moneyness_distance", "_dte_rank"]
    return (
        data.sort_values(sort_cols)
        .groupby(["source_ticker", "source_day"], as_index=False, sort=False)
        .head(1).drop(columns=["_moneyness_distance", "_dte_rank"])
        .reset_index(drop=True)
    )


def assign_point_in_time_regimes(frame: pd.DataFrame) -> pd.DataFrame:
    """Add interpretable regimes using only features already available at t."""
    if frame is None or frame.empty:return frame
    data=frame.copy().sort_values("minute").reset_index(drop=True)
    def numeric(column: str) -> pd.Series:
        if column not in data:return pd.Series(np.nan,index=data.index,dtype=float)
        return pd.to_numeric(data[column],errors="coerce")
    ticker=data.get("source_ticker",pd.Series("UNKNOWN",index=data.index)).astype(str).str.upper()
    data["sector"]=ticker.map(SECTOR_MAP).fillna("Unknown")
    trend_column="sma_30_ratio" if "sma_30_ratio" in data else "stock_return_60m"
    trend=numeric(trend_column)
    data["market_regime"]=np.select(
        [trend>=.01,trend<=-.01],["bull","bear"],default="sideways"
    )
    volatility=numeric("realized_vol_60m")
    if volatility.isna().all():volatility=numeric("realized_vol_30m")
    prior_low=volatility.expanding(min_periods=30).quantile(.33).shift(1)
    prior_high=volatility.expanding(min_periods=30).quantile(.67).shift(1)
    fallback_low,fallback_high=.30,.70
    low=prior_low.fillna(fallback_low);high=prior_high.fillna(fallback_high)
    data["volatility_regime"]=np.select(
        [volatility<=low,volatility>=high],["low","high"],default="medium"
    )
    earnings=numeric("earnings_event_today").fillna(0)>0
    news=numeric("news_count_60m").fillna(0)>0
    data["event_type"]=np.select([earnings,news],["earnings","news"],default="none")
    return data


def _session_values(frame: pd.DataFrame, time_column: str) -> list[date]:
    values = pd.to_datetime(frame[time_column], errors="coerce").dt.date
    return sorted(value for value in pd.unique(values.dropna()))


def reserve_untouched_holdout(frame: pd.DataFrame, config: WalkForwardConfig,
                              time_column: str = "source_day") -> tuple[pd.DataFrame, pd.DataFrame]:
    sessions = _session_values(frame, time_column)
    wanted = max(config.minimum_holdout_sessions, int(math.ceil(len(sessions) * config.holdout_fraction)))
    if len(sessions) < config.min_train_sessions + wanted + config.purge_sessions + 1:
        raise ValueError(
            f"Onvoldoende sessies voor train plus untouched holdout: {len(sessions)} gevonden."
        )
    holdout_days = set(sessions[-wanted:])
    boundary = len(sessions) - wanted
    research_days = set(sessions[:max(0, boundary - config.purge_sessions)])
    row_days=pd.to_datetime(frame[time_column],errors="coerce").dt.date
    return frame[row_days.isin(research_days)].copy(), frame[row_days.isin(holdout_days)].copy()


def purged_walk_forward_splits(frame: pd.DataFrame, config: WalkForwardConfig,
                               time_column: str = "source_day") -> list[dict[str, Any]]:
    sessions = _session_values(frame, time_column)
    first = config.min_train_sessions + config.purge_sessions
    fold_width=config.embargo_sessions+config.test_sessions
    possible = list(range(first, len(sessions) - fold_width + 1, fold_width))
    if not possible:
        raise ValueError("Onvoldoende sessies voor een purged walk-forward fold.")
    if len(possible) > config.n_splits:
        selected = np.linspace(0, len(possible) - 1, config.n_splits, dtype=int)
        possible = [possible[index] for index in sorted(set(selected.tolist()))]
    folds = []
    row_days=pd.to_datetime(frame[time_column],errors="coerce").dt.date
    for fold_number, test_start in enumerate(possible, 1):
        train_stop = test_start - config.purge_sessions
        train_days = set(sessions[:train_stop])
        test_begin=test_start+config.embargo_sessions
        test_days = set(sessions[test_begin:test_begin + config.test_sessions])
        train = frame[row_days.isin(train_days)].copy()
        test = frame[row_days.isin(test_days)].copy()
        if len(_session_values(train,time_column)) < config.min_train_sessions or test.empty:
            continue
        folds.append({
            "fold": fold_number, "train": train, "test": test,
            "train_start": str(min(train_days)), "train_end": str(max(train_days)),
            "test_start": str(min(test_days)), "test_end": str(max(test_days)),
        })
    return folds


def _pipeline(alpha: float = 25.0) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
        ("model", Ridge(alpha=alpha)),
    ])


def _normal_probability_positive(mean: np.ndarray, sigma: float) -> np.ndarray:
    scale = max(float(sigma), 1e-8)
    return np.asarray([.5 * (1 + math.erf(float(value) / (scale * math.sqrt(2)))) for value in mean])


def _normal_probability_above(mean: np.ndarray, sigma: float, threshold: float) -> np.ndarray:
    return _normal_probability_positive(np.asarray(mean,dtype=float)-float(threshold),sigma)


def estimate_roundtrip_costs(frame: pd.DataFrame, target: str,
                             model: CostModel = CostModel()) -> tuple[np.ndarray, np.ndarray]:
    def numeric(column: str) -> pd.Series:
        if column not in frame:
            return pd.Series(np.nan,index=frame.index,dtype=float)
        return pd.to_numeric(frame[column],errors="coerce")
    side = "call" if "call" in target.lower() else "put"
    price_column = f"{side}_close_option" if f"{side}_close_option" in frame else f"{side}_extrinsic"
    price = numeric(price_column).abs().replace(0, np.nan)
    bid = numeric(f"{side}_bid")
    ask = numeric(f"{side}_ask")
    has_quotes = bid.notna() & ask.notna() & (ask >= bid) & (price > 0)
    exit_bid=numeric(f"future_{side}_bid")
    exit_ask=numeric(f"future_{side}_ask")
    has_exit_quotes=exit_bid.notna() & exit_ask.notna() & (exit_ask>=exit_bid)
    spread_return = pd.Series(model.fallback_roundtrip_spread_pct, index=frame.index, dtype=float)
    quoted_spread=((ask-bid).clip(lower=model.minimum_tick)/price)
    spread_return.loc[has_quotes] = quoted_spread.loc[has_quotes]
    fees = 2 * (model.commission_per_contract_side + model.regulatory_fee_per_contract_side)
    fee_return = fees / (price * model.contract_multiplier)
    fee_return = fee_return.replace([np.inf, -np.inf], np.nan).fillna(0.02)
    slippage = 2 * (
        model.slippage_bps_per_side+model.market_impact_bps_per_side+model.adverse_fill_bps_per_side
    ) / 10000
    costs = (spread_return + fee_return + slippage).clip(lower=0).to_numpy(float)
    # Both entry and exit quotes plus depth are required for a non-provisional
    # execution claim.  Current Pair Store rows do not fabricate either.
    depth_available=(numeric(f"{side}_bid_size").notna() & numeric(f"{side}_ask_size").notna())
    provisional = (~(has_quotes & has_exit_quotes & depth_available)).to_numpy(bool)
    return costs, provisional


def economic_metrics(actual_gross_return: Any, expected_return: Any, costs: Any,
                     traded: Any) -> dict[str, Any]:
    actual = np.asarray(actual_gross_return, dtype=float)
    expected = np.asarray(expected_return, dtype=float)
    cost = np.asarray(costs, dtype=float)
    take = np.asarray(traded, dtype=bool)
    valid = np.isfinite(actual) & np.isfinite(expected) & np.isfinite(cost)
    actual, expected, cost, take = actual[valid], expected[valid], cost[valid], take[valid]
    net = actual[take] - cost[take]
    gross = actual[take]
    trade_cost = cost[take]
    wins, losses = net[net > 0], net[net <= 0]
    p_win = float(len(wins) / len(net)) if len(net) else 0.0
    average_win = float(np.mean(wins)) if len(wins) else 0.0
    average_loss_abs = float(abs(np.mean(losses))) if len(losses) else 0.0
    ev_formula = p_win * average_win - (1 - p_win) * average_loss_abs
    gross_wins,gross_losses=gross[gross>0],gross[gross<=0]
    gross_p_win=float(len(gross_wins)/len(gross)) if len(gross) else 0.0
    ev_net_formula=(
        gross_p_win*(float(np.mean(gross_wins)) if len(gross_wins) else 0.0)
        -(1-gross_p_win)*(float(abs(np.mean(gross_losses))) if len(gross_losses) else 0.0)
        -(float(np.mean(trade_cost)) if len(trade_cost) else 0.0)
    )
    if len(net):
        curve = np.cumprod(1 + np.clip(net, -0.999, None))
        peak = np.maximum.accumulate(curve)
        max_drawdown = float(np.max((peak - curve) / np.maximum(peak, 1e-12)))
    else:
        curve, max_drawdown = np.asarray([]), 0.0
    std = float(np.std(net, ddof=1)) if len(net) > 1 else 0.0
    downside = net[net < 0]
    downside_std = float(np.std(downside, ddof=1)) if len(downside) > 1 else 0.0
    return {
        "observations": int(len(actual)), "number_of_trades": int(len(net)),
        "trade_rate": float(np.mean(take)) if len(take) else 0.0,
        "net_return": float(curve[-1] - 1) if len(curve) else 0.0,
        "expected_value_per_trade": float(np.mean(net)) if len(net) else 0.0,
        "ev_formula": ev_formula,
        "EV_net":ev_net_formula,
        "gross_expected_value_before_costs": float(np.mean(gross)) if len(gross) else 0.0,
        "average_cost_return": float(np.mean(trade_cost)) if len(trade_cost) else 0.0,
        "sharpe": float(np.mean(net) / std * math.sqrt(252)) if std > 0 else None,
        "sortino": float(np.mean(net) / downside_std * math.sqrt(252)) if downside_std > 0 else None,
        "profit_factor": float(wins.sum() / abs(losses.sum())) if len(losses) and abs(losses.sum()) > 0 else None,
        "maximum_drawdown": max_drawdown, "max_drawdown": max_drawdown, "win_rate": p_win,
        "average_win": average_win, "average_loss": -average_loss_abs,
        "payoff_ratio": average_win / average_loss_abs if average_loss_abs else None,
        "turnover": int(len(net)), "exposure": float(np.mean(take)) if len(take) else 0.0,
        "mean_expected_return": float(np.mean(expected[take])) if take.any() else 0.0,
    }


def predictive_metrics(predictions: pd.DataFrame) -> dict[str, Any]:
    actual=pd.to_numeric(predictions.get("actual_return"),errors="coerce")
    expected=pd.to_numeric(predictions.get("expected_return"),errors="coerce")
    probability=pd.to_numeric(predictions.get("probability_positive"),errors="coerce")
    valid=actual.notna() & expected.notna()
    if not valid.any():return {"n":0}
    y=actual[valid].to_numpy(float);p=expected[valid].to_numpy(float)
    ppos=np.clip(probability[valid].to_numpy(float),0,1)
    outcome=(y>0).astype(float)
    return {
        "n":int(len(y)),"mae":float(np.mean(np.abs(y-p))),
        "rmse":float(np.sqrt(np.mean((y-p)**2))),
        "direction_accuracy":float(np.mean((p>0)==(y>0))),
        "brier":float(np.mean((ppos-outcome)**2)),
        "calibration_gap":float(abs(np.mean(ppos)-np.mean(outcome))),
    }


def _breakdowns(predictions: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    mappings = {
        "period": "test_period", "ticker": "source_ticker", "sector": "sector",
        "market_regime": "market_regime", "volatility_regime": "volatility_regime",
        "event_type": "event_type", "fold": "fold",
    }
    for label, column in mappings.items():
        if column not in predictions:
            continue
        rows = []
        for value, group in predictions.groupby(column, dropna=False, sort=False):
            metrics = economic_metrics(
                group["actual_return"], group["expected_return"],
                group["expected_cost_return"], group["trade"],
            )
            rows.append({"value": str(value), **metrics})
        result[label] = rows
    return result


def _fit_predict(train: pd.DataFrame, test: pd.DataFrame, features: list[str], target: str,
                 cost_model: CostModel, model_name: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    y_train = pd.to_numeric(train[target], errors="coerce")
    y_test = pd.to_numeric(test[target], errors="coerce")
    train_ok, test_ok = y_train.notna(), y_test.notna()
    train, y_train = train.loc[train_ok], y_train.loc[train_ok]
    test, y_test = test.loc[test_ok].copy(), y_test.loc[test_ok]
    if len(train) < 40 or len(test) < 5:
        raise ValueError("Onvoldoende geldige train- of testwaarnemingen.")
    if model_name == "historical_mean":
        predicted = np.full(len(test), float(y_train.mean()))
        residual_sigma = float(y_train.std(ddof=1) or 0.0)
    elif model_name == "ridge":
        model = _pipeline()
        model.fit(train[features], y_train)
        predicted = np.asarray(model.predict(test[features]), dtype=float)
        residual = y_train.to_numpy(float) - np.asarray(model.predict(train[features]), dtype=float)
        residual_sigma = float(np.std(residual, ddof=1)) if len(residual) > 1 else 0.0
    else:
        raise ValueError(f"Onbekend benchmarkmodel: {model_name}")
    costs, provisional = estimate_roundtrip_costs(test, target, cost_model)
    uncertainty_penalty = residual_sigma * .25
    trade = predicted > costs + cost_model.safety_margin_return + uncertainty_penalty
    out = test.copy()
    out["actual_return"] = y_test.to_numpy(float)
    out["expected_return"] = predicted
    out["prediction_uncertainty"] = residual_sigma
    out["probability_positive"] = _normal_probability_positive(predicted, residual_sigma)
    out["probability_return_above_1pct"] = _normal_probability_above(predicted,residual_sigma,.01)
    out["expected_cost_return"] = costs
    out["costs_provisional"] = provisional
    out["net_expected_edge"] = predicted - costs
    out["trade"] = trade
    out["decision"] = np.where(trade, "TRADE", "NO_TRADE")
    out["actual_net_return"] = np.where(trade, y_test.to_numpy(float) - costs, 0.0)
    return out, {"residual_sigma": residual_sigma}


def evaluate_candidate(frame: pd.DataFrame, features: Iterable[str], target: str,
                       horizon: str, model_name: str = "ridge",
                       config: WalkForwardConfig = WalkForwardConfig(),
                       cost_model: CostModel = CostModel(),
                       feature_set_name: str = "all",
                       evaluate_holdout: bool = True) -> tuple[dict[str, Any], pd.DataFrame]:
    features = [str(value) for value in features]
    column_audit = audit_feature_columns(features)
    require_valid(column_audit, feature_set_name)
    if target not in frame:
        raise ValueError(f"Target ontbreekt: {target}")
    clean = frame.dropna(subset=[target, "source_day", "minute"]).copy()
    research, holdout = reserve_untouched_holdout(clean, config)
    folds = purged_walk_forward_splits(research, config)
    predictions = []
    fold_audits = []
    for fold in folds:
        train, test = fold["train"], fold["test"]
        temporal = audit_purged_split(train, test)
        pit = merge_audits(
            audit_point_in_time(train), audit_point_in_time(test),
        )
        audit = merge_audits(temporal, pit)
        require_valid(audit, f"fold {fold['fold']}")
        pred, details = _fit_predict(train, test, features, target, cost_model, model_name)
        pred["fold"] = fold["fold"]
        pred["test_period"] = f"{fold['test_start']}..{fold['test_end']}"
        predictions.append(pred)
        fold_audits.append({
            "fold": fold["fold"], "train_start": fold["train_start"],
            "train_end": fold["train_end"], "test_start": fold["test_start"],
            "test_end": fold["test_end"], "audit": audit.as_dict(), **details,
        })
    walk = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    if walk.empty:
        raise ValueError("Geen walk-forward voorspellingen geproduceerd.")

    walk_metrics = economic_metrics(
        walk["actual_return"], walk["expected_return"], walk["expected_cost_return"], walk["trade"],
    )
    walk_metrics["costs_provisional"] = bool(walk["costs_provisional"].any())
    walk_predictive=predictive_metrics(walk)
    holdout_metrics=None;holdout_details={"residual_sigma":None};holdout_audit=None
    all_predictions=walk
    if evaluate_holdout:
        # The holdout is opened only for a model selected on earlier folds (or
        # for an immutable simple baseline), never for screening every ablation.
        holdout_audit = merge_audits(
            audit_purged_split(research, holdout),
            audit_point_in_time(research), audit_point_in_time(holdout),
        )
        require_valid(holdout_audit, "untouched holdout")
        holdout_prediction, holdout_details = _fit_predict(
            research, holdout, features, target, cost_model, model_name,
        )
        holdout_prediction["fold"] = "holdout"
        holdout_prediction["test_period"] = "untouched_holdout"
        holdout_metrics = economic_metrics(
            holdout_prediction["actual_return"], holdout_prediction["expected_return"],
            holdout_prediction["expected_cost_return"], holdout_prediction["trade"],
        )
        holdout_metrics["costs_provisional"] = bool(holdout_prediction["costs_provisional"].any())
        holdout_predictive=predictive_metrics(holdout_prediction)
        all_predictions=pd.concat([walk,holdout_prediction],ignore_index=True)
    else:
        holdout_predictive=None
    experiment_id = f"exp-{utcnow():%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:10]}"
    profitable_periods = sum(
        row["expected_value_per_trade"] > 0
        for row in _breakdowns(walk).get("period", [])
    )
    manifest = {
        "experiment_id": experiment_id, "created_at": utcnow().isoformat(),
        "model": model_name, "feature_set": feature_set_name, "features": features,
        "target": target, "horizon": horizon,
        "tickers": sorted(set(clean.get("source_ticker", pd.Series(dtype=str)).astype(str))),
        "walk_forward_config": asdict(config), "cost_model": asdict(cost_model),
        "folds": fold_audits, "walk_forward_metrics": walk_metrics,
        "walk_forward_predictive_metrics":walk_predictive,
        "holdout_metrics": holdout_metrics,"holdout_predictive_metrics":holdout_predictive,
        "breakdowns": _breakdowns(walk),
        "holdout_audit": holdout_audit.as_dict() if holdout_audit else None, "leakage_valid": True,
        "untouched_holdout_used_once": bool(evaluate_holdout),
        "holdout_status":"evaluated_after_selection" if evaluate_holdout else "sealed_during_selection",
        "profitable_test_periods": int(profitable_periods),
        "holdout_residual_sigma": holdout_details["residual_sigma"],
        "claim": "research_candidate_only",
    }
    return manifest, all_predictions


def _paired_test(candidate: pd.DataFrame, baseline: pd.DataFrame) -> dict[str, Any]:
    left = candidate[["source_ticker", "source_day", "actual_net_return"]].copy()
    right = baseline[["source_ticker", "source_day", "actual_net_return"]].copy()
    merged = left.merge(right, on=["source_ticker", "source_day"], suffixes=("_candidate", "_baseline"))
    if len(merged) < 10:
        return {"n": int(len(merged)), "p_value": None, "mean_difference": None}
    result = ttest_rel(merged["actual_net_return_candidate"], merged["actual_net_return_baseline"], nan_policy="omit")
    return {
        "n": int(len(merged)), "p_value": float(result.pvalue) if math.isfinite(result.pvalue) else None,
        "mean_difference": float((merged["actual_net_return_candidate"] - merged["actual_net_return_baseline"]).mean()),
    }


def benjamini_hochberg(p_values: Iterable[float | None]) -> list[float | None]:
    values = list(p_values)
    valid = [(index, float(value)) for index, value in enumerate(values)
             if value is not None and math.isfinite(float(value))]
    adjusted: list[float | None] = [None] * len(values)
    if not valid:
        return adjusted
    ordered = sorted(valid, key=lambda item: item[1])
    running = 1.0
    for rank_from_end, (index, value) in enumerate(reversed(ordered), 1):
        rank = len(ordered) - rank_from_end + 1
        running = min(running, value * len(ordered) / rank)
        adjusted[index] = min(1.0, running)
    return adjusted


def run_research_suite(frame: pd.DataFrame, feature_sets: dict[str, list[str]],
                       targets: list[str], horizon: str,
                       config: WalkForwardConfig = WalkForwardConfig(),
                       cost_model: CostModel = CostModel()) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    experiments = []
    predictions: dict[str, pd.DataFrame] = {}
    for target in targets:
        baseline_manifest, baseline_predictions = evaluate_candidate(
            frame, [], target, horizon, model_name="historical_mean",
            config=config, cost_model=cost_model, feature_set_name="historical_mean",
            evaluate_holdout=False,
        )
        baseline_key = f"{target}|historical_mean"
        experiments.append(baseline_manifest)
        predictions[baseline_key] = baseline_predictions
        for name, features in feature_sets.items():
            manifest, pred = evaluate_candidate(
                frame, features, target, horizon, model_name="ridge",
                config=config, cost_model=cost_model, feature_set_name=name,
                evaluate_holdout=False,
            )
            test = _paired_test(
                pred[pred["fold"] != "holdout"],
                baseline_predictions[baseline_predictions["fold"] != "holdout"],
            )
            manifest["paired_vs_historical_mean"] = test
            experiments.append(manifest)
            predictions[f"{target}|{name}"] = pred

        # Select one complex candidate solely on the repeated earlier tests.
        target_candidates=[item for item in experiments if item.get("target")==target
                           and item.get("model")!="historical_mean"]
        eligible=[item for item in target_candidates
                  if item["walk_forward_metrics"]["number_of_trades"]>=20
                  and item["profitable_test_periods"]>=2]
        chosen=max(
            eligible or target_candidates,
            key=lambda item:(
                item["walk_forward_metrics"]["expected_value_per_trade"],
                -item["walk_forward_metrics"]["maximum_drawdown"],
            ),
            default=None,
        )
        # The immutable mean baseline and the one preselected complex model are
        # now allowed to see the final holdout. All other ablations stay sealed.
        baseline_final,baseline_final_predictions=evaluate_candidate(
            frame,[],target,horizon,model_name="historical_mean",config=config,
            cost_model=cost_model,feature_set_name="historical_mean",evaluate_holdout=True,
        )
        baseline_index=next(i for i,item in enumerate(experiments)
                            if item.get("target")==target and item.get("model")=="historical_mean")
        experiments[baseline_index]=baseline_final
        predictions[baseline_key]=baseline_final_predictions
        if chosen is not None:
            chosen_final,chosen_predictions=evaluate_candidate(
                frame,feature_sets[chosen["feature_set"]],target,horizon,model_name="ridge",
                config=config,cost_model=cost_model,feature_set_name=chosen["feature_set"],
                evaluate_holdout=True,
            )
            chosen_final["selected_before_holdout"]=True
            chosen_index=next(i for i,item in enumerate(experiments)
                              if item.get("experiment_id")==chosen.get("experiment_id"))
            chosen_final["paired_vs_historical_mean"]=experiments[chosen_index].get("paired_vs_historical_mean")
            experiments[chosen_index]=chosen_final
            predictions[f"{target}|{chosen['feature_set']}"]=chosen_predictions
    adjusted = benjamini_hochberg([
        (item.get("paired_vs_historical_mean") or {}).get("p_value") for item in experiments
    ])
    for item, q_value in zip(experiments, adjusted):
        if item.get("paired_vs_historical_mean") is not None:
            item["paired_vs_historical_mean"]["fdr_q_value"] = q_value
    stable = [item for item in experiments if item["model"] != "historical_mean"
              and item.get("holdout_metrics")
              and item["holdout_metrics"]["expected_value_per_trade"] > 0
              and item["holdout_metrics"]["number_of_trades"] >= 20
              and item["profitable_test_periods"] >= 2]
    return {
        "research_version": "alpha-v1", "created_at": utcnow().isoformat(),
        "horizon": horizon, "experiments": experiments,
        "stable_candidates": [item["experiment_id"] for item in stable],
        "selection_rule": "positive cost-aware untouched-holdout EV, multiple periods, leakage-valid",
        "live_trading_authorized": False,
    }, predictions


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def save_research_suite(suite: dict[str, Any], predictions: dict[str, pd.DataFrame]) -> dict[str, Any]:
    created = pd.to_datetime(suite.get("created_at"), utc=True, errors="coerce")
    if pd.isna(created):
        created = pd.Timestamp(utcnow())
    run_id = f"run-{created:%Y%m%dT%H%M%S%fZ}-{uuid.uuid4().hex[:10]}"
    prediction_keys = {}
    for name, frame in predictions.items():
        digest = sha256(frame.to_csv(index=False).encode()).hexdigest()[:12]
        key = f"{EXPERIMENT_ROOT}/predictions/{created:%Y}/{created:%m}/{run_id}_{_safe_name(name)}_{digest}.csv.gz"
        put_df(key, frame)
        prediction_keys[name] = key
    manifest = {**suite, "run_id": run_id, "prediction_keys": prediction_keys, "append_only": True}
    key = f"{EXPERIMENT_ROOT}/runs/{created:%Y}/{created:%m}/{run_id}.json"
    put_json(key, manifest)
    return {"run_id": run_id, "manifest_key": key, "prediction_keys": prediction_keys}


def latest_research_suite() -> dict[str, Any] | None:
    keys = sorted(key for key in list_keys(f"{EXPERIMENT_ROOT}/runs/") if key.endswith(".json"))
    if not keys:
        return None
    value = get_json(keys[-1])
    return {**value, "manifest_key": keys[-1]} if isinstance(value, dict) else None
