"""Paper/live model, feature and execution-quality monitoring."""

from __future__ import annotations

from datetime import datetime, timezone
import math
from typing import Any

import numpy as np
import pandas as pd

from core.config import MONITORING_ROOT
from core.storage import get_json,list_keys,put_json


def _finite_array(values: Any) -> np.ndarray:
    x = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    return x[np.isfinite(x)]


def population_stability_index(reference: Any, current: Any, bins: int = 10) -> float | None:
    ref, cur = _finite_array(reference), _finite_array(current)
    if len(ref) < 30 or len(cur) < 20:
        return None
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    ref_hist = np.histogram(ref, bins=edges)[0] / len(ref)
    cur_hist = np.histogram(cur, bins=edges)[0] / len(cur)
    ref_hist = np.clip(ref_hist, 1e-6, None)
    cur_hist = np.clip(cur_hist, 1e-6, None)
    return float(np.sum((cur_hist - ref_hist) * np.log(cur_hist / ref_hist)))


def maximum_drawdown(returns: Any) -> float | None:
    x = _finite_array(returns)
    if not len(x):
        return None
    curve = np.cumprod(1 + np.clip(x, -0.999, None))
    peak = np.maximum.accumulate(curve)
    return float(np.max((peak - curve) / np.maximum(peak, 1e-12)))


def monitor_performance(records: pd.DataFrame,
                        feature_reference: dict[str, Any] | None = None,
                        feature_columns: list[str] | None = None) -> dict[str, Any]:
    """Compare expected and realised results and emit kill-switch triggers."""
    data = records.copy() if records is not None else pd.DataFrame()
    feature_reference = feature_reference or {}
    feature_columns = feature_columns or []
    result: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "rows": int(len(data)), "triggers": [], "feature_psi": {},
    }
    if data.empty:
        result["triggers"].append("geen_monitoringdata")
        result["kill_switch"] = True
        return result

    def numeric(column: str) -> pd.Series:
        if column not in data:
            return pd.Series(np.nan,index=data.index,dtype=float)
        return pd.to_numeric(data[column],errors="coerce")

    expected = numeric("expected_net_return")
    actual = numeric("actual_net_return")
    usable = expected.notna() & actual.notna()
    if usable.any():
        exp, act = expected[usable].to_numpy(float), actual[usable].to_numpy(float)
        error = act - exp
        result.update({
            "expected_pnl_mean": float(np.mean(exp)),
            "actual_pnl_mean": float(np.mean(act)),
            "pnl_forecast_error_mean": float(np.mean(error)),
            "pnl_forecast_mae": float(np.mean(np.abs(error))),
            "actual_sharpe": (
                float(np.mean(act) / np.std(act, ddof=1) * np.sqrt(252))
                if len(act) > 2 and np.std(act, ddof=1) > 0 else None
            ),
            "maximum_drawdown": maximum_drawdown(act),
            "direction_accuracy": float(np.mean((exp > 0) == (act > 0))),
        })
        if len(act) >= 30 and float(np.mean(act)) < 0:
            result["triggers"].append("negatieve_live_ev")
        if result["maximum_drawdown"] is not None and result["maximum_drawdown"] > 0.12:
            result["triggers"].append("drawdown_limiet")

    probability = numeric("probability_positive")
    outcome = numeric("positive_outcome")
    calibrated = probability.notna() & outcome.notna()
    if calibrated.any():
        p = np.clip(probability[calibrated].to_numpy(float), 0, 1)
        y = outcome[calibrated].to_numpy(float)
        result["brier"] = float(np.mean((p - y) ** 2))
        result["calibration_gap"] = float(abs(np.mean(p) - np.mean(y)))
        if len(p) >= 50 and result["calibration_gap"] > 0.12:
            result["triggers"].append("slechte_kalibratie")

    if "prediction_reference" in feature_reference and probability.notna().any():
        psi = population_stability_index(feature_reference["prediction_reference"], probability.dropna())
        result["prediction_psi"] = psi
        if psi is not None and psi > 0.25:
            result["triggers"].append("prediction_drift")

    for column in feature_columns:
        if column not in data or column not in feature_reference:
            continue
        psi = population_stability_index(feature_reference[column], data[column])
        result["feature_psi"][column] = psi
        if psi is not None and psi > 0.25:
            result["triggers"].append(f"feature_drift:{column}")

    quoted = numeric("expected_cost_return")
    realised = numeric("actual_cost_return")
    costs = quoted.notna() & realised.notna()
    if costs.any():
        result["execution_cost_ratio"] = float(
            realised[costs].sum() / max(abs(quoted[costs].sum()), 1e-12)
        )
        if int(costs.sum()) >= 20 and result["execution_cost_ratio"] > 1.5:
            result["triggers"].append("execution_kosten_afwijking")

    spreads = numeric("relative_spread")
    if spreads.notna().any():
        result["mean_relative_spread"] = float(spreads.mean())
        result["p95_relative_spread"] = float(spreads.quantile(.95))
        if int(spreads.notna().sum()) >= 20 and result["p95_relative_spread"] > .15:
            result["triggers"].append("uitzonderlijke_spreads")
    liquidity = numeric("open_interest")
    if liquidity.notna().any():
        result["median_open_interest"] = float(liquidity.median())
        if result["median_open_interest"] < 30:
            result["triggers"].append("liquiditeit_verslechterd")

    result["triggers"] = sorted(set(result["triggers"]))
    result["kill_switch"] = bool(result["triggers"])
    return result


def save_monitoring_snapshot(snapshot: dict[str, Any]) -> str:
    stamp=pd.to_datetime(snapshot.get("generated_at") or datetime.now(timezone.utc),utc=True,errors="coerce")
    if pd.isna(stamp):stamp=pd.Timestamp(datetime.now(timezone.utc))
    key=f"{MONITORING_ROOT}/snapshots/{stamp:%Y}/{stamp:%m}/{stamp:%Y%m%dT%H%M%S%fZ}.json"
    put_json(key,{**snapshot,"append_only":True})
    return key


def latest_monitoring_snapshot() -> dict[str, Any] | None:
    keys=sorted(key for key in list_keys(f"{MONITORING_ROOT}/snapshots/") if key.endswith(".json"))
    if not keys:return None
    value=get_json(keys[-1])
    return {**value,"key":keys[-1]} if isinstance(value,dict) else None
