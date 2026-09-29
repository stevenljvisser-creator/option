"""Target-safe relationship discovery for OptionEdge.

This module deliberately separates discovery and confirmation in time.  It is
not a causal-proof engine; it produces hypotheses with out-of-sample evidence,
effect sizes and multiple-testing control.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, brier_score_loss, mean_absolute_error,
    mean_squared_error, roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


EXCLUDE_TOKENS = ("future_", "target", "label", "outcome", "realized_payoff", "realised_payoff")


def benjamini_hochberg(p_values: Iterable[float]) -> np.ndarray:
    p = np.asarray(list(p_values), dtype=float)
    if p.size == 0:
        return p
    p = np.nan_to_num(p, nan=1.0, posinf=1.0, neginf=1.0)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = ranked * p.size / np.arange(1, p.size + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    out = np.empty_like(adjusted)
    out[order] = np.clip(adjusted, 0.0, 1.0)
    return out


def _chronological(frame: pd.DataFrame) -> pd.DataFrame:
    d = frame.copy()
    for candidate in ["as_of_utc", "minute", "date", "ts"]:
        if candidate in d:
            d["__time"] = pd.to_datetime(d[candidate], utc=True, errors="coerce")
            break
    if "__time" not in d:
        d["__time"] = pd.RangeIndex(len(d))
    return d.sort_values("__time").reset_index(drop=True)


def safe_features(frame: pd.DataFrame, target: str, max_features: int = 1200) -> list[str]:
    out = []
    for col in frame.columns:
        low = str(col).lower()
        if col == target or col == "__time" or any(token in low for token in EXCLUDE_TOKENS):
            continue
        s = pd.to_numeric(frame[col], errors="coerce")
        if s.notna().sum() >= max(30, int(len(frame) * 0.2)) and s.nunique(dropna=True) > 1:
            out.append(col)
        if len(out) >= max_features:
            break
    return out


def _corr(feature: pd.Series, target: pd.Series) -> tuple[float, float, int]:
    x = pd.to_numeric(feature, errors="coerce")
    y = pd.to_numeric(target, errors="coerce")
    keep = x.notna() & y.notna()
    if int(keep.sum()) < 20 or x[keep].nunique() < 2 or y[keep].nunique() < 2:
        return 0.0, 1.0, int(keep.sum())
    rho, p = spearmanr(x[keep].to_numpy(), y[keep].to_numpy())
    return float(np.nan_to_num(rho)), float(np.nan_to_num(p, nan=1.0)), int(keep.sum())


def _within_group_corr(frame: pd.DataFrame, feature: str, target: str,
                       group: str = "source_ticker") -> tuple[float, float, int]:
    """Within-ticker association, removing persistent ticker-level means."""
    if group not in frame:
        return _corr(frame[feature], frame[target])
    x = pd.to_numeric(frame[feature], errors="coerce")
    y = pd.to_numeric(frame[target], errors="coerce")
    keep = x.notna() & y.notna() & frame[group].notna()
    if int(keep.sum()) < 20:
        return 0.0, 1.0, int(keep.sum())
    g = frame.loc[keep, group]
    xw = x[keep] - x[keep].groupby(g).transform("mean")
    yw = y[keep] - y[keep].groupby(g).transform("mean")
    return _corr(xw, yw)


def _rho_interval(rho: float, n: int) -> list[float | None]:
    if n <= 4 or not np.isfinite(rho):
        return [None, None]
    bounded = float(np.clip(rho, -.999999, .999999))
    z = np.arctanh(bounded); radius = 1.96 / math.sqrt(n - 3)
    return [float(np.tanh(z - radius)), float(np.tanh(z + radius))]


def _effect_binary(feature: pd.Series, target: pd.Series) -> dict[str, float]:
    x = pd.to_numeric(feature, errors="coerce")
    y = pd.to_numeric(target, errors="coerce")
    keep = x.notna() & y.notna()
    x = x[keep]
    y = y[keep]
    if len(x) < 30:
        return {"high_mean": math.nan, "low_mean": math.nan, "difference": math.nan}
    q1, q3 = x.quantile([0.25, 0.75])
    low = y[x <= q1]
    high = y[x >= q3]
    return {
        "high_mean": float(high.mean()) if len(high) else math.nan,
        "low_mean": float(low.mean()) if len(low) else math.nan,
        "difference": float(high.mean() - low.mean()) if len(high) and len(low) else math.nan,
    }


def relationship_scan(frame: pd.DataFrame, target: str, max_features: int = 1200,
                      min_support: int = 80, purge_sessions: int = 1) -> dict[str, Any]:
    d = _chronological(frame)
    if target not in d:
        raise ValueError(f"Target ontbreekt: {target}")
    d[target] = pd.to_numeric(d[target], errors="coerce")
    d = d[d[target].notna()].reset_index(drop=True)
    if len(d) < max(200, min_support * 2):
        raise ValueError(f"Te weinig tijdgeordende observaties ({len(d)}).")

    # Split on complete calendar groups so observations from the same market day
    # can never appear on both sides merely because multiple tickers are present.
    grouped_time = pd.to_datetime(d["__time"], utc=True, errors="coerce").dt.floor("D")
    unique_time = sorted(grouped_time.dropna().unique())
    if len(unique_time) >= 20:
        cut_index=max(1,int(len(unique_time)*.70))
        purge=max(1,int(purge_sessions))
        early=unique_time[:cut_index]
        late=unique_time[cut_index:]
        discovery_days=set(early[:-purge] if len(early)>purge else [])
        confirmation_days=set(late[purge:] if len(late)>purge else [])
        discovery=d[grouped_time.isin(discovery_days)].copy()
        confirmation=d[grouped_time.isin(confirmation_days)].copy()
    else:
        cut = max(1, int(len(d) * 0.70))
        discovery = d.iloc[:cut].copy()
        confirmation = d.iloc[cut:].copy()
    features = safe_features(discovery, target, max_features=max_features)
    binary = set(pd.unique(d[target].dropna())).issubset({0, 1})

    discovery_stats = []
    for col in features:
        rho, p, n = _corr(discovery[col], discovery[target])
        wrho, wp, wn = _within_group_corr(discovery, col, target)
        discovery_stats.append({
            "feature": col, "discovery_rho": rho, "discovery_p": p, "discovery_n": n,
            "discovery_within_ticker_rho": wrho, "discovery_within_ticker_p": wp,
            "discovery_within_ticker_n": wn,
        })
    q_values = benjamini_hochberg(x["discovery_p"] for x in discovery_stats)
    within_q_values = benjamini_hochberg(x["discovery_within_ticker_p"] for x in discovery_stats)
    for row, q, within_q in zip(discovery_stats, q_values, within_q_values):
        row["discovery_q"] = float(q)
        row["discovery_within_ticker_q"] = float(within_q)

    # Only hypotheses selected in the discovery period are touched in confirmation.
    selected = sorted(
        [x for x in discovery_stats if x["discovery_n"] >= min_support
         and min(x["discovery_q"], x["discovery_within_ticker_q"]) <= .10],
        key=lambda x: (min(x["discovery_q"], x["discovery_within_ticker_q"]),
                       -max(abs(x["discovery_rho"]), abs(x["discovery_within_ticker_rho"]))),
    )[:200]
    confirmation_p = []
    confirmation_within_p = []
    for row in selected:
        rho, p, n = _corr(confirmation[row["feature"]], confirmation[target])
        wrho, wp, wn = _within_group_corr(confirmation, row["feature"], target)
        row.update({
            "confirmation_rho": rho, "confirmation_p": p, "confirmation_n": n,
            "confirmation_rho_ci95": _rho_interval(rho, n),
            "confirmation_within_ticker_rho": wrho, "confirmation_within_ticker_p": wp,
            "confirmation_within_ticker_n": wn,
            "confirmation_within_ticker_rho_ci95": _rho_interval(wrho, wn),
        })
        row["stable_direction"] = bool(np.sign(rho) == np.sign(row["discovery_rho"]) and rho != 0)
        row["within_ticker_stable_direction"] = bool(
            np.sign(wrho) == np.sign(row["discovery_within_ticker_rho"]) and wrho != 0
        )
        row["quartile_effect"] = _effect_binary(confirmation[row["feature"]], confirmation[target])
        confirmation_p.append(p)
        confirmation_within_p.append(wp)
    confirm_q = benjamini_hochberg(confirmation_p)
    confirm_within_q = benjamini_hochberg(confirmation_within_p)
    for row, q, within_q in zip(selected, confirm_q, confirm_within_q):
        row["confirmation_q"] = float(q)
        row["confirmation_within_ticker_q"] = float(within_q)
        row["confirmed_global"] = bool(row["stable_direction"] and q <= 0.05 and row["confirmation_n"] >= max(30, min_support // 2))
        row["confirmed_within_ticker"] = bool(
            row["within_ticker_stable_direction"] and within_q <= 0.05
            and row["confirmation_within_ticker_n"] >= max(30, min_support // 2)
        )
        row["confirmed"] = bool(row["confirmed_global"] and row["confirmed_within_ticker"])

    confirmed = sorted(
        [x for x in selected if x.get("confirmed")],
        key=lambda x: (x["confirmation_q"], -abs(x["confirmation_rho"])),
    )

    # Mutual information is descriptive and fitted on discovery only.  It is not
    # allowed to promote an unconfirmed hypothesis by itself.
    mi_rows = []
    if features:
        x = discovery[features].apply(pd.to_numeric, errors="coerce")
        x_arr = SimpleImputer(strategy="median").fit_transform(x)
        y = discovery[target].to_numpy()
        mi = (mutual_info_classif(x_arr, y.astype(int), random_state=42)
              if binary else mutual_info_regression(x_arr, y, random_state=42))
        mi_rows = [
            {"feature": feature, "mutual_information_discovery": float(value)}
            for feature, value in sorted(zip(features, mi), key=lambda z: z[1], reverse=True)[:50]
        ]

    return {
        "method_version": "oe20-deep-relations-v1",
        "target": target,
        "target_type": "binary" if binary else "continuous",
        "rows": len(d),
        "discovery_rows": len(discovery),
        "confirmation_rows": len(confirmation),
        "tested_features": len(features),
        "selected_hypotheses": len(selected),
        "confirmed_hypotheses": len(confirmed),
        "multiple_testing": "Benjamini-Hochberg FDR, afzonderlijk in discovery en confirmation",
        "purge_embargo_sessions": int(max(1,purge_sessions)),
        "ticker_confounding_guard": "globaal én within-ticker bevestigd",
        "confirmed": confirmed,
        "all_selected": selected,
        "mutual_information": mi_rows,
        "causal_claim": False,
        "warning": "Een bevestigd voorspellend verband bewijst geen oorzakelijk effect.",
    }


@dataclass(frozen=True)
class TimeSplit:
    train: np.ndarray
    test: np.ndarray


def purged_walk_forward_splits(n_rows: int, folds: int = 5, purge: int = 5,
                               min_train_fraction: float = 0.40) -> list[TimeSplit]:
    if n_rows < 100:
        return []
    first_test = max(30, int(n_rows * min_train_fraction))
    remaining = n_rows - first_test
    width = max(20, remaining // max(1, folds))
    out = []
    for start in range(first_test, n_rows, width):
        stop = min(n_rows, start + width)
        train_stop = max(0, start - purge)
        if train_stop < 30 or stop - start < 10:
            continue
        out.append(TimeSplit(np.arange(0, train_stop), np.arange(start, stop)))
    return out[:folds]


def evaluate_dimension_matrix(x: np.ndarray, y: np.ndarray, times: Iterable[Any],
                              purge: int = 5) -> dict[str, Any]:
    order = np.argsort(pd.to_datetime(pd.Series(list(times)), utc=True, errors="coerce").fillna(pd.Timestamp.min.tz_localize("UTC")).to_numpy())
    x = np.asarray(x, dtype=float)[order]
    y = np.asarray(y)[order]
    binary = set(pd.unique(pd.Series(y).dropna())).issubset({0, 1})
    rows = []
    for fold, split in enumerate(purged_walk_forward_splits(len(y), purge=purge), 1):
        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, C=0.3, class_weight="balanced") if binary else Ridge(alpha=10.0)),
        ])
        model.fit(x[split.train], y[split.train])
        if binary:
            probability = model.predict_proba(x[split.test])[:, 1]
            pred = (probability >= 0.5).astype(int)
            rows.append({
                "fold": fold, "n_train": len(split.train), "n_test": len(split.test),
                "brier": float(brier_score_loss(y[split.test], probability)),
                "auc": float(roc_auc_score(y[split.test], probability)) if len(np.unique(y[split.test])) > 1 else math.nan,
                "accuracy": float(accuracy_score(y[split.test], pred)),
                "balanced_accuracy": float(balanced_accuracy_score(y[split.test], pred)),
            })
        else:
            pred = model.predict(x[split.test])
            rows.append({
                "fold": fold, "n_train": len(split.train), "n_test": len(split.test),
                "mae": float(mean_absolute_error(y[split.test], pred)),
                "rmse": float(mean_squared_error(y[split.test], pred) ** 0.5),
            })
    summary = {}
    if rows:
        for key in rows[0]:
            if key not in {"fold", "n_train", "n_test"}:
                vals = [x[key] for x in rows if np.isfinite(x.get(key, math.nan))]
                summary[key] = float(np.mean(vals)) if vals else math.nan
    return {"target_type": "binary" if binary else "continuous", "folds": rows, "summary": summary}
