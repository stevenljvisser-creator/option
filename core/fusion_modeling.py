"""Train the close-of-day fusion model and dimension benchmark.

Daily vectors are only joined to the final point-in-time observation of a
trading day.  They are never joined to earlier intraday rows because doing so
would leak later same-day news or market information.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
import gc
import math
import pickle
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score, brier_score_loss, mean_absolute_error, mean_squared_error,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from core.config import DAILY_LABEL_ROOT, DAILY_VECTOR_ROOT
from core.multi_horizon import _daily_close_anchor, build_multi_horizon_rows
from core.storage import exists, get_bytes, get_json, put_df
from core.vector_fusion import EVENT_FEATURE_NAMES, candidate_from_raw, candidate_layout


Progress = Callable[[float, str], None] | None
DEFAULT_DAILY_HORIZONS = ["D1", "D2", "D3", "W1"]
DIMENSION_CANDIDATES = [128, 256, 384, 512, 768]

CONTRACT_FEATURES = [
    "close_stock", "strike", "dte", "risk_free_rate", "benchmark_sigma",
    "source_staleness_minutes",
    "call_extrinsic", "put_extrinsic", "bs_call_extrinsic", "bs_put_extrinsic",
    "call_expectation_gap", "put_expectation_gap", "put_call_parity_residual",
    "call_option_volume", "put_option_volume", "call_trade_volume", "put_trade_volume",
    "call_trade_count", "put_trade_count", "call_oi_open_interest", "put_oi_open_interest",
    "call_snapshot_iv", "put_snapshot_iv", "call_snapshot_delta", "put_snapshot_delta",
    "call_snapshot_gamma", "put_snapshot_gamma", "call_snapshot_theta", "put_snapshot_theta",
    "call_snapshot_vega", "put_snapshot_vega", "call_put_volume_ratio",
    "call_put_trade_volume_ratio", "call_put_extrinsic_ratio",
]


def _notify(callback: Progress, value: float, message: str) -> None:
    if callback:
        callback(float(value), str(message))


def _close_rows(frame: pd.DataFrame) -> pd.DataFrame:
    d = frame.copy()
    d["minute"] = pd.to_datetime(d["minute"], utc=True, errors="coerce")
    d["source_day"] = pd.to_datetime(d["source_day"], errors="coerce").dt.date
    d = d.dropna(subset=["minute", "source_day", "source_ticker", "expiry", "strike"])
    # One row per contract/ticker/day: the last observation available that day.
    group = ["source_ticker", "source_day", "expiry", "strike"]
    return d.sort_values("minute").groupby(group, as_index=False, sort=False).tail(1).reset_index(drop=True)


def _raw_vector_key(ticker: str, day: date) -> str:
    return f"{DAILY_VECTOR_ROOT}/{ticker}/{day:%Y}/{day:%m}/{day}.npz"


def _load_raw(ticker: str, day: date, cache: dict[tuple[str, date], dict[str, np.ndarray]]
              ) -> dict[str, np.ndarray] | None:
    ident = (ticker, day)
    if ident in cache:
        return cache[ident]
    key = _raw_vector_key(ticker, day)
    if not exists(key):
        cache[ident] = None  # type: ignore[assignment]
        return None
    from io import BytesIO
    with np.load(BytesIO(get_bytes(key)), allow_pickle=False) as archive:
        item = {name: np.asarray(archive[name], dtype=np.float32) for name in archive.files}
    cache[ident] = item
    return item


def _candidate(raw: dict[str, np.ndarray], dimensions: int,
               disabled_components: list[str] | None = None) -> np.ndarray:
    return candidate_from_raw(
        raw.get("news_raw", np.empty(0)), raw.get("options_raw", np.empty(0)),
        raw.get("stock_raw", np.empty(0)), raw.get("expectation_raw", np.empty(0)),
        raw.get("quality_raw", np.empty(0)), dimensions,
        raw.get("news_event_raw", np.empty(0)),
        raw.get("earnings_raw", np.empty(0)),
        disabled_components=disabled_components,
    )


def attach_vectors(frame: pd.DataFrame, dimensions: int,
                   raw_cache: dict[tuple[str, date], dict[str, np.ndarray]],
                   include_event_features: bool = False,
                   disabled_components: list[str] | None = None,
                   include_provenance: bool = False) -> pd.DataFrame:
    d = frame.copy()
    d["source_ticker"] = d["source_ticker"].astype(str).str.upper()
    unique = d[["source_ticker", "source_day"]].drop_duplicates()
    identities = []
    vector_rows = []
    event_rows = []
    provenance_rows = []
    for row in unique.itertuples(index=False):
        ticker = str(row.source_ticker).upper()
        day = row.source_day if isinstance(row.source_day, date) else pd.Timestamp(row.source_day).date()
        raw = _load_raw(ticker, day, raw_cache)
        if raw is None:
            continue
        identities.append({"source_ticker":ticker,"source_day":day})
        vector_rows.append(_candidate(raw,dimensions,disabled_components))
        if include_event_features:
            event_raw=np.asarray(raw.get("news_event_raw",np.empty(0)),dtype=np.float32).reshape(-1)
            padded=np.zeros(len(EVENT_FEATURE_NAMES),dtype=np.float32)
            padded[:min(len(padded),event_raw.size)]=event_raw[:len(padded)]
            event_rows.append(padded)
        if include_provenance:
            meta_key=f"{DAILY_VECTOR_ROOT}/{ticker}/{day:%Y}/{day:%m}/{day}.json"
            try:
                metadata=get_json(meta_key) if exists(meta_key) else {}
            except Exception:
                metadata={}
            news=metadata.get("news") or {}
            earnings=metadata.get("earnings") or {}
            provenance_rows.append({
                "vector_as_of_utc":metadata.get("as_of_utc"),
                "vector_metadata_present":bool(metadata),
                "vector_future_values_used_as_features":bool(
                    metadata.get("future_values_used_as_features",True)
                ),
                "news_availability_basis":news.get("availability_basis"),
                "news_present":bool(news.get("present")),
                "earnings_point_in_time_enforced":bool(
                    not earnings.get("present") or earnings.get("point_in_time_enforced")
                ),
                "vector_version":metadata.get("vector_version"),
                "vector_schema_hash":metadata.get("schema_hash"),
            })
    if not identities:
        return d.iloc[0:0]
    vector_frame=pd.DataFrame(
        np.vstack(vector_rows),columns=[f"fusion_{index:03d}" for index in range(dimensions)]
    )
    vectors=pd.concat([pd.DataFrame(identities),vector_frame],axis=1)
    if include_event_features:
        event_frame=pd.DataFrame(
            np.vstack(event_rows),columns=[f"llm_event_{name}" for name in EVENT_FEATURE_NAMES]
        )
        vectors=pd.concat([vectors,event_frame],axis=1)
    if include_provenance:
        vectors=pd.concat([vectors,pd.DataFrame(provenance_rows)],axis=1)
    return d.merge(vectors,on=["source_ticker","source_day"],how="inner")


def daily_relationship_frame(frame: pd.DataFrame, dimensions: int = 384) -> pd.DataFrame:
    """One statistically independent row per ticker-day for pattern discovery.

    Contract rows are summarized before the daily vector is attached.  This
    prevents a day with many strikes from being counted hundreds of times and
    keeps the analysis aligned with the one-vector-per-day research contract.
    """
    close = _close_rows(frame)
    if close.empty:
        return close
    group = ["source_ticker", "source_day"]
    numeric = []
    for col in close.columns:
        if col in group:
            continue
        values = pd.to_numeric(close[col], errors="coerce")
        if values.notna().any():
            close[col] = values
            numeric.append(col)
    daily = close.groupby(group, as_index=False, sort=False)[numeric].median(numeric_only=True)
    minute = close.groupby(group, as_index=False, sort=False)["minute"].max()
    daily = daily.drop(columns=["minute"], errors="ignore").merge(minute, on=group, how="left")
    if "future_stock_return" in daily:
        daily["future_stock_up"] = (daily["future_stock_return"] > 0).astype(int)
    return attach_vectors(daily, dimensions, {}, include_event_features=True)


def _time_partitions(frame: pd.DataFrame, purge_sessions: int = 1) -> dict[str, pd.DataFrame]:
    d = frame.sort_values(["source_day", "minute"]).reset_index(drop=True)
    days = sorted(pd.unique(d["source_day"]))
    if len(days) < 80:
        raise ValueError(f"Minimaal 80 ticker-handelsdagen nodig; gevonden: {len(days)}.")
    i1, i2, i3 = int(len(days) * .55), int(len(days) * .70), int(len(days) * .85)
    train_list, validation_list = days[:i1], days[i1:i2]
    calibration_list, test_list = days[i2:i3], days[i3:]
    # Purge the end of every earlier partition. Its forward target may otherwise
    # overlap the first labels in the next chronological partition.
    purge = max(1, int(purge_sessions))
    train_days = set(train_list[:-purge] if len(train_list) > purge else [])
    validation_days = set(validation_list[purge:-purge] if len(validation_list) > 2*purge else [])
    calibration_days = set(calibration_list[purge:-purge] if len(calibration_list) > 2*purge else [])
    test_days = set(test_list[purge:] if len(test_list) > purge else [])
    return {
        "train": d[d["source_day"].isin(train_days)].copy(),
        "validation": d[d["source_day"].isin(validation_days)].copy(),
        "calibration": d[d["source_day"].isin(calibration_days)].copy(),
        "test": d[d["source_day"].isin(test_days)].copy(),
    }


def _columns(frame: pd.DataFrame, dimensions: int, with_news: bool = True) -> list[str]:
    base = [x for x in CONTRACT_FEATURES if x in frame]
    vector = [f"fusion_{i:03d}" for i in range(dimensions) if f"fusion_{i:03d}" in frame]
    if not with_news:
        news_dims = candidate_layout(dimensions)["news"]
        vector = [x for x in vector if int(x.rsplit("_", 1)[1]) >= news_dims]
    return base + vector


def _pipeline() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(
            solver="liblinear", max_iter=500, C=.15, class_weight="balanced", random_state=42,
        )),
    ])


def _regression_pipeline() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
        ("model", Ridge(alpha=10.0)),
    ])


def _calibrator(probability: np.ndarray, y: np.ndarray) -> LogisticRegression | None:
    if len(y) < 80 or len(np.unique(y)) < 2:
        return None
    cal = LogisticRegression(solver="lbfgs", max_iter=300, C=1.0)
    cal.fit(np.asarray(probability).reshape(-1, 1), y)
    return cal


def _probability(model: Pipeline, calibrator: LogisticRegression | None, x: pd.DataFrame) -> np.ndarray:
    raw = model.predict_proba(x)[:, 1]
    return calibrator.predict_proba(raw.reshape(-1, 1))[:, 1] if calibrator is not None else raw


def _metrics(y: np.ndarray, p: np.ndarray) -> dict[str, Any]:
    pred = (p >= .5).astype(int)
    return {
        "n": int(len(y)),
        "positive_rate": float(np.mean(y)),
        "brier": float(brier_score_loss(y, p)),
        "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else None,
        "accuracy": float(accuracy_score(y, pred)),
        "mean_probability": float(np.mean(p)),
    }


def _fit_and_test(parts: dict[str, pd.DataFrame], features: list[str], label: str
                  ) -> tuple[dict[str, Any], dict[str, Any]]:
    train = pd.concat([parts["train"], parts["validation"]], ignore_index=True)
    calibration, test = parts["calibration"], parts["test"]
    y_train = train[label].astype(int).to_numpy()
    y_cal = calibration[label].astype(int).to_numpy()
    y_test = test[label].astype(int).to_numpy()
    if min(len(train), len(calibration), len(test)) < 30 or len(np.unique(y_train)) < 2:
        raise ValueError("Onvoldoende chronologische data of slechts één targetklasse.")
    model = _pipeline()
    model.fit(train[features], y_train)
    raw_cal = model.predict_proba(calibration[features])[:, 1]
    cal = _calibrator(raw_cal, y_cal)
    p = _probability(model, cal, test[features])
    return _metrics(y_test, p), {"model": model, "calibrator": cal, "features": features}


def _selection_score(parts: dict[str, pd.DataFrame], features: list[str], label: str) -> float:
    train, validation = parts["train"], parts["validation"]
    y_train = train[label].astype(int).to_numpy()
    y_val = validation[label].astype(int).to_numpy()
    if min(len(train), len(validation)) < 30 or len(np.unique(y_train)) < 2:
        return math.inf
    model = _pipeline()
    model.fit(train[features], y_train)
    p = model.predict_proba(validation[features])[:, 1]
    return float(brier_score_loss(y_val, p))


def _fit_regression(parts: dict[str, pd.DataFrame], features: list[str], label: str
                    ) -> tuple[dict[str, Any], Pipeline]:
    train = pd.concat([parts["train"], parts["validation"], parts["calibration"]], ignore_index=True)
    test = parts["test"]
    y_train = pd.to_numeric(train[label], errors="coerce").clip(-1, 3)
    y_test = pd.to_numeric(test[label], errors="coerce").clip(-1, 3)
    keep_train, keep_test = y_train.notna(), y_test.notna()
    if int(keep_train.sum()) < 60 or int(keep_test.sum()) < 30:
        raise ValueError(f"Onvoldoende continue targets voor {label}.")
    model = _regression_pipeline()
    model.fit(train.loc[keep_train, features], y_train[keep_train].to_numpy())
    prediction = np.clip(model.predict(test.loc[keep_test, features]), -1, 3)
    actual = y_test[keep_test].to_numpy()
    return {
        "n": int(len(actual)),
        "mae": float(mean_absolute_error(actual, prediction)),
        "rmse": float(mean_squared_error(actual, prediction) ** .5),
        "mean_actual_return": float(np.mean(actual)),
        "mean_predicted_return": float(np.mean(prediction)),
    }, model


def _save_labels(frame: pd.DataFrame, horizon: str) -> int:
    count = 0
    columns = [
        "source_ticker", "source_day", "minute", "expiry", "strike",
        "future_call_extrinsic_return", "future_put_extrinsic_return", "future_stock_return",
        "label_call_positive", "label_put_positive", "target_horizon",
    ]
    for (ticker, day), group in frame.groupby(["source_ticker", "source_day"], sort=False):
        d = day if isinstance(day, date) else pd.Timestamp(day).date()
        key = f"{DAILY_LABEL_ROOT}/{ticker}/{d:%Y}/{d:%m}/{d}_{horizon}.csv.gz"
        put_df(key, group[[x for x in columns if x in group]].copy())
        count += len(group)
    return count


def train_fusion_model(tickers: list[str], start: date, end: date,
                       horizons: list[str] | None = None, max_rows_per_horizon: int = 30000,
                       callback: Progress = None) -> tuple[dict[str, Any], dict[str, Any]]:
    horizons = [x for x in (horizons or DEFAULT_DAILY_HORIZONS) if x in DEFAULT_DAILY_HORIZONS]
    if not horizons:
        raise ValueError("Kies minimaal één dagelijkse horizon: D1, D2, D3 of W1.")
    _notify(callback, .01, "Point-in-time dagtargets opbouwen…")
    rows_by_horizon = build_multi_horizon_rows(
        tickers, start, end, horizons=horizons,
        max_rows_per_horizon=max_rows_per_horizon,
        use_all_available=True,
        callback=lambda value, message: _notify(callback, .01 + value * .28, message),
    )
    raw_cache: dict[tuple[str, date], dict[str, np.ndarray]] = {}
    metrics: dict[str, Any] = {
        "horizons": {}, "dimension_candidates": DIMENSION_CANDIDATES,
        "data_policy":"all_available_eligible_rows_no_hidden_cap",
        "data_usage":{
            h:dict((rows_by_horizon.get(h).attrs or {}).get("data_usage") or {})
            for h in horizons if h in rows_by_horizon and rows_by_horizon[h] is not None
        },
    }
    models: dict[str, Any] = {}

    usable = [h for h in horizons if h in rows_by_horizon and not rows_by_horizon[h].empty]
    for hi, horizon in enumerate(usable):
        base = _close_rows(rows_by_horizon[horizon])
        base["label_call_positive"] = (pd.to_numeric(base["future_call_extrinsic_return"], errors="coerce") > 0).astype(int)
        base["label_put_positive"] = (pd.to_numeric(base["future_put_extrinsic_return"], errors="coerce") > 0).astype(int)
        base = base.dropna(subset=["future_call_extrinsic_return", "future_put_extrinsic_return"])
        _save_labels(base, horizon)

        dimension_rows = []
        for di, dimensions in enumerate(DIMENSION_CANDIDATES):
            progress = .30 + .50 * ((hi + di / len(DIMENSION_CANDIDATES)) / max(1, len(usable)))
            _notify(callback, progress, f"{horizon}: {dimensions}D kandidaat point-in-time valideren…")
            joined = attach_vectors(base, dimensions, raw_cache)
            if joined.empty:
                dimension_rows.append({"dimensions": dimensions, "status": "geen vectoren"})
                continue
            purge={"D1":1,"D2":2,"D3":3,"W1":5}.get(horizon,1)
            parts = _time_partitions(joined,purge_sessions=purge)
            features = _columns(joined, dimensions, with_news=True)
            call_brier = _selection_score(parts, features, "label_call_positive")
            put_brier = _selection_score(parts, features, "label_put_positive")
            score = float(np.mean([call_brier, put_brier]))
            dimension_rows.append({
                "dimensions": dimensions, "validation_call_brier": call_brier,
                "validation_put_brier": put_brier, "validation_mean_brier": score,
                "rows": len(joined), "days": int(joined["source_day"].nunique()), "status": "getest",
            })
            del joined,parts
            gc.collect()
        valid = [x for x in dimension_rows if np.isfinite(x.get("validation_mean_brier", math.inf))]
        if not valid:
            metrics["horizons"][horizon] = {"status": "geen trainbare dimensiekandidaat", "dimension_benchmark": dimension_rows}
            continue
        best = min(valid, key=lambda x: (x["validation_mean_brier"], x["dimensions"]))
        dimensions = int(best["dimensions"])
        joined=attach_vectors(base,dimensions,raw_cache)
        purge={"D1":1,"D2":2,"D3":3,"W1":5}.get(horizon,1)
        parts=_time_partitions(joined,purge_sessions=purge)
        full_features = _columns(joined, dimensions, with_news=True)
        no_news_features = _columns(joined, dimensions, with_news=False)

        call_metrics, call_model = _fit_and_test(parts, full_features, "label_call_positive")
        put_metrics, put_model = _fit_and_test(parts, full_features, "label_put_positive")
        call_return_metrics, call_return_model = _fit_regression(
            parts, full_features, "future_call_extrinsic_return"
        )
        put_return_metrics, put_return_model = _fit_regression(
            parts, full_features, "future_put_extrinsic_return"
        )
        call_no_news, _ = _fit_and_test(parts, no_news_features, "label_call_positive")
        put_no_news, _ = _fit_and_test(parts, no_news_features, "label_put_positive")
        metrics["horizons"][horizon] = {
            "status": "gereed", "selected_dimensions": dimensions,
            "dimension_benchmark": dimension_rows,
            "final_holdout_call": call_metrics, "final_holdout_put": put_metrics,
            "final_holdout_call_return": call_return_metrics,
            "final_holdout_put_return": put_return_metrics,
            "ablation_without_news_call": call_no_news,
            "ablation_without_news_put": put_no_news,
            "news_brier_gain_call": float(call_no_news["brier"] - call_metrics["brier"]),
            "news_brier_gain_put": float(put_no_news["brier"] - put_metrics["brier"]),
            "rows": len(joined), "ticker_days": int(joined[["source_ticker", "source_day"]].drop_duplicates().shape[0]),
            "future_values_used_as_features": False,
        }
        models[horizon] = {
            "dimensions": dimensions, "call": call_model, "put": put_model,
            "call_return_model": call_return_model,
            "put_return_model": put_return_model,
            "vector_version": "oe20-fusion-384-v1",
        }

    _notify(callback, .98, "Fusiebenchmark en finale hold-out afronden…")
    metrics["overall"] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "usable_horizons": list(models),
        "research_claim": "voorspellend verband; geen causaliteitsclaim",
        "quotes_included": False,
        "net_profit_claim_allowed": False,
    }
    return metrics, models


def serialize_bundle(bundle: dict[str, Any]) -> bytes:
    return pickle.dumps(bundle, protocol=pickle.HIGHEST_PROTOCOL)


def predict_fusion_rows(pair_frame: pd.DataFrame, raw_vector: dict[str, np.ndarray],
                        horizon_model: dict[str, Any]) -> pd.DataFrame:
    """Predict calibrated positive-outcome probabilities and expected value.

    `pair_frame` contains only information available at the selected day's
    close. The stored daily raw blocks are recomposed with the dimension count
    selected on validation data during training.
    """
    if pair_frame is None or pair_frame.empty:
        raise ValueError("De actuele Pair Store is leeg.")
    required = {"dimensions", "call", "put", "call_return_model", "put_return_model"}
    if not required.issubset(horizon_model):
        raise ValueError("Dit fusieartifact mist kans- of verwachtingswaardemodellen; train v20 opnieuw.")
    d = pair_frame.copy()
    missing_columns = sorted({"minute", "expiry", "strike", "call_extrinsic", "put_extrinsic"} - set(d))
    if missing_columns:
        raise ValueError("De actuele Pair Store mist verplichte kolommen: " + ", ".join(missing_columns))
    d["minute"] = pd.to_datetime(d.get("minute"), utc=True, errors="coerce")
    if "source_ticker" not in d:
        if "ticker" not in d:
            raise ValueError("Tickerkolom ontbreekt in de actuele Pair Store.")
        d["source_ticker"] = d["ticker"].astype(str).str.upper()
    if "source_day" not in d:
        d["source_day"] = d["minute"].dt.date
    close_parts=[]
    for source_day,group in d.groupby("source_day",sort=False):
        if pd.isna(source_day):
            continue
        close_parts.append(_daily_close_anchor(group,source_day,max_staleness_minutes=30))
    close=pd.concat(close_parts,ignore_index=True) if close_parts else d.iloc[0:0]
    if close.empty:
        raise ValueError("Geen call/put-paar met een waarneming in de laatste 30 minuten vóór de dagsluiting.")
    dimensions = int(horizon_model["dimensions"])
    vector = _candidate(raw_vector, dimensions)
    vector_frame = pd.DataFrame(
        np.tile(vector, (len(close), 1)),
        columns=[f"fusion_{index:03d}" for index in range(dimensions)],
        index=close.index,
    )
    close = pd.concat([close, vector_frame], axis=1)

    call = horizon_model["call"]
    put = horizon_model["put"]
    call_features = list(call["features"])
    put_features = list(put["features"])
    for feature in set(call_features + put_features):
        if feature not in close:
            close[feature] = np.nan
    call_probability = _probability(call["model"], call.get("calibrator"), close[call_features])
    put_probability = _probability(put["model"], put.get("calibrator"), close[put_features])
    call_return = np.clip(horizon_model["call_return_model"].predict(close[call_features]), -1, 3)
    put_return = np.clip(horizon_model["put_return_model"].predict(close[put_features]), -1, 3)
    call_now = pd.to_numeric(close.get("call_extrinsic"), errors="coerce").to_numpy(dtype=float)
    put_now = pd.to_numeric(close.get("put_extrinsic"), errors="coerce").to_numpy(dtype=float)

    close["call_positive_probability"] = call_probability
    close["put_positive_probability"] = put_probability
    close["expected_call_extrinsic_return"] = call_return
    close["expected_put_extrinsic_return"] = put_return
    close["model_expected_call_extrinsic"] = call_now * (1 + call_return)
    close["model_expected_put_extrinsic"] = put_now * (1 + put_return)
    close["call_gross_edge"] = close["model_expected_call_extrinsic"] - call_now
    close["put_gross_edge"] = close["model_expected_put_extrinsic"] - put_now
    close["best_side"] = np.where(call_probability >= put_probability, "CALL", "PUT")
    close["best_positive_probability"] = np.maximum(call_probability, put_probability)
    close["best_gross_edge"] = np.where(
        close["best_side"].eq("CALL"), close["call_gross_edge"], close["put_gross_edge"]
    )
    close["fusion_dimensions"] = dimensions
    return close.sort_values(
        ["best_positive_probability", "best_gross_edge"], ascending=[False, False]
    ).reset_index(drop=True)
