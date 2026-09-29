"""Build one auditable multimodal vector per ticker and trading day.

The 384D vector is a modelling contract, not a claim that 384 is universally
optimal.  Raw 1024D news embeddings and pre-projection market summaries remain
in the NPZ sidecar so dimension ablations can be fitted inside each training
fold without re-downloading source data.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from functools import lru_cache
from hashlib import blake2b, sha256
from io import BytesIO
import json
import math
from typing import Any, Iterable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from core.config import (
    DAILY_VECTOR_ROOT, FEATURE_ROOT, NEWS_EMBEDDING_ROOT, NEWS_EVENT_ROOT, PAIR_ROOT,
)
from core.research_design import RAW_NEWS_DIMENSIONS, VECTOR_BLOCKS
from core.storage import exists, get_bytes, get_df, get_json, put_bytes, put_df, put_json
from core.earnings import EARNINGS_VECTOR_FEATURES,earnings_vector_frame


VECTOR_VERSION = "oe20-fusion-384-v2-earnings"
FORBIDDEN_INPUT_TOKENS = (
    "future_", "target", "label", "realized_payoff", "realised_payoff",
    "next_", "forward_", "outcome",
)


@dataclass
class FusionResult:
    vector: np.ndarray
    news_raw: np.ndarray
    news_event_raw: np.ndarray
    options_raw: np.ndarray
    stock_raw: np.ndarray
    expectation_raw: np.ndarray
    earnings_raw: np.ndarray
    quality_raw: np.ndarray
    metadata: dict[str, Any]


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def market_close_utc(day: date) -> datetime:
    """Regular US close with DST handled by the IANA timezone database."""
    return datetime(day.year, day.month, day.day, 16, 0, tzinfo=ZoneInfo("America/New_York")).astimezone(timezone.utc)


def _stable_seed(name: str) -> int:
    return int.from_bytes(blake2b(name.encode("utf-8"), digest_size=8).digest(), "little")


@lru_cache(maxsize=64)
def _projection_matrix(source_dim: int, target_dim: int, name: str) -> np.ndarray:
    """Deterministic sparse random projection for the stored baseline vector.

    Learned PCA/autoencoder projections are fitted later *inside* each temporal
    training fold.  This projection only provides a reproducible, immediately
    inspectable 384D baseline and never sees a target.
    """
    if source_dim <= 0 or target_dim <= 0:
        return np.zeros((max(0, source_dim), max(0, target_dim)), dtype=np.float32)
    rng = np.random.default_rng(_stable_seed(f"{VECTOR_VERSION}:{name}:{source_dim}:{target_dim}"))
    # Achlioptas-style sparse projection: mostly zero, symmetric non-zero values.
    u = rng.random((source_dim, target_dim), dtype=np.float32)
    mat = np.zeros_like(u, dtype=np.float32)
    scale = math.sqrt(3.0 / max(1, target_dim))
    mat[u < (1 / 6)] = scale
    mat[u > (5 / 6)] = -scale
    return mat


def _finite(values: Iterable[Any]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=np.float32).reshape(-1)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def _compress(values: Iterable[Any], dimensions: int, name: str) -> np.ndarray:
    x = _finite(values)
    if dimensions <= 0:
        return np.empty(0, dtype=np.float32)
    if x.size == 0:
        return np.zeros(dimensions, dtype=np.float32)
    if x.size == dimensions:
        y = x.astype(np.float32, copy=True)
    else:
        y = x @ _projection_matrix(x.size, dimensions, name)
    # Bound pathological source values while retaining sign and relative scale.
    scale = float(np.sqrt(np.mean(np.square(y))) + 1e-6)
    return np.tanh(y / (3.0 * scale)).astype(np.float32)


def _safe_numeric_columns(frame: pd.DataFrame, preferred: Iterable[str] | None = None) -> list[str]:
    if frame is None or frame.empty:
        return []
    candidates = list(preferred) if preferred is not None else list(frame.columns)
    out = []
    for col in candidates:
        if col not in frame:
            continue
        low = str(col).lower()
        if any(token in low for token in FORBIDDEN_INPUT_TOKENS):
            continue
        series = pd.to_numeric(frame[col], errors="coerce")
        if series.notna().any():
            out.append(col)
    return out


def _slope(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if x.size < 2:
        return 0.0
    t = np.linspace(-1.0, 1.0, x.size, dtype=np.float64)
    denom = float(np.dot(t, t))
    return float(np.dot(t, x - np.mean(x)) / denom) if denom else 0.0


def summarize_frame(frame: pd.DataFrame, preferred: Iterable[str] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    """Create a stable raw summary without looking at future/target columns."""
    columns = _safe_numeric_columns(frame, preferred)
    values: list[float] = []
    used: list[str] = []
    missing_cells = 0
    total_cells = 0
    for col in columns:
        s = pd.to_numeric(frame[col], errors="coerce")
        total_cells += len(s)
        missing_cells += int(s.isna().sum())
        x = s.dropna().to_numpy(dtype=np.float64)
        if x.size == 0:
            continue
        used.append(col)
        values.extend([
            float(np.mean(x)), float(np.std(x)), float(np.min(x)), float(np.max(x)),
            float(np.median(x)), float(np.quantile(x, 0.10)), float(np.quantile(x, 0.90)),
            float(x[-1]), _slope(x),
        ])
    return _finite(values), {
        "rows": int(0 if frame is None else len(frame)),
        "columns": used,
        "raw_dimensions": len(values),
        "missing_fraction": (missing_cells / total_cells) if total_cells else 1.0,
    }


def _softmax(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    z = np.clip(x - np.max(x), -30, 30)
    e = np.exp(z)
    return e / max(float(np.sum(e)), 1e-12)


def pool_news_embeddings(embeddings: np.ndarray, quality: np.ndarray | None = None,
                         relevance: np.ndarray | None = None, importance: np.ndarray | None = None,
                         novelty: np.ndarray | None = None, recency: np.ndarray | None = None
                         ) -> tuple[np.ndarray, dict[str, Any]]:
    emb = np.asarray(embeddings, dtype=np.float32)
    if emb.ndim != 2 or emb.shape[0] == 0:
        return np.zeros(RAW_NEWS_DIMENSIONS, dtype=np.float32), {
            "chunks": 0, "embedding_dimensions": RAW_NEWS_DIMENSIONS, "effective_chunks": 0.0,
        }
    if emb.shape[1] != RAW_NEWS_DIMENSIONS:
        raise ValueError(f"Nieuws-embedding heeft {emb.shape[1]} dimensies; verwacht {RAW_NEWS_DIMENSIONS}.")
    n = emb.shape[0]

    def signal(value: np.ndarray | None, default: float) -> np.ndarray:
        if value is None:
            return np.full(n, default, dtype=np.float32)
        out = np.asarray(value, dtype=np.float32).reshape(-1)
        if out.size != n:
            return np.full(n, default, dtype=np.float32)
        return np.nan_to_num(out, nan=default, posinf=default, neginf=default)

    q = np.clip(signal(quality, 0.5), 0, 1)
    r = np.clip(signal(relevance, 0.5), 0, 1)
    i = np.clip(signal(importance, 0.5), 0, 1)
    v = np.clip(signal(novelty, 0.5), 0, 1)
    c = np.clip(signal(recency, 0.5), 0, 1)
    logits = 1.4 * q + 1.8 * r + 1.5 * i + 0.8 * v + 0.5 * c
    weights = _softmax(logits)
    pooled = np.sum(emb * weights[:, None], axis=0)
    norm = float(np.linalg.norm(pooled))
    if norm > 0:
        pooled = pooled / norm
    entropy = -float(np.sum(weights * np.log(np.maximum(weights, 1e-12))))
    return pooled.astype(np.float32), {
        "chunks": int(n),
        "embedding_dimensions": int(emb.shape[1]),
        "effective_chunks": float(np.exp(entropy)),
        "max_attention": float(np.max(weights)),
        "mean_quality": float(np.mean(q)),
        "mean_relevance": float(np.mean(r)),
        "mean_importance": float(np.mean(i)),
        "mean_novelty": float(np.mean(v)),
    }


EVENT_CATEGORY_VALUES = {
    "event_type": ("earnings", "guidance", "mna", "regulatory", "legal", "analyst",
                   "product", "management", "capital", "macro", "operations", "other"),
    "direction": ("positive", "negative", "mixed", "neutral", "unclear"),
    "market_impact": ("low", "medium", "high", "unknown"),
    "time_horizon": ("intraday", "days", "weeks", "months", "long_term", "unknown"),
    "expected_vs_surprise": ("expected", "surprise", "unclear"),
}
EVENT_FEATURE_NAMES = tuple(
    [f"{field}_{value}_share" for field, values in EVENT_CATEGORY_VALUES.items() for value in values]
    + [f"{field}_{value}_importance_share" for field, values in EVENT_CATEGORY_VALUES.items() for value in values]
    + [f"{field}_{stat}" for field in ["ticker_relevance", "importance", "uncertainty", "novelty"]
       for stat in ["mean", "max"]]
    + ["event_count_log", "facts_mean", "causal_chain_mean", "llm_error_share"]
)


def summarize_news_events(payload: dict[str, Any] | None, as_of_utc: datetime
                          ) -> tuple[np.ndarray, dict[str, Any]]:
    """Encode Qwen event JSON without targets and enforce its availability time."""
    source_events = list((payload or {}).get("events") or [])
    events = []
    excluded_after_asof = 0
    excluded_missing_time = 0
    publication_time_fallbacks = 0
    for event in source_events:
        raw_time = event.get("available_utc")
        if not raw_time:
            raw_time = event.get("published_utc")
            publication_time_fallbacks += 1
        timestamp = pd.to_datetime(raw_time, utc=True, errors="coerce")
        if pd.isna(timestamp):
            excluded_missing_time += 1
            continue
        if timestamp.to_pydatetime() > as_of_utc:
            excluded_after_asof += 1
            continue
        events.append(event)
    if not events:
        return np.zeros(len(EVENT_FEATURE_NAMES), dtype=np.float32), {
            "events": 0,"source_events": len(source_events),
            "raw_dimensions": len(EVENT_FEATURE_NAMES),
            "excluded_after_asof": excluded_after_asof,
            "excluded_missing_time": excluded_missing_time,
            "publication_time_fallbacks": publication_time_fallbacks,
        }

    n = float(len(events))
    importance = np.asarray([
        float(np.clip(pd.to_numeric(event.get("importance", .5), errors="coerce"), 0, 1))
        if pd.notna(pd.to_numeric(event.get("importance", .5), errors="coerce")) else .5
        for event in events
    ], dtype=np.float32)
    importance_total = max(float(importance.sum()), 1e-8)
    values: list[float] = []
    for field, categories in EVENT_CATEGORY_VALUES.items():
        observed = [str(event.get(field, categories[-1])).lower() for event in events]
        values.extend(sum(value == category for value in observed) / n for category in categories)
    for field, categories in EVENT_CATEGORY_VALUES.items():
        observed = [str(event.get(field, categories[-1])).lower() for event in events]
        values.extend(
            float(importance[[value == category for value in observed]].sum()) / importance_total
            for category in categories
        )
    for field in ["ticker_relevance", "importance", "uncertainty", "novelty"]:
        numeric = np.asarray([
            float(pd.to_numeric(event.get(field, .5), errors="coerce"))
            if pd.notna(pd.to_numeric(event.get(field, .5), errors="coerce")) else .5
            for event in events
        ], dtype=np.float32)
        numeric = np.clip(numeric, 0, 1)
        values.extend([float(numeric.mean()), float(numeric.max())])
    values.extend([
        min(1.0, math.log1p(len(events)) / math.log(101)),
        float(np.mean([len(event.get("facts") or []) for event in events])) / 8.0,
        float(np.mean([len(event.get("causal_chain") or []) for event in events])) / 6.0,
        float(np.mean([bool(event.get("llm_error")) for event in events])),
    ])
    vector = _finite(values)
    if vector.size != len(EVENT_FEATURE_NAMES):
        raise AssertionError(f"LLM-eventsamenvatting heeft {vector.size} in plaats van {len(EVENT_FEATURE_NAMES)} dimensies.")
    return vector, {
        "events": len(events),"source_events": len(source_events),
        "raw_dimensions": len(EVENT_FEATURE_NAMES),
        "excluded_after_asof": excluded_after_asof,
        "excluded_missing_time": excluded_missing_time,
        "publication_time_fallbacks": publication_time_fallbacks,
    }


def _load_news_events(ticker: str, day: date, as_of_utc: datetime
                      ) -> tuple[np.ndarray, dict[str, Any]]:
    key = f"{NEWS_EVENT_ROOT}/{ticker}/{day:%Y}/{day:%m}/{day}.json.gz"
    if not exists(key):
        return np.zeros(len(EVENT_FEATURE_NAMES), dtype=np.float32), {
            "key":key,"present":False,"events":0,"raw_dimensions":len(EVENT_FEATURE_NAMES),
        }
    try:
        vector, metadata = summarize_news_events(get_json(key), as_of_utc)
        return vector, {**metadata,"key":key,"present":True,"as_of_utc":as_of_utc.isoformat()}
    except Exception as exc:
        return np.zeros(len(EVENT_FEATURE_NAMES), dtype=np.float32), {
            "key":key,"present":True,"events":0,"raw_dimensions":len(EVENT_FEATURE_NAMES),
            "error":f"{type(exc).__name__}: {exc}",
        }


def _load_news_npz(ticker: str, day: date, as_of_utc: datetime) -> tuple[np.ndarray, dict[str, Any]]:
    key = f"{NEWS_EMBEDDING_ROOT}/{ticker}/{day:%Y}/{day:%m}/{day}.npz"
    if not exists(key):
        return np.empty((0, RAW_NEWS_DIMENSIONS), dtype=np.float32), {"key": key, "present": False}
    raw = get_bytes(key)
    with np.load(BytesIO(raw), allow_pickle=False) as archive:
        embeddings = np.asarray(archive["embeddings"], dtype=np.float32)
        kwargs = {name: np.asarray(archive[name]) for name in ["quality", "relevance", "importance", "novelty", "recency"] if name in archive.files}
        excluded_after_asof = 0
        availability_basis = "missing"
        availability = None
        if "available_unix" in archive.files:
            availability = np.asarray(archive["available_unix"], dtype=np.float64).reshape(-1)
            availability_basis = "effective_available_at"
        elif "published_unix" in archive.files and "first_seen_unix" in archive.files:
            published = np.asarray(archive["published_unix"], dtype=np.float64).reshape(-1)
            first_seen = np.asarray(archive["first_seen_unix"], dtype=np.float64).reshape(-1)
            if published.size == first_seen.size:
                availability = np.maximum(published, first_seen)
                availability_basis = "max_published_first_seen"
        elif "published_unix" in archive.files:
            # Compatibility for pre-v20-agent files. This is reported as a
            # weaker timestamp basis in metadata rather than silently claiming
            # that local first-seen was audited.
            availability = np.asarray(archive["published_unix"], dtype=np.float64).reshape(-1)
            availability_basis = "legacy_publication_only"
        if availability is not None:
            keep = availability <= as_of_utc.timestamp()
            if keep.size == embeddings.shape[0]:
                excluded_after_asof = int((~keep).sum())
                embeddings = embeddings[keep]
                kwargs = {name: values[keep] if values.shape[0] == keep.size else values for name, values in kwargs.items()}
    pooled, meta = pool_news_embeddings(embeddings, **kwargs)
    return pooled, {**meta, "key": key, "present": True,
                    "as_of_utc": as_of_utc.isoformat(),
                    "availability_basis": availability_basis,
                    "excluded_after_asof": excluded_after_asof}


OPTION_PREFERRED = [
    "call_close_option", "put_close_option", "call_extrinsic", "put_extrinsic",
    "call_option_volume", "put_option_volume", "call_trade_count", "put_trade_count",
    "call_trade_volume", "put_trade_volume", "call_oi_open_interest", "put_oi_open_interest",
    "call_snapshot_iv", "put_snapshot_iv", "call_snapshot_delta", "put_snapshot_delta",
    "call_snapshot_gamma", "put_snapshot_gamma", "call_snapshot_theta", "put_snapshot_theta",
    "call_snapshot_vega", "put_snapshot_vega", "call_put_volume_ratio",
    "call_put_trade_volume_ratio", "call_put_extrinsic_ratio", "dte", "strike", "close_stock",
]

STOCK_PREFERRED = [
    "close_stock", "stock_return_1m", "stock_return_5m", "stock_return_30m", "stock_return_60m",
    "realized_vol_30m", "realized_vol_60m", "sma_10_ratio", "sma_30_ratio", "macd_ratio",
    "rsi14", "stock_volume_z30", "intraday_range", "event_stock_breakout_up_30m",
    "event_stock_breakdown_30m", "event_high_realized_vol", "event_stock_volume_spike",
]

EXPECTATION_PREFERRED = [
    "bs_call_price", "bs_put_price", "bs_call_extrinsic", "bs_put_extrinsic",
    "theoretical_call_expected_value", "theoretical_put_expected_value",
    "market_call_expected_value", "market_put_expected_value",
    "practical_call_time_value", "practical_put_time_value",
    "call_theory_market_gap", "put_theory_market_gap",
    "call_close_option", "put_close_option", "call_extrinsic", "put_extrinsic",
    "call_expectation_gap", "put_expectation_gap", "put_call_parity_residual",
    "call_put_extrinsic_ratio", "benchmark_sigma", "risk_free_rate", "dte", "strike", "close_stock",
]


def _load_sources(ticker: str, day: date, horizon_minutes: int = 30
                  ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    feature_key = f"{FEATURE_ROOT}/{ticker}/{day:%Y}/{day:%m}/{day}.csv.gz"
    pair_key = f"{PAIR_ROOT}/h{horizon_minutes}/{ticker}/{day:%Y}/{day:%m}/{day}.csv.gz"
    features = get_df(feature_key) if exists(feature_key) else pd.DataFrame()
    pairs = get_df(pair_key) if exists(pair_key) else pd.DataFrame()
    return features, pairs, {"features": feature_key, "pairs": pair_key}


def _at_or_before(frame: pd.DataFrame, as_of_utc: datetime,
                  candidates: tuple[str, ...]) -> pd.DataFrame:
    if frame is None or frame.empty:
        return frame
    d=frame.copy()
    for column in candidates:
        if column in d:
            timestamp=pd.to_datetime(d[column],utc=True,errors="coerce")
            return d[timestamp.notna() & (timestamp<=as_of_utc)].copy()
    return d


def _quality_vector(news_meta: dict[str, Any], feature_meta: dict[str, Any], pair_meta: dict[str, Any],
                    has_expectation: bool, quote_available: bool = False) -> np.ndarray:
    # Fixed semantic coordinates; do not random-project this block.
    return _finite([
        1.0 if news_meta.get("present") else 0.0,
        1.0 if feature_meta.get("rows", 0) else 0.0,
        1.0 if pair_meta.get("rows", 0) else 0.0,
        1.0 if has_expectation else 0.0,
        min(1.0, math.log1p(news_meta.get("chunks", 0)) / math.log(101)),
        min(1.0, math.log1p(feature_meta.get("rows", 0)) / math.log(10001)),
        min(1.0, math.log1p(pair_meta.get("rows", 0)) / math.log(10001)),
        float(news_meta.get("mean_quality", 0.0) or 0.0),
        float(news_meta.get("mean_relevance", 0.0) or 0.0),
        float(news_meta.get("mean_importance", 0.0) or 0.0),
        float(news_meta.get("mean_novelty", 0.0) or 0.0),
        1.0 - float(feature_meta.get("missing_fraction", 1.0) or 1.0),
        1.0 - float(pair_meta.get("missing_fraction", 1.0) or 1.0),
        float(news_meta.get("max_attention", 0.0) or 0.0),
        1.0 if quote_available else 0.0,
        1.0,  # schema/version mask; zero is reserved for legacy vectors.
    ])


def candidate_layout(dimensions: int) -> dict[str, int]:
    if dimensions < 32:
        raise ValueError("Een fusievector moet minimaal 32 dimensies hebben.")
    ratios = {"news": 0.50, "options": 0.25, "stock": 0.125, "expectation": 1/12}
    layout = {key: int(round(dimensions * ratio)) for key, ratio in ratios.items()}
    layout["quality"] = dimensions - sum(layout.values())
    if layout["quality"] < 8:
        take = 8 - layout["quality"]
        layout["news"] -= take
        layout["quality"] = 8
    return layout


def build_daily_vector(ticker: str, day: date, horizon_minutes: int = 30,
                       dimensions: int = 384) -> FusionResult:
    ticker = ticker.strip().upper()
    as_of = market_close_utc(day)
    news_raw, news_meta = _load_news_npz(ticker, day, as_of)
    if news_raw.ndim == 2:  # missing-news path
        news_raw = np.zeros(RAW_NEWS_DIMENSIONS, dtype=np.float32)
    news_event_raw, event_meta = _load_news_events(ticker, day, as_of)
    features, pairs, keys = _load_sources(ticker, day, horizon_minutes)
    features=_at_or_before(features,as_of,("ts","minute","as_of_utc"))
    pairs=_at_or_before(pairs,as_of,("minute","ts","as_of_utc"))

    if not news_meta.get("present") and features.empty and pairs.empty:
        raise FileNotFoundError(f"Geen point-in-time brondata voor {ticker} op {day}.")

    options_raw, options_meta = summarize_frame(pairs, OPTION_PREFERRED)
    stock_source = features if not features.empty else pairs
    stock_raw, stock_meta = summarize_frame(stock_source, STOCK_PREFERRED)
    expectation_raw, expectation_meta = summarize_frame(pairs, EXPECTATION_PREFERRED)
    earnings_frame, earnings_source_meta = earnings_vector_frame(ticker, as_of)
    earnings_raw, earnings_meta = summarize_frame(earnings_frame, EARNINGS_VECTOR_FEATURES)
    earnings_meta = {**earnings_meta, **earnings_source_meta}
    has_expectation = bool(expectation_meta.get("columns") or earnings_meta.get("columns"))
    quality_raw = _quality_vector(news_meta, stock_meta, options_meta, has_expectation)

    layout = candidate_layout(dimensions)
    blocks = {
        "news": _compress(np.concatenate([news_raw, news_event_raw]), layout["news"], "news"),
        "options": _compress(options_raw, layout["options"], "options"),
        "stock": _compress(stock_raw, layout["stock"], "stock"),
        "expectation": _compress(
            np.concatenate([expectation_raw, earnings_raw]), layout["expectation"], "expectation"
        ),
        "quality": _compress(quality_raw, layout["quality"], "quality"),
    }
    vector = np.concatenate([blocks[x] for x in ["news", "options", "stock", "expectation", "quality"]])
    if vector.size != dimensions:
        raise AssertionError(f"Vector heeft {vector.size} dimensies in plaats van {dimensions}.")

    metadata = {
        "ticker": ticker,
        "date": str(day),
        "as_of_utc": as_of.isoformat(),
        "created_at": utcnow().isoformat(),
        "vector_version": VECTOR_VERSION,
        "dimensions": dimensions,
        "layout": layout,
        "source_keys": {**keys, "news_embeddings": news_meta.get("key"),
                        "news_events": event_meta.get("key")},
        "news": news_meta,
        "news_events": event_meta,
        "stock": stock_meta,
        "options": options_meta,
        "expectation": expectation_meta,
        "earnings": earnings_meta,
        "quality_mask": quality_raw.tolist(),
        "future_values_used_as_features": False,
    }
    metadata["schema_hash"] = sha256(json.dumps({
        "version": VECTOR_VERSION, "layout": layout,
        "stock_columns": stock_meta.get("columns"),
        "option_columns": options_meta.get("columns"),
        "expectation_columns": expectation_meta.get("columns"),
        "earnings_columns": earnings_meta.get("columns"),
    }, sort_keys=True).encode("utf-8")).hexdigest()

    return FusionResult(
        vector=vector.astype(np.float32), news_raw=news_raw.astype(np.float32),
        news_event_raw=news_event_raw.astype(np.float32),
        options_raw=options_raw, stock_raw=stock_raw,
        expectation_raw=expectation_raw, earnings_raw=earnings_raw, quality_raw=quality_raw,
        metadata=metadata,
    )


def _npz_bytes(result: FusionResult) -> bytes:
    buffer = BytesIO()
    np.savez_compressed(
        buffer,
        vector=result.vector,
        news_raw=result.news_raw,
        news_event_raw=result.news_event_raw,
        options_raw=result.options_raw,
        stock_raw=result.stock_raw,
        expectation_raw=result.expectation_raw,
        earnings_raw=result.earnings_raw,
        quality_raw=result.quality_raw,
    )
    return buffer.getvalue()


def vector_key(ticker: str, day: date, suffix: str = "csv.gz") -> str:
    return f"{DAILY_VECTOR_ROOT}/{ticker}/{day:%Y}/{day:%m}/{day}.{suffix}"


def save_daily_vector(ticker: str, day: date, horizon_minutes: int = 30) -> dict[str, Any]:
    result = build_daily_vector(ticker, day, horizon_minutes=horizon_minutes, dimensions=384)
    row = {
        "ticker": ticker.upper(), "date": str(day), "as_of_utc": result.metadata["as_of_utc"],
        "vector_version": VECTOR_VERSION, "schema_hash": result.metadata["schema_hash"],
        **{f"vector_{i:03d}": float(x) for i, x in enumerate(result.vector)},
    }
    csv_key = vector_key(ticker.upper(), day)
    npz_key = vector_key(ticker.upper(), day, "npz")
    meta_key = vector_key(ticker.upper(), day, "json")
    put_df(csv_key, pd.DataFrame([row]))
    put_bytes(npz_key, _npz_bytes(result), "application/octet-stream")
    put_json(meta_key, result.metadata)
    return {"csv_key": csv_key, "npz_key": npz_key, "meta_key": meta_key, **result.metadata}


def candidate_from_raw(news_raw: np.ndarray, options_raw: np.ndarray, stock_raw: np.ndarray,
                       expectation_raw: np.ndarray, quality_raw: np.ndarray,
                       dimensions: int, news_event_raw: np.ndarray | None = None,
                       earnings_raw: np.ndarray | None = None,
                       disabled_components: Iterable[str] | None = None) -> np.ndarray:
    """Recompose a candidate, including exact target-free feature ablations.

    Supported ablations are ``news``, ``embeddings``, ``options``, ``stock`` /
    ``technical``, ``expectation`` / ``theoretical`` and ``earnings``.  The
    output shape remains identical so every comparison can use the same folds.
    """
    disabled={str(value).strip().lower() for value in (disabled_components or [])}
    layout = candidate_layout(dimensions)
    news_values=[] if ("news" in disabled or "embeddings" in disabled) else news_raw
    event_values=[] if "news" in disabled else (news_event_raw if news_event_raw is not None else [])
    combined_news = np.concatenate([
        _finite(news_values),
        _finite(event_values),
    ])
    option_values=[] if "options" in disabled else options_raw
    stock_values=[] if ({"stock","technical"} & disabled) else stock_raw
    expectation_values=[] if ({"expectation","theoretical"} & disabled) else expectation_raw
    earnings_values=[] if "earnings" in disabled else (earnings_raw if earnings_raw is not None else [])
    blocks = [
        _compress(combined_news, layout["news"], f"news-d{dimensions}"),
        _compress(option_values, layout["options"], f"options-d{dimensions}"),
        _compress(stock_values, layout["stock"], f"stock-d{dimensions}"),
        _compress(
            np.concatenate([_finite(expectation_values), _finite(earnings_values)]),
            layout["expectation"], f"expectation-d{dimensions}",
        ),
        _compress(quality_raw, layout["quality"], f"quality-d{dimensions}"),
    ]
    vector = np.concatenate(blocks).astype(np.float32)
    if vector.size != dimensions:
        raise AssertionError(f"Candidate heeft {vector.size} in plaats van {dimensions} dimensies.")
    return vector


def load_raw_vector(ticker: str, day: date) -> dict[str, np.ndarray]:
    key = vector_key(ticker.upper(), day, "npz")
    raw = get_bytes(key)
    with np.load(BytesIO(raw), allow_pickle=False) as archive:
        return {name: np.asarray(archive[name], dtype=np.float32) for name in archive.files}


def primary_layout_is_valid() -> bool:
    return sum(block.dimensions for block in VECTOR_BLOCKS) == 384
