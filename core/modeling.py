from __future__ import annotations
from datetime import timedelta
import pickle, math, re
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, brier_score_loss

from core.config import FEATURE_ROOT
from core.storage import get_df, list_keys
from core.quotes import QUOTE_FEATURES

FEATURE_SETS = {
    "A": [
        "stock_return_1m","stock_return_5m","stock_return_30m","stock_return_60m",
        "realized_vol_30m","realized_vol_60m","sma_10_ratio","sma_30_ratio",
        "macd_ratio","rsi14","stock_volume_z30","intraday_range"
    ],
    "B": [
        "stock_return_1m","stock_return_5m","stock_return_30m","stock_return_60m",
        "realized_vol_30m","realized_vol_60m","sma_10_ratio","sma_30_ratio",
        "macd_ratio","rsi14","stock_volume_z30","intraday_range",
        "is_call","dte","strike","log_moneyness","close_option","intrinsic","extrinsic",
        "option_volume","option_transactions","option_return_1m","option_return_5m",
        "option_return_30m","trade_count","trade_volume","trade_dollar_volume",
        "max_trade_size","trade_vwap"
    ],
    "C1": [
        "stock_return_1m","stock_return_5m","stock_return_30m","stock_return_60m",
        "realized_vol_30m","realized_vol_60m","sma_10_ratio","sma_30_ratio",
        "macd_ratio","rsi14","stock_volume_z30","intraday_range",
        "is_call","dte","strike","log_moneyness","close_option","intrinsic","extrinsic",
        "option_volume","option_transactions","option_return_1m","option_return_5m",
        "option_return_30m","trade_count","trade_volume","trade_dollar_volume",
        "max_trade_size","trade_vwap",
        "oi_open_interest","snapshot_iv","snapshot_delta","snapshot_gamma",
        "snapshot_theta","snapshot_vega","days_to_earnings","days_since_earnings",
        "eps_surprise_percent","revenue_surprise_percent",
        "sentiment_latest","sentiment_15m","sentiment_60m","news_count_15m",
        "news_count_60m","emotion_60m","surprise_60m"
    ]
}
FEATURE_SETS["C2"] = FEATURE_SETS["C1"] + QUOTE_FEATURES

KEY_RE = re.compile(
    re.escape(FEATURE_ROOT) + r"/([^/]+)/(\d{4})/(\d{2})/(\d{4}-\d{2}-\d{2})\.csv\.gz$"
)

def _notify(callback, progress, message):
    if callback:
        callback(float(progress), str(message))

def feature_catalog(tickers, start, end, callback=None):
    wanted = set(tickers)
    rows = []
    seen_objects = 0
    _notify(callback, 0.02, "Featurecatalogus in Hetzner opbouwen…")

    # A LIST operation is much faster than HEAD for every ticker/day combination.
    for key in list_keys(FEATURE_ROOT + "/"):
        seen_objects += 1
        m = KEY_RE.match(key)
        if not m:
            continue
        ticker, _, _, ds = m.groups()
        if ticker not in wanted:
            continue
        try:
            day = pd.Timestamp(ds).date()
        except Exception:
            continue
        if start <= day <= end:
            rows.append((ticker, day, key))

        if seen_objects % 2000 == 0:
            _notify(
                callback, 0.03,
                f"Featurecatalogus: {seen_objects:,} objecten bekeken • {len(rows):,} bruikbaar"
            )

    rows.sort(key=lambda x: (x[1], x[0]))
    return rows

def _even_pick(items, n):
    if len(items) <= n:
        return list(items)
    idx = np.linspace(0, len(items)-1, n, dtype=int)
    out = []
    used = set()
    for i in idx:
        i = int(i)
        if i not in used:
            out.append(items[i])
            used.add(i)
    return out

def select_feature_files(catalog, tickers, max_rows):
    if not catalog:
        return []

    # Approximately 250 sampled rows per file.
    # 100k rows => ~400 ticker/day files rather than potentially >20,000 GETs.
    target_files = max(100, min(1000, int(math.ceil(max_rows / 250))))

    by_ticker = {t: [] for t in tickers}
    for item in catalog:
        if item[0] in by_ticker:
            by_ticker[item[0]].append(item)

    active = [t for t, vals in by_ticker.items() if vals]
    if not active:
        return []

    per_ticker = max(1, target_files // len(active))
    selected = []
    for ticker in active:
        selected.extend(_even_pick(by_ticker[ticker], per_ticker))

    # Top up if some tickers have fewer available days.
    target = min(target_files, len(catalog))
    if len(selected) < target:
        used = {x[2] for x in selected}
        remaining = [x for x in catalog if x[2] not in used]
        selected.extend(_even_pick(remaining, target-len(selected)))

    return sorted(selected, key=lambda x: (x[1], x[0]))

def load_training_rows(
    tickers, start, end, max_rows=100000, callback=None,
    use_all_available=True, target=None,
):
    catalog = feature_catalog(tickers, start, end, callback)
    if not catalog:
        raise RuntimeError(
            "Geen Feature Store-bestanden gevonden voor deze tickers/periode. "
            "Draai eerst Data Agents → Technische + model features."
        )

    # Full-data mode is the OptionEdge default.  The old representative sampler
    # remains available only for explicit backwards-compatible/debug runs.
    selected = list(catalog) if use_all_available else select_feature_files(catalog, tickers, max_rows)
    if not selected:
        raise RuntimeError("Er zijn geen bruikbare Feature Store-bestanden geselecteerd.")

    rows_per_file = None if use_all_available else max(40, int(math.ceil(max_rows * 1.08 / len(selected))))
    _notify(
        callback, 0.05,
        f"{len(catalog):,} featurebestanden beschikbaar • "
        + (f"alle {len(selected):,} bestanden worden volledig gebruikt."
           if use_all_available else f"{len(selected):,} representatieve bestanden geselecteerd.")
    )

    # Parse only columns any legacy model can actually consume.  A callable
    # usecols is tolerant of older feature files that do not contain every new
    # feature yet, while still preserving every ROW from every selected file.
    required={"ts"}
    if target:
        required.add(str(target))
    for cols in FEATURE_SETS.values():
        required.update(cols)
    usecols=lambda c: c in required

    frames = []
    loaded = 0
    total = len(selected)

    for i, (ticker, day, key) in enumerate(selected, 1):
        try:
            frame = get_df(key,usecols=usecols)
            if frame.empty:
                continue

            frame["source_ticker"] = ticker

            # Sampling is now opt-in only.  Production/full-data runs retain
            # every imported eligible observation.
            if rows_per_file is not None and len(frame) > rows_per_file:
                idx = np.linspace(0, len(frame)-1, rows_per_file, dtype=int)
                frame = frame.iloc[idx].copy()

            frames.append(frame)
            loaded += len(frame)
        except Exception:
            # A single damaged/missing object should not kill a multi-year run.
            continue

        if i == 1 or i % 5 == 0 or i == total:
            progress = 0.06 + 0.44 * (i / total)
            _notify(
                callback,
                progress,
                f"Feature Store laden: {i:,}/{total:,} bestanden • "
                f"{loaded:,} regels geladen"
            )

    if not frames:
        raise RuntimeError("Feature Store-bestanden gevonden, maar geen ervan kon worden gelezen.")

    df = pd.concat(frames, ignore_index=True)
    del frames

    if "ts" not in df.columns:
        raise RuntimeError("Feature Store bevat geen 'ts'-kolom.")

    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts"]).sort_values("ts")

    if not use_all_available and len(df) > max_rows:
        idx = np.linspace(0, len(df)-1, max_rows, dtype=int)
        df = df.iloc[idx].copy()

    df.attrs["data_usage"]={
        "catalog_files":int(len(catalog)),
        "loaded_files":int(len(selected)),
        "loaded_rows":int(len(df)),
        "use_all_available":bool(use_all_available),
        "hidden_row_cap_applied":bool(not use_all_available and len(df)>=max_rows),
    }

    _notify(
        callback,
        0.51,
        f"Feature Store gereed: {len(df):,} trainingsregels uit "
        f"{len(selected):,} ticker/dag-bestanden"
        + (" • geen sampling/row-cap." if use_all_available else ".")
    )
    return df

def _metrics(y, pred, probability=None):
    out = {
        "rows": int(len(y)),
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(mean_squared_error(y, pred) ** 0.5),
        "r2": float(r2_score(y, pred)),
        "direction_accuracy": float(np.mean((pred >= 0) == (y >= 0))),
        "correlation": (
            float(np.corrcoef(y, pred)[0,1])
            if len(y) > 2 and np.std(pred) > 0 and np.std(y) > 0
            else None
        )
    }
    if probability is not None:
        out["brier"] = float(brier_score_loss((y >= 0).astype(int), probability))
    return out

def train_all(df, target="target_extrinsic_return_30m", callback=None):
    if target not in df.columns:
        raise RuntimeError(f"Target ontbreekt in de Feature Store: {target}")

    df = df.dropna(subset=[target]).sort_values("ts").copy()
    if len(df) < 2000:
        raise RuntimeError(
            f"Te weinig bruikbare trainingsregels ({len(df):,}). "
            "Bouw eerst meer feature-dagen."
        )

    cut = int(len(df) * 0.75)
    validation_start = df.iloc[cut]["ts"]
    purge_start = validation_start - pd.Timedelta(days=1)

    train = df[df["ts"] < purge_start]
    valid = df[df["ts"] >= validation_start]
    if len(train) < 1000 or len(valid) < 300:
        train = df.iloc[:cut]
        valid = df.iloc[cut:]

    results = {}
    artifacts = {}
    stages = list(FEATURE_SETS.items())

    for stage_index, (stage, features) in enumerate(stages):
        base = 0.53 + stage_index * 0.09
        _notify(callback, base, f"Model {stage} voorbereiden…")

        present = [c for c in features if c in df.columns]

        if stage == "C2":
            quote_present = [
                c for c in QUOTE_FEATURES
                if c in df.columns and df[c].notna().any()
            ]
            if not quote_present:
                results[stage] = {
                    "status": "skipped",
                    "reason": "Quotes zijn nog niet beschikbaar; C2 blijft gereserveerd."
                }
                _notify(callback, base + 0.07, "Model C2 overgeslagen: quote-laag is nog leeg.")
                continue

        if not present:
            results[stage] = {"status":"skipped","reason":"Geen features beschikbaar."}
            continue

        X_train = train[present].apply(pd.to_numeric, errors="coerce")
        X_valid = valid[present].apply(pd.to_numeric, errors="coerce")
        y_train = pd.to_numeric(train[target], errors="coerce")
        y_valid = pd.to_numeric(valid[target], errors="coerce")

        train_mask = y_train.notna()
        valid_mask = y_valid.notna()
        X_train = X_train.loc[train_mask]
        y_train = y_train.loc[train_mask]
        X_valid = X_valid.loc[valid_mask]
        y_valid = y_valid.loc[valid_mask]

        if len(X_train) < 500 or len(X_valid) < 100:
            results[stage] = {
                "status":"skipped",
                "reason":"Te weinig bruikbare regels voor deze feature-set."
            }
            continue

        reg = HistGradientBoostingRegressor(
            learning_rate=0.07,
            max_iter=140,
            max_leaf_nodes=31,
            l2_regularization=0.8,
            random_state=42
        )
        clf = HistGradientBoostingClassifier(
            learning_rate=0.07,
            max_iter=110,
            max_leaf_nodes=31,
            l2_regularization=0.8,
            random_state=42
        )

        _notify(callback, base + 0.02, f"Model {stage}: prijsmodel trainen…")
        reg.fit(X_train, y_train)

        _notify(callback, base + 0.045, f"Model {stage}: richtingsmodel trainen…")
        clf.fit(X_train, (y_train >= 0).astype(int))

        pred = reg.predict(X_valid)
        probability = clf.predict_proba(X_valid)[:,1]

        metrics = _metrics(y_valid.to_numpy(), pred, probability)
        metrics.update({
            "status":"trained",
            "features":present,
            "train_rows":int(len(X_train)),
            "validation_rows":int(len(X_valid)),
            "train_until":str(train["ts"].max()),
            "validation_from":str(valid["ts"].min())
        })
        results[stage] = metrics
        artifacts[stage] = {
            "regressor":reg,
            "classifier":clf,
            "features":present
        }

        _notify(callback, base + 0.075, f"Model {stage} klaar.")

        del X_train, X_valid, y_train, y_valid, pred, probability

    _notify(callback, 0.91, "Modelvergelijking afgerond; modelbestand opslaan…")
    return results, artifacts

def serialize_bundle(bundle):
    return pickle.dumps(bundle, protocol=pickle.HIGHEST_PROTOCOL)
