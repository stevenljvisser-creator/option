"""Point-in-time earnings ingestion for SEC Company Facts and FMP.

The storage contract is deliberately append-only:

* ``market-data/v5/earnings/raw/sec`` keeps every SEC response;
* ``market-data/v5/earnings/raw/fmp`` keeps every observed FMP snapshot;
* ``market-data/v5/earnings/processed`` keeps immutable merged snapshots;
* ``market-data/v5/earnings/status/runs`` keeps immutable run manifests.

FMP estimates are never silently replaced.  A merged event selects the latest
estimate that OptionEdge actually observed before the earnings event.  A
historical estimate first observed after the event is retained but explicitly
marked ``backfill_unverified`` and is unavailable to earlier feature dates.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time as clock_time, timedelta, timezone
from hashlib import sha256
import gzip
import json
import math
import re
import time
from typing import Any, Iterable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from core.config import (
    EARNINGS_FMP_RAW_ROOT,
    EARNINGS_PROCESSED_ROOT,
    EARNINGS_SEC_RAW_ROOT,
    EARNINGS_STATUS_ROOT,
)
from core.runtime_settings import get_setting
from core.storage import exists, get_df, get_json, list_keys, put_bytes, put_df, put_json


SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_COMPANY_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
FMP_EARNINGS_URL = "https://financialmodelingprep.com/stable/earnings"

SEC_CONCEPTS: dict[str, tuple[str, ...]] = {
    "revenue": (
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
    ),
    "net_income": ("NetIncomeLoss", "ProfitLoss"),
    "eps_actual_sec": ("EarningsPerShareDiluted", "EarningsPerShareBasic"),
    "assets": ("Assets",),
    "liabilities": ("Liabilities",),
    "equity": ("StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"),
    "cash_flow": ("NetCashProvidedByUsedInOperatingActivities",),
}
INSTANT_METRICS = {"assets", "liabilities", "equity"}
# EPS is a per-share ratio with a period-specific weighted share count and is
# therefore not additive.  It must never be quarterized by subtracting earlier
# quarters from annual EPS.
FLOW_METRICS = {"revenue", "net_income", "cash_flow"}
ALLOWED_FORMS = {"10-Q", "10-Q/A", "10-K", "10-K/A"}
QUARTER_NUMBER = {"Q1": 1, "Q2": 2, "Q3": 3, "FY": 4, "Q4": 4}

PROCESSED_COLUMNS = [
    "date", "ticker", "fiscal_year", "fiscal_quarter", "period_end",
    "earnings_date", "filing_date", "form_type", "earnings_time",
    "eps_estimated", "eps_actual", "eps_surprise", "eps_surprise_pct",
    "revenue_estimated", "revenue_actual", "revenue_surprise",
    "revenue_surprise_pct", "net_income", "assets", "liabilities", "cash_flow",
    "market_expectation_gap_eps", "market_expectation_gap_revenue",
    "eps_actual_sec", "eps_actual_fmp", "revenue_actual_sec", "revenue_actual_fmp",
    "actual_source_eps", "actual_source_revenue", "cik", "accession_number",
    "event_id", "revision_number", "sec_available_utc", "event_schedule_available_utc",
    "estimate_available_utc", "actual_available_utc", "surprise_available_utc",
    "record_available_utc", "estimate_observed_at_utc", "estimate_point_in_time_safe",
    "estimate_pit_status", "fmp_last_updated", "sec_period_derivation",
    "match_method", "match_distance_days", "sec_observed_at_utc",
]


class EarningsSourceError(RuntimeError):
    """A source response was unavailable or invalid; stored data is untouched."""


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _utc_timestamp(value: Any) -> pd.Timestamp | pd.NaT:
    return pd.to_datetime(value, utc=True, errors="coerce")


def _iso_utc(value: Any) -> str | None:
    ts = _utc_timestamp(value)
    return None if pd.isna(ts) else ts.isoformat()


def _number(value: Any) -> float:
    out = pd.to_numeric(value, errors="coerce")
    return float(out) if pd.notna(out) and math.isfinite(float(out)) else math.nan


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str,
    ).encode("utf-8")


def _stamp(value: datetime | pd.Timestamp | None = None) -> str:
    ts = _utc_timestamp(value or utcnow())
    return ts.strftime("%Y%m%dT%H%M%S%fZ")


def _request_session() -> requests.Session:
    retry = Retry(
        total=4, connect=4, read=4, status=4,
        backoff_factor=0.8,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        respect_retry_after_header=True,
    )
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def _json_response(response: requests.Response, source: str) -> Any:
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise EarningsSourceError(f"{source} HTTP {response.status_code}") from exc
    try:
        payload = response.json()
    except Exception as exc:
        raise EarningsSourceError(f"{source} gaf geen geldige JSON terug") from exc
    if isinstance(payload, dict) and any(key in payload for key in ("Error Message", "error")):
        message = payload.get("Error Message") or payload.get("error")
        raise EarningsSourceError(f"{source}: {message}")
    return payload


class SecEdgarClient:
    """Small, rate-limited SEC client with the declared User-Agent required by EDGAR."""

    def __init__(self, user_agent: str | None = None, session: requests.Session | None = None,
                 min_interval_seconds: float = 0.15):
        self.user_agent = str(user_agent or get_setting("SEC_USER_AGENT", "")).strip()
        if not self.user_agent or "@" not in self.user_agent:
            raise EarningsSourceError(
                "SEC_USER_AGENT ontbreekt. Vul op de website een naam plus contact-e-mailadres in, "
                "bijvoorbeeld 'OptionEdge PWS naam@example.nl'."
            )
        self.session = session or _request_session()
        self.min_interval_seconds = max(0.11, float(min_interval_seconds))
        self._last_request = 0.0
        self._ticker_map_cache: dict[str, dict[str, Any]] | None = None

    def _get(self, url: str) -> Any:
        wait = self.min_interval_seconds - (time.monotonic() - self._last_request)
        if wait > 0:
            time.sleep(wait)
        response = self.session.get(
            url,
            headers={
                "User-Agent": self.user_agent,
                "Accept": "application/json",
                "Accept-Encoding": "gzip, deflate",
            },
            timeout=(10, 90),
        )
        self._last_request = time.monotonic()
        return _json_response(response, "SEC EDGAR")

    def ticker_map(self) -> dict[str, dict[str, Any]]:
        if self._ticker_map_cache is not None:
            return self._ticker_map_cache
        payload = self._get(SEC_TICKERS_URL)
        values = payload.values() if isinstance(payload, dict) else payload
        out = {}
        for item in values or []:
            if not isinstance(item, dict):
                continue
            ticker = str(item.get("ticker", "")).upper().strip()
            cik = item.get("cik_str")
            if ticker and cik is not None:
                out[ticker] = {**item, "cik": str(int(cik)).zfill(10)}
        self._ticker_map_cache = out
        return self._ticker_map_cache

    def fetch_ticker(self, ticker: str) -> dict[str, Any]:
        symbol = ticker.strip().upper()
        item = self.ticker_map().get(symbol)
        if not item:
            raise EarningsSourceError(f"SEC CIK niet gevonden voor {symbol}")
        cik = item["cik"]
        return {
            "ticker": symbol,
            "cik": cik,
            "title": item.get("title"),
            "companyfacts": self._get(SEC_COMPANY_FACTS_URL.format(cik=cik)),
            "submissions": self._get(SEC_SUBMISSIONS_URL.format(cik=cik)),
        }


class FmpEarningsClient:
    def __init__(self, api_key: str | None = None, session: requests.Session | None = None):
        self.api_key = str(api_key or get_setting("FMP_API_KEY", "")).strip()
        if not self.api_key:
            raise EarningsSourceError("FMP_API_KEY ontbreekt. Vul de FMP API-key in op de Systeempagina.")
        self.session = session or _request_session()

    def fetch_ticker(self, ticker: str) -> list[dict[str, Any]]:
        response = self.session.get(
            FMP_EARNINGS_URL,
            params={
                "symbol": ticker.strip().upper(),
                "limit": 1000,
                "includeReportTimes": "true",
                "apikey": self.api_key,
            },
            headers={"Accept": "application/json", "User-Agent": "OptionEdge/2.0"},
            timeout=(10, 90),
        )
        payload = _json_response(response, "FMP earnings")
        if not isinstance(payload, list):
            raise EarningsSourceError("FMP earnings gaf geen lijst met events terug")
        return [item for item in payload if isinstance(item, dict)]


def _acceptance_map(submissions: dict[str, Any]) -> dict[str, dict[str, Any]]:
    recent = ((submissions or {}).get("filings") or {}).get("recent") or {}
    if not isinstance(recent, dict):
        return {}
    count = max((len(v) for v in recent.values() if isinstance(v, list)), default=0)
    out = {}
    for index in range(count):
        row = {key: values[index] for key, values in recent.items()
               if isinstance(values, list) and index < len(values)}
        accession = str(row.get("accessionNumber", ""))
        if accession:
            out[accession] = row
    return out


def _period_days(item: dict[str, Any]) -> float:
    start = pd.to_datetime(item.get("start"), errors="coerce")
    end = pd.to_datetime(item.get("end"), errors="coerce")
    if pd.isna(start) or pd.isna(end):
        return math.nan
    return float((end - start).days + 1)


def _unit_rank(metric: str, unit: str) -> int:
    value = str(unit).lower().replace(" ", "")
    if metric == "eps_actual_sec":
        return 0 if value in {"usd/shares", "usd-per-shares", "usd/share"} else 5
    return 0 if value == "usd" else 5


def _fact_entries(companyfacts: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    taxonomy = ((companyfacts or {}).get("facts") or {}).get("us-gaap") or {}
    out: dict[str, list[dict[str, Any]]] = {metric: [] for metric in SEC_CONCEPTS}
    for metric, concepts in SEC_CONCEPTS.items():
        for concept_rank, concept in enumerate(concepts):
            units = ((taxonomy.get(concept) or {}).get("units") or {})
            for unit, values in units.items():
                for item in values or []:
                    if not isinstance(item, dict) or str(item.get("form")) not in ALLOWED_FORMS:
                        continue
                    value = _number(item.get("val"))
                    if not math.isfinite(value) or not item.get("accn"):
                        continue
                    out[metric].append({
                        **item,
                        "metric": metric,
                        "concept": concept,
                        "concept_rank": concept_rank,
                        "unit": unit,
                        "unit_rank": _unit_rank(metric, unit),
                        "period_days": _period_days(item),
                        "numeric_value": value,
                    })
    return out


def _pick_fact(items: list[dict[str, Any]], fp: str, metric: str,
               report_date: Any = None) -> dict[str, Any] | None:
    if not items:
        return None
    if metric in INSTANT_METRICS:
        return min(items, key=lambda item: (
            _days_between(item.get("end"), report_date),
            item.get("unit_rank", 9), item.get("concept_rank", 9),
            0 if not item.get("start") else 1,
        ))

    def score(item: dict[str, Any]) -> tuple[float, int, int]:
        days = item.get("period_days")
        duration = float(days) if pd.notna(days) else 9999.0
        frame = str(item.get("frame") or "").upper()
        if fp == "FY":
            direct_q4 = 60 <= duration <= 130 and bool(re.search(r"Q4$", frame))
            duration_score = abs(duration - 91) - 1000 if direct_q4 else abs(duration - 365)
        else:
            direct = 60 <= duration <= 130
            target = 91 if direct else 91 * QUARTER_NUMBER.get(fp, 1)
            duration_score = abs(duration - target) + (0 if direct else 100)
        end_score = _days_between(item.get("end"), report_date)
        return duration_score + min(end_score, 1000) * 10, int(item.get("unit_rank", 9)), int(item.get("concept_rank", 9))

    return min(items, key=score)


def _latest_prior(rows: list[dict[str, Any]], current: dict[str, Any], quarter: int
                  ) -> dict[str, Any] | None:
    filed = pd.to_datetime(current.get("filing_date"), errors="coerce")
    candidates = []
    for row in rows:
        if row.get("fiscal_year") != current.get("fiscal_year"):
            continue
        if QUARTER_NUMBER.get(str(row.get("_fp")), 0) != quarter:
            continue
        row_filed = pd.to_datetime(row.get("filing_date"), errors="coerce")
        if pd.isna(filed) or pd.isna(row_filed) or row_filed <= filed:
            candidates.append(row)
    return max(candidates, key=lambda row: str(row.get("filing_date") or ""), default=None)


def normalize_sec_companyfacts(ticker: str, raw: dict[str, Any], observed_at: Any
                               ) -> pd.DataFrame:
    """Convert SEC facts to versioned fiscal-quarter rows.

    Flow values reported year-to-date are quarterized.  Q4 is taken directly
    from a quarterly context where available, otherwise derived from the annual
    10-K value minus Q1-Q3.  The derivation is recorded explicitly.
    """
    companyfacts = raw.get("companyfacts") or {}
    submissions = raw.get("submissions") or {}
    cik = str(raw.get("cik") or companyfacts.get("cik") or "").zfill(10)
    acceptance = _acceptance_map(submissions)
    metrics = _fact_entries(companyfacts)
    accessions = sorted({str(item.get("accn")) for values in metrics.values() for item in values})
    rows: list[dict[str, Any]] = []

    for accession in accessions:
        all_items = [item for values in metrics.values() for item in values
                     if str(item.get("accn")) == accession]
        if not all_items:
            continue
        fp_values = [str(item.get("fp")) for item in all_items if item.get("fp")]
        fy_values = [int(item.get("fy")) for item in all_items
                     if pd.notna(pd.to_numeric(item.get("fy"), errors="coerce"))]
        fp = max(set(fp_values), key=fp_values.count) if fp_values else ""
        fy = max(set(fy_values), key=fy_values.count) if fy_values else None
        if fp not in QUARTER_NUMBER or fy is None:
            continue
        filing = acceptance.get(accession, {})
        filed = filing.get("filingDate") or min(
            (str(item.get("filed")) for item in all_items if item.get("filed")), default=None,
        )
        period_end = filing.get("reportDate") or max(
            (str(item.get("end")) for item in all_items if item.get("end")), default=None,
        )
        accepted = _utc_timestamp(filing.get("acceptanceDateTime"))
        if pd.isna(accepted) and filed:
            # Company Facts exposes only the filing date.  Midnight on the next
            # UTC day is conservative and cannot leak the filing into prior bars.
            accepted = pd.Timestamp(filed, tz="UTC") + pd.Timedelta(days=1)
        row: dict[str, Any] = {
            "ticker": ticker.upper(), "cik": cik, "company_name": companyfacts.get("entityName"),
            "accession_number": accession, "fiscal_year": fy, "_fp": fp,
            "fiscal_quarter": "Q4" if fp == "FY" else fp,
            "period_end": period_end, "filing_date": filed,
            "form_type": filing.get("form") or all_items[0].get("form"),
            "sec_available_utc": _iso_utc(accepted),
            "sec_observed_at_utc": _iso_utc(observed_at),
            "sec_period_derivation": "reported_quarter",
        }
        for metric, values in metrics.items():
            candidates = [item for item in values if str(item.get("accn")) == accession]
            chosen = _pick_fact(candidates, fp, metric, period_end)
            if not chosen:
                row[metric] = math.nan
                continue
            row[metric] = chosen["numeric_value"]
            row[f"_{metric}_raw"] = chosen["numeric_value"]
            row[f"_{metric}_days"] = chosen.get("period_days")
            row[f"{metric}_concept"] = chosen.get("concept")
            row[f"{metric}_unit"] = chosen.get("unit")
        if not math.isfinite(_number(row.get("liabilities"))):
            assets,equity=_number(row.get("assets")),_number(row.get("equity"))
            if math.isfinite(assets) and math.isfinite(equity):
                row["liabilities"]=assets-equity
                row["liabilities_concept"]="derived_assets_minus_equity"
        rows.append(row)

    rows.sort(key=lambda row: (
        int(row.get("fiscal_year") or 0), QUARTER_NUMBER.get(str(row.get("_fp")), 0),
        str(row.get("filing_date") or ""), str(row.get("accession_number") or ""),
    ))

    # Quarterize cumulative 10-Q cash-flow facts and annual 10-K flow facts.
    for row in rows:
        quarter = QUARTER_NUMBER.get(str(row.get("_fp")), 0)
        eps_duration = _number(row.get("_eps_actual_sec_days"))
        if math.isfinite(_number(row.get("_eps_actual_sec_raw"))):
            if not math.isfinite(eps_duration) or eps_duration > 130:
                row["eps_actual_sec"] = math.nan
                row["eps_actual_sec_derivation"] = "non_additive_period_rejected"
        for metric in FLOW_METRICS:
            raw_value = _number(row.get(f"_{metric}_raw"))
            duration = _number(row.get(f"_{metric}_days"))
            if not math.isfinite(raw_value):
                continue
            direct = math.isfinite(duration) and duration <= 130
            if quarter in (2, 3) and not direct:
                prior_values = []
                for prior_quarter in range(1, quarter):
                    prior = _latest_prior(rows, row, prior_quarter)
                    value = _number(prior.get(metric)) if prior else math.nan
                    if math.isfinite(value):
                        prior_values.append(value)
                if len(prior_values) == quarter - 1:
                    row[metric] = raw_value - sum(prior_values)
                    row["sec_period_derivation"] = "quarterized_ytd"
            elif quarter == 4 and not direct:
                prior_values = []
                for prior_quarter in (1, 2, 3):
                    prior = _latest_prior(rows, row, prior_quarter)
                    value = _number(prior.get(metric)) if prior else math.nan
                    if math.isfinite(value):
                        prior_values.append(value)
                if len(prior_values) == 3:
                    row[metric] = raw_value - sum(prior_values)
                    row["sec_period_derivation"] = "derived_q4_from_annual"
                else:
                    # An annual value is not silently mislabeled as a quarter.
                    row[metric] = math.nan
                    row["sec_period_derivation"] = "q4_not_derivable"

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame = frame.drop(columns=[column for column in frame if column.startswith("_")], errors="ignore")
    for column in ["period_end", "filing_date"]:
        frame[column] = pd.to_datetime(frame[column], errors="coerce").dt.date
    return frame.sort_values(
        ["fiscal_year", "fiscal_quarter", "filing_date", "accession_number"],
        na_position="last",
    ).reset_index(drop=True)


def _release_label(item: dict[str, Any]) -> str:
    raw = item.get("time") or item.get("reportingTime") or item.get("reportTime") or ""
    value = str(raw).strip().lower().replace(" ", "_")
    mapping = {
        "bmo": "before_market_open", "before_market_open": "before_market_open",
        "before market open": "before_market_open", "amc": "after_market_close",
        "after_market_close": "after_market_close", "after market close": "after_market_close",
        "dmh": "during_market_hours", "during_market_hours": "during_market_hours",
    }
    if value in mapping:
        return mapping[value]
    if re.fullmatch(r"\d{1,2}:\d{2}(:\d{2})?", value):
        return value
    return "unknown"


def _event_time(event_date: Any, release_label: str) -> pd.Timestamp | pd.NaT:
    day = pd.to_datetime(event_date, errors="coerce")
    if pd.isna(day):
        return pd.NaT
    eastern = ZoneInfo("America/New_York")
    if release_label == "before_market_open":
        local = datetime.combine(day.date(), clock_time(8, 0), tzinfo=eastern)
    elif release_label == "after_market_close":
        local = datetime.combine(day.date(), clock_time(16, 5), tzinfo=eastern)
    elif release_label == "during_market_hours":
        local = datetime.combine(day.date(), clock_time(12, 0), tzinfo=eastern)
    elif re.fullmatch(r"\d{1,2}:\d{2}(:\d{2})?", release_label):
        parts = [int(x) for x in release_label.split(":")]
        local = datetime.combine(day.date(), clock_time(*parts), tzinfo=eastern)
    else:
        # With no reliable time, expose results only from the next UTC day.
        return pd.Timestamp(day.date(), tz="UTC") + pd.Timedelta(days=1)
    return pd.Timestamp(local.astimezone(timezone.utc))


def _estimate_cutoff(event_date: Any, release_label: str) -> pd.Timestamp | pd.NaT:
    """Latest safe instant for a pre-event estimate.

    If FMP does not supply a reliable release time, the start of the earnings
    day is deliberately used.  A snapshot observed later that day may already
    contain the result and must not masquerade as a historical consensus.
    """
    day = pd.to_datetime(event_date, errors="coerce")
    if pd.isna(day):
        return pd.NaT
    if release_label == "unknown":
        return pd.Timestamp(day.date(), tz="UTC")
    return _event_time(event_date, release_label)


def normalize_fmp_snapshot(ticker: str, payload: Iterable[dict[str, Any]], observed_at: Any
                           ) -> pd.DataFrame:
    observed = _utc_timestamp(observed_at)
    rows = []
    for item in payload or []:
        symbol = str(item.get("symbol") or ticker).upper().strip()
        if symbol != ticker.upper():
            continue
        event_date = pd.to_datetime(item.get("date"), errors="coerce")
        if pd.isna(event_date):
            continue
        release = _release_label(item)
        event_time = _event_time(event_date, release)
        estimate_cutoff = _estimate_cutoff(event_date, release)
        rows.append({
            "ticker": ticker.upper(),
            "earnings_date": event_date.date(),
            "earnings_time": release,
            "earnings_event_utc": _iso_utc(event_time),
            "estimate_cutoff_utc": _iso_utc(estimate_cutoff),
            "fiscal_date_ending": pd.to_datetime(
                item.get("fiscalDateEnding") or item.get("fiscalDate"), errors="coerce",
            ).date() if pd.notna(pd.to_datetime(
                item.get("fiscalDateEnding") or item.get("fiscalDate"), errors="coerce",
            )) else None,
            "eps_estimated": _number(item.get("epsEstimated")),
            "eps_actual_fmp": _number(item.get("epsActual")),
            "revenue_estimated": _number(item.get("revenueEstimated")),
            "revenue_actual_fmp": _number(item.get("revenueActual")),
            "fmp_last_updated": item.get("lastUpdated"),
            "observed_at_utc": _iso_utc(observed),
        })
    return pd.DataFrame(rows)


def select_fmp_point_in_time(history: pd.DataFrame) -> pd.DataFrame:
    """One event row while preserving the final estimate actually seen pre-event."""
    if history is None or history.empty:
        return pd.DataFrame()
    data = history.copy()
    data["observed_at_utc"] = pd.to_datetime(data["observed_at_utc"], utc=True, errors="coerce")
    data["earnings_event_utc"] = pd.to_datetime(data["earnings_event_utc"], utc=True, errors="coerce")
    data["estimate_cutoff_utc"] = pd.to_datetime(
        data.get("estimate_cutoff_utc", data["earnings_event_utc"]), utc=True, errors="coerce"
    )
    data["earnings_date"] = pd.to_datetime(data["earnings_date"], errors="coerce").dt.date
    rows = []
    for (ticker, event_date), group in data.dropna(subset=["earnings_date"]).groupby(
        ["ticker", "earnings_date"], sort=True,
    ):
        group = group.sort_values("observed_at_utc")
        scheduled = group.iloc[0]
        estimate_mask = group[["eps_estimated", "revenue_estimated"]].notna().any(axis=1)
        pre_event = estimate_mask & (group["observed_at_utc"] <= group["estimate_cutoff_utc"])
        if pre_event.any():
            estimate = group[pre_event].iloc[-1]
            pit_safe = True
            pit_status = "observed_pre_event"
        elif estimate_mask.any():
            estimate = group[estimate_mask].iloc[0]
            pit_safe = False
            pit_status = "backfill_unverified"
        else:
            estimate = scheduled
            pit_safe = False
            pit_status = "estimate_missing"
        actual_mask = group[["eps_actual_fmp", "revenue_actual_fmp"]].notna().any(axis=1)
        actual = group[actual_mask].iloc[0] if actual_mask.any() else group.iloc[-1]
        event_time = next(
            (value for value in group["earnings_event_utc"] if pd.notna(value)), pd.NaT,
        )
        rows.append({
            "ticker": ticker, "earnings_date": event_date,
            "earnings_time": next((str(x) for x in group["earnings_time"] if str(x) != "unknown"), "unknown"),
            "earnings_event_utc": _iso_utc(event_time),
            "fiscal_date_ending": estimate.get("fiscal_date_ending") or actual.get("fiscal_date_ending"),
            "eps_estimated": estimate.get("eps_estimated"),
            "revenue_estimated": estimate.get("revenue_estimated"),
            "eps_actual_fmp": actual.get("eps_actual_fmp"),
            "revenue_actual_fmp": actual.get("revenue_actual_fmp"),
            "estimate_observed_at_utc": _iso_utc(estimate.get("observed_at_utc")),
            "estimate_available_utc": _iso_utc(estimate.get("observed_at_utc")),
            "event_schedule_available_utc": _iso_utc(scheduled.get("observed_at_utc")),
            "fmp_actual_available_utc": _iso_utc(actual.get("observed_at_utc")) if actual_mask.any() else None,
            "estimate_point_in_time_safe": bool(pit_safe),
            "estimate_pit_status": pit_status,
            "fmp_last_updated": actual.get("fmp_last_updated") or estimate.get("fmp_last_updated"),
        })
    return pd.DataFrame(rows)


def _days_between(left: Any, right: Any) -> float:
    a = pd.to_datetime(left, errors="coerce")
    b = pd.to_datetime(right, errors="coerce")
    return abs(float((a - b).days)) if pd.notna(a) and pd.notna(b) else math.inf


def _match_fmp(sec_row: pd.Series, fmp: pd.DataFrame) -> tuple[pd.Series | None, str, float]:
    if fmp is None or fmp.empty:
        return None, "sec_only", math.nan
    fiscal_end = sec_row.get("period_end")
    exact = fmp[
        fmp["fiscal_date_ending"].notna()
        & (pd.to_datetime(fmp["fiscal_date_ending"], errors="coerce")
           == pd.to_datetime(fiscal_end, errors="coerce"))
    ] if "fiscal_date_ending" in fmp else pd.DataFrame()
    if not exact.empty:
        item = exact.sort_values("earnings_date").iloc[-1]
        return item, "fiscal_period_end", _days_between(item.get("earnings_date"), sec_row.get("filing_date"))

    period = pd.to_datetime(fiscal_end, errors="coerce")
    filed = pd.to_datetime(sec_row.get("filing_date"), errors="coerce")
    candidates = fmp.copy()
    event_dates = pd.to_datetime(candidates["earnings_date"], errors="coerce")
    if pd.notna(period):
        candidates = candidates[(event_dates >= period - pd.Timedelta(days=7))
                                & (event_dates <= period + pd.Timedelta(days=190))]
        event_dates = pd.to_datetime(candidates["earnings_date"], errors="coerce")
    if pd.notna(filed) and not candidates.empty:
        close = (event_dates - filed).abs().dt.days
        candidates = candidates[close <= 45]
    if candidates.empty:
        return None, "sec_only", math.nan
    scores = candidates.apply(
        lambda row: _days_between(row.get("earnings_date"), filed) * 10
        + _days_between(row.get("earnings_date"), period), axis=1,
    )
    index = scores.idxmin()
    item = candidates.loc[index]
    return item, "nearest_filing_and_period", _days_between(item.get("earnings_date"), filed)


def _max_utc(*values: Any) -> str | None:
    timestamps = [_utc_timestamp(value) for value in values]
    timestamps = [value for value in timestamps if pd.notna(value)]
    return _iso_utc(max(timestamps)) if timestamps else None


def _surprise(actual: Any, estimate: Any) -> tuple[float, float]:
    actual_value, estimate_value = _number(actual), _number(estimate)
    if not math.isfinite(actual_value) or not math.isfinite(estimate_value):
        return math.nan, math.nan
    absolute = actual_value - estimate_value
    percentage = absolute / abs(estimate_value) * 100 if estimate_value != 0 else math.nan
    return absolute, percentage


def combine_earnings(ticker: str, sec: pd.DataFrame, fmp_history: pd.DataFrame,
                     start: date | None = None, end: date | None = None) -> pd.DataFrame:
    fmp = select_fmp_point_in_time(fmp_history)
    sec = sec.copy() if sec is not None else pd.DataFrame()
    rows: list[dict[str, Any]] = []
    matched_fmp: set[int] = set()

    for _, sec_row in sec.iterrows():
        fmp_row, method, distance = _match_fmp(sec_row, fmp)
        if fmp_row is not None:
            matched_fmp.add(int(fmp_row.name))
        row = sec_row.to_dict()
        if fmp_row is not None:
            row.update({key: fmp_row.get(key) for key in fmp_row.index})
        row["match_method"] = method
        row["match_distance_days"] = distance
        rows.append(row)

    for index, fmp_row in fmp.iterrows():
        if int(index) in matched_fmp:
            continue
        rows.append({**fmp_row.to_dict(), "ticker": ticker.upper(), "match_method": "fmp_only",
                     "match_distance_days": math.nan})

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=PROCESSED_COLUMNS)
    for column in [
        "eps_estimated", "eps_actual_sec", "eps_actual_fmp", "revenue_estimated",
        "revenue", "revenue_actual_fmp", "net_income", "assets", "liabilities", "cash_flow",
    ]:
        if column not in out:
            out[column] = math.nan
        out[column] = pd.to_numeric(out[column], errors="coerce")

    out["revenue_actual_sec"] = out["revenue"]
    out["eps_actual"] = out["eps_actual_sec"].combine_first(out["eps_actual_fmp"])
    out["revenue_actual"] = out["revenue_actual_sec"].combine_first(out["revenue_actual_fmp"])
    out["actual_source_eps"] = np.where(out["eps_actual_sec"].notna(), "SEC", np.where(out["eps_actual_fmp"].notna(), "FMP", None))
    out["actual_source_revenue"] = np.where(out["revenue_actual_sec"].notna(), "SEC", np.where(out["revenue_actual_fmp"].notna(), "FMP", None))

    eps = [_surprise(actual, estimate) for actual, estimate in zip(out["eps_actual"], out["eps_estimated"])]
    revenue = [_surprise(actual, estimate) for actual, estimate in zip(out["revenue_actual"], out["revenue_estimated"])]
    out["eps_surprise"] = [value[0] for value in eps]
    out["eps_surprise_pct"] = [value[1] for value in eps]
    out["revenue_surprise"] = [value[0] for value in revenue]
    out["revenue_surprise_pct"] = [value[1] for value in revenue]
    out["market_expectation_gap_eps"] = out["eps_surprise"]
    out["market_expectation_gap_revenue"] = out["revenue_surprise"]

    if "earnings_date" not in out:
        out["earnings_date"] = pd.NaT
    if "filing_date" not in out:
        out["filing_date"] = pd.NaT
    out["date"] = pd.to_datetime(out["earnings_date"], errors="coerce").dt.date
    out["date"] = out["date"].combine_first(pd.to_datetime(out["filing_date"], errors="coerce").dt.date)
    out["ticker"] = ticker.upper()

    sec_available = out.get("sec_available_utc", pd.Series(index=out.index, dtype=object))
    fmp_actual_available = out.get("fmp_actual_available_utc", pd.Series(index=out.index, dtype=object))
    out["actual_available_utc"] = [
        _max_utc(sec_value if source == "SEC" else fmp_value)
        for sec_value, fmp_value, source in zip(sec_available, fmp_actual_available, out["actual_source_eps"])
    ]
    # Revenue can occasionally be available while EPS is missing.  Include both source clocks.
    out["actual_available_utc"] = [
        _max_utc(current, sec_value if rev_source == "SEC" else fmp_value)
        for current, sec_value, fmp_value, rev_source in zip(
            out["actual_available_utc"], sec_available, fmp_actual_available, out["actual_source_revenue"],
        )
    ]
    estimate_available = out.get("estimate_available_utc", pd.Series(index=out.index, dtype=object))
    out["surprise_available_utc"] = [
        _max_utc(actual_value, estimate_value)
        for actual_value, estimate_value in zip(out["actual_available_utc"], estimate_available)
    ]
    schedule_available = out.get("event_schedule_available_utc", pd.Series(index=out.index, dtype=object))
    out["record_available_utc"] = [
        _max_utc(sec_value, estimate_value, actual_value, schedule_value)
        for sec_value, estimate_value, actual_value, schedule_value in zip(
            sec_available, estimate_available, out["actual_available_utc"], schedule_available,
        )
    ]

    fy = pd.to_numeric(out.get("fiscal_year"), errors="coerce")
    fq = out.get("fiscal_quarter", pd.Series(index=out.index, dtype=object)).astype("string")
    fallback = pd.to_datetime(out["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    out["event_id"] = [
        f"{ticker.upper()}|FY{int(year)}|{quarter}" if pd.notna(year) and pd.notna(quarter)
        else f"{ticker.upper()}|{day}"
        for year, quarter, day in zip(fy, fq, fallback)
    ]
    out = out.sort_values(["event_id", "record_available_utc", "filing_date"], na_position="last")
    out["revision_number"] = out.groupby("event_id", sort=False).cumcount() + 1

    if start or end:
        relevant = pd.to_datetime(out["date"], errors="coerce").dt.date
        mask = pd.Series(True, index=out.index)
        if start:
            mask &= relevant >= start
        if end:
            mask &= relevant <= end
        out = out[mask]

    for column in PROCESSED_COLUMNS:
        if column not in out:
            out[column] = None
    return out[PROCESSED_COLUMNS].sort_values(
        ["date", "fiscal_year", "fiscal_quarter", "revision_number"], na_position="last",
    ).reset_index(drop=True)


def build_earnings_dataset(ticker: str, sec_raw: dict[str, Any],
                           fmp_snapshots: Iterable[tuple[Any, Iterable[dict[str, Any]]]],
                           sec_observed_at: Any, start: date | None = None,
                           end: date | None = None) -> pd.DataFrame:
    """Pure end-to-end transform used by the worker and the NVDA contract test."""
    sec = normalize_sec_companyfacts(ticker, sec_raw, sec_observed_at)
    fmp_frames = [normalize_fmp_snapshot(ticker, payload, observed)
                  for observed, payload in fmp_snapshots]
    history = pd.concat([frame for frame in fmp_frames if not frame.empty], ignore_index=True) \
        if any(not frame.empty for frame in fmp_frames) else pd.DataFrame()
    return combine_earnings(ticker, sec, history, start, end)


def append_raw_snapshot(source: str, ticker: str, payload: Any, observed_at: Any,
                        metadata: dict[str, Any] | None = None) -> str:
    observed = _utc_timestamp(observed_at)
    wrapper = {
        "source": source, "ticker": ticker.upper(), "observed_at_utc": _iso_utc(observed),
        "metadata": metadata or {}, "payload": payload,
    }
    raw = _canonical_json(wrapper)
    digest = sha256(raw).hexdigest()
    root = EARNINGS_SEC_RAW_ROOT if source == "sec" else EARNINGS_FMP_RAW_ROOT
    key = (
        f"{root}/{ticker.upper()}/{observed:%Y}/{observed:%m}/{observed:%d}/"
        f"{_stamp(observed)}_{digest[:16]}.json.gz"
    )
    if not exists(key):
        put_bytes(key, gzip.compress(raw, compresslevel=6), "application/json", "gzip")
    return key


def _raw_wrappers(source: str, ticker: str, limit: int | None = None) -> list[dict[str, Any]]:
    root = EARNINGS_SEC_RAW_ROOT if source == "sec" else EARNINGS_FMP_RAW_ROOT
    keys = sorted((key for key in list_keys(f"{root}/{ticker.upper()}/") if key.endswith(".json.gz")))
    if limit:
        keys = keys[-limit:]
    out = []
    for key in keys:
        try:
            value = get_json(key)
            if isinstance(value, dict):
                out.append({**value, "_key": key})
        except Exception:
            continue
    return out


def processed_key(ticker: str, start: date, end: date, observed_at: Any, frame: pd.DataFrame) -> str:
    digest = sha256(frame.to_csv(index=False).encode("utf-8")).hexdigest()[:16]
    ts = _utc_timestamp(observed_at)
    return (
        f"{EARNINGS_PROCESSED_ROOT}/{ticker.upper()}/{start}_{end}/"
        f"{_stamp(ts)}_{digest}.csv.gz"
    )


def load_processed_earnings(ticker: str) -> pd.DataFrame:
    keys = sorted(key for key in list_keys(f"{EARNINGS_PROCESSED_ROOT}/{ticker.upper()}/")
                  if key.endswith(".csv.gz"))
    if not keys:
        return pd.DataFrame(columns=PROCESSED_COLUMNS)
    # Range folders are not chronological ("2022_2026" sorts after
    # "2025_2025").  The immutable basename begins with the observation stamp.
    latest=max(keys,key=lambda key:key.rsplit("/",1)[-1].split("_",1)[0])
    return get_df(latest)


@dataclass
class ImportTickerResult:
    ticker: str
    status: str
    sec: dict[str, Any]
    fmp: dict[str, Any]
    processed: dict[str, Any]
    errors: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "ticker": self.ticker, "status": self.status, "sec": self.sec,
            "fmp": self.fmp, "processed": self.processed, "errors": self.errors,
        }


def import_earnings_ticker(ticker: str, start: date, end: date,
                           sec_client: SecEdgarClient | None = None,
                           fmp_client: FmpEarningsClient | None = None,
                           observed_at: datetime | None = None) -> ImportTickerResult:
    """Fetch, append, merge and persist one ticker without deleting old objects."""
    ticker = ticker.strip().upper()
    observed = observed_at or utcnow()
    errors: list[str] = []
    sec_result: dict[str, Any] = {"success": False, "rows": 0, "key": None, "error": None}
    fmp_result: dict[str, Any] = {"success": False, "rows": 0, "key": None, "error": None}

    try:
        sec_client = sec_client or SecEdgarClient()
        sec_payload = sec_client.fetch_ticker(ticker)
        sec_key = append_raw_snapshot("sec", ticker, sec_payload, observed, {
            "companyfacts_endpoint": SEC_COMPANY_FACTS_URL.format(cik=sec_payload.get("cik")),
            "submissions_endpoint": SEC_SUBMISSIONS_URL.format(cik=sec_payload.get("cik")),
        })
        sec_result.update(success=True, key=sec_key, fetched_at=_iso_utc(observed))
    except Exception as exc:
        message = f"SEC: {type(exc).__name__}: {exc}"
        sec_result["error"] = message
        errors.append(message)

    try:
        fmp_client = fmp_client or FmpEarningsClient()
        fmp_payload = fmp_client.fetch_ticker(ticker)
        fmp_key = append_raw_snapshot("fmp", ticker, fmp_payload, observed, {
            "endpoint": FMP_EARNINGS_URL, "api_key_stored": False,
        })
        fmp_result.update(success=True, key=fmp_key, rows=len(fmp_payload), fetched_at=_iso_utc(observed))
    except Exception as exc:
        message = f"FMP: {type(exc).__name__}: {exc}"
        fmp_result["error"] = message
        errors.append(message)

    sec_wrappers = _raw_wrappers("sec", ticker, limit=1)
    fmp_wrappers = _raw_wrappers("fmp", ticker)
    processed_result: dict[str, Any] = {"success": False, "rows": 0, "events": 0, "key": None}
    if sec_wrappers or fmp_wrappers:
        sec_frame = pd.DataFrame()
        if sec_wrappers:
            wrapper = sec_wrappers[-1]
            sec_raw = wrapper.get("payload") or {}
            sec_frame = normalize_sec_companyfacts(ticker, sec_raw, wrapper.get("observed_at_utc"))
            sec_result["rows"] = len(sec_frame)
        fmp_frames = []
        for wrapper in fmp_wrappers:
            frame = normalize_fmp_snapshot(
                ticker, wrapper.get("payload") or [], wrapper.get("observed_at_utc"),
            )
            if not frame.empty:
                fmp_frames.append(frame)
        history = pd.concat(fmp_frames, ignore_index=True) if fmp_frames else pd.DataFrame()
        # Every snapshot is a complete current view of all retained history.
        # A narrow rerun may never make older quarters disappear from the latest
        # processed dataset; callers can filter dates when reading.
        combined = combine_earnings(ticker, sec_frame, history, None, None)
        if not combined.empty and (sec_result["success"] or fmp_result["success"]):
            key = processed_key(ticker, start, end, observed, combined)
            if not exists(key):
                put_df(key, combined)
            processed_result.update(
                success=True, rows=len(combined), events=int(combined["event_id"].nunique()), key=key,
                point_in_time_safe_events=int(
                    combined.groupby("event_id")["estimate_point_in_time_safe"].max().fillna(False).sum()
                ),
            )

    fully_current = bool(sec_result["success"] and fmp_result["success"] and processed_result["success"])
    status = "done" if fully_current else "warning" if processed_result["success"] else "error"
    return ImportTickerResult(ticker, status, sec_result, fmp_result, processed_result, errors)


def save_earnings_run(manifest: dict[str, Any]) -> str:
    finished = _utc_timestamp(manifest.get("finished_at") or utcnow())
    run_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(manifest.get("run_id") or "run"))
    key = f"{EARNINGS_STATUS_ROOT}/runs/{finished:%Y}/{finished:%m}/{_stamp(finished)}_{run_id}.json"
    put_json(key, manifest)
    return key


def latest_earnings_run() -> dict[str, Any] | None:
    keys = sorted(key for key in list_keys(f"{EARNINGS_STATUS_ROOT}/runs/") if key.endswith(".json"))
    if not keys:
        return None
    value = get_json(keys[-1])
    return {**value, "status_key": keys[-1]} if isinstance(value, dict) else None


def _prepare_earnings_for_asof(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize point-in-time clocks once for many intraday as-of lookups."""
    if frame is None or frame.empty:
        return pd.DataFrame()
    data=frame.copy()
    for column in [
        "event_schedule_available_utc", "estimate_available_utc", "actual_available_utc",
        "surprise_available_utc", "sec_available_utc", "record_available_utc",
    ]:
        source = data[column] if column in data.columns else pd.Series(
            pd.NaT, index=data.index, dtype="datetime64[ns, UTC]"
        )
        data[column] = pd.to_datetime(source, utc=True, errors="coerce")
    data["earnings_date"] = pd.to_datetime(data.get("earnings_date"), errors="coerce").dt.date
    return data


def earnings_features_at(frame: pd.DataFrame, as_of: Any, _prepared: bool=False) -> dict[str, float]:
    """Return only values that were available by ``as_of``.

    ``_prepared`` is an internal fast path used by the Clean Feature Store.  It
    avoids repeatedly copying/parsing the same earnings history for every
    minute timestamp while preserving the exact fail-closed PIT rules.
    """
    defaults = {
        "days_to_earnings": math.nan, "days_since_earnings": math.nan,
        "earnings_event_today": 0.0, "earnings_time_bmo": 0.0, "earnings_time_amc": 0.0,
        "eps_estimated": math.nan, "eps_actual": math.nan, "eps_surprise": math.nan,
        "eps_surprise_pct": math.nan, "revenue_estimated": math.nan,
        "revenue_actual": math.nan, "revenue_surprise": math.nan,
        "revenue_surprise_pct": math.nan, "net_income": math.nan, "assets": math.nan,
        "liabilities": math.nan, "cash_flow": math.nan,
        "earnings_point_in_time_safe": 0.0, "earnings_data_age_days": math.nan,
    }
    if frame is None or frame.empty:
        return defaults
    stamp = _utc_timestamp(as_of)
    if pd.isna(stamp):
        return defaults
    today = stamp.date()
    data = frame if _prepared else _prepare_earnings_for_asof(frame)

    schedule = data[
        data["earnings_date"].notna()
        & data["event_schedule_available_utc"].notna()
        & (data["event_schedule_available_utc"] <= stamp)
    ].sort_values(["earnings_date", "event_schedule_available_utc"])
    schedule=schedule.drop_duplicates(
        "event_id" if "event_id" in schedule else "earnings_date",keep="first"
    )
    future = schedule[schedule["earnings_date"] >= today]
    past = schedule[schedule["earnings_date"] <= today]
    next_event = future.iloc[0] if not future.empty else None
    last_event = past.iloc[-1] if not past.empty else None
    if next_event is not None:
        defaults["days_to_earnings"] = float((next_event["earnings_date"] - today).days)
        defaults["earnings_event_today"] = float(next_event["earnings_date"] == today)
        defaults["earnings_time_bmo"] = float(next_event.get("earnings_time") == "before_market_open")
        defaults["earnings_time_amc"] = float(next_event.get("earnings_time") == "after_market_close")
        if pd.notna(next_event.get("estimate_available_utc")) and next_event["estimate_available_utc"] <= stamp:
            if bool(next_event.get("estimate_point_in_time_safe")):
                defaults["eps_estimated"] = _number(next_event.get("eps_estimated"))
                defaults["revenue_estimated"] = _number(next_event.get("revenue_estimated"))
                defaults["earnings_point_in_time_safe"] = 1.0
    if last_event is not None:
        defaults["days_since_earnings"] = float((today - last_event["earnings_date"]).days)

    fundamentals = data[data["sec_available_utc"].notna() & (data["sec_available_utc"] <= stamp)]
    if not fundamentals.empty:
        item = fundamentals.sort_values(["sec_available_utc", "filing_date"]).iloc[-1]
        for column in ["net_income", "assets", "liabilities", "cash_flow"]:
            defaults[column] = _number(item.get(column))

    actuals=data[
        data["actual_available_utc"].notna()
        & (data["actual_available_utc"]<=stamp)
        & data["earnings_date"].notna()
        & (data["earnings_date"]<=today)
    ]
    if not actuals.empty:
        item=actuals.sort_values(["earnings_date","actual_available_utc"]).iloc[-1]
        defaults["eps_actual"]=_number(item.get("eps_actual"))
        defaults["revenue_actual"]=_number(item.get("revenue_actual"))

    safe_estimate=data.get(
        "estimate_point_in_time_safe",pd.Series(False,index=data.index)
    ).fillna(False).astype(bool)
    surprises = data[
        data["surprise_available_utc"].notna()
        & (data["surprise_available_utc"] <= stamp)
        & (data["earnings_date"].notna())
        & (data["earnings_date"] <= today)
        & safe_estimate
    ]
    if not surprises.empty:
        item = surprises.sort_values(["earnings_date", "surprise_available_utc"]).iloc[-1]
        for column in [
            "eps_estimated", "eps_actual", "eps_surprise", "eps_surprise_pct",
            "revenue_estimated", "revenue_actual", "revenue_surprise", "revenue_surprise_pct",
        ]:
            defaults[column] = _number(item.get(column))
        defaults["earnings_point_in_time_safe"] = float(bool(item.get("estimate_point_in_time_safe")))
        available = item.get("surprise_available_utc")
        if pd.notna(available):
            defaults["earnings_data_age_days"] = float((stamp - available).total_seconds() / 86400)
    return defaults


def attach_earnings_features(base: pd.DataFrame, earnings: pd.DataFrame,
                             timestamp_column: str = "ts") -> pd.DataFrame:
    if base is None or base.empty:
        return base
    out = base.copy()
    timestamps = pd.to_datetime(out[timestamp_column], utc=True, errors="coerce")
    unique = pd.Series(timestamps.dropna().unique()).sort_values()
    prepared=_prepare_earnings_for_asof(earnings)
    features = pd.DataFrame([
        {timestamp_column: stamp, **earnings_features_at(prepared, stamp, _prepared=True)} for stamp in unique
    ])
    if features.empty:
        for column, value in earnings_features_at(pd.DataFrame(), utcnow()).items():
            out[column] = value
        return out
    out[timestamp_column] = timestamps
    return out.merge(features, on=timestamp_column, how="left")


EARNINGS_VECTOR_FEATURES = [
    "days_to_earnings", "days_since_earnings", "earnings_event_today",
    "earnings_time_bmo", "earnings_time_amc", "eps_estimated", "eps_actual",
    "eps_surprise", "eps_surprise_pct", "revenue_estimated", "revenue_actual",
    "revenue_surprise", "revenue_surprise_pct", "net_income", "assets",
    "liabilities", "cash_flow", "earnings_point_in_time_safe", "earnings_data_age_days",
]


def earnings_vector_frame(ticker: str, as_of: Any) -> tuple[pd.DataFrame, dict[str, Any]]:
    try:
        data = load_processed_earnings(ticker)
    except Exception as exc:
        return pd.DataFrame(), {"present": False, "error": f"{type(exc).__name__}: {exc}"}
    if data.empty:
        return pd.DataFrame(), {"present": False, "rows": 0}
    values = earnings_features_at(data, as_of)
    return pd.DataFrame([values]), {
        "present": True, "rows": len(data), "events": int(data.get("event_id", pd.Series()).nunique()),
        "as_of_utc": _iso_utc(as_of), "point_in_time_enforced": True,
    }
