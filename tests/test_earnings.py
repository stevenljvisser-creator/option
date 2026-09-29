import math
import unittest
from datetime import date
from unittest.mock import patch

import pandas as pd

from core.earnings import (
    PROCESSED_COLUMNS,build_earnings_dataset,earnings_features_at,normalize_fmp_snapshot,
    normalize_sec_companyfacts,select_fmp_point_in_time,
    append_raw_snapshot,
    load_processed_earnings,
)


def sec_payload():
    accn="0001045810-25-000001"
    common={"accn":accn,"fy":2025,"fp":"Q1","form":"10-Q","filed":"2025-05-01"}
    def duration(value,concept_unit="USD"):
        return {concept_unit:[{**common,"start":"2025-01-27","end":"2025-04-27","val":value}]}
    def instant(value):
        return {"USD":[{**common,"end":"2025-04-27","val":value}]}
    return {
        "ticker":"NVDA","cik":"0001045810",
        "companyfacts":{"entityName":"NVIDIA CORP","facts":{"us-gaap":{
            "RevenueFromContractWithCustomerExcludingAssessedTax":{"units":duration(44_000_000_000)},
            "NetIncomeLoss":{"units":duration(19_000_000_000)},
            "EarningsPerShareDiluted":{"units":duration(0.95,"USD/shares")},
            "Assets":{"units":instant(140_000_000_000)},
            "Liabilities":{"units":instant(50_000_000_000)},
            "NetCashProvidedByUsedInOperatingActivities":{"units":duration(18_000_000_000)},
        }}},
        "submissions":{"filings":{"recent":{
            "accessionNumber":[accn],"filingDate":["2025-05-01"],
            "reportDate":["2025-04-27"],"form":["10-Q"],
            "acceptanceDateTime":["2025-05-01T20:15:00Z"],
        }}},
    }


class EarningsContractTests(unittest.TestCase):
    def test_nvda_sec_fmp_combined_schema_and_surprises(self):
        fmp=[{
            "symbol":"NVDA","date":"2025-05-28","fiscalDateEnding":"2025-04-27",
            "time":"amc","epsEstimated":.90,"epsActual":.95,
            "revenueEstimated":43_000_000_000,"revenueActual":44_000_000_000,
        }]
        combined=build_earnings_dataset(
            "NVDA",sec_payload(),[("2025-05-20T12:00:00Z",fmp)],
            "2025-05-02T00:00:00Z",
        )
        self.assertTrue(set(PROCESSED_COLUMNS).issubset(combined.columns))
        row=combined.iloc[0]
        self.assertAlmostEqual(row["eps_surprise"],.05)
        self.assertAlmostEqual(row["revenue_surprise"],1_000_000_000)
        self.assertEqual(row["actual_source_revenue"],"SEC")
        self.assertTrue(bool(row["estimate_point_in_time_safe"]))
        self.assertEqual(row["form_type"],"10-Q")

    def test_pre_event_estimate_is_not_replaced_by_post_event_revision(self):
        before=[{"symbol":"NVDA","date":"2025-05-28","time":"amc","epsEstimated":.90,"revenueEstimated":43}]
        after=[{"symbol":"NVDA","date":"2025-05-28","time":"amc","epsEstimated":.97,"revenueEstimated":45,"epsActual":.95}]
        history=pd.concat([
            normalize_fmp_snapshot("NVDA",before,"2025-05-20T12:00:00Z"),
            normalize_fmp_snapshot("NVDA",after,"2025-05-29T12:00:00Z"),
        ],ignore_index=True)
        selected=select_fmp_point_in_time(history).iloc[0]
        self.assertEqual(selected["eps_estimated"],.90)
        self.assertEqual(selected["estimate_pit_status"],"observed_pre_event")
        self.assertEqual(selected["eps_actual_fmp"],.95)

    def test_unknown_release_time_uses_start_of_day_cutoff(self):
        payload=[{"symbol":"NVDA","date":"2025-05-28","epsEstimated":.90,"epsActual":.95}]
        history=normalize_fmp_snapshot("NVDA",payload,"2025-05-28T12:00:00Z")
        selected=select_fmp_point_in_time(history).iloc[0]
        self.assertFalse(bool(selected["estimate_point_in_time_safe"]))
        self.assertEqual(selected["estimate_pit_status"],"backfill_unverified")
        selected["actual_available_utc"]="2025-05-29T00:00:00Z"
        selected["surprise_available_utc"]="2025-05-29T00:00:00Z"
        selected["eps_actual"]=.95;selected["eps_surprise"]=.05
        selected["revenue_actual"]=44;selected["record_available_utc"]="2025-05-29T00:00:00Z"
        features=earnings_features_at(pd.DataFrame([selected]),"2025-05-30T20:00:00Z")
        self.assertTrue(math.isnan(features["eps_surprise"]))
        self.assertEqual(features["eps_actual"],.95)

    def test_annual_eps_is_never_subtracted_into_q4(self):
        raw=sec_payload();facts=raw["companyfacts"]["facts"]["us-gaap"]
        annual={"accn":"annual","fy":2025,"fp":"FY","form":"10-K","filed":"2026-02-20",
                "start":"2025-01-01","end":"2025-12-31","val":4.0}
        facts["EarningsPerShareDiluted"]["units"]["USD/shares"].append(annual)
        raw["submissions"]["filings"]["recent"]={
            "accessionNumber":["0001045810-25-000001","annual"],
            "filingDate":["2025-05-01","2026-02-20"],
            "reportDate":["2025-04-27","2025-12-31"],"form":["10-Q","10-K"],
            "acceptanceDateTime":["2025-05-01T20:15:00Z","2026-02-20T20:15:00Z"],
        }
        frame=normalize_sec_companyfacts("NVDA",raw,"2026-02-21T00:00:00Z")
        q4=frame[frame["fiscal_quarter"]=="Q4"].iloc[0]
        self.assertTrue(math.isnan(float(q4["eps_actual_sec"])))

    def test_raw_snapshot_is_content_addressed_and_append_only(self):
        stored=[]
        with patch("core.earnings.exists",return_value=False),patch(
            "core.earnings.put_bytes",side_effect=lambda key,*args:stored.append(key)
        ):
            key=append_raw_snapshot("sec","NVDA",sec_payload(),"2025-05-02T00:00:00Z")
        self.assertEqual(stored,[key])
        self.assertTrue(key.startswith("market-data/v5/earnings/raw/sec/NVDA/2025/05/02/"))
        with patch("core.earnings.exists",return_value=True),patch("core.earnings.put_bytes") as put:
            same=append_raw_snapshot("sec","NVDA",sec_payload(),"2025-05-02T00:00:00Z")
        self.assertEqual(key,same);put.assert_not_called()

    def test_latest_processed_snapshot_uses_observation_stamp_not_range_folder(self):
        older="market-data/v5/earnings/processed/NVDA/2025-01-01_2025-12-31/20250102T000000000000Z_old.csv.gz"
        newer="market-data/v5/earnings/processed/NVDA/2022-01-01_2026-12-31/20260102T000000000000Z_new.csv.gz"
        with patch("core.earnings.list_keys",return_value=iter([newer,older])),patch(
            "core.earnings.get_df",return_value=pd.DataFrame({"ticker":["NVDA"]})
        ) as get:
            load_processed_earnings("NVDA")
        get.assert_called_once_with(newer)


if __name__=="__main__":
    unittest.main()

class EarningsAllTickerContractTests(unittest.TestCase):
    def test_company_pipeline_is_not_hardcoded_to_nvda(self):
        tickers=["NVDA","AMD","AVGO","AAPL","MSFT","GOOGL","META","AMZN","TSLA","JPM","BAC","GS","LLY","UNH","XOM","CVX","CAT","BA","WMT","COST"]
        for ticker in tickers:
            fmp=[{
                "symbol":ticker,"date":"2025-05-28","fiscalDateEnding":"2025-04-27",
                "time":"amc","epsEstimated":.90,"epsActual":.95,
                "revenueEstimated":43_000_000_000,"revenueActual":44_000_000_000,
            }]
            combined=build_earnings_dataset(
                ticker,sec_payload(),[("2025-05-20T12:00:00Z",fmp)],
                "2025-05-02T00:00:00Z",
            )
            self.assertFalse(combined.empty,ticker)
            self.assertEqual(set(combined["ticker"].dropna().astype(str)),{ticker})
            self.assertAlmostEqual(float(combined.iloc[0]["eps_surprise"]),.05)
