from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from unittest import TestCase, mock

import numpy as np
import pandas as pd

from api.main import _latest_fusion_inputs
from core.data_readiness import DEFAULT_TICKERS, REQUIREMENTS, readiness
from core.deep_relationships import benjamini_hochberg, relationship_scan
from core.fusion_modeling import _time_partitions, daily_relationship_frame, predict_fusion_rows
from core.multi_horizon import _daily_close_anchor
from core.research_design import design_payload
from core.storage_inventory import classify_object
from core.vector_fusion import (
    EVENT_FEATURE_NAMES, FORBIDDEN_INPUT_TOKENS, candidate_from_raw, candidate_layout,
    market_close_utc, pool_news_embeddings, primary_layout_is_valid,
    summarize_frame, summarize_news_events,
)


class VectorContractTests(TestCase):
    def test_primary_layout(self):
        payload = design_payload()
        self.assertTrue(primary_layout_is_valid())
        self.assertEqual(payload["primary_daily_dimensions"], 384)
        self.assertEqual([x["dimensions"] for x in payload["vector_layout"]], [192, 96, 48, 32, 16])

    def test_all_dimension_candidates_are_exact(self):
        rng = np.random.default_rng(42)
        raw = [rng.normal(size=n).astype(np.float32) for n in [1024, 210, 120, 80, 16]]
        for dimensions in [128, 256, 384, 512, 768]:
            layout = candidate_layout(dimensions)
            self.assertEqual(sum(layout.values()), dimensions)
            vector = candidate_from_raw(*raw, dimensions)
            self.assertEqual(vector.shape, (dimensions,))
            self.assertTrue(np.isfinite(vector).all())

    def test_future_columns_are_never_summarized(self):
        frame = pd.DataFrame({
            "safe_feature": [1, 2, 3],
            "future_return": [999, 999, 999],
            "target_payoff": [999, 999, 999],
            "label_up": [1, 1, 1],
        })
        _, metadata = summarize_frame(frame)
        self.assertEqual(metadata["columns"], ["safe_feature"])
        self.assertTrue(all(any(token in col for token in FORBIDDEN_INPUT_TOKENS)
                            for col in ["future_return", "target_payoff", "label_up"]))

    def test_market_close_handles_dst(self):
        self.assertEqual(market_close_utc(date(2025, 1, 15)).hour, 21)
        self.assertEqual(market_close_utc(date(2025, 7, 15)).hour, 20)

    def test_attention_pooling(self):
        rng = np.random.default_rng(7)
        embeddings = rng.normal(size=(6, 1024)).astype(np.float32)
        pooled, metadata = pool_news_embeddings(embeddings, relevance=np.linspace(0, 1, 6))
        self.assertEqual(pooled.shape, (1024,))
        self.assertAlmostEqual(float(np.linalg.norm(pooled)), 1.0, places=5)
        self.assertEqual(metadata["chunks"], 6)

    def test_llm_events_are_encoded_and_filtered_at_asof(self):
        payload = {"events": [
            {"event_type": "guidance", "direction": "positive", "market_impact": "high",
             "time_horizon": "days", "expected_vs_surprise": "surprise", "importance": .9,
             "ticker_relevance": .95, "uncertainty": .1, "novelty": .8,
             "available_utc": "2025-01-02T19:00:00Z", "facts": ["raised outlook"]},
            {"event_type": "legal", "direction": "negative", "market_impact": "high",
             "time_horizon": "months", "expected_vs_surprise": "surprise", "importance": 1,
             "available_utc": "2025-01-02T22:00:00Z"},
        ]}
        vector, metadata = summarize_news_events(
            payload, datetime(2025, 1, 2, 21, 0, tzinfo=timezone.utc)
        )
        self.assertEqual(vector.shape, (len(EVENT_FEATURE_NAMES),))
        self.assertEqual(metadata["events"], 1)
        self.assertEqual(metadata["excluded_after_asof"], 1)
        self.assertAlmostEqual(vector[EVENT_FEATURE_NAMES.index("event_type_guidance_share")], 1.0)

    def test_llm_event_summary_changes_news_projection(self):
        raw = [np.ones(n, dtype=np.float32) for n in [1024, 210, 120, 80, 16]]
        plain = candidate_from_raw(*raw, 384, np.zeros(len(EVENT_FEATURE_NAMES), dtype=np.float32))
        eventful = candidate_from_raw(*raw, 384, np.ones(len(EVENT_FEATURE_NAMES), dtype=np.float32))
        self.assertFalse(np.allclose(plain[:192], eventful[:192]))
        self.assertTrue(np.allclose(plain[192:], eventful[192:]))


class StatisticalGuardTests(TestCase):
    def test_daily_target_uses_fixed_close_and_drops_stale_contracts(self):
        frame = pd.DataFrame({
            "minute": ["2025-01-02T20:00:00Z", "2025-01-02T20:50:00Z", "2025-01-02T19:00:00Z"],
            "expiry": pd.to_datetime(["2025-02-21"] * 3, utc=True),
            "strike": [140.0, 140.0, 150.0],
            "call_extrinsic": [2.0, 2.5, 1.0], "put_extrinsic": [1.0, 1.2, 2.0],
            "close_stock": [139.0, 140.0, 140.0],
        })
        result = _daily_close_anchor(frame, date(2025, 1, 2), max_staleness_minutes=30)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["strike"], 140.0)
        self.assertEqual(result.iloc[0]["minute"], pd.Timestamp("2025-01-02T21:00:00Z"))
        self.assertEqual(result.iloc[0]["source_observed_minute"], pd.Timestamp("2025-01-02T20:50:00Z"))
        self.assertAlmostEqual(result.iloc[0]["source_staleness_minutes"], 10.0)

    def test_benjamini_hochberg(self):
        result = benjamini_hochberg([.001, .01, .04, .7])
        self.assertTrue(np.all((result >= 0) & (result <= 1)))
        self.assertTrue(np.all(np.diff(np.sort(result)) >= 0))
        self.assertLessEqual(result[0], .01)

    def test_relationship_requires_later_and_within_ticker_confirmation(self):
        rng = np.random.default_rng(21)
        rows = []
        start = datetime(2023, 1, 2, 21, tzinfo=timezone.utc)
        tickers = [f"T{i:02d}" for i in range(10)]
        for day_index in range(90):
            for ticker_index, ticker in enumerate(tickers):
                signal = rng.normal() + ticker_index * .08
                rows.append({
                    "source_ticker": ticker,
                    "minute": start + timedelta(days=day_index),
                    "signal": signal,
                    "noise": rng.normal(),
                    "future_call_extrinsic_return": .8 * signal + rng.normal(scale=.25),
                })
        result = relationship_scan(pd.DataFrame(rows), "future_call_extrinsic_return", min_support=60)
        names = {x["feature"] for x in result["confirmed"]}
        self.assertIn("signal", names)
        signal = next(x for x in result["confirmed"] if x["feature"] == "signal")
        self.assertTrue(signal["confirmed_global"])
        self.assertTrue(signal["confirmed_within_ticker"])
        self.assertFalse(result["causal_claim"])

    def test_partition_purges_earlier_tail(self):
        days = [date(2024, 1, 1) + timedelta(days=i) for i in range(100)]
        frame = pd.DataFrame({"source_day": days, "minute": pd.to_datetime(days, utc=True)})
        parts = _time_partitions(frame, purge_sessions=5)
        self.assertLess(max(parts["train"]["source_day"]), min(parts["validation"]["source_day"]))
        self.assertGreater((min(parts["validation"]["source_day"]) - max(parts["train"]["source_day"])).days, 5)
        self.assertGreater((min(parts["test"]["source_day"]) - max(parts["calibration"]["source_day"])).days, 5)

    def test_contract_rows_collapse_to_one_ticker_day(self):
        rows = []
        for ticker in ["AAA", "BBB"]:
            for strike in [90, 100, 110]:
                rows.append({
                    "source_ticker": ticker, "source_day": date(2025, 1, 2),
                    "minute": "2025-01-02T20:59:00Z", "expiry": "2025-02-01",
                    "strike": strike, "safe_feature": strike / 100,
                    "future_stock_return": .01 if ticker == "AAA" else -.01,
                    "future_call_extrinsic_return": .02, "future_put_extrinsic_return": -.01,
                })
        with mock.patch(
            "core.fusion_modeling.attach_vectors",
            side_effect=lambda frame, dimensions, raw_cache, **kwargs: frame,
        ):
            daily = daily_relationship_frame(pd.DataFrame(rows))
        self.assertEqual(len(daily), 2)
        self.assertEqual(set(daily["future_stock_up"]), {0, 1})


class ReadinessTests(TestCase):
    def test_gpu_coverage_receipt_is_inventory_data(self):
        dataset, match = classify_object(
            "market-data/v5/gpu-news-agent/receipts/NVDA/2025/2025-01-01_2025-01-31.json"
        )
        self.assertEqual(dataset, "news_coverage")
        self.assertEqual(match.groups(), ("NVDA", "2025-01-01", "2025-01-31"))

    def test_empty_inventory_blocks_training(self):
        result = readiness({}, DEFAULT_TICKERS, date(2025, 1, 1))
        self.assertFalse(result["training_ready"])
        self.assertGreater(result["blocker_count"], 0)

    def test_complete_required_inventory_opens_gate(self):
        datasets = {}
        all_tickers = DEFAULT_TICKERS + ["SPY", "QQQ"]
        for requirement in REQUIREMENTS:
            if requirement["needed"] != "verplicht":
                continue
            name = requirement["datasets"][0]
            datasets[name] = {
                "objects": 100, "bytes": 1000, "date_count": 800,
                "first_date": "2021-09-01", "last_date": "2025-01-02",
                "tickers": all_tickers, "ticker_count": len(all_tickers),
            }
        result = readiness({"generated_at": "2025-01-03T00:00:00Z", "datasets": datasets},
                           DEFAULT_TICKERS, date(2025, 1, 1))
        self.assertTrue(result["training_ready"], result["blockers"])
        self.assertEqual(result["blocker_count"], 0)

    def test_v4_compatibility_news_does_not_satisfy_v5_raw_news_gate(self):
        result = readiness({
            "generated_at": "2025-01-03T00:00:00Z",
            "datasets": {
                "professional_news": {
                    "objects": 100, "first_date": "2021-09-01", "last_date": "2025-01-02",
                    "tickers": DEFAULT_TICKERS, "ticker_count": len(DEFAULT_TICKERS),
                }
            },
        }, DEFAULT_TICKERS, date(2025, 1, 1))
        row = next(x for x in result["rows"] if x["id"] == "news_raw")
        self.assertEqual(row["state"], "ontbreekt")
        self.assertIsNone(row["dataset_found"])


class _ConstantProbability:
    def __init__(self, probability):
        self.probability = probability

    def predict_proba(self, frame):
        positive = np.full(len(frame), self.probability, dtype=float)
        return np.column_stack([1 - positive, positive])


class _ConstantReturn:
    def __init__(self, value):
        self.value = value

    def predict(self, frame):
        return np.full(len(frame), self.value, dtype=float)


class FusionInferenceTests(TestCase):
    def test_status_selects_newest_date_shared_by_vector_and_pair_store(self):
        vectors = iter([
            "market-data/v5/daily-vectors/NVDA/2025/01/2025-01-03.npz",
            "market-data/v5/daily-vectors/NVDA/2025/01/2025-01-02.npz",
        ])
        expected_pair = "market-data/v4/option-pairs/h30/NVDA/2025/01/2025-01-02.csv.gz"
        with mock.patch("core.storage.list_keys", return_value=vectors), mock.patch(
            "api.main.exists", side_effect=lambda key: key == expected_pair
        ):
            result = _latest_fusion_inputs("nvda")
        self.assertEqual(result["day"], date(2025, 1, 2))
        self.assertEqual(result["pair_key"], expected_pair)

    def test_prediction_exposes_probability_expected_value_and_edge(self):
        pair = pd.DataFrame({
            "source_ticker": ["NVDA", "NVDA"],
            "source_day": [date(2025, 1, 2), date(2025, 1, 2)],
            "minute": ["2025-01-02T20:58:00Z", "2025-01-02T20:59:00Z"],
            "expiry": ["2025-02-21", "2025-02-21"],
            "strike": [140.0, 140.0],
            "call_extrinsic": [2.0, 2.5],
            "put_extrinsic": [1.5, 1.25],
        })
        raw = {
            "news_raw": np.ones(1024, dtype=np.float32),
            "options_raw": np.ones(210, dtype=np.float32),
            "stock_raw": np.ones(120, dtype=np.float32),
            "expectation_raw": np.ones(80, dtype=np.float32),
            "quality_raw": np.ones(16, dtype=np.float32),
        }
        features = ["call_extrinsic", "put_extrinsic"]
        model = {
            "dimensions": 128,
            "call": {"model": _ConstantProbability(.72), "calibrator": None, "features": features},
            "put": {"model": _ConstantProbability(.41), "calibrator": None, "features": features},
            "call_return_model": _ConstantReturn(.20),
            "put_return_model": _ConstantReturn(-.10),
        }
        result = predict_fusion_rows(pair, raw, model)
        self.assertEqual(len(result), 1)
        row = result.iloc[0]
        self.assertAlmostEqual(row["call_positive_probability"], .72)
        self.assertAlmostEqual(row["model_expected_call_extrinsic"], 3.0)
        self.assertAlmostEqual(row["call_gross_edge"], .5)
        self.assertEqual(row["best_side"], "CALL")
        self.assertEqual(row["fusion_dimensions"], 128)

    def test_missing_pair_columns_fail_explicitly(self):
        with self.assertRaisesRegex(ValueError, "put_extrinsic"):
            predict_fusion_rows(
                pd.DataFrame({"minute": ["2025-01-02T20:59:00Z"], "expiry": ["2025-02-21"],
                              "strike": [140.0], "call_extrinsic": [2.0]}),
                {}, {"dimensions": 128, "call": {}, "put": {},
                     "call_return_model": object(), "put_return_model": object()},
            )
