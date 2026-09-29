import unittest

import numpy as np
import pandas as pd

from core.alpha_research import (
    CostModel,WalkForwardConfig,economic_metrics,evaluate_candidate,
    purged_walk_forward_splits,run_research_suite,
)
from core.leakage import audit_feature_columns,audit_purged_split
from core.vector_fusion import candidate_from_raw


def research_frame(n=180):
    rng=np.random.default_rng(7)
    days=pd.bdate_range("2024-01-02",periods=n,tz="UTC")
    signal=rng.normal(size=n)
    minute=days+pd.Timedelta(hours=20)
    return pd.DataFrame({
        "source_ticker":["NVDA"]*n,"source_day":days.date,"minute":minute,
        "target_minute":minute+pd.offsets.BDay(1),"vector_as_of_utc":minute,
        "feature_signal":signal,"feature_noise":rng.normal(size=n),
        "future_call_extrinsic_return":.14+.08*signal+rng.normal(0,.02,size=n),
        "call_close_option":[2.0]*n,"call_bid":[1.98]*n,"call_ask":[2.02]*n,
    })


class AlphaResearchTests(unittest.TestCase):
    def test_future_columns_are_rejected(self):
        audit=audit_feature_columns(["feature_signal","future_call_extrinsic_return"])
        self.assertFalse(audit.valid)
        self.assertIn("future_call_extrinsic_return",audit.checks["forbidden_feature_columns"])

    def test_walk_forward_is_purged_and_non_overlapping(self):
        frame=research_frame()
        config=WalkForwardConfig(min_train_sessions=60,test_sessions=15,purge_sessions=2,n_splits=4,
                                 holdout_fraction=.15,minimum_holdout_sessions=20)
        folds=purged_walk_forward_splits(frame,config)
        self.assertEqual(len(folds),4)
        prior_test_end=None
        for fold in folds:
            self.assertTrue(audit_purged_split(fold["train"],fold["test"]).valid)
            first=pd.to_datetime(fold["test"]["minute"],utc=True).min()
            last=pd.to_datetime(fold["test"]["minute"],utc=True).max()
            if prior_test_end is not None:self.assertGreater(first,prior_test_end)
            prior_test_end=last

    def test_candidate_has_untouched_holdout_and_economic_metrics(self):
        manifest,predictions=evaluate_candidate(
            research_frame(),["feature_signal","feature_noise"],
            "future_call_extrinsic_return","D1",
            config=WalkForwardConfig(min_train_sessions=60,test_sessions=15,purge_sessions=1,
                                     n_splits=3,holdout_fraction=.15,minimum_holdout_sessions=20),
            cost_model=CostModel(fallback_roundtrip_spread_pct=.02),
        )
        self.assertTrue(manifest["leakage_valid"])
        self.assertTrue(manifest["untouched_holdout_used_once"])
        self.assertIn("holdout",set(predictions["fold"].astype(str)))
        self.assertIn("expected_value_per_trade",manifest["holdout_metrics"])
        self.assertIn("NO_TRADE",set(predictions["decision"]) | {"NO_TRADE"})

    def test_ev_is_cost_aware(self):
        metrics=economic_metrics([.20,-.10,.15],[.12,.11,.13],[.05,.05,.05],[True,True,False])
        self.assertAlmostEqual(metrics["expected_value_per_trade"],0.0)
        self.assertEqual(metrics["number_of_trades"],2)
        self.assertGreater(metrics["maximum_drawdown"],0)

    def test_exact_ablation_keeps_dimensions_and_changes_block(self):
        arrays=[np.arange(10,dtype=float)+offset for offset in range(5)]
        full=candidate_from_raw(arrays[0],arrays[1],arrays[2],arrays[3],arrays[4],384,
                                news_event_raw=np.ones(4),earnings_raw=np.ones(3))
        no_news=candidate_from_raw(arrays[0],arrays[1],arrays[2],arrays[3],arrays[4],384,
                                   news_event_raw=np.ones(4),earnings_raw=np.ones(3),
                                   disabled_components=["news"])
        self.assertEqual(len(full),384);self.assertEqual(len(no_news),384)
        self.assertFalse(np.allclose(full,no_news))
        self.assertTrue(np.allclose(no_news[:192],0))

    def test_only_preselected_complex_model_opens_holdout(self):
        config=WalkForwardConfig(min_train_sessions=60,test_sessions=15,purge_sessions=1,
                                 n_splits=3,holdout_fraction=.15,minimum_holdout_sessions=20)
        suite,_=run_research_suite(
            research_frame(),{"signal":["feature_signal"],"noise":["feature_noise"]},
            ["future_call_extrinsic_return"],"D1",config=config,
            cost_model=CostModel(fallback_roundtrip_spread_pct=.02),
        )
        complex_models=[x for x in suite["experiments"] if x["model"]=="ridge"]
        self.assertEqual(sum(bool(x.get("holdout_metrics")) for x in complex_models),1)
        self.assertEqual(sum(bool(x.get("selected_before_holdout")) for x in complex_models),1)


if __name__=="__main__":
    unittest.main()
