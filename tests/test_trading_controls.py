import unittest

import numpy as np
import pandas as pd

from core.feature_engine import _attach_future_option_targets
from core.monitoring import monitor_performance,population_stability_index
from core.trading_controls import (
    PortfolioState,RiskPolicy,TradeCandidate,evaluate_trade,live_promotion_gate,
)


class TradingControlTests(unittest.TestCase):
    def candidate(self,**changes):
        values=dict(
            ticker="NVDA",side="CALL",expected_return=.25,expected_cost_return=.04,
            uncertainty=.05,option_price=2.0,relative_spread=.04,
            open_interest=1000,daily_volume=500,data_age_minutes=1,
        )
        values.update(changes)
        return TradeCandidate(**values)

    def test_edge_and_risk_layer_can_trade_but_sizes_position(self):
        decision=evaluate_trade(self.candidate(),PortfolioState(equity=100_000))
        self.assertTrue(decision.allowed)
        self.assertGreater(decision.contracts,0)

    def test_kill_switch_for_leakage(self):
        decision=evaluate_trade(
            self.candidate(leakage_valid=False),PortfolioState(equity=100_000)
        )
        self.assertFalse(decision.allowed)
        self.assertTrue(decision.kill_switch)
        self.assertIn("leakage_audit_mislukt",decision.reasons)

    def test_unknown_spread_is_no_trade(self):
        decision=evaluate_trade(self.candidate(relative_spread=None),PortfolioState(equity=100_000))
        self.assertEqual(decision.decision,"NO_TRADE")
        self.assertIn("spread_onbekend",decision.reasons)

    def test_live_gate_is_fail_closed(self):
        gate=live_promotion_gate(None,None,data_healthy=False)
        self.assertFalse(gate["eligible"])
        self.assertFalse(gate["live_execution_enabled"])

    def test_monitoring_detects_bad_live_ev(self):
        frame=pd.DataFrame({
            "expected_net_return":[.02]*40,"actual_net_return":[-.03]*40,
            "probability_positive":[.8]*40,"positive_outcome":[0]*40,
            "relative_spread":[.2]*40,"open_interest":[10]*40,
        })
        status=monitor_performance(frame)
        self.assertTrue(status["kill_switch"])
        self.assertIn("negatieve_live_ev",status["triggers"])

    def test_population_stability_index(self):
        ref=np.linspace(0,1,100);cur=np.linspace(2,3,100)
        self.assertGreater(population_stability_index(ref,cur),.25)


class FeatureStoreSpeedContractTests(unittest.TestCase):
    def test_vectorized_target_join_is_contract_isolated(self):
        rows=[]
        base=pd.Timestamp("2025-01-02 14:30",tz="UTC")
        for contract,offset in [("O:NVDA260101C00100000",0),("O:NVDA260101P00100000",100)]:
            for minute in [0,1,2,4,5]:
                rows.append({"ticker":contract,"ts":base+pd.Timedelta(minutes=minute),
                             "close_option":offset+minute+1.0,"extrinsic":offset+minute+.5})
        frame=pd.DataFrame(rows)
        result=_attach_future_option_targets(frame,2)
        first=result[(result["ticker"].str.contains("C"))&(result["ts"]==base)].iloc[0]
        self.assertEqual(first["target_ts"],base+pd.Timedelta(minutes=2))
        self.assertEqual(first["future_option_close"],3.0)
        self.assertEqual(len(result),len(frame))


if __name__=="__main__":
    unittest.main()
