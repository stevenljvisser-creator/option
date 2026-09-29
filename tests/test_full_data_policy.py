import unittest
from datetime import date
from unittest.mock import patch
import pandas as pd

from core.modeling import load_training_rows
from core import storage


class FullDataPolicyTests(unittest.TestCase):
    def _frame(self,start):
        return pd.DataFrame({
            "ts":pd.date_range(start,periods=10,freq="min",tz="UTC"),
            "target_extrinsic_return_30m":[.01]*10,
            "stock_return_1m":[.001]*10,
        })

    def test_full_data_mode_ignores_legacy_row_cap(self):
        catalog=[
            ("NVDA",date(2025,1,2),"k1"),
            ("NVDA",date(2025,1,3),"k2"),
        ]
        frames={"k1":self._frame("2025-01-02 14:30"),"k2":self._frame("2025-01-03 14:30")}
        with patch("core.modeling.feature_catalog",return_value=catalog), \
             patch("core.modeling.get_df",side_effect=lambda key,**kw:frames[key].copy()):
            full=load_training_rows(
                ["NVDA"],date(2025,1,1),date(2025,1,5),max_rows=5,
                use_all_available=True,target="target_extrinsic_return_30m",
            )
        self.assertEqual(len(full),20)
        self.assertTrue(full.attrs["data_usage"]["use_all_available"])
        self.assertFalse(full.attrs["data_usage"]["hidden_row_cap_applied"])

    def test_s3_client_is_reused_for_identical_settings(self):
        storage._cached_client.cache_clear()
        settings={
            "HETZNER_S3_ENDPOINT":"https://example.invalid",
            "HETZNER_S3_REGION":"hel1",
            "HETZNER_S3_BUCKET":"bucket",
            "HETZNER_S3_ACCESS_KEY":"a",
            "HETZNER_S3_SECRET_KEY":"s",
        }
        def get_setting(key,default=None):
            return settings.get(key,default)
        sentinel=object()
        with patch("core.storage.get_setting",side_effect=get_setting), \
             patch("core.storage.boto3.client",return_value=sentinel) as make:
            self.assertIs(storage.client(),sentinel)
            self.assertIs(storage.client(),sentinel)
            self.assertEqual(make.call_count,1)
