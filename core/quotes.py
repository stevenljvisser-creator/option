"""
Reserved quote feature layer.

MarketScope 12 deliberately trains without quote data when the Massive plan
does not provide historical quotes. The schema is already fixed so a future
quote-agent can populate these columns without changing/re-downloading the
other datasets.
"""
import numpy as np

QUOTE_FEATURES=[
    "bid","ask","midprice","spread","relative_spread","bid_size","ask_size"
]

def ensure_quote_columns(df):
    for c in QUOTE_FEATURES:
        if c not in df.columns:
            df[c]=np.nan
    return df

def quotes_available(df):
    return any(c in df.columns and df[c].notna().any() for c in QUOTE_FEATURES)
