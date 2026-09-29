import os

DATABASE_URL=os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg://marketscope:marketscope_internal@postgres:5432/marketscope"
)

# Existing MarketScope 10/11 raw data stays exactly where it is.
RAW_ROOT="market-data/v2"

# All new modular datasets/features live here.
DATA_V3_ROOT="market-data/v3"
OPTIONEDGE_ROOT="market-data/v4"
TRADES_ROOT=f"{DATA_V3_ROOT}/option-trades"
TRADE_FEATURE_ROOT=f"{DATA_V3_ROOT}/option-trade-minute"
OI_ROOT=f"{DATA_V3_ROOT}/open-interest"
EARNINGS_ROOT=f"{DATA_V3_ROOT}/earnings"
SENTIMENT_ROOT=f"{DATA_V3_ROOT}/sentiment"
FEATURE_ROOT=f"{OPTIONEDGE_ROOT}/features"
QUOTE_ROOT=f"{DATA_V3_ROOT}/quotes"          # reserved for later
MODEL_ROOT=f"{DATA_V3_ROOT}/models"

MACRO_ROOT=f"{DATA_V3_ROOT}/macro"
PAIR_ROOT=f"{OPTIONEDGE_ROOT}/option-pairs"
RELATIONSHIP_ROOT=f"{OPTIONEDGE_ROOT}/relationships"
PAIR_MODEL_ROOT=f"{OPTIONEDGE_ROOT}/pair-models"

GF_ROOT=f"{DATA_V3_ROOT}/google-finance"
GF_NEWS_ROOT=f"{GF_ROOT}/news"
GF_SNAPSHOT_ROOT=f"{GF_ROOT}/snapshots"
RICH_NEWS_ROOT=f"{DATA_V3_ROOT}/news-rich"
INVENTORY_ROOT=f"{DATA_V3_ROOT}/inventory"
INVENTORY_RECEIPT_ROOT=f"{INVENTORY_ROOT}/receipts"
INVENTORY_SNAPSHOT_ROOT=f"{INVENTORY_ROOT}/snapshots"

DEFAULT_IMPORT_CHUNK_ROWS=300000
SETTINGS_KEY_FILE=os.getenv("SETTINGS_KEY_FILE","/marketscope-secrets/fernet.key")

GPU_SENTIMENT_ROOT=f"{OPTIONEDGE_ROOT}/professional-news"


# OptionEdge 1.0 clean research layer.
# Legacy v2/v3 source data remains untouched. Legacy news/features are deliberately
# excluded from the new model because the user rejected their quality.
PRO_NEWS_ROOT=f"{OPTIONEDGE_ROOT}/professional-news"
PRO_NEWS_STATUS_ROOT=f"{OPTIONEDGE_ROOT}/gpu-agent"
SIGNAL_EVENT_ROOT=f"{OPTIONEDGE_ROOT}/events"
CLEAN_SENTIMENT_ROOT=PRO_NEWS_ROOT
CLEAN_FEATURE_ROOT=f"{OPTIONEDGE_ROOT}/features"
CLEAN_PAIR_ROOT=f"{OPTIONEDGE_ROOT}/option-pairs"
CLEAN_RELATIONSHIP_ROOT=f"{OPTIONEDGE_ROOT}/relationships"
CLEAN_PAIR_MODEL_ROOT=f"{OPTIONEDGE_ROOT}/pair-models"

# OptionEdge 2.0 fusion research layer.  v2/v3/v4 remain immutable inputs so
# an upgrade never destroys the user's historical data or earlier experiments.
FUSION_ROOT="market-data/v5"
NEWS_ARTICLE_ROOT=f"{FUSION_ROOT}/news-articles"
NEWS_CHUNK_ROOT=f"{FUSION_ROOT}/news-chunks"
NEWS_EMBEDDING_ROOT=f"{FUSION_ROOT}/news-embeddings"
NEWS_EVENT_ROOT=f"{FUSION_ROOT}/news-events"
DAILY_VECTOR_ROOT=f"{FUSION_ROOT}/daily-vectors"
DAILY_LABEL_ROOT=f"{FUSION_ROOT}/daily-labels"
DEEP_RELATIONSHIP_ROOT=f"{FUSION_ROOT}/deep-relationships"
FUSION_MODEL_ROOT=f"{FUSION_ROOT}/fusion-models"
GPU_LLM_STATUS_ROOT=f"{FUSION_ROOT}/gpu-news-agent"

# Append-only earnings layer.  The legacy v3 Massive/Benzinga files above stay
# untouched for audit/backwards compatibility; OptionEdge 2.0 reads these v5
# SEC/FMP objects for new imports and point-in-time features.
EARNINGS_V5_ROOT=f"{FUSION_ROOT}/earnings"
EARNINGS_SEC_RAW_ROOT=f"{EARNINGS_V5_ROOT}/raw/sec"
EARNINGS_FMP_RAW_ROOT=f"{EARNINGS_V5_ROOT}/raw/fmp"
EARNINGS_PROCESSED_ROOT=f"{EARNINGS_V5_ROOT}/processed"
EARNINGS_STATUS_ROOT=f"{EARNINGS_V5_ROOT}/status"

# Alpha-first research and trading-control layers.  These prefixes are new and
# append-only; existing v2/v3/v4/v5 observations and model artifacts remain
# untouched and continue to act as inputs/benchmarks.
ALPHA_RESEARCH_ROOT=f"{FUSION_ROOT}/alpha-research"
EXPERIMENT_ROOT=f"{ALPHA_RESEARCH_ROOT}/experiments"
PAPER_TRADING_ROOT=f"{FUSION_ROOT}/paper-trading"
TRADING_CONTROL_ROOT=f"{FUSION_ROOT}/trading-control"
MONITORING_ROOT=f"{FUSION_ROOT}/monitoring"
