"""Authoritative research design exposed by the API and Streamlit UI.

Keep the research contract in one place.  The application may change its UI,
but the meaning of a vector coordinate, the data gates, and the leakage rules
must remain versioned and auditable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


RESEARCH_VERSION = "20.0"
RESEARCH_START = "2022-09-16"
NEWS_CONTEXT_START = "2021-09-16"


@dataclass(frozen=True)
class VectorBlock:
    code: str
    label: str
    start: int
    stop: int
    source: str
    meaning: str

    @property
    def dimensions(self) -> int:
        return self.stop - self.start


VECTOR_BLOCKS = (
    VectorBlock(
        "news", "Nieuws & gebeurtenissen", 0, 192, "1024D chunk embeddings + LLM event JSON",
        "Attention-pooling van uitsluitend nieuws dat vóór de as-of-grens bekend was.",
    ),
    VectorBlock(
        "options", "Optie-oppervlak", 192, 288, "call/put, moneyness, expiratie, IV, volume en trades",
        "Vaste samenvatting van de optie-surface en intraday optieactiviteit.",
    ),
    VectorBlock(
        "stock", "Aandeel & markt", 288, 336, "OHLCV, returns, volatiliteit en marktcontext",
        "Point-in-time prijs-, volume- en relatieve-marktkenmerken.",
    ),
    VectorBlock(
        "expectation", "Verwachtingswaarde", 336, 368, "theorie versus waarneembare marktprijs",
        "Theoretisch verdisconteerde waarde én praktisch waargenomen optie-/tijdwaarde, parity residual en expectation gaps.",
    ),
    VectorBlock(
        "quality", "Kwaliteit & masks", 368, 384, "dekking, ontbrekende modaliteiten en tijdkwaliteit",
        "Maakt ontbrekende of onbetrouwbare input expliciet in plaats van die stil met nul te vullen.",
    ),
)

DIMENSION_CANDIDATES = (128, 256, 384, 512, 768)
RAW_NEWS_DIMENSIONS = 1024
PRIMARY_OBJECTIVE = (
    "Een robuust handelsmodel onderzoeken dat uitsluitend na positieve out-of-sample "
    "verwachtingswaarde, realistische kosten, aanvaardbaar risico en overtuigende paper trading "
    "in aanmerking kan komen voor beperkte live inzet. Winstgevendheid is niet gegarandeerd."
)

ECONOMIC_METRICS = (
    "EV_net", "net_return", "expected_value_per_trade", "Sharpe", "Sortino",
    "profit_factor", "maximum_drawdown", "win_rate", "average_win",
    "average_loss", "payoff_ratio", "turnover", "exposure", "number_of_trades",
)

TRADING_ARCHITECTURE = (
    "data", "features", "model", "expected_edge", "risk_manager", "execution",
)

DEVELOPMENT_PRIORITIES = (
    "datakwaliteit", "timestamps_en_leakage", "betrouwbare_features", "simpele_baselines",
    "alpha_research", "purged_walk_forward", "realistische_kosten", "informatiefusie",
    "risicomodel", "paper_trading", "live_monitoring", "beperkte_live_trading",
)

LIVE_PROMOTION_GATES = (
    "positieve out-of-sample EV", "positief na realistische kosten",
    "stabiel over meerdere perioden/tickers/regimes", "acceptabele drawdown",
    "niet gedreven door enkele uitzonderlijke trades", "paper trading bevestigt",
    "stabiele datastroom", "automatische risicocontrole", "kill switch getest",
)


LEAKAGE_RULES = (
    "Elke observatie krijgt een as_of_utc. Alleen records met effective_available_utc <= as_of_utc zijn input; voor gewone webbackfills is dat minimaal first_seen_utc.",
    "Een artikelupdate wordt een nieuwe revisie; de latere tekst vervangt de historische tekst niet.",
    "De baselineprojectie en nieuwsattention zijn target-vrij en versie-vast; schaling, modelschatting en kalibratie worden alleen op de toegestane trainings-/kalibratieperioden gefit.",
    "De gerealiseerde payoff, toekomstige extrinsieke waarde en toekomstige koers zijn alleen labels.",
    "Train/validatie/test zijn chronologisch, met purge en embargo rondom iedere targethorizon.",
    "Nieuwsduplicaten en syndicatienieuws worden vóór pooling geclusterd om kunstmatig gewicht te voorkomen.",
)


TARGETS = (
    {
        "code": "realized_discounted_payoff",
        "label": "Werkelijke verdisconteerde payoff op expiratie",
        "role": "target_only",
        "note": "Nooit als input in de vector van dezelfde observatie.",
    },
    {
        "code": "future_extrinsic_return",
        "label": "Toekomstige verandering extrinsieke waarde",
        "role": "target_only",
        "note": "Apart per call/put en per horizon.",
    },
    {
        "code": "future_abnormal_stock_return",
        "label": "Toekomstig abnormaal aandeelrendement",
        "role": "target_only",
        "note": "Aandeelrendement minus vooraf gedefinieerde markt/sectorbenchmark.",
    },
    {
        "code": "positive_net_option_outcome",
        "label": "Positieve netto optie-uitkomst",
        "role": "target_only_after_quotes",
        "note": "Pas valide na bid/ask, spread, slippage en uitvoerbaarheidsregels.",
    },
)


RELATIONSHIP_METHODS = (
    {
        "method": "Tijdgestuurde ablaties",
        "question": "Welke laag voegt economische waarde toe boven de simpele baseline?",
        "guard": "Exact dezelfde purged folds voor alle lagen; slechts één vooraf geselecteerd complex model opent de hold-out.",
        "status": "geïmplementeerd: nieuws, embeddings, opties, earnings, technisch en theoretische waarde",
    },
    {
        "method": "Embedding-neighbours",
        "question": "Wat gebeurde na historisch vergelijkbare ticker-dagen?",
        "guard": "Alleen eerdere dagen als buren; ticker- en tijdsafstand rapporteren.",
        "status": "nog te implementeren na voldoende dagvectoren",
    },
    {
        "method": "Event studies",
        "question": "Welke abnormal returns volgen op specifieke LLM-events?",
        "guard": "Vooraf vastgelegde eventvensters en marktbenchmark.",
        "status": "gedeeltelijk: eventvelden in FDR-scan; formele eventstudy nog nodig",
    },
    {
        "method": "Lagged association",
        "question": "Loopt nieuws, optieactiviteit of koersreactie voor?",
        "guard": "Lags vooraf definiëren; Benjamini-Hochberg-correctie.",
        "status": "nog te implementeren na exchange-calendar en sectordata",
    },
    {
        "method": "Conditionele patterns",
        "question": "Welke combinaties werken alleen onder bepaalde regimes?",
        "guard": "Ontdekken op train; éénmalig toetsen op validatie en eindtest.",
        "status": "basis geïmplementeerd: periode, ticker, sector, markt-, volatiliteits- en eventregime; interactiemodellen volgen na voldoende support",
    },
    {
        "method": "Gekalibreerde voorspelling",
        "question": "Zijn voorspelde kansen empirisch betrouwbaar?",
        "guard": "Brier, calibration slope/intercept, Wilson-CI en final hold-out.",
        "status": "gedeeltelijk: Brier/AUC/calibratie; slope/intercept en CI volgen",
    },
)


def vector_layout() -> list[dict[str, Any]]:
    return [dict(asdict(block), dimensions=block.dimensions) for block in VECTOR_BLOCKS]


def design_payload() -> dict[str, Any]:
    return {
        "version": RESEARCH_VERSION,
        "research_start": RESEARCH_START,
        "news_context_start": NEWS_CONTEXT_START,
        "primary_daily_dimensions": 384,
        "raw_news_dimensions": RAW_NEWS_DIMENSIONS,
        "dimension_candidates": list(DIMENSION_CANDIDATES),
        "vector_layout": vector_layout(),
        "leakage_rules": list(LEAKAGE_RULES),
        "targets": list(TARGETS),
        "relationship_methods": list(RELATIONSHIP_METHODS),
        "primary_objective": PRIMARY_OBJECTIVE,
        "economic_metrics": list(ECONOMIC_METRICS),
        "trading_architecture": list(TRADING_ARCHITECTURE),
        "development_priorities": list(DEVELOPMENT_PRIORITIES),
        "live_promotion_gates": list(LIVE_PROMOTION_GATES),
        "live_trading_enabled": False,
        "dimension_decision": (
            "384D is de vooraf gekozen hoofdvariant. Meer dimensies zijn alleen beter als zij in exact "
            "dezelfde purged walk-forward perioden aantoonbaar beter kalibreren én generaliseren. "
            "Daarom blijven de ruwe 1024D nieuwschunks en bronfeatures beschikbaar voor een "
            "vooraf vastgelegde vergelijking van 128/256/384/512/768D."
        ),
    }
