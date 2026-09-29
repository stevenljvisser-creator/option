HORIZONS = {
    "M15": {"label":"15 minuten", "kind":"minutes", "value":15, "equivalent_market_minutes":15},
    "M30": {"label":"30 minuten", "kind":"minutes", "value":30, "equivalent_market_minutes":30},
    "H1":  {"label":"1 uur",      "kind":"minutes", "value":60, "equivalent_market_minutes":60},
    "H2":  {"label":"2 uur",      "kind":"minutes", "value":120, "equivalent_market_minutes":120},
    "H4":  {"label":"4 uur",      "kind":"minutes", "value":240, "equivalent_market_minutes":240},
    "D1":  {"label":"1 handelsdag", "kind":"sessions", "value":1, "equivalent_market_minutes":390},
    "D2":  {"label":"2 handelsdagen","kind":"sessions", "value":2, "equivalent_market_minutes":780},
    "D3":  {"label":"3 handelsdagen","kind":"sessions", "value":3, "equivalent_market_minutes":1170},
    "W1":  {"label":"1 handelsweek","kind":"sessions", "value":5, "equivalent_market_minutes":1950},
}

DEFAULT_HORIZONS=list(HORIZONS.keys())

def horizon_label(code):
    return HORIZONS.get(code,{}).get("label",code)

def horizon_spec(code):
    if code not in HORIZONS:
        raise ValueError(f"Onbekende horizon: {code}")
    return HORIZONS[code]
