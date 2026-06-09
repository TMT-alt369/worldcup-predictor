from __future__ import annotations

import pandas as pd


MODEL_WEIGHT = 0.70
MARKET_WEIGHT = 0.30


def implied_probabilities(home_odds: float, draw_odds: float, away_odds: float) -> dict[str, float]:
    for value in [home_odds, draw_odds, away_odds]:
        if float(value) <= 1:
            raise ValueError("Odds must be greater than 1.")
    raw = {
        "home": 1 / float(home_odds),
        "draw": 1 / float(draw_odds),
        "away": 1 / float(away_odds),
    }
    total = sum(raw.values())
    return {key: round(value / total, 4) for key, value in raw.items()}


def fuse_probabilities(model: dict[str, float], market: dict[str, float], model_weight: float = MODEL_WEIGHT) -> dict[str, float]:
    market_weight = 1 - model_weight
    fused = {
        key: model_weight * float(model[key]) + market_weight * float(market[key])
        for key in ["home", "draw", "away"]
    }
    total = sum(fused.values())
    return {key: round(value / total, 4) for key, value in fused.items()}


def market_probability_table(row: pd.Series, prediction) -> pd.DataFrame:
    market = implied_probabilities(row["home_odds"], row["draw_odds"], row["away_odds"])
    model = {
        "home": prediction.home_win_probability,
        "draw": prediction.draw_probability,
        "away": prediction.away_win_probability,
    }
    fused = fuse_probabilities(model, market)
    labels = {"home": "主勝", "draw": "和局", "away": "客勝"}
    return pd.DataFrame(
        [
            {
                "market": labels[key],
                "model_probability": model[key],
                "market_probability": market[key],
                "fused_probability": fused[key],
            }
            for key in ["home", "draw", "away"]
        ]
    )
