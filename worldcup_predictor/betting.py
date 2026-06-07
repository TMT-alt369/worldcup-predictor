from __future__ import annotations

from dataclasses import dataclass

from worldcup_predictor.model import Prediction


@dataclass(frozen=True)
class BettingSignal:
    market: str
    model_probability: float
    implied_probability: float
    edge: float
    odds: float
    recommendation: str
    risk_level: str


def implied_probability(decimal_odds: float) -> float:
    if decimal_odds <= 1:
        return 0.0
    return round(1 / decimal_odds, 4)


def risk_level(confidence: float, edge: float) -> str:
    if confidence >= 0.52 and edge >= 0.08:
        return "低"
    if confidence >= 0.43 and edge >= 0.04:
        return "中"
    return "高"


def recommendation(edge: float, risk: str) -> str:
    if edge >= 0.08 and risk == "低":
        return "強烈關注"
    if edge >= 0.04:
        return "可列入觀察"
    return "不建議投注"


def analyze_1x2(
    prediction: Prediction,
    home_odds: float,
    draw_odds: float,
    away_odds: float,
) -> list[BettingSignal]:
    markets = [
        ("主勝", prediction.home_win_probability, home_odds),
        ("和局", prediction.draw_probability, draw_odds),
        ("客勝", prediction.away_win_probability, away_odds),
    ]

    signals = []
    for market, model_prob, odds in markets:
        implied = implied_probability(float(odds))
        edge = round(model_prob - implied, 4)
        risk = risk_level(prediction.confidence, edge)
        signals.append(
            BettingSignal(
                market=market,
                model_probability=model_prob,
                implied_probability=implied,
                edge=edge,
                odds=float(odds),
                recommendation=recommendation(edge, risk),
                risk_level=risk,
            )
        )
    return signals
