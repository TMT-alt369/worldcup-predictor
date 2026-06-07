from __future__ import annotations

from dataclasses import dataclass
from math import exp, factorial

import numpy as np
import pandas as pd

from worldcup_predictor.history import worldcup_performance_score


@dataclass(frozen=True)
class Prediction:
    home_team: str
    away_team: str
    expected_home_goals: float
    expected_away_goals: float
    predicted_home_goals: int
    predicted_away_goals: int
    home_win_probability: float
    draw_probability: float
    away_win_probability: float
    confidence: float


def poisson_probability(lam: float, goals: int) -> float:
    return (lam**goals * exp(-lam)) / factorial(goals)


def team_strength(matches: pd.DataFrame, team: str) -> dict[str, float]:
    team_matches = matches[
        (matches["home_team"] == team) | (matches["away_team"] == team)
    ].sort_values("date")

    if team_matches.empty:
        return {"attack": 1.25, "defense": 1.25, "elo": 1700, "form": 0.5}

    goals_for = []
    goals_against = []
    points = []
    elos = []

    for _, match in team_matches.tail(6).iterrows():
        is_home = match["home_team"] == team
        scored = int(match["home_goals"] if is_home else match["away_goals"])
        conceded = int(match["away_goals"] if is_home else match["home_goals"])
        elo = float(match["home_elo"] if is_home else match["away_elo"])

        goals_for.append(scored)
        goals_against.append(conceded)
        elos.append(elo)
        if scored > conceded:
            points.append(3)
        elif scored == conceded:
            points.append(1)
        else:
            points.append(0)

    return {
        "attack": float(np.mean(goals_for)),
        "defense": float(np.mean(goals_against)),
        "elo": float(np.mean(elos)),
        "form": float(np.mean(points) / 3),
    }


def predict_match(
    matches: pd.DataFrame,
    home_team: str,
    away_team: str,
    worldcup_team_stats: pd.DataFrame | None = None,
) -> Prediction:
    home = team_strength(matches, home_team)
    away = team_strength(matches, away_team)

    elo_gap = (home["elo"] - away["elo"]) / 400
    form_gap = home["form"] - away["form"]
    history_gap = 0.0
    if worldcup_team_stats is not None:
        home_history = worldcup_performance_score(worldcup_team_stats, home_team)
        away_history = worldcup_performance_score(worldcup_team_stats, away_team)
        history_gap = home_history - away_history

    expected_home = (
        0.58 * home["attack"]
        + 0.30 * away["defense"]
        + 0.18
        + 0.20 * elo_gap
        + 0.15 * form_gap
        + 0.18 * history_gap
    )
    expected_away = (
        0.55 * away["attack"]
        + 0.32 * home["defense"]
        - 0.10 * elo_gap
        - 0.12 * form_gap
        - 0.14 * history_gap
    )

    expected_home = float(np.clip(expected_home, 0.25, 3.8))
    expected_away = float(np.clip(expected_away, 0.20, 3.5))

    score_probs = {}
    home_win = draw = away_win = 0.0
    for home_goals in range(6):
        for away_goals in range(6):
            prob = poisson_probability(expected_home, home_goals) * poisson_probability(
                expected_away, away_goals
            )
            score_probs[(home_goals, away_goals)] = prob
            if home_goals > away_goals:
                home_win += prob
            elif home_goals == away_goals:
                draw += prob
            else:
                away_win += prob

    total = home_win + draw + away_win
    home_win, draw, away_win = home_win / total, draw / total, away_win / total
    predicted_score = max(score_probs, key=score_probs.get)
    confidence = max(home_win, draw, away_win)

    return Prediction(
        home_team=home_team,
        away_team=away_team,
        expected_home_goals=round(expected_home, 2),
        expected_away_goals=round(expected_away, 2),
        predicted_home_goals=predicted_score[0],
        predicted_away_goals=predicted_score[1],
        home_win_probability=round(home_win, 4),
        draw_probability=round(draw, 4),
        away_win_probability=round(away_win, 4),
        confidence=round(confidence, 4),
    )


def prediction_to_frame(prediction: Prediction) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"結果": "主勝", "機率": prediction.home_win_probability},
            {"結果": "和局", "機率": prediction.draw_probability},
            {"結果": "客勝", "機率": prediction.away_win_probability},
        ]
    )
