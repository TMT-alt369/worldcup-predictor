from __future__ import annotations

import pandas as pd


K_FACTOR = 32
PREDICTION_WEIGHTS = {
    "elo": 0.50,
    "recent_form": 0.30,
    "worldcup_history": 0.20,
    "form": 0.30,
    "history": 0.20,
}


def expected_score(rating_a: float, rating_b: float) -> float:
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))


def actual_score(goals_for: int, goals_against: int) -> float:
    if goals_for > goals_against:
        return 1.0
    if goals_for == goals_against:
        return 0.5
    return 0.0


def update_elo(home_elo: float, away_elo: float, home_score: int, away_score: int) -> tuple[float, float]:
    home_expected = expected_score(home_elo, away_elo)
    away_expected = 1 - home_expected
    home_actual = actual_score(home_score, away_score)
    away_actual = 1 - home_actual
    margin = max(abs(home_score - away_score), 1)
    multiplier = 1 + min(margin - 1, 3) * 0.15
    return (
        round(home_elo + K_FACTOR * multiplier * (home_actual - home_expected), 1),
        round(away_elo + K_FACTOR * multiplier * (away_actual - away_expected), 1),
    )


def calculate_elo_updates(results: pd.DataFrame, team_meta: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame()
    current = dict(zip(team_meta["team"], team_meta["elo"]))
    rows = []
    for _, match in results.sort_values("date").iterrows():
        home = match["home_team"]
        away = match["away_team"]
        home_before = float(match.get("home_elo_before", current.get(home, 1700)))
        away_before = float(match.get("away_elo_before", current.get(away, 1700)))
        home_after, away_after = update_elo(home_before, away_before, int(match["home_score"]), int(match["away_score"]))
        current[home] = home_after
        current[away] = away_after
        rows.append(
            {
                "date": match["date"],
                "home_team": home,
                "away_team": away,
                "home_score": int(match["home_score"]),
                "away_score": int(match["away_score"]),
                "tournament": match.get("tournament", "International"),
                "home_elo_before": home_before,
                "away_elo_before": away_before,
                "home_elo_after": home_after,
                "away_elo_after": away_after,
            }
        )
    return pd.DataFrame(rows)


def team_recent_form(results: pd.DataFrame, team: str, limit: int = 10) -> str:
    if results.empty:
        return "尚無近期資料"
    team_rows = results[(results["home_team"] == team) | (results["away_team"] == team)].tail(limit)
    form = []
    for _, row in team_rows.iterrows():
        is_home = row["home_team"] == team
        gf = int(row["home_score"] if is_home else row["away_score"])
        ga = int(row["away_score"] if is_home else row["home_score"])
        if gf > ga:
            form.append("W")
        elif gf == ga:
            form.append("D")
        else:
            form.append("L")
    return "-".join(form) if form else "尚無近期資料"


def elo_ranking_with_updates(results: pd.DataFrame, team_meta: pd.DataFrame) -> pd.DataFrame:
    updates = calculate_elo_updates(results, team_meta)
    rows = []
    for _, team in team_meta.iterrows():
        name = team["team"]
        original = float(team.get("elo", 1700))
        related = updates[(updates["home_team"] == name) | (updates["away_team"] == name)] if not updates.empty else pd.DataFrame()
        if related.empty:
            updated = original
        else:
            latest = related.iloc[-1]
            updated = float(latest["home_elo_after"] if latest["home_team"] == name else latest["away_elo_after"])
        rows.append(
            {
                "team": name,
                "team_zh": team.get("team_zh", name),
                "flag_emoji": team.get("flag_emoji", ""),
                "original_elo": round(original, 1),
                "updated_elo": round(updated, 1),
                "elo_change": round(updated - original, 1),
                "recent_10": team_recent_form(updates, name),
            }
        )
    return pd.DataFrame(rows).sort_values("updated_elo", ascending=False).reset_index(drop=True)
