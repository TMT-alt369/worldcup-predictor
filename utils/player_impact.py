from __future__ import annotations

import pandas as pd


def normalize_player_impact(players: pd.DataFrame) -> pd.DataFrame:
    data = players.copy()
    aliases = {
        "player_name": "player",
        "national_caps": "appearances",
        "national_goals": "goals",
    }
    for source, target in aliases.items():
        if target not in data.columns and source in data.columns:
            data[target] = data[source]
    defaults = {
        "player": "資料待補",
        "team": "Unknown",
        "position": "資料待補",
        "age": 0,
        "club": "資料待補",
        "appearances": 0,
        "goals": 0,
        "assists": 0,
        "recent_form_rating": 0.5,
        "is_available": True,
    }
    for column, default in defaults.items():
        if column not in data.columns:
            data[column] = default
    for column in ["age", "appearances", "goals", "assists", "recent_form_rating"]:
        data[column] = pd.to_numeric(data[column], errors="coerce").fillna(defaults[column])
    if "goal_rate" not in data.columns:
        data["goal_rate"] = data["goals"] / data["appearances"].replace(0, pd.NA)
    if "assist_rate" not in data.columns:
        data["assist_rate"] = data["assists"] / data["appearances"].replace(0, pd.NA)
    data["goal_rate"] = pd.to_numeric(data["goal_rate"], errors="coerce").fillna(0.0)
    data["assist_rate"] = pd.to_numeric(data["assist_rate"], errors="coerce").fillna(0.0)
    if "impact_score" not in data.columns:
        position_bonus = data["position"].map({"Forward": 1.15, "Goalkeeper": 1.08, "Midfielder": 1.05, "Defender": 1.0}).fillna(1.0)
        data["impact_score"] = (
            data["recent_form_rating"] * 8
            + data["goal_rate"] * 35
            + data["assist_rate"] * 25
            + data["appearances"].clip(upper=100) * 0.12
        ) * position_bonus
    data["impact_score"] = pd.to_numeric(data["impact_score"], errors="coerce").fillna(0.0).round(2)
    data["is_available"] = data["is_available"].astype(str).str.lower().isin(["true", "1", "yes", "y", "available"])
    data["availability_note"] = data.apply(_availability_note, axis=1)
    return data


def _availability_note(row: pd.Series) -> str:
    if bool(row.get("is_available", True)):
        return "可出賽"
    position = str(row.get("position", ""))
    if position == "Forward":
        return "核心前鋒缺陣，進攻效率下修"
    if position == "Goalkeeper":
        return "主力門將缺陣，失球風險上升"
    return "主力缺陣，整體強度下修"


def team_impact(players: pd.DataFrame, team: str, top_n: int = 5) -> pd.DataFrame:
    data = normalize_player_impact(players)
    team_rows = data[data["team"] == team].copy()
    if team_rows.empty:
        return team_rows
    return team_rows.sort_values(["is_available", "impact_score"], ascending=False).head(top_n)


def win_probability_adjustment(players: pd.DataFrame, team: str) -> float:
    top = team_impact(players, team, top_n=5)
    if top.empty:
        return 0.0
    available_score = top.loc[top["is_available"], "impact_score"].sum()
    unavailable = top.loc[~top["is_available"]].copy()
    unavailable_score = unavailable["impact_score"].sum()
    forward_penalty = unavailable.loc[unavailable["position"] == "Forward", "impact_score"].sum() * 0.25
    goalkeeper_penalty = unavailable.loc[unavailable["position"] == "Goalkeeper", "impact_score"].sum() * 0.35
    adjustment = (available_score - unavailable_score * 1.35 - forward_penalty - goalkeeper_penalty) / 10000
    return round(float(max(min(adjustment, 0.04), -0.06)), 4)
