from __future__ import annotations

from pathlib import Path

import pandas as pd


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
POSITION_WEIGHT = {
    "Forward": 1.18,
    "Midfielder": 1.08,
    "Defender": 1.02,
    "Goalkeeper": 1.12,
}


def load_player_pool() -> pd.DataFrame:
    for source in [DATA_DIR / "players_2026.csv", DATA_DIR / "player_database.csv", DATA_DIR / "players.csv"]:
        if source.exists():
            return pd.read_csv(source)
    return pd.DataFrame()


def load_player_status() -> pd.DataFrame:
    source = DATA_DIR / "player_status.csv"
    if not source.exists():
        return pd.DataFrame(columns=["player_name", "team", "status", "is_available", "note"])
    data = pd.read_csv(source)
    if "is_available" not in data.columns:
        data["is_available"] = True
    return data


def normalize_players(players: pd.DataFrame | None = None, status: pd.DataFrame | None = None) -> pd.DataFrame:
    data = load_player_pool() if players is None else players.copy()
    if data.empty:
        return pd.DataFrame(
            columns=[
                "player_name",
                "team",
                "team_zh",
                "position",
                "age",
                "club",
                "appearances",
                "goals",
                "assists",
                "recent_form",
                "goal_rate",
                "assist_rate",
                "impact_score",
                "is_available",
            ]
        )
    rename = {
        "player": "player_name",
        "national_caps": "appearances",
        "caps": "appearances",
        "national_goals": "goals",
        "recent_form_rating": "recent_form",
        "recent_form_score": "recent_form",
    }
    data = data.rename(columns={source: target for source, target in rename.items() if source in data.columns})
    defaults = {
        "player_name": "Unknown Player",
        "team": "Unknown",
        "team_zh": "Unknown",
        "position": "Midfielder",
        "age": 0,
        "club": "N/A",
        "appearances": 0,
        "goals": 0,
        "assists": 0,
        "recent_form": 6.5,
    }
    for column, default in defaults.items():
        if column not in data.columns:
            data[column] = default
    for column in ["age", "appearances", "goals", "assists", "recent_form"]:
        data[column] = pd.to_numeric(data[column], errors="coerce").fillna(defaults[column])
    safe_apps = data["appearances"].replace(0, pd.NA)
    if "goal_rate" not in data.columns:
        data["goal_rate"] = data["goals"] / safe_apps
    if "assist_rate" not in data.columns:
        data["assist_rate"] = data["assists"] / safe_apps
    data["goal_rate"] = pd.to_numeric(data["goal_rate"], errors="coerce").fillna(0.0)
    data["assist_rate"] = pd.to_numeric(data["assist_rate"], errors="coerce").fillna(0.0)
    status_data = load_player_status() if status is None else status.copy()
    if not status_data.empty and {"player_name", "team"}.issubset(status_data.columns):
        status_data["is_available"] = status_data.get("is_available", True).astype(str).str.lower().isin(["true", "1", "yes", "y", "available"])
        data = data.merge(status_data[["player_name", "team", "status", "is_available", "note"]], on=["player_name", "team"], how="left")
    if "is_available" not in data.columns:
        data["is_available"] = True
    data["is_available"] = data["is_available"].map(lambda value: str(value).lower() in ["true", "1", "yes", "y", "available"]).fillna(True)
    if "status" not in data.columns:
        data["status"] = "Available"
    if "note" not in data.columns:
        data["note"] = ""
    data["status"] = data["status"].fillna("Available")
    data["note"] = data["note"].fillna("")
    data["position_weight"] = data["position"].map(POSITION_WEIGHT).fillna(1.0)
    data["impact_score"] = (
        data["recent_form"].clip(0, 10) * 6
        + data["goal_rate"] * 32
        + data["assist_rate"] * 24
        + data["appearances"].clip(0, 120) * 0.10
    ) * data["position_weight"]
    data["impact_score"] = pd.to_numeric(data["impact_score"], errors="coerce").fillna(0).round(2)
    return data


def team_player_impact(players: pd.DataFrame, team: str, unavailable_players: list[str] | None = None, top_n: int = 8) -> pd.DataFrame:
    data = normalize_players(players)
    team_rows = data[data["team"].astype(str).eq(str(team))].copy()
    if team_rows.empty:
        return team_rows
    unavailable = set(unavailable_players or [])
    if unavailable:
        team_rows.loc[team_rows["player_name"].isin(unavailable), "is_available"] = False
        team_rows.loc[team_rows["player_name"].isin(unavailable), "status"] = "Simulated Out"
    return team_rows.sort_values("impact_score", ascending=False).head(top_n).reset_index(drop=True)


def team_strength_delta(players: pd.DataFrame, team: str, unavailable_players: list[str] | None = None) -> dict[str, float]:
    base = team_player_impact(players, team, [], top_n=8)
    scenario = team_player_impact(players, team, unavailable_players or [], top_n=8)
    if base.empty:
        return {"impact_loss": 0.0, "win_probability_delta": 0.0, "xg_delta": 0.0, "champion_probability_delta": 0.0}
    base_score = float(base["impact_score"].sum())
    available_score = float(scenario.loc[scenario["is_available"], "impact_score"].sum())
    impact_loss = max(0.0, base_score - available_score)
    forward_loss = float(scenario.loc[(~scenario["is_available"]) & (scenario["position"].eq("Forward")), "impact_score"].sum())
    keeper_loss = float(scenario.loc[(~scenario["is_available"]) & (scenario["position"].eq("Goalkeeper")), "impact_score"].sum())
    win_delta = -min(0.14, impact_loss / 850)
    xg_delta = -min(0.65, (impact_loss + forward_loss * 0.55) / 170)
    champion_delta = -min(0.035, (impact_loss + keeper_loss * 0.35) / 2800)
    return {
        "impact_loss": round(impact_loss, 2),
        "win_probability_delta": round(win_delta, 4),
        "xg_delta": round(xg_delta, 3),
        "champion_probability_delta": round(champion_delta, 4),
    }


def match_player_impact(
    players: pd.DataFrame,
    home_team: str,
    away_team: str,
    home_unavailable: list[str] | None = None,
    away_unavailable: list[str] | None = None,
) -> dict[str, object]:
    home_delta = team_strength_delta(players, home_team, home_unavailable)
    away_delta = team_strength_delta(players, away_team, away_unavailable)
    home_win_change = home_delta["win_probability_delta"] - away_delta["win_probability_delta"] * 0.45
    away_win_change = away_delta["win_probability_delta"] - home_delta["win_probability_delta"] * 0.45
    return {
        "home_team": home_team,
        "away_team": away_team,
        "home": home_delta,
        "away": away_delta,
        "home_win_probability_change": round(home_win_change, 4),
        "away_win_probability_change": round(away_win_change, 4),
        "home_xg_change": home_delta["xg_delta"],
        "away_xg_change": away_delta["xg_delta"],
    }
