from __future__ import annotations

from math import exp, factorial
from pathlib import Path
import re
import unicodedata

import pandas as pd


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
POSITION_WEIGHT = {
    "Forward": 1.18,
    "Midfielder": 1.08,
    "Defender": 1.02,
    "Goalkeeper": 1.12,
}
POSITION_LABELS = {
    "Forward": "前鋒",
    "Midfielder": "中場",
    "Defender": "後衛",
    "Goalkeeper": "門將",
}


def _name_key(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(char for char in text if not unicodedata.combining(char))
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def _team_key(value: object) -> str:
    return str(value or "").strip().lower()


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if pd.isna(number):
        return default
    return number


def _poisson(lam: float, goals: int) -> float:
    lam = max(0.05, float(lam))
    return (lam**goals * exp(-lam)) / factorial(goals)


def _score_projection(home_xg: float, away_xg: float, max_goals: int = 8) -> dict[str, float | int]:
    home_win = draw = away_win = 0.0
    score_probs: dict[tuple[int, int], float] = {}
    for home_goals in range(max_goals + 1):
        for away_goals in range(max_goals + 1):
            probability = _poisson(home_xg, home_goals) * _poisson(away_xg, away_goals)
            score_probs[(home_goals, away_goals)] = probability
            if home_goals > away_goals:
                home_win += probability
            elif home_goals == away_goals:
                draw += probability
            else:
                away_win += probability
    total = max(home_win + draw + away_win, 0.000001)
    predicted_home, predicted_away = max(score_probs, key=score_probs.get)
    return {
        "home_win_probability": round(home_win / total, 4),
        "draw_probability": round(draw / total, 4),
        "away_win_probability": round(away_win / total, 4),
        "predicted_home_goals": int(predicted_home),
        "predicted_away_goals": int(predicted_away),
    }


def _prediction_base(prediction: object | None) -> dict[str, float]:
    return {
        "home_xg": round(_safe_float(getattr(prediction, "expected_home_goals", None), 1.35), 3),
        "away_xg": round(_safe_float(getattr(prediction, "expected_away_goals", None), 1.15), 3),
        "home_win_probability": round(_safe_float(getattr(prediction, "home_win_probability", None), 0.38), 4),
        "draw_probability": round(_safe_float(getattr(prediction, "draw_probability", None), 0.27), 4),
        "away_win_probability": round(_safe_float(getattr(prediction, "away_win_probability", None), 0.35), 4),
        "predicted_home_goals": int(_safe_float(getattr(prediction, "predicted_home_goals", None), 1)),
        "predicted_away_goals": int(_safe_float(getattr(prediction, "predicted_away_goals", None), 1)),
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
        data["_player_key"] = data["player_name"].map(_name_key)
        data["_team_key"] = data["team"].map(_team_key)
        status_data["_player_key"] = status_data["player_name"].map(_name_key)
        status_data["_team_key"] = status_data["team"].map(_team_key)
        data = data.merge(
            status_data[["_player_key", "_team_key", "status", "is_available", "note"]],
            on=["_player_key", "_team_key"],
            how="left",
        )
        data = data.drop(columns=["_player_key", "_team_key"], errors="ignore")
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
    unavailable = {_name_key(player) for player in (unavailable_players or [])}
    if unavailable:
        selected = team_rows["player_name"].map(_name_key).isin(unavailable)
        team_rows.loc[selected, "is_available"] = False
        team_rows.loc[selected, "status"] = "Simulated Out"
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
    prediction: object | None = None,
    home_champion_probability: float = 0.0,
    away_champion_probability: float = 0.0,
) -> dict[str, object]:
    base = _prediction_base(prediction)
    home_rows = team_player_impact(players, home_team, home_unavailable, top_n=60)
    away_rows = team_player_impact(players, away_team, away_unavailable, top_n=60)
    home_out = home_rows.loc[~home_rows["is_available"]].copy()
    away_out = away_rows.loc[~away_rows["is_available"]].copy()

    adjusted_home_xg = float(base["home_xg"])
    adjusted_away_xg = float(base["away_xg"])
    player_effects: list[dict[str, object]] = []

    def apply_player(row: pd.Series, side: str) -> None:
        nonlocal adjusted_home_xg, adjusted_away_xg
        impact_score = _safe_float(row.get("impact_score"), 0.0)
        modifier = max(0.0, min(1.35, impact_score / 100))
        position = str(row.get("position", "Midfielder"))
        player = str(row.get("player_name", "Unknown Player"))
        team = home_team if side == "home" else away_team
        before_home = adjusted_home_xg
        before_away = adjusted_away_xg
        if position == "Forward":
            if side == "home":
                adjusted_home_xg *= 1 - modifier * 0.15
            else:
                adjusted_away_xg *= 1 - modifier * 0.15
        elif position == "Midfielder":
            if side == "home":
                adjusted_home_xg *= 1 - modifier * 0.10
            else:
                adjusted_away_xg *= 1 - modifier * 0.10
        elif position == "Defender":
            if side == "home":
                adjusted_away_xg *= 1 + modifier * 0.10
            else:
                adjusted_home_xg *= 1 + modifier * 0.10
        elif position == "Goalkeeper":
            if side == "home":
                adjusted_away_xg *= 1 + modifier * 0.15
            else:
                adjusted_home_xg *= 1 + modifier * 0.15
        player_effects.append(
            {
                "player_name": player,
                "team": team,
                "position": POSITION_LABELS.get(position, position),
                "impact_score": round(impact_score, 2),
                "home_xg_delta": round(adjusted_home_xg - before_home, 3),
                "away_xg_delta": round(adjusted_away_xg - before_away, 3),
                "note": "Simulated Out",
            }
        )

    for _, row in home_out.iterrows():
        apply_player(row, "home")
    for _, row in away_out.iterrows():
        apply_player(row, "away")

    adjusted_home_xg = round(max(0.15, min(5.0, adjusted_home_xg)), 3)
    adjusted_away_xg = round(max(0.15, min(5.0, adjusted_away_xg)), 3)
    adjusted = _score_projection(adjusted_home_xg, adjusted_away_xg)
    adjusted["home_xg"] = adjusted_home_xg
    adjusted["away_xg"] = adjusted_away_xg

    home_loss = float(home_out["impact_score"].sum()) if not home_out.empty else 0.0
    away_loss = float(away_out["impact_score"].sum()) if not away_out.empty else 0.0
    home_xg_change = adjusted_home_xg - float(base["home_xg"])
    away_xg_change = adjusted_away_xg - float(base["away_xg"])
    home_win_change = float(adjusted["home_win_probability"]) - float(base["home_win_probability"])
    away_win_change = float(adjusted["away_win_probability"]) - float(base["away_win_probability"])
    draw_change = float(adjusted["draw_probability"]) - float(base["draw_probability"])

    home_champion_delta = -min(0.12, max(0.0, _safe_float(home_champion_probability) * (abs(min(home_xg_change, 0)) * 0.50 + home_loss / 320)))
    away_champion_delta = -min(0.12, max(0.0, _safe_float(away_champion_probability) * (abs(min(away_xg_change, 0)) * 0.50 + away_loss / 320)))
    if home_loss and home_champion_delta == 0:
        home_champion_delta = -min(0.015, home_loss / 9000)
    if away_loss and away_champion_delta == 0:
        away_champion_delta = -min(0.015, away_loss / 9000)

    home_delta = {
        "impact_loss": round(home_loss, 2),
        "win_probability_delta": round(home_win_change, 4),
        "xg_delta": round(home_xg_change, 3),
        "champion_probability_delta": round(home_champion_delta, 4),
    }
    away_delta = {
        "impact_loss": round(away_loss, 2),
        "win_probability_delta": round(away_win_change, 4),
        "xg_delta": round(away_xg_change, 3),
        "champion_probability_delta": round(away_champion_delta, 4),
    }
    return {
        "home_team": home_team,
        "away_team": away_team,
        "base": base,
        "adjusted": adjusted,
        "draw_probability_change": round(draw_change, 4),
        "home": home_delta,
        "away": away_delta,
        "home_win_probability_change": round(home_win_change, 4),
        "away_win_probability_change": round(away_win_change, 4),
        "home_xg_change": round(home_xg_change, 3),
        "away_xg_change": round(away_xg_change, 3),
        "predicted_score_change": f"{int(base['predicted_home_goals'])}:{int(base['predicted_away_goals'])} -> {int(adjusted['predicted_home_goals'])}:{int(adjusted['predicted_away_goals'])}",
        "player_effects": player_effects,
    }


def match_absence_ranking(
    players: pd.DataFrame,
    home_team: str,
    away_team: str,
    prediction: object | None = None,
    top_n: int = 10,
) -> pd.DataFrame:
    base = _prediction_base(prediction)
    rows: list[dict[str, object]] = []
    for side, team in [("home", home_team), ("away", away_team)]:
        team_rows = team_player_impact(players, team, [], top_n=60)
        for _, player in team_rows.iterrows():
            player_name = str(player.get("player_name", "Unknown Player"))
            scenario = match_player_impact(
                players,
                home_team,
                away_team,
                [player_name] if side == "home" else [],
                [player_name] if side == "away" else [],
                prediction=prediction,
            )
            if side == "home":
                win_loss = float(base["home_win_probability"]) - float(scenario["adjusted"]["home_win_probability"])
            else:
                win_loss = float(base["away_win_probability"]) - float(scenario["adjusted"]["away_win_probability"])
            rows.append(
                {
                    "player_name": player_name,
                    "team": team,
                    "position": POSITION_LABELS.get(str(player.get("position", "")), str(player.get("position", ""))),
                    "impact_score": round(_safe_float(player.get("impact_score")), 2),
                    "win_probability_loss": round(max(0.0, win_loss), 4),
                }
            )
    if not rows:
        return pd.DataFrame(columns=["player_name", "team", "position", "impact_score", "win_probability_loss"])
    return pd.DataFrame(rows).sort_values(["win_probability_loss", "impact_score"], ascending=False).head(top_n).reset_index(drop=True)
