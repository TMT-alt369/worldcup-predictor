from __future__ import annotations

import pandas as pd


POSITION_ZH = {
    "Goalkeeper": "守門員",
    "Defender": "後衛",
    "Midfielder": "中場",
    "Forward": "前鋒",
}


def ensure_player_columns(players: pd.DataFrame) -> pd.DataFrame:
    data = players.copy()
    defaults = {
        "jersey_number": 0,
        "age": 0,
        "height_cm": 0,
        "market_value_eur_m": 0.0,
        "club": "待官方名單公布",
        "nationality": "",
        "data_source": "local_fallback_players",
        "squad_status": "fallback_demo_partial",
        "recent_form_rating": 7.0,
        "national_caps": 0,
        "national_goals": 0,
    }
    for column, default in defaults.items():
        if column not in data.columns:
            data[column] = default

    if "goal_rate" not in data.columns:
        data["goal_rate"] = (
            data["national_goals"] / data["national_caps"].replace(0, pd.NA)
        ).fillna(0)

    if "player_name" not in data.columns and "name" in data.columns:
        data["player_name"] = data["name"]

    return data


def player_database(players: pd.DataFrame, team_meta: pd.DataFrame) -> pd.DataFrame:
    meta = team_meta[["team", "team_zh", "flag_emoji", "fifa_ranking", "elo"]].copy()
    data = ensure_player_columns(players)
    data = data.merge(meta, on="team", how="left", suffixes=("", "_meta"))

    if "team_zh_meta" in data.columns:
        data["team_zh"] = data["team_zh_meta"].fillna(data.get("team_zh", data["team"]))
    elif "team_zh" not in data.columns:
        data["team_zh"] = data["team"]

    data["flag_emoji"] = data["flag_emoji"].fillna("🏳️")
    data["nationality"] = data["nationality"].replace("", pd.NA).fillna(data["team_zh"])
    data["national_team"] = data["flag_emoji"] + " " + data["team_zh"]
    data["position_zh"] = data["position"].map(POSITION_ZH).fillna(data["position"])

    if (data["jersey_number"].fillna(0) == 0).all():
        data["jersey_number"] = data.groupby("team").cumcount() + 7
        data.loc[data["position"] == "Goalkeeper", "jersey_number"] = 1
    if (data["age"].fillna(0) == 0).all():
        data["age"] = 26 + (data.groupby("team").cumcount() % 8)

    columns = [
        "player_name",
        "jersey_number",
        "position",
        "position_zh",
        "age",
        "height_cm",
        "market_value_eur_m",
        "club",
        "team",
        "team_zh",
        "nationality",
        "national_team",
        "fifa_ranking",
        "elo",
        "national_caps",
        "national_goals",
        "goal_rate",
        "recent_form_rating",
        "data_source",
        "squad_status",
    ]
    return data[columns].sort_values(["team_zh", "position", "player_name"]).reset_index(drop=True)


def squad_summary(player_db: pd.DataFrame) -> pd.DataFrame:
    summary = (
        player_db.groupby(["team", "team_zh", "national_team"], as_index=False)
        .agg(
            players=("player_name", "count"),
            average_age=("age", "mean"),
            total_market_value_eur_m=("market_value_eur_m", "sum"),
            fifa_ranking=("fifa_ranking", "first"),
            elo=("elo", "first"),
            data_source=("data_source", "first"),
        )
        .sort_values(["fifa_ranking", "elo"], ascending=[True, False])
    )
    summary["average_age"] = summary["average_age"].round(1)
    summary["total_market_value_eur_m"] = summary["total_market_value_eur_m"].round(1)
    return summary.reset_index(drop=True)
