from __future__ import annotations

import pandas as pd


POSITION_ZH = {
    "Goalkeeper": "守門員",
    "Defender": "後衛",
    "Midfielder": "中場",
    "Forward": "前鋒",
}


def player_database(players: pd.DataFrame, team_meta: pd.DataFrame) -> pd.DataFrame:
    meta = team_meta[["team", "team_zh", "flag_emoji"]].copy()
    data = players.copy()
    data = data.merge(meta, on="team", how="left", suffixes=("", "_meta"))
    data["team_zh"] = data["team_zh_meta"].fillna(data.get("team_zh", data["team"]))
    data["flag_emoji"] = data["flag_emoji"].fillna("🏳️")
    data["national_team"] = data["flag_emoji"] + " " + data["team_zh"]
    data["position_zh"] = data["position"].map(POSITION_ZH).fillna(data["position"])
    data["jersey_number"] = data.groupby("team").cumcount() + 7
    data.loc[data["position"] == "Goalkeeper", "jersey_number"] = 1
    data["age"] = 26 + (data.groupby("team").cumcount() % 8)
    data["club"] = "待官方名單公布"
    return data[
        [
            "player_name",
            "jersey_number",
            "position",
            "position_zh",
            "age",
            "club",
            "team",
            "team_zh",
            "national_team",
            "national_caps",
            "national_goals",
            "goal_rate",
            "recent_form_rating",
        ]
    ].sort_values(["team_zh", "position", "player_name"]).reset_index(drop=True)
