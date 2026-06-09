from __future__ import annotations

import pandas as pd


POSITION_ZH = {
    "Goalkeeper": "守門員",
    "Defender": "後衛",
    "Midfielder": "中場",
    "Forward": "前鋒",
}

TEXT_FALLBACK = "資料待補"
MOJIBAKE_MARKERS = ("嚙", "蝛", "蝳", "鞈", "�", "????", "????????", "??")


def clean_display_text(value, fallback: str = TEXT_FALLBACK) -> str:
    text = str(value or "").strip()
    if not text or text.lower() == "nan":
        return fallback
    if any(marker in text for marker in MOJIBAKE_MARKERS):
        return fallback
    return text


def _safe_rate(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    safe_denominator = pd.to_numeric(denominator, errors="coerce").replace(0, pd.NA)
    rate = pd.to_numeric(numerator, errors="coerce") / safe_denominator
    return pd.to_numeric(rate, errors="coerce").fillna(0.0)


def ensure_player_columns(players: pd.DataFrame) -> pd.DataFrame:
    data = players.copy()
    alias_map = {
        "team_en": "team",
        "appearances": "national_caps",
        "caps": "national_caps",
        "goals": "national_goals",
        "recent_form_score": "recent_form_rating",
    }
    for source, target in alias_map.items():
        if target not in data.columns and source in data.columns:
            data[target] = data[source]

    defaults = {
        "team": "Unknown",
        "team_zh": "",
        "player_name": TEXT_FALLBACK,
        "jersey_number": 0,
        "position": TEXT_FALLBACK,
        "age": 0,
        "height_cm": 0,
        "preferred_foot": TEXT_FALLBACK,
        "market_value_eur_m": 0.0,
        "club": TEXT_FALLBACK,
        "nationality": "",
        "data_source": "local_players",
        "squad_status": "local_data",
        "recent_form_rating": 0.5,
        "national_caps": 0,
        "national_goals": 0,
        "assists": 0,
    }
    for column, default in defaults.items():
        if column not in data.columns:
            data[column] = default

    if "name" in data.columns:
        data["player_name"] = data["player_name"].where(data["player_name"].notna(), data["name"])

    numeric_columns = [
        "national_caps",
        "national_goals",
        "assists",
        "recent_form_rating",
        "market_value_eur_m",
        "height_cm",
        "age",
    ]
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce").fillna(defaults.get(column, 0))

    if "goal_rate" not in data.columns:
        data["goal_rate"] = _safe_rate(data["national_goals"], data["national_caps"])
    else:
        data["goal_rate"] = pd.to_numeric(data["goal_rate"], errors="coerce").fillna(0.0)

    if "assist_rate" not in data.columns:
        data["assist_rate"] = _safe_rate(data["assists"], data["national_caps"])
    else:
        data["assist_rate"] = pd.to_numeric(data["assist_rate"], errors="coerce").fillna(0.0)

    for column in ["player_name", "team_zh", "nationality", "club", "preferred_foot", "position"]:
        data[column] = data[column].map(clean_display_text)

    data["team_zh"] = data["team_zh"].where(data["team_zh"] != TEXT_FALLBACK, data["team"])
    return data


def player_database(players: pd.DataFrame, team_meta: pd.DataFrame) -> pd.DataFrame:
    meta_columns = [
        column
        for column in ["team", "team_zh", "flag_emoji", "fifa_ranking", "elo"]
        if column in team_meta.columns
    ]
    meta = team_meta[meta_columns].copy()
    data = ensure_player_columns(players)
    data = data.merge(meta, on="team", how="left", suffixes=("", "_meta"))

    if "team_zh_meta" in data.columns:
        data["team_zh"] = data["team_zh_meta"].fillna(data["team_zh"])
    data["team_zh"] = data["team_zh"].replace("", pd.NA).fillna(data["team"])
    if "flag_emoji" not in data.columns:
        data["flag_emoji"] = ""
    data["flag_emoji"] = data["flag_emoji"].fillna("")
    data["nationality"] = data["nationality"].replace("", pd.NA).fillna(data["team_zh"])
    data["national_team"] = (data["flag_emoji"] + " " + data["team_zh"]).str.strip()
    data["position_zh"] = data["position"].map(POSITION_ZH).fillna(data["position"])
    data["height_cm"] = pd.to_numeric(data["height_cm"], errors="coerce").fillna(0).astype(int)

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
        "preferred_foot",
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
        "assists",
        "goal_rate",
        "assist_rate",
        "recent_form_rating",
        "data_source",
        "squad_status",
        "flag_emoji",
    ]
    for column in columns:
        if column not in data.columns:
            data[column] = 0 if column in {"fifa_ranking", "elo"} else TEXT_FALLBACK
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
