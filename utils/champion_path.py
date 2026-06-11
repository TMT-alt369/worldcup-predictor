from __future__ import annotations

import pandas as pd


STAGE_COLUMNS = [
    ("小組出線率", "group_qualified_probability"),
    ("32 強率", "round_32_probability"),
    ("16 強率", "round_16_probability"),
    ("8 強率", "round_8_probability"),
    ("4 強率", "semi_final_probability"),
    ("決賽率", "final_probability"),
    ("冠軍率", "champion_probability"),
]


def _value(row: pd.Series, column: str) -> float:
    if column not in row.index:
        if column == "round_32_probability":
            column = "group_qualified_probability"
        else:
            return 0.0
    return float(pd.to_numeric(pd.Series([row.get(column, 0)]), errors="coerce").fillna(0).iloc[0])


def stage_probability_table(team_row: pd.Series) -> pd.DataFrame:
    rows = [{"階段": label, "機率": _value(team_row, column)} for label, column in STAGE_COLUMNS]
    return pd.DataFrame(rows)


def likely_knockout_path(team_row: pd.Series, simulation_df: pd.DataFrame, max_opponents: int = 4) -> pd.DataFrame:
    if simulation_df is None or simulation_df.empty:
        return pd.DataFrame(columns=["階段", "可能對手", "對手冠軍率", "對手 Elo"])

    team = str(team_row.get("team", ""))
    candidates = simulation_df[simulation_df["team"].astype(str) != team].copy()
    if candidates.empty:
        return pd.DataFrame(columns=["階段", "可能對手", "對手冠軍率", "對手 Elo"])

    candidates["champion_probability"] = pd.to_numeric(candidates["champion_probability"], errors="coerce").fillna(0)
    candidates["elo"] = pd.to_numeric(candidates.get("elo", 1700), errors="coerce").fillna(1700)
    top = candidates.sort_values(["champion_probability", "elo"], ascending=False).head(max_opponents)
    stages = ["16 強", "8 強", "4 強", "決賽"]
    rows = []
    for stage, (_, opponent) in zip(stages, top.iterrows()):
        rows.append(
            {
                "階段": stage,
                "可能對手": opponent.get("team_display", opponent.get("team_zh", opponent.get("team", "N/A"))),
                "對手冠軍率": float(opponent.get("champion_probability", 0)),
                "對手 Elo": int(float(opponent.get("elo", 0) or 0)),
            }
        )
    return pd.DataFrame(rows)


def biggest_obstacle(team_row: pd.Series, simulation_df: pd.DataFrame) -> str:
    path = likely_knockout_path(team_row, simulation_df, max_opponents=1)
    if path.empty:
        return "目前資料不足"
    return str(path.iloc[0]["可能對手"])


def champion_path_text(team_row: pd.Series, simulation_df: pd.DataFrame) -> str:
    team = str(team_row.get("team_display", team_row.get("team_zh", team_row.get("team", "該隊"))))
    obstacle = biggest_obstacle(team_row, simulation_df)
    round_8 = _value(team_row, "round_8_probability")
    semi = _value(team_row, "semi_final_probability")
    champion = _value(team_row, "champion_probability")
    if champion >= 0.10:
        tone = "具備明顯競爭力"
    elif champion >= 0.04:
        tone = "有機會扮演淘汰賽變數"
    else:
        tone = "需要連續突破高強度對手"
    return (
        f"{team} 目前冠軍路徑評估為{tone}。8 強率約 {round_8 * 100:.1f}%，"
        f"4 強率約 {semi * 100:.1f}%，最大阻礙可能是 {obstacle}。"
        "若能通過該階段，後續奪冠機率通常會明顯提升。"
    )
