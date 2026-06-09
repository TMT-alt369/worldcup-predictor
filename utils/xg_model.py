from __future__ import annotations

import math

import pandas as pd


PREDICTION_WEIGHTS_XG = {
    "elo": 0.40,
    "recent_form": 0.20,
    "worldcup_history": 0.15,
    "xg": 0.25,
}


def calculate_shot_xg(row: pd.Series) -> float:
    situation = str(row.get("situation", "")).lower()
    if situation == "penalty":
        return 0.76

    distance = float(row.get("shot_distance", 20))
    angle = float(row.get("shot_angle", 25))
    body_part = str(row.get("body_part", "")).lower()

    value = 0.34 * math.exp(-distance / 22) + 0.22 * min(max(angle, 0), 90) / 90
    if "head" in body_part:
        value *= 0.82
    if bool(row.get("is_big_chance", False)):
        value *= 1.65
    if "outside box" in situation:
        value *= 0.55
    return round(float(min(max(value, 0.01), 0.95)), 3)


def prepare_xg_data(shots: pd.DataFrame) -> pd.DataFrame:
    data = shots.copy()
    defaults = {
        "match_id": "M001",
        "team": "Unknown",
        "player": "Unknown",
        "minute": 0,
        "shot_distance": 20,
        "shot_angle": 25,
        "body_part": "Right Foot",
        "situation": "Open Play",
        "is_big_chance": False,
        "is_goal": False,
    }
    for column, default in defaults.items():
        if column not in data.columns:
            data[column] = default

    for column in ["minute", "shot_distance", "shot_angle"]:
        data[column] = pd.to_numeric(data[column], errors="coerce").fillna(defaults[column])

    data["is_big_chance"] = data["is_big_chance"].astype(str).str.lower().isin(["true", "1", "yes", "y"])
    data["is_goal"] = data["is_goal"].astype(str).str.lower().isin(["true", "1", "yes", "y"])
    if "xg_value" not in data.columns:
        data["xg_value"] = data.apply(calculate_shot_xg, axis=1)
    data["xg_value"] = pd.to_numeric(data["xg_value"], errors="coerce").fillna(0.01).clip(0.01, 0.95)
    data["minute"] = data["minute"].astype(int)
    return data.sort_values(["match_id", "minute"]).reset_index(drop=True)


def xg_match_summary(shots: pd.DataFrame, match_id: str) -> pd.DataFrame:
    data = prepare_xg_data(shots)
    match = data[data["match_id"] == match_id]
    if match.empty:
        return pd.DataFrame()
    return (
        match.groupby("team", as_index=False)
        .agg(xg=("xg_value", "sum"), goals=("is_goal", "sum"), shots=("player", "count"))
        .assign(
            xg=lambda frame: frame["xg"].round(2),
            difference=lambda frame: (frame["goals"] - frame["xg"]).round(2),
        )
    )


def xg_analysis_text(summary: pd.DataFrame) -> list[str]:
    if summary.empty or len(summary) < 2:
        return ["目前尚未匯入足夠的 xG 射門資料。"]

    leader = summary.sort_values("xg", ascending=False).iloc[0]
    efficient = summary.sort_values("difference", ascending=False).iloc[0]
    wasteful = summary.sort_values("difference", ascending=True).iloc[0]
    lowest = summary.sort_values("xg", ascending=True).iloc[0]

    lines = [
        f"{leader['team']} 的機會品質較高，累積 xG 為 {leader['xg']:.2f}。",
        f"{efficient['team']} 的進球效率較高，實際進球比 xG 多 {efficient['difference']:.2f}。",
    ]
    if wasteful["difference"] < -0.25:
        lines.append(f"{wasteful['team']} 屬於高 xG 低進球， finishing 效率仍有改善空間。")
    if efficient["difference"] > 0.5:
        lines.append(f"{efficient['team']} 屬於低 xG 高進球，表現偏向效率型或把握度較佳。")
    if leader["xg"] - lowest["xg"] > 0.7:
        lines.append("本場比賽偏向一方壓制，射門品質差距明顯。")
    else:
        lines.append("雙方 xG 差距不大，比賽更偏向膠著與效率差異。")
    return lines
