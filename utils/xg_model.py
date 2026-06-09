from __future__ import annotations

import math

import pandas as pd


def calculate_shot_xg(row: pd.Series) -> float:
    if str(row.get("situation", "")).lower() == "penalty":
        return 0.76
    distance = float(row.get("shot_distance", 20))
    angle = float(row.get("shot_angle", 25))
    body_part = str(row.get("body_part", "")).lower()
    situation = str(row.get("situation", "")).lower()
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
    if "xg_value" not in data.columns:
        data["xg_value"] = data.apply(calculate_shot_xg, axis=1)
    data["xg_value"] = pd.to_numeric(data["xg_value"], errors="coerce").fillna(0.01).clip(0.01, 0.95)
    data["minute"] = pd.to_numeric(data["minute"], errors="coerce").fillna(0).astype(int)
    data["is_goal"] = data["is_goal"].astype(str).str.lower().isin(["true", "1", "yes", "y"])
    return data.sort_values(["match_id", "minute"]).reset_index(drop=True)


def xg_match_summary(shots: pd.DataFrame, match_id: str) -> pd.DataFrame:
    data = prepare_xg_data(shots)
    match = data[data["match_id"] == match_id]
    if match.empty:
        return pd.DataFrame()
    return (
        match.groupby("team", as_index=False)
        .agg(xg=("xg_value", "sum"), goals=("is_goal", "sum"), shots=("player", "count"))
        .assign(xg=lambda frame: frame["xg"].round(2), difference=lambda frame: (frame["goals"] - frame["xg"]).round(2))
    )


def xg_analysis_text(summary: pd.DataFrame) -> list[str]:
    if summary.empty or len(summary) < 2:
        return ["目前尚未匯入足夠 xG 射門資料。"]
    leader = summary.sort_values("xg", ascending=False).iloc[0]
    efficient = summary.sort_values("difference", ascending=False).iloc[0]
    wasteful = summary.sort_values("difference", ascending=True).iloc[0]
    lines = [
        f"{leader['team']} 的機會品質較高，累積 xG 為 {leader['xg']:.2f}。",
        f"{efficient['team']} 的進球效率較高，實際進球比 xG 多 {efficient['difference']:.2f}。",
    ]
    if wasteful["difference"] < -0.25:
        lines.append(f"{wasteful['team']} 屬於高 xG 低進球，可能是把握度不足或門將表現影響。")
    if efficient["difference"] > 0.5:
        lines.append(f"{efficient['team']} 屬於低 xG 高進球，進攻效率或臨門一腳表現突出。")
    if leader["xg"] - summary.sort_values("xg").iloc[0]["xg"] > 0.7:
        lines.append("比賽型態偏向一方壓制。")
    else:
        lines.append("比賽型態偏向效率與細節差距。")
    return lines
