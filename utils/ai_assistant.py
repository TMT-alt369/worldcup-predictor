from __future__ import annotations

from datetime import datetime

import pandas as pd


INSUFFICIENT_DATA_MESSAGE = "目前資料不足，請改問其他球隊或賽事。"


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(pd.to_numeric(pd.Series([value]), errors="coerce").fillna(default).iloc[0])
    except Exception:
        return default


def _find_team(question: str, team_meta: pd.DataFrame) -> str | None:
    if team_meta is None or team_meta.empty:
        return None
    for _, row in team_meta.iterrows():
        candidates = [str(row.get("team", "")), str(row.get("team_en", "")), str(row.get("team_zh", ""))]
        for candidate in candidates:
            if candidate and candidate != "N/A" and candidate.lower() in question.lower():
                return str(row.get("team", row.get("team_en", candidate)))
    return None


def _team_display(team: str, team_meta: pd.DataFrame) -> str:
    if team_meta is None or team_meta.empty:
        return team
    for column in ["team", "team_en"]:
        if column in team_meta.columns:
            match = team_meta[team_meta[column].astype(str).eq(str(team))]
            if not match.empty:
                row = match.iloc[0]
                return f"{row.get('flag_emoji', '')} {row.get('team_zh', team)}".strip()
    return team


def _champion_probability(team: str, simulation_df: pd.DataFrame) -> float | None:
    if simulation_df is None or simulation_df.empty or "team" not in simulation_df.columns:
        return None
    row = simulation_df[simulation_df["team"].astype(str).eq(str(team))]
    if row.empty:
        return None
    return _safe_float(row.iloc[0].get("champion_probability", 0), 0)


def _elo_row(team: str, team_meta: pd.DataFrame) -> pd.Series:
    if team_meta is None or team_meta.empty:
        return pd.Series(dtype=object)
    for column in ["team", "team_en"]:
        if column in team_meta.columns:
            match = team_meta[team_meta[column].astype(str).eq(str(team))]
            if not match.empty:
                return match.iloc[0]
    return pd.Series(dtype=object)


def _answer_champion(question: str, team_meta: pd.DataFrame, simulation_df: pd.DataFrame) -> str:
    team = _find_team(question, team_meta)
    if team:
        probability = _champion_probability(team, simulation_df)
        if probability is None:
            return INSUFFICIENT_DATA_MESSAGE
        return f"{_team_display(team, team_meta)} 的目前模擬奪冠機率約 {probability * 100:.1f}%。此結果僅供機率分析，不代表實際賽果。"
    if simulation_df is None or simulation_df.empty:
        return INSUFFICIENT_DATA_MESSAGE
    top = simulation_df.sort_values("champion_probability", ascending=False).head(5)
    lines = [
        f"{idx + 1}. {row.get('team_display', row.get('team', 'N/A'))}：{_safe_float(row.get('champion_probability')) * 100:.1f}%"
        for idx, (_, row) in enumerate(top.iterrows())
    ]
    return "目前奪冠機率前 5 名為：" + "；".join(lines) + "。"


def _answer_compare(question: str, team_meta: pd.DataFrame) -> str:
    teams = []
    if team_meta is not None and not team_meta.empty:
        for _, row in team_meta.iterrows():
            team = str(row.get("team", row.get("team_en", "")))
            names = [team, str(row.get("team_zh", "")), str(row.get("team_en", ""))]
            if any(name and name.lower() in question.lower() for name in names):
                teams.append(team)
    teams = list(dict.fromkeys(teams))[:2]
    if len(teams) < 2:
        return INSUFFICIENT_DATA_MESSAGE
    rows = [_elo_row(team, team_meta) for team in teams]
    elos = [_safe_float(row.get("elo", 1700), 1700) for row in rows]
    leader = teams[0] if elos[0] >= elos[1] else teams[1]
    gap = abs(elos[0] - elos[1])
    return f"{_team_display(leader, team_meta)} 的 Elo 較高，兩隊差距約 {gap:.0f} 分。若差距低於 35 分，仍可視為接近對戰。"


def _answer_today(fixtures_df: pd.DataFrame, team_meta: pd.DataFrame) -> str:
    if fixtures_df is None or fixtures_df.empty:
        return INSUFFICIENT_DATA_MESSAGE
    df = fixtures_df.copy().head(5)
    lines = []
    for _, row in df.iterrows():
        date_text = str(row.get("taiwan_time", row.get("date", "N/A")))
        home = _team_display(str(row.get("home_team", "")), team_meta)
        away = _team_display(str(row.get("away_team", "")), team_meta)
        lines.append(f"{date_text}：{home} vs {away}")
    return "近期賽程包含：" + "；".join(lines) + "。"


def _answer_top_elo(team_meta: pd.DataFrame) -> str:
    if team_meta is None or team_meta.empty or "elo" not in team_meta.columns:
        return INSUFFICIENT_DATA_MESSAGE
    df = team_meta.copy()
    df["elo"] = pd.to_numeric(df["elo"], errors="coerce").fillna(0)
    row = df.sort_values("elo", ascending=False).iloc[0]
    team = str(row.get("team", row.get("team_en", "")))
    return f"目前資料中 Elo 最高的是 {_team_display(team, team_meta)}，Elo 約 {int(row['elo'])} 分。"


def _answer_xg(shots_df: pd.DataFrame, team_meta: pd.DataFrame) -> str:
    if shots_df is None or shots_df.empty or "team" not in shots_df.columns:
        return INSUFFICIENT_DATA_MESSAGE
    df = shots_df.copy()
    df["xg_value"] = pd.to_numeric(df.get("xg_value", 0), errors="coerce").fillna(0)
    grouped = df.groupby("team", as_index=False)["xg_value"].sum().sort_values("xg_value", ascending=False)
    if grouped.empty:
        return INSUFFICIENT_DATA_MESSAGE
    row = grouped.iloc[0]
    return f"xG 表現最高的是 {_team_display(str(row['team']), team_meta)}，累積 xG 約 {float(row['xg_value']):.2f}。"


def answer_question(
    question: str,
    fixtures_df: pd.DataFrame,
    team_meta: pd.DataFrame,
    simulation_df: pd.DataFrame,
    shots_df: pd.DataFrame,
    players_df: pd.DataFrame,
    matches_df: pd.DataFrame,
) -> str:
    q = (question or "").strip()
    if not q:
        return "請輸入想查詢的球隊、賽程或模型問題。"

    if any(word in q for word in ["奪冠", "冠軍機率", "冠軍率"]):
        return _answer_champion(q, team_meta, simulation_df)
    if any(word in q for word in ["誰比較強", "比較強", "對戰"]):
        return _answer_compare(q, team_meta)
    if any(word in q for word in ["今天", "近期", "有哪些比賽", "賽程"]):
        return _answer_today(fixtures_df, team_meta)
    if any(word in q for word in ["Elo", "elo", "排名最高"]):
        return _answer_top_elo(team_meta)
    if any(word in q for word in ["xG", "XG", "預期進球"]):
        return _answer_xg(shots_df, team_meta)
    if "最膠著" in q:
        return "最膠著比賽可從單場分析頁查看勝平負機率差距；目前資料不足以直接排序所有比賽。"
    if "路徑最難" in q:
        return "冠軍路徑難度可在冠軍路徑模擬頁查看最大阻礙對手；目前資料不足以直接排序所有球隊。"
    return INSUFFICIENT_DATA_MESSAGE
