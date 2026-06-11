from __future__ import annotations

import pandas as pd


COMPARISON_METRICS = [
    "Elo",
    "FIFA 排名",
    "近期狀態",
    "平均年齡",
    "xG",
    "xGA",
    "冠軍率",
    "小組出線率",
]


def _safe_float(value, default: float = 0.0) -> float:
    try:
        parsed = pd.to_numeric(pd.Series([value]), errors="coerce").fillna(default).iloc[0]
        return float(parsed)
    except Exception:
        return default


def _meta_row(team: str, team_meta: pd.DataFrame) -> pd.Series:
    if team_meta is None or team_meta.empty:
        return pd.Series(dtype=object)
    for column in ["team", "team_en"]:
        if column in team_meta.columns:
            match = team_meta[team_meta[column].astype(str).eq(str(team))]
            if not match.empty:
                return match.iloc[0]
    return pd.Series(dtype=object)


def _simulation_row(team: str, simulation_df: pd.DataFrame) -> pd.Series:
    if simulation_df is None or simulation_df.empty or "team" not in simulation_df.columns:
        return pd.Series(dtype=object)
    match = simulation_df[simulation_df["team"].astype(str).eq(str(team))]
    return match.iloc[0] if not match.empty else pd.Series(dtype=object)


def _team_players(team: str, players_df: pd.DataFrame) -> pd.DataFrame:
    if players_df is None or players_df.empty:
        return pd.DataFrame()
    for column in ["team", "team_en"]:
        if column in players_df.columns:
            subset = players_df[players_df[column].astype(str).eq(str(team))]
            if not subset.empty:
                return subset.copy()
    return pd.DataFrame()


def _xg_values(team: str, shots_df: pd.DataFrame) -> tuple[float, float]:
    if shots_df is None or shots_df.empty or "team" not in shots_df.columns:
        return 1.2, 1.1
    shots = shots_df.copy()
    shots["xg_value"] = pd.to_numeric(shots.get("xg_value", 0), errors="coerce").fillna(0)
    own = float(shots.loc[shots["team"].astype(str).eq(str(team)), "xg_value"].sum())
    opponent = float(shots.loc[~shots["team"].astype(str).eq(str(team)), "xg_value"].mean() or 1.1)
    if own <= 0:
        own = 1.2
    if opponent <= 0:
        opponent = 1.1
    return own, opponent


def team_profile(
    team: str,
    team_meta: pd.DataFrame,
    players_df: pd.DataFrame,
    simulation_df: pd.DataFrame,
    wc_team_stats: pd.DataFrame,
    shots_df: pd.DataFrame,
) -> dict[str, object]:
    meta = _meta_row(team, team_meta)
    sim = _simulation_row(team, simulation_df)
    players = _team_players(team, players_df)
    xg, xga = _xg_values(team, shots_df)

    avg_age = 0.0
    if not players.empty and "age" in players.columns:
        avg_age = float(pd.to_numeric(players["age"], errors="coerce").dropna().mean() or 0)

    best_finish = "N/A"
    if wc_team_stats is not None and not wc_team_stats.empty and "team" in wc_team_stats.columns:
        stat = wc_team_stats[wc_team_stats["team"].astype(str).eq(str(team))]
        if not stat.empty:
            best_finish = str(stat.iloc[0].get("best_finish", stat.iloc[0].get("best_result", "N/A")) or "N/A")

    return {
        "team": team,
        "Elo": _safe_float(meta.get("elo", sim.get("elo", 1700)), 1700),
        "FIFA 排名": _safe_float(meta.get("fifa_ranking", 99), 99),
        "近期狀態": _safe_float(sim.get("recent_form", 0.5), 0.5),
        "平均年齡": avg_age if avg_age > 0 else 26.5,
        "xG": xg,
        "xGA": xga,
        "冠軍率": _safe_float(sim.get("champion_probability", 0), 0),
        "小組出線率": _safe_float(sim.get("group_qualified_probability", 0), 0),
        "世界盃最佳成績": best_finish,
    }


def comparison_table(profile_a: dict[str, object], profile_b: dict[str, object], name_a: str, name_b: str) -> pd.DataFrame:
    rows = []
    for metric in COMPARISON_METRICS:
        rows.append({"指標": metric, name_a: profile_a.get(metric, 0), name_b: profile_b.get(metric, 0)})
    rows.append({"指標": "世界盃最佳成績", name_a: profile_a.get("世界盃最佳成績", "N/A"), name_b: profile_b.get("世界盃最佳成績", "N/A")})
    return pd.DataFrame(rows)


def radar_values(profile: dict[str, object]) -> dict[str, float]:
    elo = (_safe_float(profile.get("Elo"), 1700) - 1400) / 600
    fifa = 1 - min(_safe_float(profile.get("FIFA 排名"), 99), 100) / 100
    form = _safe_float(profile.get("近期狀態"), 0.5)
    xg = min(_safe_float(profile.get("xG"), 1.2) / 3, 1)
    xga = 1 - min(_safe_float(profile.get("xGA"), 1.1) / 3, 1)
    champion = min(_safe_float(profile.get("冠軍率"), 0) * 8, 1)
    return {
        "Elo": max(0, min(1, elo)),
        "FIFA": max(0, min(1, fifa)),
        "近期狀態": max(0, min(1, form)),
        "xG": max(0, min(1, xg)),
        "防守": max(0, min(1, xga)),
        "冠軍率": max(0, min(1, champion)),
    }


def comparison_text(name_a: str, profile_a: dict[str, object], name_b: str, profile_b: dict[str, object]) -> str:
    elo_gap = _safe_float(profile_a.get("Elo"), 1700) - _safe_float(profile_b.get("Elo"), 1700)
    champ_gap = _safe_float(profile_a.get("冠軍率"), 0) - _safe_float(profile_b.get("冠軍率"), 0)
    xg_gap = _safe_float(profile_a.get("xG"), 0) - _safe_float(profile_b.get("xG"), 0)
    leader = name_a if elo_gap >= 0 else name_b
    if abs(elo_gap) < 35:
        strength = "雙方 Elo 差距有限，比賽可能偏向拉鋸。"
    else:
        strength = f"{leader} 在 Elo 評分上較有優勢。"
    champ_text = "冠軍率差距不大。" if abs(champ_gap) < 0.02 else f"{name_a if champ_gap > 0 else name_b} 的長線模擬表現較突出。"
    xg_text = "xG 表現接近。" if abs(xg_gap) < 0.25 else f"{name_a if xg_gap > 0 else name_b} 的進攻機會品質較高。"
    return f"{strength}{champ_text}{xg_text}整體判讀仍需搭配賽程、球員狀態與臨場戰術。"
