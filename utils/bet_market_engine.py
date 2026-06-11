from __future__ import annotations

from math import exp, factorial
from types import SimpleNamespace

import pandas as pd


COMMON_SCORES = [
    (0, 0), (1, 0), (0, 1), (1, 1), (2, 0), (0, 2), (2, 1), (1, 2),
    (2, 2), (3, 0), (0, 3), (3, 1), (1, 3),
]

STAGE_COLUMNS = [
    ("小組出線", "group_qualified_probability"),
    ("32 強", "round_32_probability"),
    ("16 強", "round_16_probability"),
    ("8 強", "round_8_probability"),
    ("4 強", "semi_final_probability"),
    ("決賽", "final_probability"),
    ("冠軍", "champion_probability"),
]

CONTINENT_LABELS = {
    "UEFA": "歐洲",
    "CONMEBOL": "南美洲",
    "CONCACAF": "中北美洲",
    "CAF": "非洲",
    "AFC": "亞洲",
    "OFC": "大洋洲",
}


def _poisson(lam: float, goals: int) -> float:
    lam = max(0.05, float(lam))
    return (lam**goals * exp(-lam)) / factorial(goals)


def score_distribution(home_xg: float, away_xg: float, max_goals: int = 6) -> pd.DataFrame:
    rows = []
    for home_goals in range(max_goals + 1):
        for away_goals in range(max_goals + 1):
            rows.append(
                {
                    "home_goals": home_goals,
                    "away_goals": away_goals,
                    "probability": _poisson(home_xg, home_goals) * _poisson(away_xg, away_goals),
                }
            )
    df = pd.DataFrame(rows)
    total = float(df["probability"].sum()) or 1.0
    df["probability"] = df["probability"] / total
    return df


def _estimated_odds(probability: float) -> float:
    probability = max(0.005, min(0.98, float(probability)))
    return round(1 / probability, 2)


def _risk(probability: float, estimated_odds: float) -> str:
    if probability >= 0.60:
        level = "低"
    elif probability >= 0.40:
        level = "中"
    else:
        level = "高"
    if estimated_odds > 6.0 and level == "低":
        level = "中"
    elif estimated_odds > 6.0 and level == "中":
        level = "高"
    return level


def _fuse(probability: float, market_odds: float | None = None) -> float:
    probability = max(0, min(1, float(probability)))
    if market_odds is None or float(market_odds or 0) <= 1:
        return probability
    market_probability = max(0, min(1, 1 / float(market_odds)))
    return 0.70 * probability + 0.30 * market_probability


def _row(
    market_id: str,
    market_name: str,
    option_name: str,
    probability: float,
    explanation: str,
    market_odds: float | None = None,
) -> dict[str, object]:
    fused = _fuse(probability, market_odds)
    odds = _estimated_odds(fused)
    return {
        "market_id": market_id,
        "market_name": market_name,
        "option_name": option_name,
        "probability": round(float(probability), 4),
        "estimated_odds": odds,
        "market_odds": round(float(market_odds), 2) if market_odds else "",
        "fused_probability": round(float(fused), 4),
        "risk_level": _risk(fused, odds),
        "explanation": explanation,
    }


def _from_rows(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _get_value(source, names: list[str], default: float = 0.0) -> float:
    for name in names:
        if isinstance(source, dict) and name in source:
            return float(source.get(name) or default)
        if isinstance(source, pd.Series) and name in source:
            return float(source.get(name) or default)
        if hasattr(source, name):
            return float(getattr(source, name) or default)
    return float(default)


def _as_prediction(prediction) -> SimpleNamespace:
    return SimpleNamespace(
        home_win_probability=_get_value(prediction, ["home_win_probability", "home_win"], 0.34),
        draw_probability=_get_value(prediction, ["draw_probability", "draw"], 0.28),
        away_win_probability=_get_value(prediction, ["away_win_probability", "away_win"], 0.38),
        expected_home_goals=_get_value(prediction, ["expected_home_goals", "predicted_home_goals", "home_xg"], 1.3),
        expected_away_goals=_get_value(prediction, ["expected_away_goals", "predicted_away_goals", "away_xg"], 1.1),
    )


def predict_1x2(prediction, fixture: pd.Series | None = None) -> pd.DataFrame:
    prediction = _as_prediction(prediction)
    odds = fixture if fixture is not None else {}
    return _from_rows(
        [
            _row("1x2_home", "不讓分勝平負", "主勝", prediction.home_win_probability, "主隊進球期望高於客隊時機率會上升。", odds.get("home_odds")),
            _row("1x2_draw", "不讓分勝平負", "和局", prediction.draw_probability, "雙方期望進球接近時和局機率較高。", odds.get("draw_odds")),
            _row("1x2_away", "不讓分勝平負", "客勝", prediction.away_win_probability, "客隊進攻與防守效率較佳時機率會上升。", odds.get("away_odds")),
        ]
    )


def predict_handicap(score_df: pd.DataFrame) -> pd.DataFrame:
    markets = [
        ("handicap_home_-1", "主隊 -1", lambda h, a: h - 1 > a),
        ("handicap_home_+1", "主隊 +1", lambda h, a: h + 1 > a),
        ("handicap_away_-1", "客隊 -1", lambda h, a: a - 1 > h),
        ("handicap_away_+1", "客隊 +1", lambda h, a: a + 1 > h),
        ("handicap_home_-2", "主隊 -2", lambda h, a: h - 2 > a),
        ("handicap_away_+2", "客隊 +2", lambda h, a: a + 2 > h),
    ]
    rows = []
    for market_id, option, predicate in markets:
        probability = score_df.loc[
            score_df.apply(lambda row: predicate(int(row.home_goals), int(row.away_goals)), axis=1),
            "probability",
        ].sum()
        rows.append(_row(market_id, "讓分盤", option, probability, "依 0~6 球比分分佈推導過盤機率。"))
    return _from_rows(rows)


def predict_over_under(score_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for line in [0.5, 1.5, 2.5, 3.5, 4.5, 8.5]:
        total_goals = score_df["home_goals"] + score_df["away_goals"]
        over = float(score_df.loc[total_goals > line, "probability"].sum())
        under = 1 - over
        rows.append(_row(f"ou_over_{line}", "大小球", f"大 {line}", over, "總進球高於盤口的模型機率。"))
        rows.append(_row(f"ou_under_{line}", "大小球", f"小 {line}", under, "總進球低於或等於盤口的模型機率。"))
    return _from_rows(rows)


def predict_both_teams_score(score_df: pd.DataFrame) -> pd.DataFrame:
    yes = float(score_df.loc[(score_df["home_goals"] > 0) & (score_df["away_goals"] > 0), "probability"].sum())
    return _from_rows(
        [
            _row("btts_yes", "兩隊是否都進球", "是", yes, "雙方至少各進一球的機率。"),
            _row("btts_no", "兩隊是否都進球", "否", 1 - yes, "至少一隊沒有進球的機率。"),
        ]
    )


def predict_half_time_1x2(prediction) -> pd.DataFrame:
    prediction = _as_prediction(prediction)
    half = score_distribution(prediction.expected_home_goals * 0.45, prediction.expected_away_goals * 0.45, 4)
    home = float(half.loc[half["home_goals"] > half["away_goals"], "probability"].sum())
    draw = float(half.loc[half["home_goals"] == half["away_goals"], "probability"].sum())
    away = float(half.loc[half["home_goals"] < half["away_goals"], "probability"].sum())
    return _from_rows(
        [
            _row("ht_home", "上半場勝平負", "主勝", home, "以半場期望進球推估。"),
            _row("ht_draw", "上半場勝平負", "和局", draw, "半場進球較少時和局機率通常較高。"),
            _row("ht_away", "上半場勝平負", "客勝", away, "以客隊半場進球分佈推估。"),
        ]
    )


def predict_half_full_time(prediction) -> pd.DataFrame:
    prediction = _as_prediction(prediction)
    first = score_distribution(prediction.expected_home_goals * 0.45, prediction.expected_away_goals * 0.45, 4)
    second = score_distribution(prediction.expected_home_goals * 0.55, prediction.expected_away_goals * 0.55, 4)
    labels = {"home": "主", "draw": "和", "away": "客"}
    probs = {(a, b): 0.0 for a in labels for b in labels}
    for _, h1 in first.iterrows():
        ht = "home" if h1.home_goals > h1.away_goals else "away" if h1.home_goals < h1.away_goals else "draw"
        for _, h2 in second.iterrows():
            home_total = int(h1.home_goals + h2.home_goals)
            away_total = int(h1.away_goals + h2.away_goals)
            ft = "home" if home_total > away_total else "away" if home_total < away_total else "draw"
            probs[(ht, ft)] += float(h1.probability) * float(h2.probability)
    rows = []
    for ht in ["home", "draw", "away"]:
        for ft in ["home", "draw", "away"]:
            option = f"{labels[ht]}/{labels[ft]}"
            rows.append(_row(f"hft_{ht}_{ft}", "半全場", option, probs[(ht, ft)], "由上半場與下半場比分分佈推導。"))
    return _from_rows(rows)


def predict_half_over_under(prediction) -> pd.DataFrame:
    prediction = _as_prediction(prediction)
    half = score_distribution(prediction.expected_home_goals * 0.45, prediction.expected_away_goals * 0.45, 4)
    rows = []
    for line in [0.5, 1.5, 2.5]:
        total_goals = half["home_goals"] + half["away_goals"]
        over = float(half.loc[total_goals > line, "probability"].sum())
        rows.append(_row(f"ht_ou_over_{line}", "上半場大小球", f"大 {line}", over, "上半場總進球高於盤口的機率。"))
        rows.append(_row(f"ht_ou_under_{line}", "上半場大小球", f"小 {line}", 1 - over, "上半場總進球低於或等於盤口的機率。"))
    return _from_rows(rows)


def predict_half_team_goals(prediction) -> pd.DataFrame:
    prediction = _as_prediction(prediction)
    half = score_distribution(prediction.expected_home_goals * 0.45, prediction.expected_away_goals * 0.45, 4)
    data = predict_team_goals(half)
    data["market_name"] = "上半場正確進球數"
    data["option_name"] = data["option_name"].str.replace("主隊", "主隊上半場").str.replace("客隊", "客隊上半場")
    return data


def predict_half_correct_score(prediction) -> pd.DataFrame:
    prediction = _as_prediction(prediction)
    half = score_distribution(prediction.expected_home_goals * 0.45, prediction.expected_away_goals * 0.45, 4)
    data = predict_correct_score(half)
    data["market_name"] = "上半場正確比數"
    return data


def predict_first_goal(prediction) -> pd.DataFrame:
    prediction = _as_prediction(prediction)
    home_rate = max(0.05, float(prediction.expected_home_goals))
    away_rate = max(0.05, float(prediction.expected_away_goals))
    total = home_rate + away_rate
    no_goal = exp(-total)
    scoring = 1 - no_goal
    home = scoring * home_rate / total
    away = scoring * away_rate / total
    return _from_rows(
        [
            _row("first_goal_home", "第一球", "主隊", home, "依雙方進球率推估第一球來源。"),
            _row("first_goal_none", "第一球", "無進球", no_goal, "全場 0:0 的模型機率。"),
            _row("first_goal_away", "第一球", "客隊", away, "依客隊進球率推估第一球來源。"),
        ]
    )


def predict_correct_score(score_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    used = 0.0
    for home, away in COMMON_SCORES:
        probability = float(score_df.loc[(score_df["home_goals"] == home) & (score_df["away_goals"] == away), "probability"].sum())
        used += probability
        rows.append(_row(f"score_{home}_{away}", "正確比數", f"{home}:{away}", probability, "由 Poisson 比分分佈直接推導。"))
    rows.append(_row("score_other", "正確比數", "其他", max(0, 1 - used), "未列入常見比分的其餘結果。"))
    return _from_rows(rows)


def predict_team_goals(score_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for side, label, column in [("home", "主隊", "home_goals"), ("away", "客隊", "away_goals")]:
        for goals in [0, 1, 2]:
            probability = float(score_df.loc[score_df[column] == goals, "probability"].sum())
            rows.append(_row(f"{side}_goals_{goals}", "球隊正確進球數", f"{label} {goals} 球", probability, "單隊進球數分佈。"))
        probability = float(score_df.loc[score_df[column] >= 3, "probability"].sum())
        rows.append(_row(f"{side}_goals_3plus", "球隊正確進球數", f"{label} 3+ 球", probability, "單隊進 3 球以上機率。"))
    return _from_rows(rows)


def predict_total_goals_range(score_df: pd.DataFrame) -> pd.DataFrame:
    total = score_df["home_goals"] + score_df["away_goals"]
    ranges = [
        ("0_1", "0-1 球", total <= 1),
        ("2_3", "2-3 球", (total >= 2) & (total <= 3)),
        ("4_5", "4-5 球", (total >= 4) & (total <= 5)),
        ("6plus", "6+ 球", total >= 6),
    ]
    return _from_rows([
        _row(f"total_range_{key}", "總進球數", option, float(score_df.loc[mask, "probability"].sum()), "由總進球分佈推導。")
        for key, option, mask in ranges
    ])


def predict_odd_even(score_df: pd.DataFrame) -> pd.DataFrame:
    total = score_df["home_goals"] + score_df["away_goals"]
    odd = float(score_df.loc[total % 2 == 1, "probability"].sum())
    return _from_rows(
        [
            _row("odd_even_odd", "總進球單雙", "單", odd, "總進球為奇數的機率。"),
            _row("odd_even_even", "總進球單雙", "雙", 1 - odd, "總進球為偶數的機率。"),
        ]
    )


def predict_group_winner(fixtures: pd.DataFrame, simulation_df: pd.DataFrame) -> pd.DataFrame:
    if fixtures is None or fixtures.empty or simulation_df is None or simulation_df.empty:
        return pd.DataFrame()
    teams = []
    for group, group_df in fixtures.groupby("group"):
        group_teams = sorted(set(group_df["home_team"].astype(str)) | set(group_df["away_team"].astype(str)))
        sim = simulation_df[simulation_df["team"].astype(str).isin(group_teams)].copy()
        if sim.empty:
            continue
        sim["strength"] = pd.to_numeric(sim.get("strength", sim.get("elo", 1700)), errors="coerce").fillna(1)
        total_strength = float(sim["strength"].sum()) or 1.0
        for _, row in sim.iterrows():
            probability = float(row["strength"]) / total_strength
            teams.append(_row(f"group_winner_{group}_{row['team']}", "小組第一", f"{group} {row.get('team_display', row['team'])}", probability, "依小組內模擬強度推估第一名機率。"))
    return _from_rows(teams)


def predict_champion(simulation_df: pd.DataFrame) -> pd.DataFrame:
    if simulation_df is None or simulation_df.empty:
        return pd.DataFrame()
    rows = []
    for _, row in simulation_df.sort_values("champion_probability", ascending=False).iterrows():
        rows.append(_row(f"champion_{row['team']}", "冠軍", str(row.get("team_display", row["team"])), row.get("champion_probability", 0), "Monte Carlo 冠軍率。"))
    return _from_rows(rows)


def predict_continent_winner(simulation_df: pd.DataFrame, team_meta: pd.DataFrame) -> pd.DataFrame:
    if simulation_df is None or simulation_df.empty or team_meta is None or team_meta.empty:
        return pd.DataFrame()
    meta_cols = [col for col in ["team", "confederation"] if col in team_meta.columns]
    if len(meta_cols) < 2:
        return pd.DataFrame()
    merged = simulation_df.merge(team_meta[meta_cols], on="team", how="left")
    merged["champion_probability"] = pd.to_numeric(merged.get("champion_probability", 0), errors="coerce").fillna(0)
    rows = []
    for confed, group in merged.groupby("confederation"):
        option = CONTINENT_LABELS.get(str(confed), str(confed))
        rows.append(_row(f"continent_{confed}", "冠軍來自哪一洲", option, float(group["champion_probability"].sum()), "依各洲參賽隊伍冠軍率加總。"))
    return _from_rows(rows).sort_values("probability", ascending=False).reset_index(drop=True)


def predict_team_reaches_stage(simulation_df: pd.DataFrame) -> pd.DataFrame:
    if simulation_df is None or simulation_df.empty:
        return pd.DataFrame()
    rows = []
    for _, row in simulation_df.iterrows():
        team = str(row.get("team_display", row.get("team", "")))
        for label, column in STAGE_COLUMNS:
            probability = row.get(column, row.get("group_qualified_probability", 0))
            rows.append(_row(f"stage_{row.get('team', team)}_{column}", "球隊晉級階段", f"{team} {label}", probability, "Monte Carlo 階段晉級率。"))
    return _from_rows(rows)


def build_match_markets(prediction, fixture: pd.Series | None = None) -> dict[str, pd.DataFrame]:
    prediction = _as_prediction(prediction)
    score_df = score_distribution(prediction.expected_home_goals, prediction.expected_away_goals)
    return {
        "1x2": predict_1x2(prediction, fixture),
        "handicap": predict_handicap(score_df),
        "over_under": predict_over_under(score_df),
        "both_teams_score": predict_both_teams_score(score_df),
        "first_goal": predict_first_goal(prediction),
        "correct_score": predict_correct_score(score_df),
        "team_goals": predict_team_goals(score_df),
        "total_goals_range": predict_total_goals_range(score_df),
        "odd_even": predict_odd_even(score_df),
        "half_time_1x2": predict_half_time_1x2(prediction),
        "half_full_time": predict_half_full_time(prediction),
        "half_over_under": predict_half_over_under(prediction),
        "half_team_goals": predict_half_team_goals(prediction),
        "half_correct_score": predict_half_correct_score(prediction),
    }
