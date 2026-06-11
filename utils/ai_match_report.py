from __future__ import annotations

import pandas as pd


def _pct(value: float) -> str:
    return f"{float(value) * 100:.1f}%"


def _team_name(team: str, team_meta: pd.DataFrame | None = None) -> str:
    if team_meta is None or team_meta.empty or "team" not in team_meta.columns:
        return str(team)
    row = team_meta[team_meta["team"] == team]
    if row.empty:
        return str(team)
    flag = str(row.iloc[0].get("flag_emoji", "") or "")
    name = str(row.iloc[0].get("team_zh", team) or team)
    return f"{flag} {name}".strip()


def _market_gap_text(market_table: pd.DataFrame | None) -> str:
    if market_table is None or market_table.empty:
        return "市場機率資料不足，暫以模型機率作為主要參考。"
    table = market_table.copy()
    table["gap"] = (
        pd.to_numeric(table["model_probability"], errors="coerce").fillna(0)
        - pd.to_numeric(table["market_probability"], errors="coerce").fillna(0)
    ).abs()
    row = table.sort_values("gap", ascending=False).iloc[0]
    return (
        f"{row['market']} 的模型與市場差距最大，模型機率為 {_pct(row['model_probability'])}，"
        f"市場機率為 {_pct(row['market_probability'])}。"
    )


def _form_text(matches: pd.DataFrame, home_team: str, away_team: str) -> str:
    def points(team: str) -> float:
        team_matches = matches[(matches["home_team"] == team) | (matches["away_team"] == team)].tail(5)
        if team_matches.empty:
            return 0.5
        scores = []
        for _, row in team_matches.iterrows():
            is_home = row["home_team"] == team
            gf = int(row["home_goals"] if is_home else row["away_goals"])
            ga = int(row["away_goals"] if is_home else row["home_goals"])
            scores.append(3 if gf > ga else 1 if gf == ga else 0)
        return sum(scores) / max(1, len(scores) * 3)

    home_form = points(home_team)
    away_form = points(away_team)
    if abs(home_form - away_form) < 0.08:
        return "近期狀態接近，勝負差距主要取決於臨場效率。"
    leader = home_team if home_form > away_form else away_team
    return f"{leader} 近期狀態略優，近況分數差距約 {abs(home_form - away_form):.2f}。"


def generate_match_report(
    fixture: pd.Series,
    prediction,
    matches: pd.DataFrame,
    team_meta: pd.DataFrame | None = None,
    market_table: pd.DataFrame | None = None,
    xg_summary: pd.DataFrame | None = None,
    confidence_result=None,
) -> list[str]:
    home = str(fixture.get("home_team", prediction.home_team))
    away = str(fixture.get("away_team", prediction.away_team))
    home_name = _team_name(home, team_meta)
    away_name = _team_name(away, team_meta)

    probabilities = {
        home_name: float(prediction.home_win_probability),
        "和局": float(prediction.draw_probability),
        away_name: float(prediction.away_win_probability),
    }
    likely_result, likely_probability = max(probabilities.items(), key=lambda item: item[1])
    uncertainty = "偏高" if float(prediction.draw_probability) >= 0.26 or likely_probability < 0.45 else "中等"
    total_goals = float(prediction.expected_home_goals) + float(prediction.expected_away_goals)
    total_goal_text = "偏向大 2.5" if total_goals >= 2.65 else "偏向小 2.5"

    home_elo = pd.to_numeric(
        matches.loc[matches["home_team"].eq(home), "home_elo"].tail(5), errors="coerce"
    ).mean()
    away_elo = pd.to_numeric(
        matches.loc[matches["away_team"].eq(away), "away_elo"].tail(5), errors="coerce"
    ).mean()
    if pd.isna(home_elo):
        home_elo = 1700
    if pd.isna(away_elo):
        away_elo = 1700
    elo_gap = float(home_elo - away_elo)
    if abs(elo_gap) < 35:
        elo_text = "Elo 差距有限，雙方基礎實力接近。"
    else:
        elo_text = f"Elo 顯示{'主隊' if elo_gap > 0 else '客隊'}約有 {abs(elo_gap):.0f} 分優勢。"

    xg_text = "xG 資料不足，暫以 Poisson 預期進球作為機會品質代理。"
    if xg_summary is not None and not xg_summary.empty and {"team", "xg"}.issubset(xg_summary.columns):
        xg_leader = xg_summary.sort_values("xg", ascending=False).iloc[0]
        xg_text = f"xG 參考資料顯示 {xg_leader['team']} 近期機會品質較高。"

    confidence_text = ""
    if confidence_result is not None:
        confidence_text = (
            f"智慧信心指數為 {confidence_result.score:.1f}%（{confidence_result.stars} {confidence_result.level}），"
            f"主要方向為 {confidence_result.favored_market}。"
        )

    lines = [
        f"本場模型最看好 {likely_result}，機率為 {_pct(likely_probability)}。",
        elo_text,
        _form_text(matches, home, away),
        xg_text,
        _market_gap_text(market_table),
        f"可能比分為 {prediction.predicted_home_goals}:{prediction.predicted_away_goals}，預估總進球約 {total_goals:.2f} 球，大小球方向{total_goal_text}。",
        f"和局機率為 {_pct(prediction.draw_probability)}，整體結果不確定性{uncertainty}。",
        "本報告僅供機率分析，不構成下注建議。",
    ]
    if confidence_text:
        lines.insert(1, confidence_text)
    return lines
