from __future__ import annotations

import pandas as pd


def _pct(value: float) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = 0.0
    if pd.isna(number):
        number = 0.0
    return f"{number * 100:.1f}%"


def _safe_prediction_value(prediction, names: list[str], default: float = 0.0) -> float:
    for name in names:
        try:
            if isinstance(prediction, dict) and name in prediction:
                value = prediction.get(name, default)
            elif hasattr(prediction, name):
                value = getattr(prediction, name, default)
            else:
                continue
            number = float(value if value is not None else default)
            return float(default) if pd.isna(number) else number
        except (TypeError, ValueError):
            continue
    return float(default)


def _safe_text(source, name: str, default: str = "") -> str:
    if isinstance(source, dict):
        value = source.get(name, default)
    else:
        value = getattr(source, name, default)
    if value is None:
        return default
    text = str(value)
    return default if "?" in text else text


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
    if not {"model_probability", "market_probability", "market"}.issubset(table.columns):
        return "市場機率資料欄位不足，暫以模型機率作為主要參考。"
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
        if matches is None or matches.empty or not {"home_team", "away_team", "home_goals", "away_goals"}.issubset(matches.columns):
            return 0.5
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
    home = str(fixture.get("home_team", _safe_text(prediction, "home_team", "主隊")))
    away = str(fixture.get("away_team", _safe_text(prediction, "away_team", "客隊")))
    home_name = _team_name(home, team_meta)
    away_name = _team_name(away, team_meta)
    home_win_probability = _safe_prediction_value(prediction, ["home_win_probability", "home_win"], 0.34)
    draw_probability = _safe_prediction_value(prediction, ["draw_probability", "draw"], 0.28)
    away_win_probability = _safe_prediction_value(prediction, ["away_win_probability", "away_win"], 0.38)
    expected_home_goals = _safe_prediction_value(prediction, ["expected_home_goals", "predicted_home_goals", "home_xg"], 1.2)
    expected_away_goals = _safe_prediction_value(prediction, ["expected_away_goals", "predicted_away_goals", "away_xg"], 1.1)
    predicted_home_goals = int(round(_safe_prediction_value(prediction, ["predicted_home_goals"], expected_home_goals)))
    predicted_away_goals = int(round(_safe_prediction_value(prediction, ["predicted_away_goals"], expected_away_goals)))

    probabilities = {
        home_name: home_win_probability,
        "和局": draw_probability,
        away_name: away_win_probability,
    }
    likely_result, likely_probability = max(probabilities.items(), key=lambda item: item[1])
    uncertainty = "偏高" if draw_probability >= 0.26 or likely_probability < 0.45 else "中等"
    total_goals = expected_home_goals + expected_away_goals
    total_goal_text = "偏向大 2.5" if total_goals >= 2.65 else "偏向小 2.5"

    if matches is not None and not matches.empty and {"home_team", "away_team", "home_elo", "away_elo"}.issubset(matches.columns):
        home_elo = pd.to_numeric(
            matches.loc[matches["home_team"].eq(home), "home_elo"].tail(5), errors="coerce"
        ).mean()
        away_elo = pd.to_numeric(
            matches.loc[matches["away_team"].eq(away), "away_elo"].tail(5), errors="coerce"
        ).mean()
    else:
        home_elo = 1700
        away_elo = 1700
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
        try:
            if isinstance(confidence_result, dict):
                score = float(confidence_result.get("score", 0) or 0)
            else:
                score = float(getattr(confidence_result, "score", 0) or 0)
            stars = _safe_text(confidence_result, "stars", "★★")
            level = _safe_text(confidence_result, "level", "基礎信心")
            favored_market = _safe_text(confidence_result, "favored_market", likely_result)
            confidence_text = (
                f"智慧信心指數為 {score:.1f}%（{stars} {level}），"
                f"主要方向為 {favored_market}。"
            )
        except Exception:
            confidence_text = "目前資料不足，僅提供基礎分析。"

    lines = [
        f"本場模型最看好 {likely_result}，機率為 {_pct(likely_probability)}。",
        elo_text,
        _form_text(matches, home, away),
        xg_text,
        _market_gap_text(market_table),
        f"可能比分為 {predicted_home_goals}:{predicted_away_goals}，預估總進球約 {total_goals:.2f} 球，大小球方向{total_goal_text}。",
        f"和局機率為 {_pct(draw_probability)}，整體結果不確定性{uncertainty}。",
        "本報告僅供機率分析，不構成下注建議。",
    ]
    if confidence_text:
        lines.insert(1, confidence_text)
    return lines
