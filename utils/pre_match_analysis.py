from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from worldcup_predictor.model import team_strength
from worldcup_predictor.team_resolver import find_team_row, resolve_team_name, team_match_key


ABILITY_ITEMS = ["進攻", "防守", "中場控制", "球星影響力", "綜合實力"]


@dataclass(frozen=True)
class TeamAbility:
    team: str
    attack: float
    defense: float
    midfield: float
    star_power: float
    overall: float
    estimated: bool
    notes: tuple[str, ...]


def _clamp(value: float, lower: float = 0.0, upper: float = 100.0) -> float:
    return float(max(lower, min(upper, value)))


def _safe_number(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _scale(value: float, minimum: float, maximum: float, out_min: float = 45.0, out_max: float = 98.0) -> float:
    if maximum <= minimum:
        return out_min
    ratio = (value - minimum) / (maximum - minimum)
    return _clamp(out_min + ratio * (out_max - out_min), out_min, out_max)


def _team_meta_scores(team: str, team_meta: pd.DataFrame | None) -> tuple[float, float, bool]:
    row = find_team_row(team_meta, team, columns=("team", "team_en", "team_zh", "country_code")) if team_meta is not None else None
    if row is None:
        return 65.0, 62.0, True
    elo = _safe_number(row.get("elo"), 1700)
    ranking = _safe_number(row.get("fifa_ranking"), 40)
    elo_score = _scale(elo, 1450, 2000, 48, 98)
    ranking_score = _clamp(102 - ranking * 1.15, 45, 98)
    return elo_score, ranking_score, False


def _history_score(team: str, history: pd.DataFrame | None) -> tuple[float, bool]:
    row = find_team_row(history, team, columns=("team", "team_en", "country")) if history is not None else None
    if row is None:
        return 58.0, True
    appearances = _safe_number(row.get("appearances", row.get("tournaments_played")), 0)
    win_rate = _safe_number(row.get("win_rate"), 0)
    wins = _safe_number(row.get("wins"), 0)
    best_finish = str(row.get("best_finish", "") or "").lower()
    best_bonus = 0.0
    if "champion" in best_finish or "冠軍" in best_finish:
        best_bonus = 14.0
    elif "runner" in best_finish or "final" in best_finish or "亞軍" in best_finish:
        best_bonus = 10.0
    elif "semi" in best_finish or "four" in best_finish or "四強" in best_finish:
        best_bonus = 7.0
    score = 46 + win_rate * 42 + min(appearances, 22) * 1.15 + min(wins, 80) * 0.08 + best_bonus
    return _clamp(score, 45, 98), False


def _xg_scores(team: str, shots: pd.DataFrame | None) -> tuple[float, float, bool]:
    if shots is None or shots.empty or "team" not in shots.columns:
        return 64.0, 64.0, True
    data = shots.copy()
    data["team_key"] = data["team"].map(team_match_key)
    target = team_match_key(team)
    team_rows = data[data["team_key"] == target].copy()
    if team_rows.empty or "xg_value" not in team_rows.columns:
        return 64.0, 64.0, True
    team_xg = pd.to_numeric(team_rows["xg_value"], errors="coerce").fillna(0).sum()
    match_count = max(1, team_rows.get("match_id", pd.Series(["M001"])).nunique())
    xg_per_match = team_xg / match_count

    opponent_xg = []
    if "match_id" in data.columns:
        for match_id in team_rows["match_id"].dropna().unique():
            match = data[data["match_id"].eq(match_id)].copy()
            opp = match[match["team_key"] != target]
            if not opp.empty and "xg_value" in opp.columns:
                opponent_xg.append(pd.to_numeric(opp["xg_value"], errors="coerce").fillna(0).sum())
    xga_per_match = sum(opponent_xg) / max(1, len(opponent_xg)) if opponent_xg else 1.25

    attack_score = _scale(xg_per_match, 0.45, 2.50, 50, 96)
    defense_score = _clamp(98 - _scale(xga_per_match, 0.35, 2.40, 0, 48), 50, 96)
    return attack_score, defense_score, False


def _player_scores(team: str, players: pd.DataFrame | None) -> tuple[float, bool]:
    if players is None or players.empty:
        return 62.0, True
    data = players.copy()
    if "team" not in data.columns and "team_en" in data.columns:
        data["team"] = data["team_en"]
    if "team" not in data.columns:
        return 62.0, True
    data["team_key"] = data["team"].map(team_match_key)
    rows = data[data["team_key"] == team_match_key(team)].copy()
    if rows.empty:
        return 62.0, True

    if "goals" not in rows.columns and "national_goals" in rows.columns:
        rows["goals"] = rows["national_goals"]
    if "appearances" not in rows.columns and "national_caps" in rows.columns:
        rows["appearances"] = rows["national_caps"]
    if "recent_form_rating" not in rows.columns and "recent_form" in rows.columns:
        rows["recent_form_rating"] = rows["recent_form"]
    if "recent_form_rating" not in rows.columns and "recent_form_score" in rows.columns:
        rows["recent_form_rating"] = rows["recent_form_score"]

    rows["goals"] = pd.to_numeric(rows.get("goals", 0), errors="coerce").fillna(0)
    rows["assists"] = pd.to_numeric(rows.get("assists", 0), errors="coerce").fillna(0)
    rows["appearances"] = pd.to_numeric(rows.get("appearances", 0), errors="coerce").fillna(0)
    rows["recent_form_rating"] = pd.to_numeric(rows.get("recent_form_rating", 6.6), errors="coerce").fillna(6.6)
    appearances = rows["appearances"].replace(0, pd.NA)
    rows["goal_rate"] = pd.to_numeric(rows.get("goal_rate", rows["goals"] / appearances), errors="coerce").fillna(0)
    rows["assist_rate"] = pd.to_numeric(rows.get("assist_rate", rows["assists"] / appearances), errors="coerce").fillna(0)

    if "impact_score" not in rows.columns:
        rows["impact_score"] = (
            rows["recent_form_rating"] * 7.0
            + rows["goal_rate"] * 42.0
            + rows["assist_rate"] * 28.0
            + rows["appearances"].clip(upper=120) * 0.09
        )
    rows["impact_score"] = pd.to_numeric(rows["impact_score"], errors="coerce").fillna(0)
    top = rows.sort_values("impact_score", ascending=False).head(5)
    score = top["impact_score"].mean() if not top.empty else 62.0
    return _clamp(score, 48, 98), False


def _form_score(matches: pd.DataFrame | None, team: str) -> tuple[float, bool]:
    if matches is None or matches.empty:
        return 62.0, True
    try:
        strength = team_strength(matches, resolve_team_name(team))
        form = _safe_number(strength.get("form"), 0.5)
        attack = _safe_number(strength.get("attack"), 1.25)
        defense = _safe_number(strength.get("defense"), 1.25)
    except Exception:
        return 62.0, True
    score = form * 46 + _scale(attack, 0.6, 2.4, 8, 34) + _clamp(24 - defense * 7, 6, 20)
    return _clamp(score, 45, 96), False


def calculate_team_ability(
    team: str,
    team_meta: pd.DataFrame | None = None,
    matches: pd.DataFrame | None = None,
    history: pd.DataFrame | None = None,
    players: pd.DataFrame | None = None,
    shots: pd.DataFrame | None = None,
) -> TeamAbility:
    notes: list[str] = []
    elo_score, ranking_score, meta_estimated = _team_meta_scores(team, team_meta)
    history_score, history_estimated = _history_score(team, history)
    xg_attack, xg_defense, xg_estimated = _xg_scores(team, shots)
    player_score, player_estimated = _player_scores(team, players)
    form_score, form_estimated = _form_score(matches, team)

    if meta_estimated:
        notes.append("Elo / 世界排名使用 fallback")
    if history_estimated:
        notes.append("歷史戰績使用 fallback")
    if xg_estimated:
        notes.append("xG 使用推估值")
    if player_estimated:
        notes.append("球員影響分使用推估值")
    if form_estimated:
        notes.append("近期狀態使用推估值")

    attack = 0.30 * elo_score + 0.28 * xg_attack + 0.26 * player_score + 0.16 * form_score
    defense = 0.28 * elo_score + 0.34 * xg_defense + 0.20 * history_score + 0.18 * form_score
    midfield = 0.32 * elo_score + 0.30 * form_score + 0.22 * player_score + 0.16 * ranking_score
    star_power = 0.56 * player_score + 0.24 * elo_score + 0.20 * form_score
    overall = 0.28 * attack + 0.24 * defense + 0.20 * midfield + 0.18 * star_power + 0.10 * history_score

    estimated = meta_estimated or history_estimated or xg_estimated or player_estimated or form_estimated
    return TeamAbility(
        team=resolve_team_name(team),
        attack=round(_clamp(attack), 1),
        defense=round(_clamp(defense), 1),
        midfield=round(_clamp(midfield), 1),
        star_power=round(_clamp(star_power), 1),
        overall=round(_clamp(overall), 1),
        estimated=estimated,
        notes=tuple(notes),
    )


def ability_comparison_table(home: TeamAbility, away: TeamAbility, home_label: str, away_label: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"項目": "進攻", home_label: home.attack, away_label: away.attack},
            {"項目": "防守", home_label: home.defense, away_label: away.defense},
            {"項目": "中場控制", home_label: home.midfield, away_label: away.midfield},
            {"項目": "球星影響力", home_label: home.star_power, away_label: away.star_power},
            {"項目": "綜合實力", home_label: home.overall, away_label: away.overall},
        ]
    )


def prediction_summary(prediction: Any) -> dict[str, Any]:
    probs = {
        "主勝": _safe_number(getattr(prediction, "home_win_probability", 0), 0),
        "和局": _safe_number(getattr(prediction, "draw_probability", 0), 0),
        "客勝": _safe_number(getattr(prediction, "away_win_probability", 0), 0),
    }
    best_name, best_prob = max(probs.items(), key=lambda item: item[1])
    sorted_probs = sorted(probs.values(), reverse=True)
    margin = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else best_prob
    draw_prob = probs["和局"]
    if best_prob < 0.43 or margin < 0.08 or draw_prob >= 0.30:
        risk = "高"
    elif best_prob < 0.55 or margin < 0.18 or draw_prob >= 0.25:
        risk = "中"
    else:
        risk = "低"
    confidence = _clamp(best_prob * 72 + margin * 22 + (1 - draw_prob) * 6, 25, 92)
    return {
        "home_win_probability": probs["主勝"],
        "draw_probability": probs["和局"],
        "away_win_probability": probs["客勝"],
        "favored_result": best_name,
        "risk_level": risk,
        "analysis_confidence": round(confidence, 1),
        "predicted_score": (
            int(getattr(prediction, "predicted_home_goals", 1)),
            int(getattr(prediction, "predicted_away_goals", 1)),
        ),
    }


def analysis_lines(home: TeamAbility, away: TeamAbility, summary: dict[str, Any], home_label: str, away_label: str) -> list[str]:
    lines: list[str] = []
    favored = home_label if summary["favored_result"] == "主勝" else away_label if summary["favored_result"] == "客勝" else "平局"
    lines.append(f"模型分析顯示，本場最高機率方向為{favored}，但仍需搭配風險等級解讀。")

    if abs(home.overall - away.overall) < 3:
        lines.append("從綜合實力來看，兩隊差距接近，比賽可能偏向小比分拉鋸。")
    elif home.overall > away.overall:
        lines.append(f"{home_label}在綜合實力與控場能力略占優勢。")
    else:
        lines.append(f"{away_label}在綜合實力與攻防平衡上更突出。")

    attack_gap = home.attack - away.attack
    defense_gap = home.defense - away.defense
    if attack_gap > 5:
        lines.append(f"{home_label}進攻評分較高，若能早早進球，勝率會明顯提升。")
    elif attack_gap < -5:
        lines.append(f"{away_label}進攻端威脅較大，主隊需要降低轉換進攻失誤。")

    if defense_gap > 5:
        lines.append(f"{home_label}防守穩定度較佳，具備壓低比分的條件。")
    elif defense_gap < -5:
        lines.append(f"{away_label}防守評分較高，可能讓比賽進入較膠著節奏。")

    if summary["draw_probability"] >= 0.25:
        lines.append(f"和局機率仍有 {summary['draw_probability'] * 100:.1f}%，若兩隊前段時間保守，平手機率會上升。")

    return lines[:5]


def build_pre_match_analysis(
    row: pd.Series | dict[str, Any],
    prediction: Any,
    team_meta: pd.DataFrame | None = None,
    matches: pd.DataFrame | None = None,
    history: pd.DataFrame | None = None,
    players: pd.DataFrame | None = None,
    shots: pd.DataFrame | None = None,
    team_label_func=None,
) -> dict[str, Any]:
    home_team = str(row.get("home_team", "Home"))
    away_team = str(row.get("away_team", "Away"))
    home_label = team_label_func(home_team) if team_label_func else resolve_team_name(home_team)
    away_label = team_label_func(away_team) if team_label_func else resolve_team_name(away_team)

    home = calculate_team_ability(home_team, team_meta, matches, history, players, shots)
    away = calculate_team_ability(away_team, team_meta, matches, history, players, shots)
    summary = prediction_summary(prediction)
    estimated = home.estimated or away.estimated
    notes = list(dict.fromkeys([*home.notes, *away.notes]))

    return {
        "home_label": home_label,
        "away_label": away_label,
        "home": home,
        "away": away,
        "comparison": ability_comparison_table(home, away, home_label, away_label),
        "summary": summary,
        "lines": analysis_lines(home, away, summary, home_label, away_label),
        "is_estimated": estimated,
        "notes": notes,
    }
