from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


CONFIDENCE_WEIGHTS = {
    "fused_probability": 0.40,
    "elo_advantage": 0.20,
    "market_support": 0.20,
    "recent_form": 0.10,
    "xg_advantage": 0.10,
}


@dataclass(frozen=True)
class ConfidenceResult:
    score: float
    stars: str
    level: str
    color: str
    favored_market: str
    source_lines: tuple[str, ...]
    risk_lines: tuple[str, ...]
    components: dict[str, float]


def _safe_float(value, default: float = 0.0) -> float:
    try:
        parsed = pd.to_numeric(pd.Series([value]), errors="coerce").fillna(default).iloc[0]
        return float(parsed)
    except Exception:
        return default


def _clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, float(value)))


def confidence_tier(score: float) -> tuple[str, str, str]:
    if score >= 80:
        return "★★★★★", "極高信心", "#22c55e"
    if score >= 65:
        return "★★★★", "高信心", "#22c55e"
    if score >= 50:
        return "★★★", "中等信心", "#facc15"
    if score >= 35:
        return "★★", "偏低信心", "#f97316"
    return "★", "低信心", "#ef4444"


def _market_key(label: str) -> str:
    if "主" in label or label.lower() == "home":
        return "home"
    if "和" in label or "平" in label or label.lower() == "draw":
        return "draw"
    return "away"


def _best_market(market_table: pd.DataFrame, prediction) -> tuple[str, str, float, float]:
    if market_table is not None and not market_table.empty and "fused_probability" in market_table.columns:
        table = market_table.copy()
        table["fused_probability"] = pd.to_numeric(table["fused_probability"], errors="coerce").fillna(0)
        table["market_probability"] = pd.to_numeric(table.get("market_probability", 0), errors="coerce").fillna(0)
        row = table.sort_values("fused_probability", ascending=False).iloc[0]
        label = str(row.get("market", "主勝"))
        return label, _market_key(label), _safe_float(row.get("fused_probability")), _safe_float(row.get("market_probability"))

    probabilities = {
        "主勝": _safe_float(getattr(prediction, "home_win_probability", 0)),
        "和局": _safe_float(getattr(prediction, "draw_probability", 0)),
        "客勝": _safe_float(getattr(prediction, "away_win_probability", 0)),
    }
    label, probability = max(probabilities.items(), key=lambda item: item[1])
    return label, _market_key(label), probability, probability


def _team_elo(team: str, team_meta: pd.DataFrame, matches: pd.DataFrame, side: str) -> float:
    if team_meta is not None and not team_meta.empty:
        for column in ["team", "team_en"]:
            if column in team_meta.columns:
                row = team_meta[team_meta[column].astype(str).eq(str(team))]
                if not row.empty:
                    return _safe_float(row.iloc[0].get("elo", 1700), 1700)
    if matches is not None and not matches.empty:
        if side == "home" and {"home_team", "home_elo"}.issubset(matches.columns):
            values = matches.loc[matches["home_team"].astype(str).eq(str(team)), "home_elo"].tail(5)
            if not values.empty:
                return _safe_float(values.mean(), 1700)
        if side == "away" and {"away_team", "away_elo"}.issubset(matches.columns):
            values = matches.loc[matches["away_team"].astype(str).eq(str(team)), "away_elo"].tail(5)
            if not values.empty:
                return _safe_float(values.mean(), 1700)
    return 1700.0


def _recent_record(matches: pd.DataFrame, team: str, limit: int = 5) -> tuple[float, str]:
    if matches is None or matches.empty or not {"home_team", "away_team", "home_goals", "away_goals"}.issubset(matches.columns):
        return 0.5, "最近五場資料不足"
    team_matches = matches[(matches["home_team"].astype(str).eq(str(team))) | (matches["away_team"].astype(str).eq(str(team)))].tail(limit)
    if team_matches.empty:
        return 0.5, "最近五場資料不足"
    wins = draws = losses = points = 0
    for _, row in team_matches.iterrows():
        is_home = str(row["home_team"]) == str(team)
        gf = int(_safe_float(row["home_goals"] if is_home else row["away_goals"], 0))
        ga = int(_safe_float(row["away_goals"] if is_home else row["home_goals"], 0))
        if gf > ga:
            wins += 1
            points += 3
        elif gf == ga:
            draws += 1
            points += 1
        else:
            losses += 1
    score = points / max(1, len(team_matches) * 3)
    return score, f"最近五場 {wins} 勝 {draws} 和 {losses} 敗"


def _xg_for_team(team: str, fixture: pd.Series, prediction, shots: pd.DataFrame | None, side: str) -> float:
    if shots is not None and not shots.empty and {"team", "xg_value"}.issubset(shots.columns):
        data = shots.copy()
        if "match_id" in data.columns and "match_id" in fixture.index:
            match_data = data[data["match_id"].astype(str).eq(str(fixture.get("match_id")))]
            if not match_data.empty:
                data = match_data
        data["xg_value"] = pd.to_numeric(data["xg_value"], errors="coerce").fillna(0)
        value = data.loc[data["team"].astype(str).eq(str(team)), "xg_value"].sum()
        if value > 0:
            return float(value)
    if side == "home":
        return _safe_float(getattr(prediction, "expected_home_goals", 1.2), 1.2)
    return _safe_float(getattr(prediction, "expected_away_goals", 1.1), 1.1)


def calculate_confidence(
    fixture: pd.Series,
    prediction,
    market_table: pd.DataFrame | None,
    team_meta: pd.DataFrame | None = None,
    matches: pd.DataFrame | None = None,
    shots: pd.DataFrame | None = None,
) -> ConfidenceResult:
    home = str(fixture.get("home_team", getattr(prediction, "home_team", "")))
    away = str(fixture.get("away_team", getattr(prediction, "away_team", "")))
    label, key, fused_probability, market_probability = _best_market(market_table, prediction)

    home_elo = _team_elo(home, team_meta, matches, "home")
    away_elo = _team_elo(away, team_meta, matches, "away")
    elo_gap = home_elo - away_elo
    if key == "home":
        elo_component = _clamp(max(0, elo_gap) / 300 * 100)
        favored_team = home
        other_team = away
        xg_gap_sign = 1
    elif key == "away":
        elo_component = _clamp(max(0, -elo_gap) / 300 * 100)
        favored_team = away
        other_team = home
        xg_gap_sign = -1
    else:
        elo_component = _clamp((1 - min(abs(elo_gap), 220) / 220) * 100)
        favored_team = home
        other_team = away
        xg_gap_sign = 0

    favored_form, favored_record = _recent_record(matches, favored_team)
    other_form, _ = _recent_record(matches, other_team)
    if key == "draw":
        recent_component = _clamp((1 - abs(favored_form - other_form)) * 100)
    else:
        recent_component = _clamp((favored_form * 0.75 + max(0, favored_form - other_form) * 0.25) * 100)

    home_xg = _xg_for_team(home, fixture, prediction, shots, "home")
    away_xg = _xg_for_team(away, fixture, prediction, shots, "away")
    xg_gap = home_xg - away_xg
    if key == "draw":
        xg_component = _clamp((1 - min(abs(xg_gap), 1.5) / 1.5) * 100)
    else:
        xg_component = _clamp(max(0, xg_gap * xg_gap_sign) / 1.5 * 100)

    components = {
        "最佳融合機率": _clamp(fused_probability * 100),
        "Elo 優勢": elo_component,
        "市場支持度": _clamp(market_probability * 100),
        "近期狀態": recent_component,
        "xG 優勢": xg_component,
    }
    score = sum(components[name] * weight for name, weight in zip(components, CONFIDENCE_WEIGHTS.values()))
    score = round(_clamp(score), 1)
    stars, level, color = confidence_tier(score)

    draw_probability = _safe_float(getattr(prediction, "draw_probability", 0), 0)
    source_lines = (
        f"Elo 優勢 {abs(elo_gap):.0f} 分",
        f"市場支持率 {market_probability * 100:.1f}%",
        f"xG 領先 {abs(xg_gap):.2f}",
        favored_record,
    )
    risks = []
    if draw_probability >= 0.22:
        risks.append(f"和局機率仍有 {draw_probability * 100:.1f}%")
    underdog_xg = away_xg if key == "home" else home_xg
    if key != "draw" and underdog_xg >= 1.15:
        risks.append("對手反擊與進攻效率仍需留意")
    if fused_probability < 0.5:
        risks.append("最佳融合機率未過半，屬於不確定性較高場次")
    if not risks:
        risks.append("主要風險來自臨場陣容與比賽節奏變化")

    return ConfidenceResult(
        score=score,
        stars=stars,
        level=level,
        color=color,
        favored_market=label,
        source_lines=source_lines,
        risk_lines=tuple(risks),
        components=components,
    )
