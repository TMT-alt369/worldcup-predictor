from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


CONFIDENCE_WEIGHTS = {
    "best_probability": 0.60,
    "probability_margin": 0.25,
    "model_support": 0.15,
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
        return float(default)


def _clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, float(value)))


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


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
    label = str(label)
    if "主" in label or label.lower() == "home":
        return "home"
    if "和" in label or "平" in label or label.lower() == "draw":
        return "draw"
    return "away"


def _prediction_probabilities(prediction) -> dict[str, float]:
    return {
        "home": _clamp01(_safe_float(getattr(prediction, "home_win_probability", 0), 0)),
        "draw": _clamp01(_safe_float(getattr(prediction, "draw_probability", 0), 0)),
        "away": _clamp01(_safe_float(getattr(prediction, "away_win_probability", 0), 0)),
    }


def _market_rows(market_table: pd.DataFrame | None, prediction) -> list[dict[str, object]]:
    labels = {"home": "主勝", "draw": "和局", "away": "客勝"}
    prediction_probs = _prediction_probabilities(prediction)
    if market_table is not None and not market_table.empty and "fused_probability" in market_table.columns:
        table = market_table.copy()
        table["fused_probability"] = pd.to_numeric(table["fused_probability"], errors="coerce").fillna(0)
        table["model_probability"] = pd.to_numeric(
            table.get("model_probability", table["fused_probability"]),
            errors="coerce",
        ).fillna(table["fused_probability"])
        table["market_probability"] = pd.to_numeric(table.get("market_probability", 0), errors="coerce").fillna(0)
        rows = []
        for _, row in table.iterrows():
            label = str(row.get("market", "主勝"))
            key = _market_key(label)
            rows.append(
                {
                    "label": label,
                    "key": key,
                    "fused_probability": _clamp01(_safe_float(row.get("fused_probability"))),
                    "model_probability": _clamp01(_safe_float(row.get("model_probability"), prediction_probs.get(key, 0))),
                    "market_probability": _clamp01(_safe_float(row.get("market_probability"))),
                }
            )
        if rows:
            return rows

    return [
        {
            "label": label,
            "key": key,
            "fused_probability": probability,
            "model_probability": probability,
            "market_probability": probability,
        }
        for key, label in labels.items()
        for probability in [prediction_probs.get(key, 0)]
    ]


def _ranked_markets(market_table: pd.DataFrame | None, prediction) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    rows = sorted(_market_rows(market_table, prediction), key=lambda item: float(item["model_probability"]), reverse=True)
    best = rows[0]
    second = rows[1] if len(rows) > 1 else rows[0]
    market_best = max(rows, key=lambda item: float(item.get("market_probability", 0)))
    return best, second, market_best


def _team_elo(team: str, team_meta: pd.DataFrame | None, matches: pd.DataFrame | None, side: str) -> float:
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


def _recent_record(matches: pd.DataFrame | None, team: str, limit: int = 5) -> tuple[float, str]:
    required = {"home_team", "away_team", "home_goals", "away_goals"}
    if matches is None or matches.empty or not required.issubset(matches.columns):
        return 0.5, "最近五場資料不足"
    team_matches = matches[
        (matches["home_team"].astype(str).eq(str(team)))
        | (matches["away_team"].astype(str).eq(str(team)))
    ].tail(limit)
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


def _support_for_direction(key: str, elo_gap: float, xg_gap: float, form_gap: float, market_alignment: bool) -> tuple[float, dict[str, float]]:
    if key == "home":
        elo_support = _clamp01(max(0, elo_gap) / 250)
        xg_support = _clamp01(max(0, xg_gap) / 1.4)
        form_support = _clamp01(0.5 + form_gap)
    elif key == "away":
        elo_support = _clamp01(max(0, -elo_gap) / 250)
        xg_support = _clamp01(max(0, -xg_gap) / 1.4)
        form_support = _clamp01(0.5 - form_gap)
    else:
        elo_support = _clamp01(1 - min(abs(elo_gap), 180) / 180)
        xg_support = _clamp01(1 - min(abs(xg_gap), 1.4) / 1.4)
        form_support = _clamp01(1 - min(abs(form_gap), 0.6) / 0.6)
    player_support = 0.50
    market_support = 1.0 if market_alignment else 0.45
    support = _clamp01(
        elo_support * 0.30
        + xg_support * 0.25
        + form_support * 0.20
        + player_support * 0.10
        + market_support * 0.15
    )
    return support, {
        "elo_support": round(elo_support, 4),
        "xg_support": round(xg_support, 4),
        "recent_form_support": round(form_support, 4),
        "player_support": round(player_support, 4),
        "market_alignment": 1.0 if market_alignment else 0.0,
    }


def _apply_probability_floor(score: float, best_probability: float, margin: float) -> float:
    if best_probability >= 0.70:
        score = max(score, 70)
    elif best_probability >= 0.60:
        score = max(score, 62)
    elif best_probability >= 0.50 and margin >= 0.15:
        score = max(score, 55)
    elif best_probability >= 0.50:
        score = max(score, 50)
    elif best_probability >= 0.40:
        score = max(score, 40)
    if margin < 0.08 and best_probability < 0.50:
        score = min(score, 45)
    return score


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
    best, second, market_best = _ranked_markets(market_table, prediction)
    label = str(best["label"])
    key = str(best["key"])
    best_probability = _clamp01(_safe_float(best["model_probability"]))
    second_probability = _clamp01(_safe_float(second["model_probability"]))
    margin = max(0.0, best_probability - second_probability)
    market_probability = _clamp01(_safe_float(best.get("market_probability", 0)))
    market_alignment = str(market_best.get("key", key)) == key

    home_elo = _team_elo(home, team_meta, matches, "home")
    away_elo = _team_elo(away, team_meta, matches, "away")
    elo_gap = home_elo - away_elo

    home_form, home_record = _recent_record(matches, home)
    away_form, away_record = _recent_record(matches, away)
    form_gap = home_form - away_form
    favored_record = away_record if key == "away" else home_record

    home_xg = _xg_for_team(home, fixture, prediction, shots, "home")
    away_xg = _xg_for_team(away, fixture, prediction, shots, "away")
    xg_gap = home_xg - away_xg

    support, support_components = _support_for_direction(key, elo_gap, xg_gap, form_gap, market_alignment)
    score = best_probability * 60 + margin * 25 + support * 15
    score = round(_clamp(_apply_probability_floor(score, best_probability, margin)), 1)
    stars, level, color = confidence_tier(score)

    draw_probability = _safe_float(getattr(prediction, "draw_probability", 0), 0)
    source_lines = (
        f"{label}為最高機率 {best_probability * 100:.1f}%",
        f"領先第二高機率 {margin * 100:.1f}%",
        f"Elo 優勢 {abs(elo_gap):.0f} 分",
        f"xG 領先 {abs(xg_gap):.2f}",
        favored_record,
    )

    risks: list[str] = []
    if draw_probability >= 0.22:
        risks.append(f"和局機率仍有 {draw_probability * 100:.1f}%")
    if margin < 0.12:
        risks.append("第一名與第二名機率接近，結果不確定性較高")
    if not market_alignment:
        risks.append("市場支持與模型方向不一致")
    if market_probability > 0 and market_probability < 0.25:
        risks.append("市場支持率偏低，代表市場與模型看法不同")
    if abs(xg_gap) < 0.25:
        risks.append("xG 差距接近，實際比賽可能偏膠著")
    underdog_xg = away_xg if key == "home" else home_xg
    if key != "draw" and underdog_xg >= 1.15:
        risks.append("對手反擊或進攻效率仍有威脅")
    if not risks:
        risks.append("主要風險集中在臨場陣容與比賽節奏變化")

    components = {
        "best_probability": round(best_probability, 4),
        "probability_margin": round(margin, 4),
        "model_support_factor": round(support, 4),
        **support_components,
    }
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
