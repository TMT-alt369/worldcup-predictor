from __future__ import annotations

from itertools import combinations
from math import prod
from typing import Callable

import pandas as pd


def _risk_from_probability(probability: float) -> str:
    if probability >= 0.58:
        return "低風險"
    if probability >= 0.43:
        return "中風險"
    return "高風險"


def _odds_for_direction(row: pd.Series, direction: str) -> float:
    if direction == "主勝":
        return float(row.get("home_odds", 1.8) or 1.8)
    if direction == "和局":
        return float(row.get("draw_odds", 3.2) or 3.2)
    return float(row.get("away_odds", 2.4) or 2.4)


def _match_label(row: pd.Series, team_name_func: Callable[[str], str] | None = None) -> str:
    home = str(row.get("home_team", "主隊"))
    away = str(row.get("away_team", "客隊"))
    if team_name_func:
        home = team_name_func(home)
        away = team_name_func(away)
    return f"{home} vs {away}"


def build_match_candidates(
    fixtures: pd.DataFrame,
    market_table_func: Callable[[pd.Series, object], pd.DataFrame],
    predict_match_func: Callable[[str, str], object],
    team_name_func: Callable[[str], str] | None = None,
    limit: int = 18,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if fixtures is None or fixtures.empty:
        return pd.DataFrame(columns=["比賽", "方向", "單場機率", "單場賠率", "風險等級"])

    for _, fixture in fixtures.head(limit).iterrows():
        try:
            prediction = predict_match_func(str(fixture["home_team"]), str(fixture["away_team"]))
            table = market_table_func(fixture, prediction)
        except Exception:
            continue
        if table is None or table.empty:
            continue

        table = table.copy()
        table["fused_probability"] = pd.to_numeric(table["fused_probability"], errors="coerce").fillna(0)
        best_rows = table.sort_values("fused_probability", ascending=False).head(2)
        for _, item in best_rows.iterrows():
            direction = str(item.get("market", "主勝"))
            probability = float(item.get("fused_probability", 0) or 0)
            odds = _odds_for_direction(fixture, direction)
            rows.append(
                {
                    "比賽": _match_label(fixture, team_name_func),
                    "方向": direction,
                    "單場機率": probability,
                    "單場賠率": odds,
                    "風險等級": _risk_from_probability(probability),
                }
            )

    if not rows:
        return pd.DataFrame(columns=["比賽", "方向", "單場機率", "單場賠率", "風險等級"])
    return pd.DataFrame(rows)


def build_parlay_combinations(candidates: pd.DataFrame, size: int = 2, max_rows: int = 8) -> pd.DataFrame:
    columns = ["組合", "比賽", "方向", "總賠率", "預估命中率", "風險等級"]
    if candidates is None or candidates.empty or len(candidates) < size:
        return pd.DataFrame(columns=columns)

    clean = candidates.copy()
    clean["單場機率"] = pd.to_numeric(clean["單場機率"], errors="coerce").fillna(0)
    clean["單場賠率"] = pd.to_numeric(clean["單場賠率"], errors="coerce").fillna(1.01)
    clean = clean.sort_values(["單場機率", "單場賠率"], ascending=[False, True]).head(12)

    rows: list[dict[str, object]] = []
    seen_match_sets: set[tuple[str, ...]] = set()
    for combo in combinations(clean.to_dict("records"), size):
        matches = tuple(item["比賽"] for item in combo)
        if len(set(matches)) < size:
            continue
        match_key = tuple(sorted(matches))
        if match_key in seen_match_sets:
            continue
        seen_match_sets.add(match_key)

        hit_rate = prod(float(item["單場機率"]) for item in combo)
        total_odds = prod(float(item["單場賠率"]) for item in combo)
        rows.append(
            {
                "組合": f"{size} 串 1",
                "比賽": " / ".join(matches),
                "方向": " + ".join(str(item["方向"]) for item in combo),
                "總賠率": round(total_odds, 2),
                "預估命中率": hit_rate,
                "風險等級": _risk_from_probability(hit_rate),
            }
        )

    if not rows:
        return pd.DataFrame(columns=columns)
    result = pd.DataFrame(rows).sort_values("預估命中率", ascending=False).head(max_rows)
    return result.reset_index(drop=True)


def parlay_summary_text(table: pd.DataFrame) -> str:
    if table is None or table.empty:
        return "目前資料不足，暫無可整理的串關候選組合。"
    top = table.iloc[0]
    return (
        f"目前機率最高的候選組合為 {top['組合']}，預估命中率約 "
        f"{float(top['預估命中率']) * 100:.1f}%，總賠率約 {float(top['總賠率']):.2f}。"
        "此結果僅供機率分析，不構成下注建議。"
    )
