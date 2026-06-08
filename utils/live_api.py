from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd
import requests


API_FOOTBALL_FIXTURES_URL = "https://v3.football.api-sports.io/fixtures"
API_FOOTBALL_EVENTS_URL = "https://v3.football.api-sports.io/fixtures/events"
API_FOOTBALL_STATISTICS_URL = "https://v3.football.api-sports.io/fixtures/statistics"
DEFAULT_WORLD_CUP_LEAGUE_ID = "1"
DEFAULT_WORLD_CUP_SEASON = "2026"
MOCK_SOURCE = "Mock 展示資料"
API_SOURCE = "API-Football 即時資料"


def _secret_value(secrets: Any, key: str) -> str | None:
    try:
        value = secrets[key]
    except Exception:
        return None
    return str(value).strip() if value else None


def _normal_status(status: str) -> str:
    mapping = {
        "NS": "未開賽",
        "TBD": "未定",
        "1H": "Live",
        "HT": "半場",
        "2H": "Live",
        "ET": "延長賽",
        "BT": "中場休息",
        "P": "點球大戰",
        "FT": "結束",
        "AET": "結束",
        "PEN": "結束",
        "PST": "延期",
        "CANC": "取消",
        "SUSP": "暫停",
        "INT": "中斷",
        "ABD": "中止",
        "AWD": "判定",
        "WO": "棄權",
        "LIVE": "Live",
    }
    return mapping.get(str(status).upper(), str(status or "未定"))


def _payload_summary(payload: dict[str, Any]) -> str:
    response = payload.get("response", [])
    errors = payload.get("errors", {})
    results = payload.get("results")
    if isinstance(response, list):
        response_count = len(response)
    elif response:
        response_count = 1
    else:
        response_count = 0
    return f"results={results}, response_count={response_count}, errors={errors or 'none'}"


def _request_json(
    url: str,
    api_key: str,
    params: dict[str, Any],
    diagnostics: dict[str, Any] | None = None,
    label: str = "request",
) -> dict[str, Any]:
    response = requests.get(
        url,
        headers={"x-apisports-key": api_key},
        params=params,
        timeout=12,
    )
    if diagnostics is not None:
        diagnostics[f"{label}_http_status"] = response.status_code
    response.raise_for_status()
    payload = response.json()
    if diagnostics is not None:
        diagnostics[f"{label}_summary"] = _payload_summary(payload)
    return payload


def _stat_value(statistics: list[dict[str, Any]], stat_name: str, default: int) -> int:
    for item in statistics:
        if str(item.get("type", "")).lower() != stat_name.lower():
            continue
        value = item.get("value")
        if value is None:
            return default
        if isinstance(value, str) and value.endswith("%"):
            value = value[:-1]
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return default
    return default


def _fixture_statistics(
    api_key: str,
    fixture_id: str,
    diagnostics: dict[str, Any] | None = None,
) -> dict[str, int]:
    payload = _request_json(
        API_FOOTBALL_STATISTICS_URL,
        api_key,
        {"fixture": fixture_id},
        diagnostics=diagnostics,
        label="statistics",
    )
    teams = payload.get("response", [])
    if len(teams) < 2:
        return {}

    home_stats = teams[0].get("statistics", []) or []
    away_stats = teams[1].get("statistics", []) or []
    return {
        "home_shots": _stat_value(home_stats, "Total Shots", 0),
        "away_shots": _stat_value(away_stats, "Total Shots", 0),
        "home_possession": _stat_value(home_stats, "Ball Possession", 50),
        "away_possession": _stat_value(away_stats, "Ball Possession", 50),
        "home_corners": _stat_value(home_stats, "Corner Kicks", 0),
        "away_corners": _stat_value(away_stats, "Corner Kicks", 0),
    }


def _event_type(api_event_type: str, detail: str) -> str:
    event_type = str(api_event_type or "").lower()
    detail_text = str(detail or "").lower()
    if "goal" in event_type or "goal" in detail_text:
        return "Goal"
    if "card" in event_type and "red" in detail_text:
        return "Red Card"
    if "card" in event_type and "yellow" in detail_text:
        return "Yellow Card"
    if "subst" in event_type:
        return "Substitution"
    return api_event_type or detail or "Event"


def _fixture_events(
    api_key: str,
    fixture_id: str,
    diagnostics: dict[str, Any] | None = None,
) -> pd.DataFrame:
    payload = _request_json(
        API_FOOTBALL_EVENTS_URL,
        api_key,
        {"fixture": fixture_id},
        diagnostics=diagnostics,
        label="events",
    )
    rows = []
    for item in payload.get("response", []):
        time = item.get("time", {}) or {}
        team = item.get("team", {}) or {}
        player = item.get("player", {}) or {}
        assist = item.get("assist", {}) or {}
        detail = str(item.get("detail") or item.get("comments") or "")
        rows.append(
            {
                "live_match_id": str(fixture_id),
                "minute": time.get("elapsed") or 0,
                "event_type": _event_type(str(item.get("type") or ""), detail),
                "team": team.get("name", "TBD"),
                "player": player.get("name") or "未提供",
                "detail": assist.get("name") or detail or str(item.get("type") or ""),
                "data_source": API_SOURCE,
            }
        )
    return pd.DataFrame(rows)


def _api_football_live_matches(
    api_key: str,
    target_date: date | None = None,
    diagnostics: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    params: dict[str, Any] = {
        "league": DEFAULT_WORLD_CUP_LEAGUE_ID,
        "season": DEFAULT_WORLD_CUP_SEASON,
    }
    if target_date is not None:
        params["date"] = target_date.isoformat()

    if diagnostics is not None:
        diagnostics["api_params"] = {"league": params.get("league"), "season": params.get("season"), "date": params.get("date")}
    payload = _request_json(
        API_FOOTBALL_FIXTURES_URL,
        api_key,
        params,
        diagnostics=diagnostics,
        label="fixtures",
    )
    rows = []
    event_frames = []
    for item in payload.get("response", []):
        fixture = item.get("fixture", {}) or {}
        fixture_id = str(fixture.get("id") or "")
        if not fixture_id:
            continue

        teams = item.get("teams", {}) or {}
        goals = item.get("goals", {}) or {}
        venue = fixture.get("venue", {}) or {}
        status = fixture.get("status", {}) or {}
        stats = {
            "home_shots": 0,
            "away_shots": 0,
            "home_possession": 50,
            "away_possession": 50,
            "home_corners": 0,
            "away_corners": 0,
        }

        try:
            stats.update(_fixture_statistics(api_key, fixture_id, diagnostics))
        except requests.RequestException:
            pass

        try:
            events = _fixture_events(api_key, fixture_id, diagnostics)
            if not events.empty:
                event_frames.append(events)
        except requests.RequestException:
            pass

        rows.append(
            {
                "live_match_id": fixture_id,
                "scheduled_time": fixture.get("date"),
                "status": _normal_status(status.get("short", "")),
                "minute": status.get("elapsed") or 0,
                "home_team": teams.get("home", {}).get("name", "TBD"),
                "away_team": teams.get("away", {}).get("name", "TBD"),
                "home_score": goals.get("home") if goals.get("home") is not None else 0,
                "away_score": goals.get("away") if goals.get("away") is not None else 0,
                "venue": venue.get("name") or "未提供",
                **stats,
                "updated_at": pd.Timestamp.now(tz="Asia/Taipei").strftime("%Y-%m-%d %H:%M"),
                "data_source": API_SOURCE,
            }
        )

    matches = pd.DataFrame(rows)
    events_df = pd.concat(event_frames, ignore_index=True) if event_frames else pd.DataFrame()
    return matches, events_df


def _fallback_live_data(
    fallback_matches: pd.DataFrame,
    fallback_events: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    matches = fallback_matches.copy()
    events = fallback_events.copy()
    matches["data_source"] = MOCK_SOURCE
    if not events.empty:
        events["data_source"] = MOCK_SOURCE
    return matches, events, MOCK_SOURCE


def load_live_matches_from_secrets(
    secrets: Any,
    fallback_matches: pd.DataFrame,
    fallback_events: pd.DataFrame,
    target_date: date | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    matches, events, source, _ = load_live_matches_with_debug(
        secrets,
        fallback_matches,
        fallback_events,
        target_date,
    )
    return matches, events, source


def load_live_matches_with_debug(
    secrets: Any,
    fallback_matches: pd.DataFrame,
    fallback_events: pd.DataFrame,
    target_date: date | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, str, dict[str, Any]]:
    diagnostics: dict[str, Any] = {
        "secret_name": "FOOTBALL_API_KEY",
        "secret_exists": False,
        "api_test_success": False,
        "http_status": None,
        "api_summary": "尚未呼叫 API",
        "fallback_reason": "",
        "error": "",
    }
    api_key = _secret_value(secrets, "FOOTBALL_API_KEY")
    diagnostics["secret_exists"] = bool(api_key)
    if not api_key:
        diagnostics["fallback_reason"] = "Streamlit Secrets 未提供 FOOTBALL_API_KEY"
        matches, events, source = _fallback_live_data(fallback_matches, fallback_events)
        return matches, events, source, diagnostics

    try:
        matches, events = _api_football_live_matches(api_key, target_date, diagnostics)
        diagnostics["http_status"] = diagnostics.get("fixtures_http_status")
        diagnostics["api_summary"] = diagnostics.get("fixtures_summary", "API 已回應，但無摘要")
        diagnostics["api_test_success"] = diagnostics.get("fixtures_http_status") == 200
    except requests.RequestException as exc:
        diagnostics["http_status"] = diagnostics.get("fixtures_http_status")
        diagnostics["api_summary"] = diagnostics.get("fixtures_summary", "API 呼叫失敗，無回傳摘要")
        diagnostics["fallback_reason"] = "API-Football 請求失敗"
        diagnostics["error"] = f"{exc.__class__.__name__}: {exc}"
        matches, events, source = _fallback_live_data(fallback_matches, fallback_events)
        return matches, events, source, diagnostics

    if matches.empty:
        diagnostics["fallback_reason"] = "API-Football 回傳 0 場比賽，改用展示資料"
        matches, events, source = _fallback_live_data(fallback_matches, fallback_events)
        return matches, events, source, diagnostics

    if events.empty:
        events = pd.DataFrame(columns=fallback_events.columns)
    return matches, events, API_SOURCE, diagnostics
