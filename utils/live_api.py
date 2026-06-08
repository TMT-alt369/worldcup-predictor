from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd
import requests


API_FOOTBALL_FIXTURES_URL = "https://v3.football.api-sports.io/fixtures"
API_FOOTBALL_EVENTS_URL = "https://v3.football.api-sports.io/fixtures/events"
API_FOOTBALL_STATISTICS_URL = "https://v3.football.api-sports.io/fixtures/statistics"
ESPN_SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/soccer/all/scoreboard"

API_FOOTBALL_SOURCE = "API-Football 即時資料"
ESPN_SOURCE = "ESPN 即時資料"
MOCK_SOURCE = "展示資料"


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
        "STATUS_SCHEDULED": "未開賽",
        "STATUS_IN_PROGRESS": "Live",
        "STATUS_HALFTIME": "半場",
        "STATUS_FINAL": "結束",
    }
    return mapping.get(str(status).upper(), str(status or "未定"))


def _payload_summary(payload: dict[str, Any]) -> str:
    response = payload.get("response", [])
    events = payload.get("events", [])
    errors = payload.get("errors", {})
    results = payload.get("results")
    response_count = len(response) if isinstance(response, list) else int(bool(response))
    event_count = len(events) if isinstance(events, list) else int(bool(events))
    return f"results={results}, response_count={response_count}, event_count={event_count}, errors={errors or 'none'}"


def _request_api_football(
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


def _request_json(
    url: str,
    params: dict[str, Any] | None = None,
    diagnostics: dict[str, Any] | None = None,
    label: str = "request",
) -> dict[str, Any]:
    response = requests.get(url, params=params or {}, timeout=12)
    if diagnostics is not None:
        diagnostics[f"{label}_http_status"] = response.status_code
    response.raise_for_status()
    payload = response.json()
    if diagnostics is not None:
        diagnostics[f"{label}_summary"] = _payload_summary(payload)
    return payload


def _stat_value(statistics: list[dict[str, Any]], stat_name: str, default: int | None = None) -> int | None:
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


def _fixture_statistics(api_key: str, fixture_id: str) -> dict[str, int | None]:
    payload = _request_api_football(API_FOOTBALL_STATISTICS_URL, api_key, {"fixture": fixture_id})
    teams = payload.get("response", [])
    if len(teams) < 2:
        return {}

    home_stats = teams[0].get("statistics", []) or []
    away_stats = teams[1].get("statistics", []) or []
    return {
        "home_shots": _stat_value(home_stats, "Total Shots"),
        "away_shots": _stat_value(away_stats, "Total Shots"),
        "home_possession": _stat_value(home_stats, "Ball Possession"),
        "away_possession": _stat_value(away_stats, "Ball Possession"),
        "home_corners": _stat_value(home_stats, "Corner Kicks"),
        "away_corners": _stat_value(away_stats, "Corner Kicks"),
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


def _fixture_events(api_key: str, fixture_id: str) -> pd.DataFrame:
    payload = _request_api_football(API_FOOTBALL_EVENTS_URL, api_key, {"fixture": fixture_id})
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
                "data_source": API_FOOTBALL_SOURCE,
            }
        )
    return pd.DataFrame(rows)


def _api_football_live_matches(
    api_key: str,
    diagnostics: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    params = {"live": "all"}
    if diagnostics is not None:
        diagnostics["api_football_params"] = params
    payload = _request_api_football(
        API_FOOTBALL_FIXTURES_URL,
        api_key,
        params,
        diagnostics=diagnostics,
        label="api_football_fixtures",
    )

    rows = []
    event_frames = []
    for item in payload.get("response", []):
        fixture = item.get("fixture", {}) or {}
        fixture_id = str(fixture.get("id") or "")
        if not fixture_id:
            continue

        league = item.get("league", {}) or {}
        teams = item.get("teams", {}) or {}
        goals = item.get("goals", {}) or {}
        venue = fixture.get("venue", {}) or {}
        status = fixture.get("status", {}) or {}
        stats = {
            "home_shots": None,
            "away_shots": None,
            "home_possession": None,
            "away_possession": None,
            "home_corners": None,
            "away_corners": None,
        }

        try:
            stats.update(_fixture_statistics(api_key, fixture_id))
        except requests.RequestException:
            pass

        try:
            events = _fixture_events(api_key, fixture_id)
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
                "league_name": league.get("name") or "未提供",
                "home_team": teams.get("home", {}).get("name", "TBD"),
                "away_team": teams.get("away", {}).get("name", "TBD"),
                "home_score": goals.get("home") if goals.get("home") is not None else 0,
                "away_score": goals.get("away") if goals.get("away") is not None else 0,
                "venue": venue.get("name") or "未提供",
                **stats,
                "updated_at": pd.Timestamp.now(tz="Asia/Taipei").strftime("%Y-%m-%d %H:%M"),
                "data_source": API_FOOTBALL_SOURCE,
            }
        )

    events_df = pd.concat(event_frames, ignore_index=True) if event_frames else pd.DataFrame()
    return pd.DataFrame(rows), events_df


def _espn_scoreboard_matches(
    diagnostics: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    payload = _request_json(ESPN_SCOREBOARD_URL, diagnostics=diagnostics, label="espn_scoreboard")
    rows = []
    event_rows = []
    for item in payload.get("events", []) or []:
        competitions = item.get("competitions", []) or []
        competition = competitions[0] if competitions else {}
        competitors = competition.get("competitors", []) or []
        if len(competitors) < 2:
            continue

        home = next((team for team in competitors if team.get("homeAway") == "home"), competitors[0])
        away = next((team for team in competitors if team.get("homeAway") == "away"), competitors[1])
        home_team = home.get("team", {}) or {}
        away_team = away.get("team", {}) or {}
        status = competition.get("status", {}).get("type", {}) or item.get("status", {}).get("type", {}) or {}
        league = item.get("league", {}) or {}
        live_match_id = str(item.get("id") or competition.get("id") or "")

        rows.append(
            {
                "live_match_id": live_match_id,
                "scheduled_time": item.get("date") or competition.get("date"),
                "status": _normal_status(status.get("name") or status.get("shortDetail") or status.get("description")),
                "minute": 0,
                "league_name": league.get("name") or league.get("abbreviation") or "足球賽事",
                "home_team": home_team.get("displayName") or home_team.get("name") or "TBD",
                "away_team": away_team.get("displayName") or away_team.get("name") or "TBD",
                "home_score": int(home.get("score") or 0),
                "away_score": int(away.get("score") or 0),
                "venue": (competition.get("venue", {}) or {}).get("fullName") or "未提供",
                "home_shots": None,
                "away_shots": None,
                "home_possession": None,
                "away_possession": None,
                "home_corners": None,
                "away_corners": None,
                "updated_at": pd.Timestamp.now(tz="Asia/Taipei").strftime("%Y-%m-%d %H:%M"),
                "data_source": ESPN_SOURCE,
            }
        )

        for detail in competition.get("details", []) or []:
            event_rows.append(
                {
                    "live_match_id": live_match_id,
                    "minute": detail.get("clock", {}).get("displayValue") or 0,
                    "event_type": _event_type(detail.get("type", {}).get("text", ""), detail.get("text", "")),
                    "team": (detail.get("team", {}) or {}).get("displayName") or "未提供",
                    "player": (detail.get("athletes", [{}])[0] or {}).get("displayName") if detail.get("athletes") else "未提供",
                    "detail": detail.get("text") or "",
                    "data_source": ESPN_SOURCE,
                }
            )

    return pd.DataFrame(rows), pd.DataFrame(event_rows)


def _fallback_live_data(
    fallback_matches: pd.DataFrame,
    fallback_events: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    matches = fallback_matches.copy()
    events = fallback_events.copy()
    matches["league_name"] = matches.get("league_name", "世界盃展示資料")
    matches["data_source"] = MOCK_SOURCE
    if not events.empty:
        events["data_source"] = MOCK_SOURCE
    return matches, events, MOCK_SOURCE


def load_live_matches_from_secrets(
    secrets: Any,
    fallback_matches: pd.DataFrame,
    fallback_events: pd.DataFrame,
    target_date: date | None = None,
    mode: str = "real",
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    matches, events, source, _ = load_live_matches_with_debug(
        secrets,
        fallback_matches,
        fallback_events,
        target_date,
        mode=mode,
    )
    return matches, events, source


def load_live_matches_with_debug(
    secrets: Any,
    fallback_matches: pd.DataFrame,
    fallback_events: pd.DataFrame,
    target_date: date | None = None,
    mode: str = "real",
) -> tuple[pd.DataFrame, pd.DataFrame, str, dict[str, Any]]:
    diagnostics: dict[str, Any] = {
        "secret_name": "FOOTBALL_API_KEY",
        "secret_exists": False,
        "api_test_success": False,
        "http_status": None,
        "api_summary": "尚未呼叫 API",
        "fallback_reason": "",
        "error": "",
        "mode": mode,
    }

    if mode != "real":
        diagnostics["fallback_reason"] = "使用者選擇世界盃展示資料"
        matches, events, source = _fallback_live_data(fallback_matches, fallback_events)
        return matches, events, source, diagnostics

    api_key = _secret_value(secrets, "FOOTBALL_API_KEY")
    diagnostics["secret_exists"] = bool(api_key)
    if api_key:
        try:
            matches, events = _api_football_live_matches(api_key, diagnostics)
            diagnostics["http_status"] = diagnostics.get("api_football_fixtures_http_status")
            diagnostics["api_summary"] = diagnostics.get("api_football_fixtures_summary", "API-Football 已回應")
            diagnostics["api_test_success"] = diagnostics.get("api_football_fixtures_http_status") == 200
            if not matches.empty:
                return matches, events, API_FOOTBALL_SOURCE, diagnostics
            diagnostics["fallback_reason"] = "API-Football 無可用即時賽事，改查 ESPN"
        except requests.RequestException as exc:
            diagnostics["error"] = f"{exc.__class__.__name__}: {exc}"
            diagnostics["fallback_reason"] = "API-Football 請求失敗，改查 ESPN"
    else:
        diagnostics["fallback_reason"] = "未設定 FOOTBALL_API_KEY，改查 ESPN"

    try:
        matches, events = _espn_scoreboard_matches(diagnostics)
        diagnostics["http_status"] = diagnostics.get("espn_scoreboard_http_status")
        diagnostics["api_summary"] = diagnostics.get("espn_scoreboard_summary", "ESPN 已回應")
        diagnostics["api_test_success"] = diagnostics.get("espn_scoreboard_http_status") == 200
        if not matches.empty:
            return matches, events, ESPN_SOURCE, diagnostics
        diagnostics["fallback_reason"] = "ESPN 無可用賽事，改用展示資料"
    except requests.RequestException as exc:
        diagnostics["error"] = f"{exc.__class__.__name__}: {exc}"
        diagnostics["fallback_reason"] = "ESPN 請求失敗，改用展示資料"

    matches, events, source = _fallback_live_data(fallback_matches, fallback_events)
    return matches, events, source, diagnostics
