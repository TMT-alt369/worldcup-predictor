from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd
import requests


FOOTBALL_DATA_MATCHES_URL = "https://api.football-data.org/v4/competitions/WC/matches"
API_FOOTBALL_FIXTURES_URL = "https://v3.football.api-sports.io/fixtures"
API_FOOTBALL_EVENTS_URL = "https://v3.football.api-sports.io/fixtures/events"
API_FOOTBALL_STATISTICS_URL = "https://v3.football.api-sports.io/fixtures/statistics"


def secret_value(secrets: Any, key: str) -> str | None:
    try:
        value = secrets.get(key)
    except Exception:
        value = None
    return str(value).strip() if value else None


def normalize_status(status: str) -> str:
    mapping = {
        "SCHEDULED": "未開賽",
        "TIMED": "未開賽",
        "IN_PLAY": "Live",
        "PAUSED": "半場",
        "FINISHED": "結束",
        "LIVE": "Live",
        "1H": "Live",
        "2H": "Live",
        "HT": "半場",
        "FT": "結束",
        "AET": "結束",
        "PEN": "結束",
        "NS": "未開賽",
        "TBD": "未開賽",
        "PST": "延期",
        "CANC": "取消",
    }
    return mapping.get(str(status).upper(), str(status))


def fallback_live_data(
    fallback_matches: pd.DataFrame,
    fallback_events: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    matches = fallback_matches.copy()
    events = fallback_events.copy()
    matches["data_source"] = "本地 fallback 展示資料"
    return matches, events, "本地 fallback 展示資料"


def request_json(url: str, headers: dict[str, str], params: dict[str, Any]) -> dict[str, Any]:
    response = requests.get(url, headers=headers, params=params, timeout=12)
    response.raise_for_status()
    return response.json()


def football_data_matches(api_key: str, target_date: date | None = None) -> pd.DataFrame:
    params = {}
    if target_date is not None:
        params["dateFrom"] = target_date.isoformat()
        params["dateTo"] = target_date.isoformat()
    payload = request_json(
        FOOTBALL_DATA_MATCHES_URL,
        headers={"X-Auth-Token": api_key},
        params=params,
    )
    rows = []
    for item in payload.get("matches", []):
        score = item.get("score", {}).get("fullTime", {})
        rows.append(
            {
                "live_match_id": str(item.get("id")),
                "scheduled_time": item.get("utcDate"),
                "status": normalize_status(item.get("status", "")),
                "minute": 0,
                "home_team": item.get("homeTeam", {}).get("name", "TBD"),
                "away_team": item.get("awayTeam", {}).get("name", "TBD"),
                "home_score": score.get("home") if score.get("home") is not None else 0,
                "away_score": score.get("away") if score.get("away") is not None else 0,
                "venue": item.get("venue") or "待公布",
                "home_shots": 0,
                "away_shots": 0,
                "home_possession": 50,
                "away_possession": 50,
                "home_corners": 0,
                "away_corners": 0,
                "updated_at": pd.Timestamp.now(tz="Asia/Taipei").strftime("%Y-%m-%d %H:%M"),
                "data_source": "football-data.org",
            }
        )
    return pd.DataFrame(rows)


def api_football_headers(api_key: str) -> dict[str, str]:
    return {"x-apisports-key": api_key}


def api_football_fixture_events(api_key: str, fixture_id: str) -> pd.DataFrame:
    payload = request_json(
        API_FOOTBALL_EVENTS_URL,
        headers=api_football_headers(api_key),
        params={"fixture": fixture_id},
    )
    rows = []
    for item in payload.get("response", []):
        time = item.get("time", {}) or {}
        team = item.get("team", {}) or {}
        player = item.get("player", {}) or {}
        assist = item.get("assist", {}) or {}
        event_type = str(item.get("type") or "")
        detail = str(item.get("detail") or "")
        rows.append(
            {
                "live_match_id": str(fixture_id),
                "minute": time.get("elapsed") or 0,
                "event_type": detail or event_type,
                "team": team.get("name", "TBD"),
                "player": player.get("name") or "待公布",
                "detail": assist.get("name") or detail or event_type,
            }
        )
    return pd.DataFrame(rows)


def statistic_value(statistics: list[dict[str, Any]], name: str, default: int) -> int:
    for item in statistics:
        if str(item.get("type", "")).lower() == name.lower():
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


def api_football_fixture_statistics(api_key: str, fixture_id: str) -> dict[str, int]:
    payload = request_json(
        API_FOOTBALL_STATISTICS_URL,
        headers=api_football_headers(api_key),
        params={"fixture": fixture_id},
    )
    response = payload.get("response", [])
    if len(response) < 2:
        return {}
    home_stats = response[0].get("statistics", []) or []
    away_stats = response[1].get("statistics", []) or []
    return {
        "home_shots": statistic_value(home_stats, "Total Shots", 0),
        "away_shots": statistic_value(away_stats, "Total Shots", 0),
        "home_possession": statistic_value(home_stats, "Ball Possession", 50),
        "away_possession": statistic_value(away_stats, "Ball Possession", 50),
        "home_corners": statistic_value(home_stats, "Corner Kicks", 0),
        "away_corners": statistic_value(away_stats, "Corner Kicks", 0),
    }


def api_football_matches(
    api_key: str,
    target_date: date | None = None,
    league_id: str = "1",
    season: str = "2026",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    params = {"league": league_id, "season": season}
    if target_date is not None:
        params["date"] = target_date.isoformat()
    payload = request_json(
        API_FOOTBALL_FIXTURES_URL,
        headers=api_football_headers(api_key),
        params=params,
    )
    rows = []
    event_frames = []
    for item in payload.get("response", []):
        fixture = item.get("fixture", {}) or {}
        fixture_id = str(fixture.get("id"))
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
            stats.update(api_football_fixture_statistics(api_key, fixture_id))
        except requests.RequestException:
            pass

        try:
            events = api_football_fixture_events(api_key, fixture_id)
            if not events.empty:
                event_frames.append(events)
        except requests.RequestException:
            pass

        rows.append(
            {
                "live_match_id": fixture_id,
                "scheduled_time": fixture.get("date"),
                "status": normalize_status(status.get("short", "")),
                "minute": status.get("elapsed") or 0,
                "home_team": teams.get("home", {}).get("name", "TBD"),
                "away_team": teams.get("away", {}).get("name", "TBD"),
                "home_score": goals.get("home") if goals.get("home") is not None else 0,
                "away_score": goals.get("away") if goals.get("away") is not None else 0,
                "venue": venue.get("name") or "待公布",
                **stats,
                "updated_at": pd.Timestamp.now(tz="Asia/Taipei").strftime("%Y-%m-%d %H:%M"),
                "data_source": "API-Football",
            }
        )

    events_df = pd.concat(event_frames, ignore_index=True) if event_frames else pd.DataFrame()
    return pd.DataFrame(rows), events_df


def load_live_data(
    secrets: Any,
    fallback_matches: pd.DataFrame,
    fallback_events: pd.DataFrame,
    target_date: date | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    api_football_key = secret_value(secrets, "FOOTBALL_API_KEY")
    api_football_league = secret_value(secrets, "API_FOOTBALL_LEAGUE_ID") or "1"
    api_football_season = secret_value(secrets, "API_FOOTBALL_SEASON") or "2026"
    football_data_key = secret_value(secrets, "FOOTBALL_DATA_API_KEY")

    if api_football_key:
        try:
            matches, events = api_football_matches(
                api_football_key,
                target_date,
                league_id=api_football_league,
                season=api_football_season,
            )
            if not matches.empty:
                if events.empty:
                    events = pd.DataFrame(columns=fallback_events.columns)
                return matches, events, "API-Football"
        except requests.RequestException:
            pass

    if football_data_key:
        try:
            matches = football_data_matches(football_data_key, target_date)
            if not matches.empty:
                return matches, pd.DataFrame(columns=fallback_events.columns), "football-data.org"
        except requests.RequestException:
            pass

    return fallback_live_data(fallback_matches, fallback_events)
