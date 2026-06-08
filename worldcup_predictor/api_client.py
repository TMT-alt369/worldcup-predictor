from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd
import requests


FOOTBALL_DATA_MATCHES_URL = "https://api.football-data.org/v4/competitions/WC/matches"
API_FOOTBALL_FIXTURES_URL = "https://v3.football.api-sports.io/fixtures"


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
        "HT": "半場",
        "FT": "結束",
        "NS": "未開賽",
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


def football_data_matches(api_key: str, target_date: date | None = None) -> pd.DataFrame:
    params = {}
    if target_date is not None:
        params["dateFrom"] = target_date.isoformat()
        params["dateTo"] = target_date.isoformat()
    response = requests.get(
        FOOTBALL_DATA_MATCHES_URL,
        headers={"X-Auth-Token": api_key},
        params=params,
        timeout=12,
    )
    response.raise_for_status()
    payload = response.json()
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


def api_football_matches(api_key: str, target_date: date | None = None) -> pd.DataFrame:
    params = {}
    if target_date is not None:
        params["date"] = target_date.isoformat()
    response = requests.get(
        API_FOOTBALL_FIXTURES_URL,
        headers={"x-apisports-key": api_key},
        params=params,
        timeout=12,
    )
    response.raise_for_status()
    payload = response.json()
    rows = []
    for item in payload.get("response", []):
        fixture = item.get("fixture", {})
        teams = item.get("teams", {})
        goals = item.get("goals", {})
        venue = fixture.get("venue", {}) or {}
        status = fixture.get("status", {}) or {}
        rows.append(
            {
                "live_match_id": str(fixture.get("id")),
                "scheduled_time": fixture.get("date"),
                "status": normalize_status(status.get("short", "")),
                "minute": status.get("elapsed") or 0,
                "home_team": teams.get("home", {}).get("name", "TBD"),
                "away_team": teams.get("away", {}).get("name", "TBD"),
                "home_score": goals.get("home") if goals.get("home") is not None else 0,
                "away_score": goals.get("away") if goals.get("away") is not None else 0,
                "venue": venue.get("name") or "待公布",
                "home_shots": 0,
                "away_shots": 0,
                "home_possession": 50,
                "away_possession": 50,
                "home_corners": 0,
                "away_corners": 0,
                "updated_at": pd.Timestamp.now(tz="Asia/Taipei").strftime("%Y-%m-%d %H:%M"),
                "data_source": "API-Football",
            }
        )
    return pd.DataFrame(rows)


def load_live_data(
    secrets: Any,
    fallback_matches: pd.DataFrame,
    fallback_events: pd.DataFrame,
    target_date: date | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    football_data_key = secret_value(secrets, "FOOTBALL_DATA_API_KEY")
    api_football_key = secret_value(secrets, "API_FOOTBALL_KEY")

    if football_data_key:
        try:
            matches = football_data_matches(football_data_key, target_date)
            if not matches.empty:
                return matches, pd.DataFrame(columns=fallback_events.columns), "football-data.org"
        except requests.RequestException:
            pass

    if api_football_key:
        try:
            matches = api_football_matches(api_football_key, target_date)
            if not matches.empty:
                return matches, pd.DataFrame(columns=fallback_events.columns), "API-Football"
        except requests.RequestException:
            pass

    return fallback_live_data(fallback_matches, fallback_events)
