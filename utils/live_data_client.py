from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from utils.realtime_worldcup_center import calculate_group_standings, normalize_match_results


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MOCK_MATCH_RESULTS = DATA_DIR / "match_results.csv"

LIVE_COLUMNS = [
    "match_id",
    "match_time",
    "date",
    "group",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "status",
    "minute",
    "venue",
]

STATUS_MAP = {
    "not_started": "scheduled",
    "scheduled": "scheduled",
    "upcoming": "scheduled",
    "ns": "scheduled",
    "live": "live",
    "in_progress": "live",
    "1h": "live",
    "2h": "live",
    "halftime": "halftime",
    "half_time": "halftime",
    "ht": "halftime",
    "finished": "finished",
    "completed": "finished",
    "final": "finished",
    "ft": "finished",
    "postponed": "postponed",
    "pst": "postponed",
    "cancelled": "cancelled",
    "canceled": "cancelled",
    "canc": "cancelled",
}

STATUS_LABELS = {
    "scheduled": "未開始",
    "live": "進行中",
    "halftime": "中場",
    "finished": "已結束",
    "postponed": "延賽",
    "cancelled": "取消",
}


@dataclass
class LiveDataResult:
    data: pd.DataFrame
    source_mode: str
    provider: str
    fallback_used: bool
    message: str
    updated_at: str


def _now_text() -> str:
    return pd.Timestamp.now(tz="Asia/Taipei").strftime("%Y/%m/%d %H:%M:%S")


def _secret_value(secrets: Any, key: str, default: str = "") -> str:
    try:
        value = secrets[key]
    except Exception:
        return default
    return str(value).strip() if value is not None else default


def provider_settings(secrets: Any | None = None) -> dict[str, str]:
    if secrets is None:
        secrets = {}
    provider = _secret_value(secrets, "FOOTBALL_API_PROVIDER", "mock").lower() or "mock"
    api_key = _secret_value(secrets, "FOOTBALL_API_KEY", "")
    base_url = _secret_value(secrets, "FOOTBALL_API_BASE_URL", "")
    return {
        "provider": provider,
        "api_key": api_key,
        "base_url": base_url,
    }


def normalize_status(value: object) -> str:
    status = str(value or "scheduled").strip().lower()
    return STATUS_MAP.get(status, status if status in STATUS_LABELS else "scheduled")


def normalize_live_matches(df: pd.DataFrame | None) -> pd.DataFrame:
    data = df.copy() if df is not None else pd.DataFrame()
    for column in LIVE_COLUMNS:
        if column not in data.columns:
            data[column] = pd.NA
    data = data[LIVE_COLUMNS].copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["match_time"] = pd.to_datetime(data["match_time"], errors="coerce")
    data["status"] = data["status"].map(normalize_status)
    data["home_score"] = pd.to_numeric(data["home_score"], errors="coerce")
    data["away_score"] = pd.to_numeric(data["away_score"], errors="coerce")
    data["minute"] = pd.to_numeric(data["minute"], errors="coerce").fillna(0).astype(int)
    for column in ["match_id", "group", "home_team", "away_team", "venue"]:
        data[column] = data[column].fillna("").astype(str).str.strip()
    data["status_label"] = data["status"].map(STATUS_LABELS).fillna("未開始")
    data["score_display"] = data.apply(
        lambda row: "待開賽"
        if pd.isna(row["home_score"]) or pd.isna(row["away_score"])
        else f"{int(row['home_score'])} - {int(row['away_score'])}",
        axis=1,
    )
    return data


def _load_mock_matches() -> pd.DataFrame:
    if not MOCK_MATCH_RESULTS.exists():
        return pd.DataFrame(columns=LIVE_COLUMNS)
    raw = pd.read_csv(MOCK_MATCH_RESULTS)
    if "match_time" not in raw.columns:
        raw["match_time"] = raw.get("date", "")
    if "minute" not in raw.columns:
        raw["minute"] = 0
    if "venue" not in raw.columns:
        raw["venue"] = "TBD"
    return normalize_live_matches(raw)


def _api_get_json(url: str, api_key: str) -> Any:
    headers = {"x-apisports-key": api_key} if api_key else {}
    response = requests.get(url, headers=headers, timeout=12)
    response.raise_for_status()
    return response.json()


def _records_from_payload(payload: Any) -> pd.DataFrame:
    if isinstance(payload, list):
        return pd.DataFrame(payload)
    if not isinstance(payload, dict):
        return pd.DataFrame()
    for key in ["matches", "fixtures", "response", "data", "events"]:
        value = payload.get(key)
        if isinstance(value, list):
            if key == "response":
                return _api_football_response_to_rows(value)
            return pd.DataFrame(value)
    return pd.DataFrame()


def _api_football_response_to_rows(items: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for item in items:
        fixture = item.get("fixture", {}) or {}
        teams = item.get("teams", {}) or {}
        goals = item.get("goals", {}) or {}
        league = item.get("league", {}) or {}
        status = fixture.get("status", {}) or {}
        venue = fixture.get("venue", {}) or {}
        rows.append(
            {
                "match_id": str(fixture.get("id", "")),
                "match_time": fixture.get("date"),
                "date": fixture.get("date"),
                "group": league.get("round") or league.get("name") or "Group TBD",
                "home_team": (teams.get("home", {}) or {}).get("name", ""),
                "away_team": (teams.get("away", {}) or {}).get("name", ""),
                "home_score": goals.get("home"),
                "away_score": goals.get("away"),
                "status": status.get("short") or status.get("long") or "scheduled",
                "minute": status.get("elapsed") or 0,
                "venue": venue.get("name") or "TBD",
            }
        )
    return pd.DataFrame(rows)


def _fetch_live_from_api(settings: dict[str, str]) -> pd.DataFrame:
    base_url = settings.get("base_url", "").rstrip("/")
    api_key = settings.get("api_key", "")
    if not api_key:
        raise ValueError("missing API key")
    if not base_url:
        raise ValueError("missing API base URL")
    payload = _api_get_json(f"{base_url}/fixtures?live=all", api_key)
    return normalize_live_matches(_records_from_payload(payload))


def fetch_live_matches(secrets: Any | None = None) -> LiveDataResult:
    settings = provider_settings(secrets)
    provider = settings["provider"]
    if provider == "mock":
        return LiveDataResult(_load_mock_matches(), "Mock", "mock", False, "Mock 資料", _now_text())

    try:
        matches = _fetch_live_from_api(settings)
        if matches.empty:
            return LiveDataResult(_load_mock_matches(), "Mock", provider, True, "目前使用備援資料：API 無可用賽事", _now_text())
        return LiveDataResult(matches, "Live API", provider, False, "Live API 即時資料", _now_text())
    except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
        return LiveDataResult(_load_mock_matches(), "Mock", provider, True, f"目前使用備援資料：{exc.__class__.__name__}", _now_text())


def fetch_match_results(secrets: Any | None = None) -> LiveDataResult:
    result = fetch_live_matches(secrets)
    finished = result.data[result.data["status"].eq("finished")].copy()
    return LiveDataResult(finished, result.source_mode, result.provider, result.fallback_used, result.message, result.updated_at)


def fetch_group_standings(secrets: Any | None = None) -> LiveDataResult:
    result = fetch_live_matches(secrets)
    standings_input = result.data[
        [
            "match_id",
            "date",
            "match_time",
            "group",
            "home_team",
            "away_team",
            "home_score",
            "away_score",
            "status",
            "minute",
            "venue",
        ]
    ].copy()
    standings = calculate_group_standings(normalize_match_results(standings_input))
    return LiveDataResult(standings, result.source_mode, result.provider, result.fallback_used, result.message, result.updated_at)


def fetch_team_stats(secrets: Any | None = None) -> LiveDataResult:
    result = fetch_group_standings(secrets)
    return result


def fetch_player_stats(secrets: Any | None = None) -> LiveDataResult:
    settings = provider_settings(secrets)
    empty = pd.DataFrame(columns=["player", "team", "position", "minutes", "goals", "assists", "rating"])
    return LiveDataResult(empty, "Mock" if settings["provider"] == "mock" else "Live API", settings["provider"], settings["provider"] != "mock", "球員即時資料待 API 支援", _now_text())
