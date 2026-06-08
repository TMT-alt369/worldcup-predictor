from pathlib import Path

import pandas as pd


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
WORLDCUP_DIR = DATA_DIR / "worldcup"


def load_fixtures() -> pd.DataFrame:
    real_fixtures = DATA_DIR / "fixtures_real_2026.csv"
    source = real_fixtures if real_fixtures.exists() else DATA_DIR / "fixtures.csv"
    fixtures = pd.read_csv(source)
    fixtures["date"] = pd.to_datetime(fixtures["date"])
    if "datetime_taipei" in fixtures.columns:
        fixtures["datetime_taipei"] = pd.to_datetime(fixtures["datetime_taipei"])
    else:
        fixtures["datetime_taipei"] = fixtures["date"]
    return fixtures.sort_values("datetime_taipei").reset_index(drop=True)


def load_historical_matches() -> pd.DataFrame:
    matches = pd.read_csv(DATA_DIR / "historical_matches.csv", parse_dates=["date"])
    return matches.sort_values("date").reset_index(drop=True)


def load_odds() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "odds.csv")


def load_fixture_with_odds() -> pd.DataFrame:
    fixtures = load_fixtures()
    odds = load_odds()
    merged = fixtures.merge(odds, on="match_id", how="left")
    merged["bookmaker"] = merged["bookmaker"].fillna("DemoOdds")
    merged["home_odds"] = merged["home_odds"].fillna(2.10)
    merged["draw_odds"] = merged["draw_odds"].fillna(3.30)
    merged["away_odds"] = merged["away_odds"].fillna(3.40)
    return merged


def team_options() -> list[str]:
    fixtures = load_fixtures()
    teams = set(fixtures["home_team"]).union(set(fixtures["away_team"]))
    return sorted(teams)


def load_worldcup_matches() -> pd.DataFrame:
    matches = pd.read_csv(
        WORLDCUP_DIR / "worldcup_matches_2002_2022.csv",
        parse_dates=["date"],
    )
    return matches.sort_values(["tournament_year", "date"]).reset_index(drop=True)


def load_worldcup_team_stats() -> pd.DataFrame:
    return pd.read_csv(WORLDCUP_DIR / "worldcup_team_stats_2002_2022.csv")


def load_worldcup_champions() -> pd.DataFrame:
    return pd.read_csv(WORLDCUP_DIR / "worldcup_champions_2002_2022.csv")


def load_worldcup_top4() -> pd.DataFrame:
    return pd.read_csv(WORLDCUP_DIR / "worldcup_top4_2002_2022.csv")


def load_worldcup_head_to_head() -> pd.DataFrame:
    return pd.read_csv(WORLDCUP_DIR / "worldcup_head_to_head_2002_2022.csv")


def load_team_meta() -> pd.DataFrame:
    team_meta = pd.read_csv(DATA_DIR / "team_meta.csv")
    if "team_en" in team_meta.columns:
        team_meta["team"] = team_meta["team_en"]
    if "team_zh" not in team_meta.columns:
        team_meta["team_zh"] = team_meta["team"]
    return team_meta


def load_worldcup_team_history() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "worldcup_team_history_2002_2022.csv")


def load_players() -> pd.DataFrame:
    source = DATA_DIR / "players_2026.csv"
    if not source.exists():
        source = DATA_DIR / "players.csv"
    players = pd.read_csv(source)
    rename_map = {
        "team_en": "team",
        "caps": "national_caps",
        "goals": "national_goals",
        "recent_form_score": "recent_form_rating",
    }
    players = players.rename(columns={k: v for k, v in rename_map.items() if k in players.columns})
    if "team_zh" not in players.columns:
        players["team_zh"] = players["team"]
    if "data_source" not in players.columns:
        players["data_source"] = "local_fallback_players"
    if "squad_status" not in players.columns:
        players["squad_status"] = "fallback_demo_partial"
    caps = pd.to_numeric(players["national_caps"], errors="coerce").replace(0, pd.NA)
    goals = pd.to_numeric(players["national_goals"], errors="coerce").fillna(0)
    players["goal_rate"] = (goals / caps).astype("Float64").fillna(0).astype(float)
    return players


def load_live_matches() -> pd.DataFrame:
    source = DATA_DIR / "mock_live_matches.csv"
    if not source.exists():
        source = DATA_DIR / "live_matches.csv"
    return pd.read_csv(source)


def load_live_events() -> pd.DataFrame:
    source = DATA_DIR / "mock_live_events.csv"
    if not source.exists():
        source = DATA_DIR / "live_events.csv"
    return pd.read_csv(source)


def load_live_stats() -> pd.DataFrame:
    live_matches = load_live_matches()
    return live_matches[
        [
            "live_match_id",
            "home_shots",
            "away_shots",
            "home_possession",
            "away_possession",
            "home_corners",
            "away_corners",
        ]
    ]
