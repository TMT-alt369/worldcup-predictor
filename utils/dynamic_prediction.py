from __future__ import annotations

from pathlib import Path

import pandas as pd

from utils.simulation import run_worldcup_monte_carlo


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
STAGE_COLUMNS = [
    "group_qualified_probability",
    "round_32_probability",
    "round_16_probability",
    "round_8_probability",
    "semi_final_probability",
    "final_probability",
    "champion_probability",
]


def load_match_results(path: str | Path | None = None) -> pd.DataFrame:
    source = Path(path) if path is not None else DATA_DIR / "match_results.csv"
    columns = ["match_id", "date", "group", "home_team", "away_team", "home_score", "away_score", "status"]
    if not source.exists():
        return pd.DataFrame(columns=columns)
    data = pd.read_csv(source)
    for column in columns:
        if column not in data.columns:
            data[column] = "" if column not in ["home_score", "away_score"] else 0
    data["home_score"] = pd.to_numeric(data["home_score"], errors="coerce").fillna(0).astype(int)
    data["away_score"] = pd.to_numeric(data["away_score"], errors="coerce").fillna(0).astype(int)
    data["status"] = data["status"].fillna("").astype(str)
    completed = data[data["status"].str.lower().isin(["completed", "finished", "final", "ft", "結束", "已完成"])].copy()
    return completed[columns].reset_index(drop=True)


def combine_results(*frames: pd.DataFrame) -> pd.DataFrame:
    valid = [frame for frame in frames if frame is not None and not frame.empty]
    if not valid:
        return pd.DataFrame(columns=["match_id", "date", "group", "home_team", "away_team", "home_score", "away_score", "status"])
    combined = pd.concat(valid, ignore_index=True)
    combined["match_id"] = combined["match_id"].astype(str)
    combined = combined.drop_duplicates(subset=["match_id"], keep="last")
    return combined.reset_index(drop=True)


def group_standings(results: pd.DataFrame) -> pd.DataFrame:
    rows: dict[tuple[str, str], dict[str, object]] = {}
    for _, match in results.iterrows():
        group = str(match.get("group", "") or "未分組")
        home = str(match.get("home_team", ""))
        away = str(match.get("away_team", ""))
        home_score = int(match.get("home_score", 0))
        away_score = int(match.get("away_score", 0))
        for team in [home, away]:
            rows.setdefault(
                (group, team),
                {"group": group, "team": team, "played": 0, "wins": 0, "draws": 0, "losses": 0, "gf": 0, "ga": 0, "gd": 0, "points": 0},
            )
        home_row = rows[(group, home)]
        away_row = rows[(group, away)]
        home_row["played"] += 1
        away_row["played"] += 1
        home_row["gf"] += home_score
        home_row["ga"] += away_score
        away_row["gf"] += away_score
        away_row["ga"] += home_score
        if home_score > away_score:
            home_row["wins"] += 1
            away_row["losses"] += 1
            home_row["points"] += 3
        elif away_score > home_score:
            away_row["wins"] += 1
            home_row["losses"] += 1
            away_row["points"] += 3
        else:
            home_row["draws"] += 1
            away_row["draws"] += 1
            home_row["points"] += 1
            away_row["points"] += 1
        home_row["gd"] = home_row["gf"] - home_row["ga"]
        away_row["gd"] = away_row["gf"] - away_row["ga"]
    if not rows:
        return pd.DataFrame(columns=["group", "team", "played", "wins", "draws", "losses", "gf", "ga", "gd", "points"])
    table = pd.DataFrame(rows.values())
    return table.sort_values(["group", "points", "gd", "gf"], ascending=[True, False, False, False]).reset_index(drop=True)


def _expected_score(home_elo: float, away_elo: float) -> float:
    return 1 / (1 + 10 ** ((away_elo - home_elo) / 400))


def update_elo(team_meta: pd.DataFrame, results: pd.DataFrame, k_factor: float = 28.0) -> pd.DataFrame:
    data = team_meta.copy()
    if "team" not in data.columns and "team_en" in data.columns:
        data["team"] = data["team_en"]
    data["elo"] = pd.to_numeric(data.get("elo", 1650), errors="coerce").fillna(1650).astype(float)
    data["original_elo"] = data["elo"]
    elo_map = dict(zip(data["team"], data["elo"]))
    for _, match in results.iterrows():
        home = str(match.get("home_team", ""))
        away = str(match.get("away_team", ""))
        if home not in elo_map or away not in elo_map:
            continue
        home_elo = float(elo_map[home])
        away_elo = float(elo_map[away])
        expected_home = _expected_score(home_elo, away_elo)
        home_score = int(match.get("home_score", 0))
        away_score = int(match.get("away_score", 0))
        actual_home = 1.0 if home_score > away_score else 0.5 if home_score == away_score else 0.0
        margin_multiplier = 1 + min(2.0, abs(home_score - away_score)) * 0.12
        change = k_factor * margin_multiplier * (actual_home - expected_home)
        elo_map[home] = home_elo + change
        elo_map[away] = away_elo - change
    data["updated_elo"] = data["team"].map(elo_map).fillna(data["elo"])
    data["elo_change"] = data["updated_elo"] - data["original_elo"]
    data["elo"] = data["updated_elo"].round(0).astype(int)
    return data


def append_results_to_recent_matches(recent_matches: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    if results is None or results.empty:
        return recent_matches.copy()
    additions = results.rename(columns={"home_score": "home_goals", "away_score": "away_goals"}).copy()
    additions["date"] = pd.to_datetime(additions.get("date", pd.Timestamp.today()), errors="coerce")
    additions["home_elo"] = 1700
    additions["away_elo"] = 1700
    needed = ["date", "home_team", "away_team", "home_goals", "away_goals", "home_elo", "away_elo"]
    for column in needed:
        if column not in additions.columns:
            additions[column] = 0
    base = recent_matches.copy()
    for column in needed:
        if column not in base.columns:
            base[column] = 0
    return pd.concat([base, additions[needed]], ignore_index=True)


def rerun_dynamic_simulation(
    fixtures: pd.DataFrame,
    team_meta: pd.DataFrame,
    worldcup_team_stats: pd.DataFrame,
    recent_matches: pd.DataFrame,
    results: pd.DataFrame,
    simulations: int = 1000,
) -> pd.DataFrame:
    updated_meta = update_elo(team_meta, results)
    updated_matches = append_results_to_recent_matches(recent_matches, results)
    simulated = run_worldcup_monte_carlo(fixtures, updated_meta, worldcup_team_stats, updated_matches, simulations=simulations)
    for column in STAGE_COLUMNS:
        if column not in simulated.columns:
            simulated[column] = 0.0
    return simulated


def probability_delta(before: pd.DataFrame, after: pd.DataFrame) -> pd.DataFrame:
    if before is None or before.empty or after is None or after.empty:
        return pd.DataFrame()
    columns = ["team", "team_display"] + [column for column in STAGE_COLUMNS if column in before.columns and column in after.columns]
    before_small = before[columns].copy()
    after_small = after[columns].copy()
    merged = before_small.merge(after_small, on="team", suffixes=("_before", "_after"))
    if "team_display_after" in merged.columns:
        merged["team_display"] = merged["team_display_after"]
    for column in STAGE_COLUMNS:
        before_col = f"{column}_before"
        after_col = f"{column}_after"
        if before_col in merged.columns and after_col in merged.columns:
            merged[f"{column}_delta"] = pd.to_numeric(merged[after_col], errors="coerce").fillna(0) - pd.to_numeric(merged[before_col], errors="coerce").fillna(0)
    keep = ["team", "team_display"] + [col for col in merged.columns if col.endswith("_before") or col.endswith("_after") or col.endswith("_delta")]
    return merged[keep].reset_index(drop=True)
