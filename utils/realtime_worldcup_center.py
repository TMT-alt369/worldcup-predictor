from __future__ import annotations

from pathlib import Path

import pandas as pd


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MATCH_RESULTS_PATH = DATA_DIR / "match_results.csv"

REQUIRED_COLUMNS = [
    "match_id",
    "date",
    "group",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "status",
]

STATUS_LABELS = {
    "scheduled": "未開始",
    "live": "進行中",
    "finished": "已結束",
}


def sample_match_results() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ["A1", "2026-06-12", "Group A", "Mexico", "South Africa", 2, 0, "finished"],
            ["A2", "2026-06-13", "Group A", "Canada", "Bosnia and Herzegovina", 1, 1, "finished"],
            ["A3", "2026-06-18", "Group A", "Mexico", "Canada", pd.NA, pd.NA, "scheduled"],
            ["A4", "2026-06-18", "Group A", "South Africa", "Bosnia and Herzegovina", 0, 0, "live"],
            ["B1", "2026-06-14", "Group B", "United States", "Paraguay", 1, 0, "finished"],
            ["B2", "2026-06-15", "Group B", "Brazil", "Germany", 2, 2, "finished"],
            ["B3", "2026-06-20", "Group B", "United States", "Brazil", pd.NA, pd.NA, "scheduled"],
            ["B4", "2026-06-20", "Group B", "Paraguay", "Germany", 1, 2, "live"],
        ],
        columns=REQUIRED_COLUMNS,
    )


def ensure_match_results_file(path: Path | None = None) -> Path:
    source = MATCH_RESULTS_PATH if path is None else Path(path)
    if not source.exists():
        source.parent.mkdir(parents=True, exist_ok=True)
        sample_match_results().to_csv(source, index=False)
    return source


def normalize_match_results(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy() if df is not None else pd.DataFrame()
    for column in REQUIRED_COLUMNS:
        if column not in data.columns:
            data[column] = pd.NA
    data = data[REQUIRED_COLUMNS].copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["status"] = data["status"].astype(str).str.strip().str.lower()
    data["status"] = data["status"].replace(
        {
            "completed": "finished",
            "complete": "finished",
            "done": "finished",
            "in_progress": "live",
            "playing": "live",
            "not_started": "scheduled",
        }
    )
    data.loc[~data["status"].isin(STATUS_LABELS), "status"] = "scheduled"
    data["home_score"] = pd.to_numeric(data["home_score"], errors="coerce")
    data["away_score"] = pd.to_numeric(data["away_score"], errors="coerce")
    text_columns = ["match_id", "group", "home_team", "away_team"]
    for column in text_columns:
        data[column] = data[column].fillna("").astype(str).str.strip()
    return data


def load_realtime_match_results(path: Path | None = None) -> pd.DataFrame:
    source = ensure_match_results_file(path)
    return normalize_match_results(pd.read_csv(source))


def scoreboard_table(results: pd.DataFrame) -> pd.DataFrame:
    data = normalize_match_results(results)
    output = data.copy()
    output["score"] = output.apply(
        lambda row: "待開賽"
        if pd.isna(row["home_score"]) or pd.isna(row["away_score"])
        else f"{int(row['home_score'])} - {int(row['away_score'])}",
        axis=1,
    )
    output["status_label"] = output["status"].map(STATUS_LABELS).fillna("未開始")
    output["date_display"] = output["date"].dt.strftime("%Y/%m/%d").fillna("日期待補")
    return output[
        [
            "match_id",
            "date_display",
            "group",
            "home_team",
            "away_team",
            "score",
            "status_label",
        ]
    ]


def calculate_group_standings(results: pd.DataFrame) -> pd.DataFrame:
    data = normalize_match_results(results)
    teams: dict[tuple[str, str], dict[str, int | str]] = {}

    for _, row in data.iterrows():
        group = str(row["group"] or "Group TBD")
        for team in [row["home_team"], row["away_team"]]:
            if not team:
                continue
            teams.setdefault(
                (group, team),
                {
                    "group": group,
                    "team": team,
                    "played": 0,
                    "wins": 0,
                    "draws": 0,
                    "losses": 0,
                    "goals_for": 0,
                    "goals_against": 0,
                    "goal_difference": 0,
                    "points": 0,
                },
            )

        if row["status"] != "finished":
            continue
        if pd.isna(row["home_score"]) or pd.isna(row["away_score"]):
            continue

        home = teams[(group, row["home_team"])]
        away = teams[(group, row["away_team"])]
        home_score = int(row["home_score"])
        away_score = int(row["away_score"])

        home["played"] += 1
        away["played"] += 1
        home["goals_for"] += home_score
        home["goals_against"] += away_score
        away["goals_for"] += away_score
        away["goals_against"] += home_score

        if home_score > away_score:
            home["wins"] += 1
            home["points"] += 3
            away["losses"] += 1
        elif home_score < away_score:
            away["wins"] += 1
            away["points"] += 3
            home["losses"] += 1
        else:
            home["draws"] += 1
            away["draws"] += 1
            home["points"] += 1
            away["points"] += 1

    standings = pd.DataFrame(teams.values())
    if standings.empty:
        return pd.DataFrame(
            columns=[
                "group",
                "rank",
                "team",
                "played",
                "wins",
                "draws",
                "losses",
                "goals_for",
                "goals_against",
                "goal_difference",
                "points",
            ]
        )
    standings["goal_difference"] = standings["goals_for"] - standings["goals_against"]
    standings = standings.sort_values(
        ["group", "points", "goal_difference", "goals_for", "team"],
        ascending=[True, False, False, False, True],
    ).reset_index(drop=True)
    standings["rank"] = standings.groupby("group").cumcount() + 1
    return standings[
        [
            "group",
            "rank",
            "team",
            "played",
            "wins",
            "draws",
            "losses",
            "goals_for",
            "goals_against",
            "goal_difference",
            "points",
        ]
    ]


def qualification_scenarios(standings: pd.DataFrame) -> pd.DataFrame:
    if standings is None or standings.empty:
        return pd.DataFrame(columns=["group", "rank", "team", "status"])
    scenarios = standings[["group", "rank", "team", "played", "points"]].copy()

    def status_for(row: pd.Series) -> str:
        if int(row["played"]) == 0:
            return "資料不足"
        if int(row["rank"]) <= 2:
            return "暫時晉級"
        if int(row["rank"]) == 3:
            return "尚有機會"
        return "已淘汰"

    scenarios["status"] = scenarios.apply(status_for, axis=1)
    return scenarios[["group", "rank", "team", "status"]]
