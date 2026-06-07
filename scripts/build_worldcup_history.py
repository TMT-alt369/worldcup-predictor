from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIR = ROOT / "data" / "source"
OUTPUT_DIR = ROOT / "data" / "worldcup"
YEARS = set(range(2002, 2023, 4))


def result_for(home_goals: int, away_goals: int) -> str:
    if home_goals > away_goals:
        return "home"
    if home_goals < away_goals:
        return "away"
    return "draw"


def canonical_pair(team_a: str, team_b: str) -> tuple[str, str]:
    return tuple(sorted([team_a, team_b]))


def build_matches(matches: pd.DataFrame) -> pd.DataFrame:
    matches = matches.copy()
    matches["year"] = matches["tournament_name"].str.extract(r"(\d{4})").astype(int)
    matches = matches[matches["year"].isin(YEARS)].copy()
    matches["match_date"] = pd.to_datetime(matches["match_date"])
    matches["winner"] = matches.apply(
        lambda row: row["home_team_name"]
        if row["home_team_score"] > row["away_team_score"]
        else row["away_team_name"]
        if row["away_team_score"] > row["home_team_score"]
        else "Draw",
        axis=1,
    )
    matches["is_draw"] = matches["home_team_score"] == matches["away_team_score"]

    output = matches[
        [
            "match_id",
            "year",
            "stage_name",
            "group_name",
            "match_date",
            "home_team_name",
            "away_team_name",
            "home_team_score",
            "away_team_score",
            "winner",
            "is_draw",
            "extra_time",
            "penalty_shootout",
            "home_team_score_penalties",
            "away_team_score_penalties",
            "stadium_name",
            "city_name",
            "country_name",
        ]
    ].rename(
        columns={
            "year": "tournament_year",
            "stage_name": "stage",
            "match_date": "date",
            "home_team_name": "home_team",
            "away_team_name": "away_team",
            "home_team_score": "home_goals",
            "away_team_score": "away_goals",
            "extra_time": "went_to_extra_time",
            "penalty_shootout": "went_to_penalties",
            "home_team_score_penalties": "home_penalties",
            "away_team_score_penalties": "away_penalties",
            "stadium_name": "stadium",
            "city_name": "city",
            "country_name": "host_country",
        }
    )
    return output.sort_values(["tournament_year", "date", "match_id"])


def build_champions(tournaments: pd.DataFrame, standings: pd.DataFrame) -> pd.DataFrame:
    tournaments = tournaments[tournaments["year"].isin(YEARS)].copy()
    standings = standings.copy()
    standings["year"] = standings["tournament_name"].str.extract(r"(\d{4})").astype(int)
    standings = standings[standings["year"].isin(YEARS)]

    rows = []
    for _, tournament in tournaments.iterrows():
        year = tournament["year"]
        year_standings = standings[standings["year"] == year].set_index("position")
        rows.append(
            {
                "year": year,
                "host": tournament["host_country"],
                "champion": year_standings.loc[1, "team_name"],
                "runner_up": year_standings.loc[2, "team_name"],
            }
        )
    return pd.DataFrame(rows).sort_values("year")


def build_top4(standings: pd.DataFrame) -> pd.DataFrame:
    standings = standings.copy()
    standings["year"] = standings["tournament_name"].str.extract(r"(\d{4})").astype(int)
    standings = standings[
        standings["year"].isin(YEARS) & standings["position"].isin([1, 2, 3, 4])
    ]

    rows = []
    for year, group in standings.groupby("year"):
        by_position = group.set_index("position")
        rows.append(
            {
                "year": year,
                "champion": by_position.loc[1, "team_name"],
                "runner_up": by_position.loc[2, "team_name"],
                "third_place": by_position.loc[3, "team_name"],
                "fourth_place": by_position.loc[4, "team_name"],
            }
        )
    return pd.DataFrame(rows).sort_values("year")


def build_team_stats(matches: pd.DataFrame, top4: pd.DataFrame) -> pd.DataFrame:
    team_rows = []
    tournament_teams = {}

    for _, row in matches.iterrows():
        tournament_teams.setdefault(row["tournament_year"], set()).update(
            [row["home_team"], row["away_team"]]
        )
        teams = [
            (row["home_team"], row["home_goals"], row["away_goals"]),
            (row["away_team"], row["away_goals"], row["home_goals"]),
        ]
        for team, goals_for, goals_against in teams:
            outcome = result_for(goals_for, goals_against)
            team_rows.append(
                {
                    "team": team,
                    "matches": 1,
                    "wins": int(outcome == "home"),
                    "draws": int(outcome == "draw"),
                    "losses": int(outcome == "away"),
                    "goals_for": goals_for,
                    "goals_against": goals_against,
                }
            )

    stats = pd.DataFrame(team_rows).groupby("team", as_index=False).sum()
    appearances = {}
    for teams in tournament_teams.values():
        for team in teams:
            appearances[team] = appearances.get(team, 0) + 1
    stats["tournaments_played"] = stats["team"].map(appearances).fillna(0).astype(int)

    title_counts = top4["champion"].value_counts()
    top4_counts = pd.concat(
        [top4["champion"], top4["runner_up"], top4["third_place"], top4["fourth_place"]]
    ).value_counts()
    stats["goal_difference"] = stats["goals_for"] - stats["goals_against"]
    stats["win_rate"] = (stats["wins"] / stats["matches"]).round(4)
    stats["titles"] = stats["team"].map(title_counts).fillna(0).astype(int)
    stats["top4_finishes"] = stats["team"].map(top4_counts).fillna(0).astype(int)
    return stats[
        [
            "team",
            "tournaments_played",
            "matches",
            "wins",
            "draws",
            "losses",
            "goals_for",
            "goals_against",
            "goal_difference",
            "win_rate",
            "titles",
            "top4_finishes",
        ]
    ].sort_values(["titles", "top4_finishes", "win_rate", "goal_difference"], ascending=False)


def build_head_to_head(matches: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in matches.iterrows():
        team_a, team_b = canonical_pair(row["home_team"], row["away_team"])
        team_a_goals = row["home_goals"] if row["home_team"] == team_a else row["away_goals"]
        team_b_goals = row["away_goals"] if row["home_team"] == team_a else row["home_goals"]
        rows.append(
            {
                "team_a": team_a,
                "team_b": team_b,
                "matches": 1,
                "team_a_wins": int(team_a_goals > team_b_goals),
                "draws": int(team_a_goals == team_b_goals),
                "team_b_wins": int(team_b_goals > team_a_goals),
                "team_a_goals": team_a_goals,
                "team_b_goals": team_b_goals,
                "last_meeting_year": row["tournament_year"],
            }
        )

    return (
        pd.DataFrame(rows)
        .groupby(["team_a", "team_b"], as_index=False)
        .agg(
            matches=("matches", "sum"),
            team_a_wins=("team_a_wins", "sum"),
            draws=("draws", "sum"),
            team_b_wins=("team_b_wins", "sum"),
            team_a_goals=("team_a_goals", "sum"),
            team_b_goals=("team_b_goals", "sum"),
            last_meeting_year=("last_meeting_year", "max"),
        )
        .sort_values(["matches", "last_meeting_year"], ascending=False)
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    raw_matches = pd.read_csv(SOURCE_DIR / "fjelstul_matches.csv")
    raw_tournaments = pd.read_csv(SOURCE_DIR / "fjelstul_tournaments.csv")
    raw_standings = pd.read_csv(SOURCE_DIR / "fjelstul_tournament_standings.csv")

    matches = build_matches(raw_matches)
    champions = build_champions(raw_tournaments, raw_standings)
    top4 = build_top4(raw_standings)
    team_stats = build_team_stats(matches, top4)
    head_to_head = build_head_to_head(matches)

    matches.to_csv(OUTPUT_DIR / "worldcup_matches_2002_2022.csv", index=False)
    champions.to_csv(OUTPUT_DIR / "worldcup_champions_2002_2022.csv", index=False)
    top4.to_csv(OUTPUT_DIR / "worldcup_top4_2002_2022.csv", index=False)
    team_stats.to_csv(OUTPUT_DIR / "worldcup_team_stats_2002_2022.csv", index=False)
    head_to_head.to_csv(OUTPUT_DIR / "worldcup_head_to_head_2002_2022.csv", index=False)


if __name__ == "__main__":
    main()
