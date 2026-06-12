import pandas as pd

from worldcup_predictor.team_resolver import find_team_row, resolve_team_name, team_match_key


def worldcup_performance_score(team_stats: pd.DataFrame, team: str) -> float:
    row = find_team_row(team_stats, team)
    if row is None:
        return 0.45

    item = row
    goal_diff_per_match = item["goal_difference"] / max(item["matches"], 1)
    score = (
        0.55 * item["win_rate"]
        + 0.18 * min(item["titles"], 2) / 2
        + 0.17 * min(item["top4_finishes"], 4) / 4
        + 0.10 * min(max((goal_diff_per_match + 2) / 4, 0), 1)
    )
    return round(float(score), 4)


def team_summary(team_stats: pd.DataFrame, team: str) -> dict[str, float | int | str]:
    row = find_team_row(team_stats, team)
    canonical_team = resolve_team_name(team)
    if row is None:
        return {
            "team": canonical_team,
            "tournaments_played": pd.NA,
            "matches": pd.NA,
            "wins": pd.NA,
            "draws": pd.NA,
            "losses": pd.NA,
            "goals_for": pd.NA,
            "goals_against": pd.NA,
            "goal_difference": pd.NA,
            "win_rate": pd.NA,
            "titles": pd.NA,
            "top4_finishes": pd.NA,
            "performance_score": 0.45,
            "has_data": False,
        }

    data = row.to_dict()
    data["team"] = resolve_team_name(data.get("team", canonical_team))
    data["performance_score"] = worldcup_performance_score(team_stats, team)
    data["has_data"] = True
    return data


def head_to_head_record(
    head_to_head: pd.DataFrame,
    team_a: str,
    team_b: str,
) -> pd.DataFrame:
    first, second = sorted([resolve_team_name(team_a), resolve_team_name(team_b)])
    first_key, second_key = team_match_key(first), team_match_key(second)
    if head_to_head.empty:
        record = pd.DataFrame()
    else:
        team_a_keys = head_to_head["team_a"].map(team_match_key)
        team_b_keys = head_to_head["team_b"].map(team_match_key)
        record = head_to_head[
            (team_a_keys == first_key) & (team_b_keys == second_key)
        ].copy()
        if record.empty:
            record = head_to_head[
                (team_a_keys == second_key) & (team_b_keys == first_key)
            ].copy()
    if record.empty:
        return pd.DataFrame(
            [
                {
                    "team_a": first,
                    "team_b": second,
                    "matches": 0,
                    "team_a_wins": 0,
                    "draws": 0,
                    "team_b_wins": 0,
                    "team_a_goals": 0,
                    "team_b_goals": 0,
                    "last_meeting_year": pd.NA,
                    "has_data": False,
                    "note": "1930-2022 World Cup no head-to-head record",
                }
            ]
        )
    record["team_a"] = record["team_a"].map(resolve_team_name)
    record["team_b"] = record["team_b"].map(resolve_team_name)
    record["has_data"] = True
    return record


def top_team_stats(team_stats: pd.DataFrame, limit: int = 12) -> pd.DataFrame:
    return team_stats.sort_values(
        ["titles", "top4_finishes", "win_rate", "goal_difference"],
        ascending=False,
    ).head(limit)
