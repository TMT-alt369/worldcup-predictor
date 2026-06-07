import pandas as pd


def worldcup_performance_score(team_stats: pd.DataFrame, team: str) -> float:
    row = team_stats[team_stats["team"] == team]
    if row.empty:
        return 0.45

    item = row.iloc[0]
    goal_diff_per_match = item["goal_difference"] / max(item["matches"], 1)
    score = (
        0.55 * item["win_rate"]
        + 0.18 * min(item["titles"], 2) / 2
        + 0.17 * min(item["top4_finishes"], 4) / 4
        + 0.10 * min(max((goal_diff_per_match + 2) / 4, 0), 1)
    )
    return round(float(score), 4)


def team_summary(team_stats: pd.DataFrame, team: str) -> dict[str, float | int | str]:
    row = team_stats[team_stats["team"] == team]
    if row.empty:
        return {
            "team": team,
            "tournaments_played": 0,
            "matches": 0,
            "wins": 0,
            "draws": 0,
            "losses": 0,
            "goals_for": 0,
            "goals_against": 0,
            "goal_difference": 0,
            "win_rate": 0.0,
            "titles": 0,
            "top4_finishes": 0,
            "performance_score": 0.45,
        }

    data = row.iloc[0].to_dict()
    data["performance_score"] = worldcup_performance_score(team_stats, team)
    return data


def head_to_head_record(
    head_to_head: pd.DataFrame,
    team_a: str,
    team_b: str,
) -> pd.DataFrame:
    first, second = sorted([team_a, team_b])
    record = head_to_head[
        (head_to_head["team_a"] == first) & (head_to_head["team_b"] == second)
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
                    "last_meeting_year": None,
                }
            ]
        )
    return record


def top_team_stats(team_stats: pd.DataFrame, limit: int = 12) -> pd.DataFrame:
    return team_stats.sort_values(
        ["titles", "top4_finishes", "win_rate", "goal_difference"],
        ascending=False,
    ).head(limit)
