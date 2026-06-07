from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from worldcup_predictor.history import worldcup_performance_score


@dataclass(frozen=True)
class TeamProfile:
    team: str
    team_zh: str
    flag_emoji: str
    elo: float
    history_score: float
    recent_form: float
    strength: float


STAGE_COLUMNS = {
    "group_qualified": "小組出線機率",
    "round_16": "16 強機率",
    "round_8": "8 強機率",
    "semi_final": "4 強機率",
    "final": "決賽機率",
    "champion": "冠軍機率",
}


def recent_form_score(matches: pd.DataFrame, team: str) -> float:
    team_matches = matches[
        (matches["home_team"] == team) | (matches["away_team"] == team)
    ].sort_values("date")
    if team_matches.empty:
        return 0.5

    points = []
    for _, match in team_matches.tail(6).iterrows():
        is_home = match["home_team"] == team
        scored = int(match["home_goals"] if is_home else match["away_goals"])
        conceded = int(match["away_goals"] if is_home else match["home_goals"])
        if scored > conceded:
            points.append(3)
        elif scored == conceded:
            points.append(1)
        else:
            points.append(0)
    return float(np.mean(points) / 3) if points else 0.5


def tournament_teams(fixtures: pd.DataFrame) -> list[str]:
    group_fixtures = fixtures[fixtures["stage"].str.startswith("Group ", na=False)]
    teams = set(group_fixtures["home_team"]).union(set(group_fixtures["away_team"]))
    return sorted(teams)


def build_profiles(
    fixtures: pd.DataFrame,
    team_meta: pd.DataFrame,
    worldcup_team_stats: pd.DataFrame,
    recent_matches: pd.DataFrame,
) -> dict[str, TeamProfile]:
    meta = team_meta.set_index("team")
    teams = tournament_teams(fixtures)
    profiles = {}

    for team in teams:
        meta_row = meta.loc[team] if team in meta.index else None
        elo = float(meta_row["elo"]) if meta_row is not None and "elo" in meta_row else 1650.0
        team_zh = str(meta_row["team_zh"]) if meta_row is not None and "team_zh" in meta_row else team
        flag_emoji = str(meta_row["flag_emoji"]) if meta_row is not None and "flag_emoji" in meta_row else "🏳️"
        history_score = worldcup_performance_score(worldcup_team_stats, team)
        form = recent_form_score(recent_matches, team)
        strength = ((elo - 1500) / 500) + (0.65 * history_score) + (0.45 * form)

        profiles[team] = TeamProfile(
            team=team,
            team_zh=team_zh,
            flag_emoji=flag_emoji,
            elo=elo,
            history_score=history_score,
            recent_form=form,
            strength=float(strength),
        )

    return profiles


def expected_goals(profile_a: TeamProfile, profile_b: TeamProfile) -> tuple[float, float]:
    strength_gap = profile_a.strength - profile_b.strength
    elo_gap = (profile_a.elo - profile_b.elo) / 400
    expected_a = 1.25 + 0.52 * strength_gap + 0.10 * elo_gap
    expected_b = 1.15 - 0.52 * strength_gap - 0.08 * elo_gap
    return float(np.clip(expected_a, 0.25, 3.8)), float(np.clip(expected_b, 0.20, 3.6))


def simulate_score(
    rng: np.random.Generator,
    profiles: dict[str, TeamProfile],
    team_a: str,
    team_b: str,
) -> tuple[int, int]:
    goals_a, goals_b = expected_goals(profiles[team_a], profiles[team_b])
    return int(rng.poisson(goals_a)), int(rng.poisson(goals_b))


def knockout_winner(
    rng: np.random.Generator,
    profiles: dict[str, TeamProfile],
    team_a: str,
    team_b: str,
) -> str:
    goals_a, goals_b = simulate_score(rng, profiles, team_a, team_b)
    if goals_a > goals_b:
        return team_a
    if goals_b > goals_a:
        return team_b

    rating_gap = profiles[team_a].strength - profiles[team_b].strength
    win_probability = 1 / (1 + np.exp(-rating_gap))
    return team_a if rng.random() < win_probability else team_b


def group_table(teams: list[str]) -> dict[str, dict[str, float]]:
    return {
        team: {"points": 0, "gf": 0, "ga": 0, "gd": 0}
        for team in teams
    }


def rank_group(table: dict[str, dict[str, float]], profiles: dict[str, TeamProfile]) -> list[str]:
    return sorted(
        table,
        key=lambda team: (
            table[team]["points"],
            table[team]["gd"],
            table[team]["gf"],
            profiles[team].strength,
        ),
        reverse=True,
    )


def simulate_group_stage(
    rng: np.random.Generator,
    fixtures: pd.DataFrame,
    profiles: dict[str, TeamProfile],
) -> list[str]:
    group_fixtures = fixtures[fixtures["stage"].str.startswith("Group ", na=False)]
    group_rankings = []
    third_place = []

    for group, group_matches in group_fixtures.groupby("stage", sort=True):
        teams = sorted(set(group_matches["home_team"]).union(set(group_matches["away_team"])))
        table = group_table(teams)

        for _, match in group_matches.iterrows():
            home = match["home_team"]
            away = match["away_team"]
            home_goals, away_goals = simulate_score(rng, profiles, home, away)

            table[home]["gf"] += home_goals
            table[home]["ga"] += away_goals
            table[away]["gf"] += away_goals
            table[away]["ga"] += home_goals
            table[home]["gd"] = table[home]["gf"] - table[home]["ga"]
            table[away]["gd"] = table[away]["gf"] - table[away]["ga"]

            if home_goals > away_goals:
                table[home]["points"] += 3
            elif away_goals > home_goals:
                table[away]["points"] += 3
            else:
                table[home]["points"] += 1
                table[away]["points"] += 1

        ranked = rank_group(table, profiles)
        group_rankings.extend(ranked[:2])
        third_place.append((ranked[2], table[ranked[2]]))

    best_thirds = sorted(
        third_place,
        key=lambda item: (
            item[1]["points"],
            item[1]["gd"],
            item[1]["gf"],
            profiles[item[0]].strength,
        ),
        reverse=True,
    )[:8]
    return group_rankings + [team for team, _ in best_thirds]


def seed_knockout(qualifiers: list[str], profiles: dict[str, TeamProfile]) -> list[str]:
    seeded = sorted(qualifiers, key=lambda team: profiles[team].strength, reverse=True)
    bracket = []
    left, right = 0, len(seeded) - 1
    while left < right:
        bracket.extend([seeded[left], seeded[right]])
        left += 1
        right -= 1
    return bracket


def play_round(
    rng: np.random.Generator,
    profiles: dict[str, TeamProfile],
    teams: list[str],
) -> list[str]:
    winners = []
    for index in range(0, len(teams), 2):
        winners.append(knockout_winner(rng, profiles, teams[index], teams[index + 1]))
    return winners


def run_tournament_simulation(
    fixtures: pd.DataFrame,
    team_meta: pd.DataFrame,
    worldcup_team_stats: pd.DataFrame,
    recent_matches: pd.DataFrame,
    simulations: int = 1000,
    seed: int = 2026,
) -> pd.DataFrame:
    profiles = build_profiles(fixtures, team_meta, worldcup_team_stats, recent_matches)
    counts = {
        team: {stage: 0 for stage in STAGE_COLUMNS}
        for team in profiles
    }
    rng = np.random.default_rng(seed)

    for _ in range(simulations):
        qualifiers = simulate_group_stage(rng, fixtures, profiles)
        for team in qualifiers:
            counts[team]["group_qualified"] += 1

        round_32 = seed_knockout(qualifiers, profiles)
        round_16 = play_round(rng, profiles, round_32)
        for team in round_16:
            counts[team]["round_16"] += 1

        round_8 = play_round(rng, profiles, round_16)
        for team in round_8:
            counts[team]["round_8"] += 1

        semi_final = play_round(rng, profiles, round_8)
        for team in semi_final:
            counts[team]["semi_final"] += 1

        finalists = play_round(rng, profiles, semi_final)
        for team in finalists:
            counts[team]["final"] += 1

        champion = play_round(rng, profiles, finalists)[0]
        counts[champion]["champion"] += 1

    rows = []
    for team, profile in profiles.items():
        row = {
            "team": team,
            "team_zh": profile.team_zh,
            "flag_emoji": profile.flag_emoji,
            "team_display": f"{profile.flag_emoji} {profile.team_zh}",
            "elo": int(round(profile.elo)),
            "history_score": round(profile.history_score, 4),
            "recent_form": round(profile.recent_form, 4),
            "strength": round(profile.strength, 4),
        }
        for stage in STAGE_COLUMNS:
            row[f"{stage}_probability"] = round(counts[team][stage] / simulations, 4)
        rows.append(row)

    return pd.DataFrame(rows).sort_values(
        ["champion_probability", "final_probability", "semi_final_probability"],
        ascending=False,
    ).reset_index(drop=True)
