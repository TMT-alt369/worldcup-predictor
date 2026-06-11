from __future__ import annotations

from pathlib import Path

import pandas as pd

from utils.dynamic_prediction import (
    combine_results,
    group_standings,
    load_match_results,
    probability_delta,
    rerun_dynamic_simulation,
    update_elo,
)
from utils.simulation import run_worldcup_monte_carlo


DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_completed_results() -> pd.DataFrame:
    return load_match_results(DATA_DIR / "match_results.csv")


def build_group_standings(results: pd.DataFrame | None = None) -> pd.DataFrame:
    data = load_completed_results() if results is None else results
    return group_standings(data)


def build_dynamic_elo(team_meta: pd.DataFrame, results: pd.DataFrame | None = None) -> pd.DataFrame:
    data = load_completed_results() if results is None else results
    return update_elo(team_meta, data)


def recalculate_probabilities(
    fixtures: pd.DataFrame,
    team_meta: pd.DataFrame,
    worldcup_team_stats: pd.DataFrame,
    recent_matches: pd.DataFrame,
    results: pd.DataFrame | None = None,
    simulations: int = 1000,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    result_data = load_completed_results() if results is None else results
    before = run_worldcup_monte_carlo(
        fixtures,
        team_meta,
        worldcup_team_stats,
        recent_matches,
        simulations=simulations,
    )
    after = rerun_dynamic_simulation(
        fixtures,
        team_meta,
        worldcup_team_stats,
        recent_matches,
        result_data,
        simulations=simulations,
    )
    changes = probability_delta(before, after)
    return before, after, changes


def append_manual_result(
    base_results: pd.DataFrame,
    fixture: pd.Series,
    home_score: int,
    away_score: int,
) -> pd.DataFrame:
    manual = pd.DataFrame(
        [
            {
                "match_id": str(fixture.get("match_id", "")),
                "date": str(fixture.get("date", ""))[:10],
                "group": str(fixture.get("stage", "未分組")),
                "home_team": str(fixture.get("home_team", "")),
                "away_team": str(fixture.get("away_team", "")),
                "home_score": int(home_score),
                "away_score": int(away_score),
                "status": "Completed",
            }
        ]
    )
    return combine_results(base_results, manual)


def probability_change_summary(changes: pd.DataFrame, limit: int = 12) -> pd.DataFrame:
    if changes is None or changes.empty:
        return pd.DataFrame()
    data = changes.copy()
    delta_columns = [column for column in data.columns if column.endswith("_delta")]
    if not delta_columns:
        return data.head(limit)
    data["total_abs_change"] = data[delta_columns].abs().sum(axis=1)
    return data.sort_values("total_abs_change", ascending=False).head(limit).reset_index(drop=True)
