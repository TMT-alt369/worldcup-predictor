from __future__ import annotations

import pandas as pd

from worldcup_predictor.tournament import run_tournament_simulation


def run_worldcup_monte_carlo(
    fixtures: pd.DataFrame,
    team_meta: pd.DataFrame,
    worldcup_team_stats: pd.DataFrame,
    recent_matches: pd.DataFrame,
    simulations: int = 1000,
) -> pd.DataFrame:
    """Thin wrapper for V7 demo exports without changing the production model."""
    results = run_tournament_simulation(
        fixtures,
        team_meta,
        worldcup_team_stats,
        recent_matches,
        simulations=simulations,
    )
    if "round_32_probability" not in results.columns and "group_qualified_probability" in results.columns:
        results["round_32_probability"] = results["group_qualified_probability"]
    results["simulation_count"] = int(simulations)
    return results
