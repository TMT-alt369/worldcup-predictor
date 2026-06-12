from __future__ import annotations

from itertools import combinations
from pathlib import Path

import pandas as pd

from utils.simulation import run_worldcup_monte_carlo


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
VALID_CONFEDERATIONS = {"UEFA", "CONMEBOL", "AFC", "CAF", "CONCACAF", "OFC"}
STAGE_COLUMNS = [
    "group_qualified_probability",
    "round_16_probability",
    "round_8_probability",
    "semi_final_probability",
    "final_probability",
    "champion_probability",
]


def run_ai_monte_carlo(
    fixtures: pd.DataFrame,
    team_meta: pd.DataFrame,
    worldcup_team_stats: pd.DataFrame,
    recent_matches: pd.DataFrame,
    simulations: int = 1000,
) -> pd.DataFrame:
    simulations = int(simulations)
    if simulations not in [1000, 5000, 10000]:
        simulations = 1000
    data = run_worldcup_monte_carlo(
        fixtures,
        team_meta,
        worldcup_team_stats,
        recent_matches,
        simulations=simulations,
    )
    if "round_32_probability" not in data.columns:
        data["round_32_probability"] = data.get("group_qualified_probability", 0)
    data["simulation_count"] = simulations
    return data


def common_final_combinations(simulation_df: pd.DataFrame, limit: int = 10) -> pd.DataFrame:
    if simulation_df is None or simulation_df.empty or "final_probability" not in simulation_df.columns:
        return pd.DataFrame(columns=["final_combo", "estimated_probability"])
    top = simulation_df.sort_values("final_probability", ascending=False).head(12).copy()
    rows = []
    for left, right in combinations(top.to_dict("records"), 2):
        probability = float(left.get("final_probability", 0)) * float(right.get("final_probability", 0))
        rows.append(
            {
                "final_combo": f"{left.get('team_display', left.get('team'))} vs {right.get('team_display', right.get('team'))}",
                "estimated_probability": probability,
            }
        )
    return pd.DataFrame(rows).sort_values("estimated_probability", ascending=False).head(limit).reset_index(drop=True)


def dark_horse_ranking(simulation_df: pd.DataFrame, limit: int = 10) -> pd.DataFrame:
    if simulation_df is None or simulation_df.empty:
        return pd.DataFrame(columns=["team_display", "elo", "champion_probability", "dark_horse_score"])
    data = simulation_df.copy()
    data["elo_rank"] = pd.to_numeric(data.get("elo", 1650), errors="coerce").rank(ascending=False, method="min")
    data["champion_probability"] = pd.to_numeric(data.get("champion_probability", 0), errors="coerce").fillna(0)
    data["dark_horse_score"] = data["champion_probability"] * (1 + (data["elo_rank"] / max(1, len(data))) * 0.75)
    return data[data["elo_rank"] > 10].sort_values("dark_horse_score", ascending=False).head(limit).reset_index(drop=True)


def continent_champion_probabilities(simulation_df: pd.DataFrame, team_meta: pd.DataFrame) -> pd.DataFrame:
    if simulation_df is None or simulation_df.empty:
        return pd.DataFrame(columns=["confederation", "champion_probability", "missing_teams"])

    mapping_path = DATA_DIR / "team_confederations.csv"
    if mapping_path.exists():
        meta = pd.read_csv(mapping_path)
        if "team_code" in meta.columns and "team" not in meta.columns:
            meta["team"] = meta["team_code"]
    else:
        meta = pd.DataFrame() if team_meta is None else team_meta.copy()

    if meta.empty:
        missing = sorted(simulation_df["team"].dropna().astype(str).unique().tolist())
        return pd.DataFrame([{"confederation": "資料待補", "champion_probability": 0.0, "missing_teams": ", ".join(missing)}])

    if "team" not in meta.columns and "team_en" in meta.columns:
        meta["team"] = meta["team_en"]
    if "team_code" not in meta.columns:
        meta["team_code"] = ""
    if "confederation" not in meta.columns:
        meta["confederation"] = ""

    meta = meta[["team", "team_code", "confederation"]].copy()
    meta["team_key"] = meta["team"].astype(str).str.strip().str.lower()
    meta["code_key"] = meta["team_code"].astype(str).str.strip().str.lower()
    meta["confederation"] = meta["confederation"].where(meta["confederation"].isin(VALID_CONFEDERATIONS), "")

    simulation = simulation_df.copy()
    simulation["team_key"] = simulation["team"].astype(str).str.strip().str.lower()
    merged = simulation.merge(meta[["team_key", "confederation"]], on="team_key", how="left")
    if merged["confederation"].isna().any() or merged["confederation"].eq("").any():
        missing_mask = merged["confederation"].isna() | merged["confederation"].eq("")
        missing_teams = sorted(merged.loc[missing_mask, "team"].dropna().astype(str).unique().tolist())
    else:
        missing_teams = []
    valid = merged[merged["confederation"].isin(VALID_CONFEDERATIONS)].copy()
    if valid.empty:
        return pd.DataFrame([{"confederation": "資料待補", "champion_probability": 0.0, "missing_teams": ", ".join(missing_teams)}])

    valid["champion_probability"] = pd.to_numeric(valid.get("champion_probability", 0), errors="coerce").fillna(0)
    result = (
        valid.groupby("confederation", as_index=False)["champion_probability"]
        .sum()
        .sort_values("champion_probability", ascending=False)
        .reset_index(drop=True)
    )
    result["missing_teams"] = ""
    if missing_teams:
        result.loc[0, "missing_teams"] = ", ".join(missing_teams)
    return result


def stage_probability_table(simulation_df: pd.DataFrame) -> pd.DataFrame:
    if simulation_df is None or simulation_df.empty:
        return pd.DataFrame()
    columns = ["team_display"] + [column for column in STAGE_COLUMNS if column in simulation_df.columns]
    return simulation_df[columns].copy()
