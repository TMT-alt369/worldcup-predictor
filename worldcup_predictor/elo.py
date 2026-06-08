import pandas as pd


def elo_tier(elo: float) -> str:
    if elo >= 1850:
        return "冠軍熱門"
    if elo >= 1780:
        return "強隊"
    if elo >= 1680:
        return "競爭者"
    if elo >= 1580:
        return "中段班"
    return "挑戰者"


def build_elo_rankings(team_meta: pd.DataFrame) -> pd.DataFrame:
    rankings = team_meta.copy()
    rankings["elo"] = pd.to_numeric(rankings["elo"], errors="coerce").fillna(1500)
    rankings["fifa_ranking"] = pd.to_numeric(
        rankings["fifa_ranking"],
        errors="coerce",
    ).fillna(999).astype(int)
    rankings["team_display"] = rankings["flag_emoji"] + " " + rankings["team_zh"]
    rankings["elo_tier"] = rankings["elo"].map(elo_tier)
    rankings = rankings.sort_values(
        ["elo", "fifa_ranking"],
        ascending=[False, True],
    ).reset_index(drop=True)
    rankings.insert(0, "rank", rankings.index + 1)
    return rankings
