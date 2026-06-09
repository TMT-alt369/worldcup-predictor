import pandas as pd

from worldcup_predictor.betting import analyze_1x2
from worldcup_predictor.model import predict_match


HIGH_CONFIDENCE_THRESHOLD = 0.58
LOW_CONFIDENCE_THRESHOLD = 0.48


def actual_result(row: pd.Series) -> str:
    if row["home_goals"] > row["away_goals"]:
        return "主勝"
    if row["home_goals"] == row["away_goals"]:
        return "和局"
    return "客勝"


def predicted_result(probabilities: dict[str, float]) -> str:
    return max(probabilities, key=probabilities.get)


def team_player_goal_feature(players: pd.DataFrame | None, team: str) -> float:
    if players is None or players.empty:
        return 0.0
    if "goal_rate" not in players.columns:
        return 0.0
    team_players = players[players["team"] == team]
    if team_players.empty:
        return 0.0
    return float(team_players.sort_values("goal_rate", ascending=False).head(3)["goal_rate"].mean())


def auxiliary_confidence(
    base_confidence: float,
    players: pd.DataFrame | None,
    home_team: str,
    away_team: str,
) -> float:
    home_goal_rate = team_player_goal_feature(players, home_team)
    away_goal_rate = team_player_goal_feature(players, away_team)
    player_gap = abs(home_goal_rate - away_goal_rate)
    adjusted = base_confidence + min(player_gap * 0.08, 0.035)
    return round(min(adjusted, 0.95), 4)


def confidence_bucket(confidence: float) -> str:
    if confidence >= HIGH_CONFIDENCE_THRESHOLD:
        return "高信心"
    if confidence < LOW_CONFIDENCE_THRESHOLD:
        return "低信心"
    return "中信心"


def summarize_bucket(rows: list[dict], bucket: str) -> tuple[int, float]:
    bucket_rows = [row for row in rows if row["confidence_bucket"] == bucket]
    if not bucket_rows:
        return 0, 0.0
    correct = sum(int(row["correct"]) for row in bucket_rows)
    return len(bucket_rows), round(correct / len(bucket_rows), 4)


def backtest(
    matches: pd.DataFrame,
    worldcup_team_stats: pd.DataFrame | None = None,
    players: pd.DataFrame | None = None,
) -> dict:
    if len(matches) < 6:
        return {
            "matches": 0,
            "accuracy": 0.0,
            "simulated_roi": 0.0,
            "average_confidence": 0.0,
            "high_confidence_matches": 0,
            "high_confidence_accuracy": 0.0,
            "low_confidence_matches": 0,
            "low_confidence_accuracy": 0.0,
            "bucket_rows": [],
        }

    correct = 0
    bets = 0
    profit = 0.0
    confidences = []
    rows = []

    for index in range(5, len(matches)):
        train = matches.iloc[:index]
        row = matches.iloc[index]
        prediction = predict_match(
            train,
            row["home_team"],
            row["away_team"],
            worldcup_team_stats,
        )
        confidence = auxiliary_confidence(
            prediction.confidence,
            players,
            row["home_team"],
            row["away_team"],
        )
        confidences.append(confidence)

        probabilities = {
            "主勝": prediction.home_win_probability,
            "和局": prediction.draw_probability,
            "客勝": prediction.away_win_probability,
        }
        pick = predicted_result(probabilities)
        result = actual_result(row)
        is_correct = pick == result
        correct += int(is_correct)

        signals = analyze_1x2(prediction, 2.0, 3.4, 3.6)
        best_signal = max(signals, key=lambda signal: signal.edge)
        if best_signal.recommendation != "不建議投注":
            bets += 1
            profit += best_signal.odds - 1 if best_signal.market == result else -1

        rows.append(
            {
                "match": f"{row['home_team']} vs {row['away_team']}",
                "prediction": pick,
                "actual": result,
                "correct": is_correct,
                "confidence": confidence,
                "confidence_bucket": confidence_bucket(confidence),
                "player_goal_feature_home": team_player_goal_feature(players, row["home_team"]),
                "player_goal_feature_away": team_player_goal_feature(players, row["away_team"]),
            }
        )

    tested = len(matches) - 5
    roi = profit / bets if bets else 0.0
    high_count, high_accuracy = summarize_bucket(rows, "高信心")
    low_count, low_accuracy = summarize_bucket(rows, "低信心")
    mid_count, mid_accuracy = summarize_bucket(rows, "中信心")

    return {
        "matches": tested,
        "accuracy": round(correct / tested, 4),
        "simulated_roi": round(roi, 4),
        "average_confidence": round(sum(confidences) / len(confidences), 4),
        "high_confidence_matches": high_count,
        "high_confidence_accuracy": high_accuracy,
        "mid_confidence_matches": mid_count,
        "mid_confidence_accuracy": mid_accuracy,
        "low_confidence_matches": low_count,
        "low_confidence_accuracy": low_accuracy,
        "bucket_rows": rows,
    }
