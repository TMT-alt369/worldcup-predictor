import html

import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components

from worldcup_predictor.backtest import backtest
from worldcup_predictor.betting import analyze_1x2
from worldcup_predictor.data_loader import (
    load_fixture_with_odds,
    load_fixtures,
    load_historical_matches,
    load_live_events,
    load_live_matches,
    load_players,
    load_team_meta,
    load_worldcup_champions,
    load_worldcup_head_to_head,
    load_worldcup_matches,
    load_worldcup_team_history,
    load_worldcup_team_stats,
    load_worldcup_top4,
)
from worldcup_predictor.elo import build_elo_rankings
from worldcup_predictor.history import head_to_head_record, team_summary, top_team_stats
from worldcup_predictor.model import poisson_probability, predict_match, prediction_to_frame, team_strength
from worldcup_predictor.players import player_database, squad_summary
from worldcup_predictor.team_resolver import find_team_row, resolve_team_name
from worldcup_predictor.tournament import run_tournament_simulation
from worldcup_predictor.ui import disclaimer_box, format_percent, signal_dataframe
from utils.simulation import run_worldcup_monte_carlo
from utils.player_impact import team_impact, win_probability_adjustment
from utils.market_probability import market_probability_table
from utils.ai_match_report import generate_match_report
from utils.parlay_analyzer import build_match_candidates, build_parlay_combinations, parlay_summary_text
from utils.champion_path import champion_path_text, likely_knockout_path, stage_probability_table
from utils.team_compare import comparison_table, comparison_text, radar_values, team_profile
from utils.mobile_style import mobile_css
from utils.ai_assistant import answer_question
from utils.confidence_engine import ConfidenceResult, calculate_confidence
from utils import bet_market_engine
from utils.dynamic_prediction import (
    combine_results,
    group_standings,
    load_match_results,
    probability_delta,
    rerun_dynamic_simulation,
    update_elo,
)
from utils.dynamic_worldcup_engine import (
    append_manual_result,
    build_dynamic_elo,
    build_group_standings,
    probability_change_summary,
    recalculate_probabilities,
)
from utils.player_impact_engine import (
    load_player_pool,
    match_absence_ranking,
    match_player_impact,
    normalize_players,
    team_player_impact,
)
from utils.ai_monte_carlo_engine import (
    common_final_combinations,
    continent_champion_probabilities,
    dark_horse_ranking,
    run_ai_monte_carlo,
    stage_probability_table as ai_stage_probability_table,
)
from utils.pre_match_analysis import build_pre_match_analysis
from utils.realtime_worldcup_center import (
    calculate_group_standings,
    qualification_scenarios,
    scoreboard_table,
)
from utils.live_data_client import (
    fetch_group_standings,
    fetch_live_matches,
    fetch_match_results,
    fetch_player_stats,
    fetch_team_stats,
)
try:
    from utils.xg_model import PREDICTION_WEIGHTS_XG, prepare_xg_data, xg_match_summary, xg_analysis_text
except ImportError:
    PREDICTION_WEIGHTS_XG = {"elo": 0.40, "xg": 0.40, "form": 0.20, "recent_form": 0.20, "worldcup_history": 0.15}

    def prepare_xg_data(*args, **kwargs) -> pd.DataFrame:
        return pd.DataFrame()

    def xg_match_summary(*args, **kwargs) -> pd.DataFrame:
        return pd.DataFrame()

    def xg_analysis_text(*args, **kwargs) -> list[str]:
        return ["目前尚未匯入足夠的 xG 射門資料。"]

try:
    from utils.elo_update import PREDICTION_WEIGHTS, elo_ranking_with_updates
except ImportError:
    PREDICTION_WEIGHTS = {
        "elo": 0.50,
        "recent_form": 0.30,
        "worldcup_history": 0.20,
        "form": 0.30,
        "history": 0.20,
    }

    def elo_ranking_with_updates(results: pd.DataFrame, team_meta: pd.DataFrame) -> pd.DataFrame:
        data = team_meta.copy()
        if "team_zh" not in data.columns:
            data["team_zh"] = data.get("team", "Unknown")
        if "flag_emoji" not in data.columns:
            data["flag_emoji"] = ""
        data["original_elo"] = pd.to_numeric(data.get("elo", 1700), errors="coerce").fillna(1700)
        data["updated_elo"] = data["original_elo"]
        data["elo_change"] = 0.0
        data["recent_10"] = "資料待補"
        return data[["team", "team_zh", "flag_emoji", "original_elo", "updated_elo", "elo_change", "recent_10"]].sort_values(
            "updated_elo", ascending=False
        ).reset_index(drop=True)


st.set_page_config(page_title="世足智慧預測中心", page_icon="⚽", layout="wide", initial_sidebar_state="collapsed")

GOLD = "#d6b25e"
GOLD_LIGHT = "#f3d98b"
NAVY = "#071426"
NAVY_2 = "#10233e"
INK = "#e9eef7"
MUTED = "#9fb0ca"

FLAGS = {
    "Argentina": "🇦🇷",
    "Brazil": "🇧🇷",
    "Canada": "🇨🇦",
    "Croatia": "🇭🇷",
    "England": "🏴",
    "France": "🇫🇷",
    "Germany": "🇩🇪",
    "Japan": "🇯🇵",
    "Mexico": "🇲🇽",
    "Morocco": "🇲🇦",
    "South Korea": "🇰🇷",
    "Spain": "🇪🇸",
    "USA": "🇺🇸",
    "United States": "🇺🇸",
    "South Africa": "🇿🇦",
    "Czechia": "🇨🇿",
    "Bosnia and Herzegovina": "🇧🇦",
    "Paraguay": "🇵🇾",
    "Qatar": "🇶🇦",
    "Switzerland": "🇨🇭",
    "Haiti": "🇭🇹",
    "Scotland": "🏴",
    "Australia": "🇦🇺",
    "Turkiye": "🇹🇷",
    "Ivory Coast": "🇨🇮",
    "Ecuador": "🇪🇨",
    "Curacao": "🇨🇼",
    "Netherlands": "🇳🇱",
    "Sweden": "🇸🇪",
    "Tunisia": "🇹🇳",
    "Iran": "🇮🇷",
    "New Zealand": "🇳🇿",
    "Belgium": "🇧🇪",
    "Egypt": "🇪🇬",
    "Saudi Arabia": "🇸🇦",
    "Uruguay": "🇺🇾",
    "Cape Verde": "🇨🇻",
    "Senegal": "🇸🇳",
    "Iraq": "🇮🇶",
    "Norway": "🇳🇴",
    "Algeria": "🇩🇿",
    "Austria": "🇦🇹",
    "Jordan": "🇯🇴",
    "Portugal": "🇵🇹",
    "DR Congo": "🇨🇩",
    "Uzbekistan": "🇺🇿",
    "Colombia": "🇨🇴",
    "Ghana": "🇬🇭",
    "Panama": "🇵🇦",
}

TEAM_ZH = {
    "Mexico": "墨西哥",
    "South Africa": "南非",
    "South Korea": "南韓",
    "Czechia": "捷克",
    "Canada": "加拿大",
    "Bosnia and Herzegovina": "波士尼亞與赫塞哥維納",
    "United States": "美國",
    "USA": "美國",
    "Paraguay": "巴拉圭",
    "Qatar": "卡達",
    "Switzerland": "瑞士",
    "Brazil": "巴西",
    "Morocco": "摩洛哥",
    "Haiti": "海地",
    "Scotland": "蘇格蘭",
    "Australia": "澳洲",
    "Turkiye": "土耳其",
    "Ivory Coast": "象牙海岸",
    "Ecuador": "厄瓜多",
    "Germany": "德國",
    "Curacao": "庫拉索",
    "Netherlands": "荷蘭",
    "Japan": "日本",
    "Sweden": "瑞典",
    "Tunisia": "突尼西亞",
    "Iran": "伊朗",
    "New Zealand": "紐西蘭",
    "Belgium": "比利時",
    "Egypt": "埃及",
    "Saudi Arabia": "沙烏地阿拉伯",
    "Uruguay": "烏拉圭",
    "Spain": "西班牙",
    "Cape Verde": "維德角",
    "France": "法國",
    "Senegal": "塞內加爾",
    "Iraq": "伊拉克",
    "Norway": "挪威",
    "Argentina": "阿根廷",
    "Algeria": "阿爾及利亞",
    "Austria": "奧地利",
    "Jordan": "約旦",
    "Portugal": "葡萄牙",
    "DR Congo": "剛果民主共和國",
    "Uzbekistan": "烏茲別克",
    "Colombia": "哥倫比亞",
    "Ghana": "迦納",
    "Panama": "巴拿馬",
    "England": "英格蘭",
    "Croatia": "克羅埃西亞",
}


def flag(team: str) -> str:
    if "TEAM_FLAG_MAP" in globals():
        return TEAM_FLAG_MAP.get(team, FLAGS.get(team, "🏳️"))
    return FLAGS.get(team, "🏳️")


def team_name(team: str, with_flag: bool = True) -> str:
    if "TEAM_NAME_MAP" in globals():
        name = TEAM_NAME_MAP.get(team, TEAM_ZH.get(team, team))
    else:
        name = TEAM_ZH.get(team, team)
    return f"{flag(team)} {name}" if with_flag else name


def matchup_text(row: pd.Series, with_flag: bool = True) -> str:
    return f"{team_name(row['home_team'], with_flag)} vs {team_name(row['away_team'], with_flag)}"


def taipei_datetime(row: pd.Series) -> pd.Timestamp:
    value = row["datetime_taipei"] if "datetime_taipei" in row.index else row["date"]
    return pd.to_datetime(value)


def taipei_time_text(value) -> str:
    return pd.to_datetime(value).strftime("%Y/%m/%d %H:%M")


def fixture_time_text(row: pd.Series) -> str:
    return taipei_time_text(taipei_datetime(row))


def live_time_text(value) -> str:
    try:
        if value is None or pd.isna(value):
            return "N/A"
        return taipei_time_text(value)
    except Exception:
        return "N/A"


def inject_theme() -> None:
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+TC:wght@400;500;700;800;900&display=swap');
        html, body, [class*="css"] {{
            font-family: "Noto Sans TC", "Segoe UI", sans-serif;
        }}
        .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {{
            background:
                linear-gradient(180deg, rgba(7, 20, 38, 0.98), rgba(5, 12, 24, 1)),
                {NAVY};
            color: {INK};
        }}
        [data-testid="stHeader"] {{
            background: rgba(7, 20, 38, 0.82);
            backdrop-filter: blur(10px);
        }}
        .block-container {{
            padding-top: 2.2rem;
            padding-bottom: 1.4rem;
            max-width: 1240px;
        }}
        [data-testid="stSidebar"] {{
            background: #06101f;
            border-right: 1px solid rgba(214, 178, 94, 0.22);
        }}
        [data-testid="stSidebar"] * {{ color: {INK}; }}
        [data-testid="stSidebar"] [role="radiogroup"] label {{
            background: rgba(16, 35, 62, 0.62);
            border: 1px solid rgba(214, 178, 94, 0.12);
            border-radius: 8px;
            padding: 5px 8px;
            margin-bottom: 4px;
        }}
        [data-testid="stSidebar"] [role="radiogroup"] label:hover {{
            border-color: rgba(214, 178, 94, 0.42);
            background: rgba(214, 178, 94, 0.10);
        }}
        h1, h2, h3 {{
            color: {INK};
            letter-spacing: 0;
        }}
        .stCaption, [data-testid="stCaptionContainer"] {{ color: {MUTED}; }}
        .hero {{
            display: grid;
            grid-template-columns: minmax(0, 1.25fr) minmax(280px, 0.75fr);
            gap: 28px;
            align-items: stretch;
            min-height: 380px;
            padding: 36px;
            border: 1px solid rgba(214, 178, 94, 0.32);
            border-radius: 8px;
            background:
                radial-gradient(circle at 84% 16%, rgba(243, 217, 139, 0.22), transparent 30%),
                linear-gradient(135deg, #071426 0%, #10233e 58%, #06101f 100%);
            box-shadow: 0 26px 66px rgba(0, 0, 0, 0.36);
            margin-bottom: 24px;
            overflow: hidden;
            position: relative;
        }}
        .hero:before {{
            content: "";
            position: absolute;
            inset: auto -8% -35% -8%;
            height: 58%;
            border-top: 2px solid rgba(214, 178, 94, 0.4);
            border-radius: 50% 50% 0 0;
        }}
        .hero-kicker {{
            color: {GOLD_LIGHT};
            font-size: 0.86rem;
            font-weight: 800;
            margin-bottom: 12px;
        }}
        .hero-title {{
            font-size: 3.2rem;
            line-height: 1.08;
            font-weight: 900;
            color: #fff;
            margin-bottom: 14px;
        }}
        .hero-copy {{
            color: #c8d3e6;
            font-size: 1.08rem;
            max-width: 720px;
            line-height: 1.72;
        }}
        .hero-visual {{
            position: relative;
            min-height: 280px;
            border-radius: 8px;
            background:
                linear-gradient(180deg, rgba(243, 217, 139, 0.18), rgba(243, 217, 139, 0.02)),
                repeating-linear-gradient(90deg, rgba(255,255,255,0.045) 0 1px, transparent 1px 18px);
            border: 1px solid rgba(214, 178, 94, 0.25);
            display: flex;
            align-items: center;
            justify-content: center;
            overflow: hidden;
        }}
        .hero-visual:before {{
            content: "";
            position: absolute;
            width: 250px;
            height: 250px;
            border: 2px solid rgba(243, 217, 139, 0.36);
            border-radius: 50%;
        }}
        .hero-visual:after {{
            content: "FIFA WORLD CUP";
            position: absolute;
            bottom: 20px;
            color: rgba(243, 217, 139, 0.76);
            font-weight: 900;
            letter-spacing: 0.16em;
            font-size: 0.78rem;
        }}
        .trophy {{
            font-size: 7.3rem;
            filter: drop-shadow(0 18px 24px rgba(0,0,0,0.46));
            z-index: 1;
        }}
        .page-title {{
            padding: 18px 0 10px;
            border-bottom: 1px solid rgba(214, 178, 94, 0.22);
            margin-bottom: 20px;
        }}
        .timezone-badge {{
            display: inline-flex;
            align-items: center;
            color: {GOLD_LIGHT};
            border: 1px solid rgba(243, 217, 139, 0.38);
            background: rgba(214, 178, 94, 0.11);
            border-radius: 999px;
            padding: 6px 10px;
            font-size: 0.82rem;
            font-weight: 800;
            margin-bottom: 10px;
        }}
        .page-title h1 {{
            margin: 0;
            font-size: 2.08rem;
            font-weight: 900;
        }}
        .page-title p {{
            margin: 7px 0 0;
            color: {MUTED};
            line-height: 1.6;
        }}
        .display-card {{
            background: linear-gradient(180deg, rgba(16, 35, 62, 0.98), rgba(8, 21, 40, 0.98));
            border: 1px solid rgba(214, 178, 94, 0.27);
            border-radius: 8px;
            padding: 18px;
            box-shadow: 0 14px 34px rgba(0,0,0,0.26);
            margin-bottom: 14px;
        }}
        .card-label {{
            color: {MUTED};
            font-size: 0.83rem;
            margin-bottom: 8px;
        }}
        .card-value {{
            color: {GOLD_LIGHT};
            font-size: 1.86rem;
            line-height: 1.1;
            font-weight: 900;
        }}
        .card-note {{
            color: {MUTED};
            font-size: 0.78rem;
            margin-top: 8px;
        }}
        .score-card {{
            text-align: center;
            padding: 34px 18px;
            border-color: rgba(243, 217, 139, 0.44);
            background:
                radial-gradient(circle at 50% 0%, rgba(243, 217, 139, 0.16), transparent 34%),
                linear-gradient(180deg, rgba(16, 35, 62, 1), rgba(6, 16, 31, 1));
        }}
        .score-teams {{
            color: #d9e3f4;
            font-size: 1.08rem;
            margin-bottom: 12px;
            font-weight: 700;
        }}
        .score-value {{
            color: #fff;
            font-size: 5.6rem;
            line-height: 1;
            font-weight: 900;
            letter-spacing: 0;
            text-shadow: 0 8px 28px rgba(0,0,0,0.42);
        }}
        .score-note {{
            color: {GOLD_LIGHT};
            margin-top: 12px;
            font-weight: 800;
        }}
        .confidence-wrap {{
            background: linear-gradient(180deg, rgba(16, 35, 62, 0.98), rgba(8, 21, 40, 0.98));
            border: 1px solid rgba(214, 178, 94, 0.27);
            border-radius: 8px;
            padding: 18px;
            box-shadow: 0 14px 34px rgba(0,0,0,0.26);
            margin-bottom: 14px;
        }}
        .confidence-wrap .card-value {{
            margin-bottom: 10px;
        }}
        .smart-confidence .card-value {{
            font-size: 2.35rem;
            color: #fff;
        }}
        .confidence-stars {{
            color: {GOLD_LIGHT};
            font-size: 1.22rem;
            font-weight: 900;
            margin: 4px 0;
        }}
        .confidence-level {{
            color: {MUTED};
            font-size: 0.92rem;
            font-weight: 800;
            margin-bottom: 12px;
        }}
        .confidence-bar-bg {{
            width: 100%;
            height: 11px;
            border-radius: 999px;
            background: rgba(255,255,255,0.12);
            border: 1px solid rgba(214,178,94,0.18);
            overflow: hidden;
            margin: 8px 0 14px;
        }}
        .confidence-bar-fill {{
            height: 100%;
            border-radius: 999px;
            box-shadow: 0 0 18px rgba(255,255,255,0.12);
        }}
        .confidence-section-title {{
            color: {GOLD_LIGHT};
            font-size: 0.86rem;
            font-weight: 900;
            margin-top: 12px;
        }}
        .risk-title {{
            color: #fca5a5;
        }}
        .confidence-list {{
            margin: 7px 0 0 0;
            padding-left: 0;
            list-style: none;
            color: {MUTED};
            font-size: 0.84rem;
            line-height: 1.58;
        }}
        .risk-list {{
            color: #f4c7c7;
        }}
        div[data-testid="stProgress"] > div > div > div {{
            background: linear-gradient(90deg, {GOLD}, {GOLD_LIGHT});
        }}
        div[data-testid="stProgress"] > div > div {{
            background: rgba(255,255,255,0.13);
            border: 1px solid rgba(214,178,94,0.20);
        }}
        .risk-badge {{
            display: inline-flex;
            align-items: center;
            padding: 7px 11px;
            border-radius: 6px;
            font-weight: 800;
            font-size: 0.86rem;
            margin-left: 8px;
            white-space: nowrap;
        }}
        .risk-low {{
            background: rgba(46, 204, 113, 0.18);
            color: #8ff0b4;
            border: 1px solid rgba(46, 204, 113, 0.34);
        }}
        .risk-mid {{
            background: rgba(243, 217, 139, 0.18);
            color: {GOLD_LIGHT};
            border: 1px solid rgba(243, 217, 139, 0.36);
        }}
        .risk-high {{
            background: rgba(231, 76, 60, 0.18);
            color: #ff9a8f;
            border: 1px solid rgba(231, 76, 60, 0.34);
        }}
        .footer {{
            margin-top: 34px;
            padding: 20px 0 10px;
            border-top: 1px solid rgba(214, 178, 94, 0.2);
            color: {MUTED};
            font-size: 0.82rem;
            line-height: 1.7;
        }}
        .footer strong {{
            color: {GOLD_LIGHT};
        }}
        div[data-testid="stDataFrame"] {{
            border: 1px solid rgba(214, 178, 94, 0.18);
            border-radius: 8px;
            overflow-x: auto;
            overflow-y: hidden;
            max-width: 100%;
        }}
        div[data-testid="stDataFrame"] > div {{
            min-width: 100%;
        }}
        @media (max-width: 900px) {{
            .hero {{
                grid-template-columns: 1fr;
                padding: 24px;
            }}
            .hero-title {{ font-size: 2.35rem; }}
            .score-value {{ font-size: 3.35rem; }}
            .risk-badge {{
                margin-left: 0;
                margin-top: 8px;
            }}
        }}
        @media (max-width: 768px) {{
            .block-container {{
                padding: 1rem 0.75rem 5.2rem;
                max-width: 100%;
            }}
            .hero {{
                min-height: auto;
                padding: 18px;
                gap: 16px;
                margin-bottom: 16px;
            }}
            .hero-title {{
                font-size: 1.92rem;
                line-height: 1.18;
            }}
            .hero-copy {{
                font-size: 0.98rem;
                line-height: 1.62;
            }}
            .hero-visual {{
                min-height: 180px;
            }}
            .trophy {{
                font-size: 5rem;
            }}
            .page-title {{
                padding: 12px 0 8px;
                margin-bottom: 14px;
            }}
            .page-title h1 {{
                font-size: 1.55rem;
            }}
            .page-title p {{
                font-size: 0.95rem;
            }}
            .display-card,
            .confidence-wrap {{
                padding: 16px;
                margin-bottom: 12px;
            }}
            .card-value {{
                font-size: 1.45rem;
                overflow-wrap: anywhere;
            }}
            .score-card {{
                padding: 28px 12px;
                margin-left: auto;
                margin-right: auto;
            }}
            .score-teams {{
                font-size: 1rem;
                line-height: 1.45;
            }}
            .score-value {{
                font-size: 4rem;
                text-align: center;
            }}
            div[data-testid="stDataFrame"] {{
                max-width: 100%;
                overflow-x: auto;
                -webkit-overflow-scrolling: touch;
            }}
            div[data-testid="stDataFrame"] iframe {{
                max-width: 100%;
            }}
            [data-testid="stHorizontalBlock"] {{
                gap: 0.7rem;
            }}
            [data-testid="stSidebar"] [role="radiogroup"] label {{
                padding: 9px 10px;
                font-size: 0.96rem;
                line-height: 1.35;
            }}
            .stSelectbox label,
            .stTextInput label,
            .stRadio label {{
                font-size: 1rem !important;
            }}
            .stSelectbox div,
            .stTextInput input {{
                font-size: 1rem !important;
            }}
            .js-plotly-plot,
            .plot-container,
            .svg-container {{
                max-width: 100% !important;
            }}
            div[data-testid="column"] {{
                min-width: 0 !important;
            }}
            .footer {{
                padding-bottom: 42px;
                font-size: 0.78rem;
            }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(mobile_css(), unsafe_allow_html=True)


def page_header(title: str, caption: str) -> None:
    st.markdown(
        f"""
        <div class="page-title">
          <div class="timezone-badge">時區：台灣時間（UTC+8）</div>
          <h1>{html.escape(title)}</h1>
          <p>{html.escape(caption)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def display_card(label: str, value: str, note: str | None = None) -> None:
    note_html = f"<div class='card-note'>{html.escape(note)}</div>" if note else ""
    st.markdown(
        f"""
        <div class="display-card">
          <div class="card-label">{html.escape(label)}</div>
          <div class="card-value">{html.escape(value)}</div>
          {note_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def confidence_card(confidence: float) -> None:
    percent = max(0, min(100, confidence * 100))
    st.markdown(
        f"""
        <div class="confidence-wrap">
          <div class="card-label">信心分數</div>
          <div class="card-value">{percent:.1f}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.progress(int(round(percent)), text="模型信心指數")


def confidence_for_fixture(row: pd.Series, prediction, market_table: pd.DataFrame | None = None) -> ConfidenceResult:
    if market_table is None:
        market_table = market_probability_table(row, prediction)
    try:
        shots_df = prepare_xg_data(pd.read_csv("data/xg_shots.csv"))
    except Exception:
        shots_df = pd.DataFrame()
    return calculate_confidence(
        row,
        prediction,
        market_table,
        team_meta=team_meta_df,
        matches=matches_df,
        shots=shots_df,
    )


def smart_confidence_card(result: ConfidenceResult, compact: bool = False) -> None:
    percent = max(0, min(100, result.score))
    source_html = "".join(f"<li>✓ {html.escape(line)}</li>" for line in result.source_lines)
    risk_html = "".join(f"<li>⚠ {html.escape(line)}</li>" for line in result.risk_lines)
    details = ""
    if not compact:
        details = f"""
          <div class="confidence-section-title">信心來源</div>
          <ul class="confidence-list">{source_html}</ul>
          <div class="confidence-section-title risk-title">風險因素</div>
          <ul class="confidence-list risk-list">{risk_html}</ul>
        """
    st.markdown(
        f"""
        <div class="confidence-wrap smart-confidence">
          <div class="card-label">模型信心指數</div>
          <div class="card-value">{percent:.1f}%</div>
          <div class="confidence-stars">{html.escape(result.stars)}</div>
          <div class="confidence-level">{html.escape(result.level)} · {html.escape(result.favored_market)}</div>
          <div class="confidence-bar-bg">
            <div class="confidence-bar-fill" style="width:{percent:.1f}%; background:{result.color};"></div>
          </div>
          {details}
        </div>
        """,
        unsafe_allow_html=True,
    )


def risk_class(risk_level: str) -> str:
    if risk_level == "低":
        return "risk-low"
    if risk_level == "中":
        return "risk-mid"
    return "risk-high"


def risk_badge(risk_level: str) -> str:
    return f"<span class='risk-badge {risk_class(risk_level)}'>風險：{html.escape(risk_level)}</span>"


def footer() -> None:
    st.markdown(
        """
        <div class="footer">
          <strong>資料來源</strong>：Fjelstul World Cup Database、fixtures_real_2026.csv 真實賽程與展示用賠率資料<br>
          <strong>模型說明</strong>：Poisson + Elo + 近期狀態 + 世界盃歷史表現輔助權重<br>
          <strong>版本資訊</strong>：World Cup Data Platform V8.6 · Streamlit · 僅供資料分析參考，不保證賽果或獲利。
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data
def cached_data() -> tuple[pd.DataFrame, ...]:
    fixtures = load_fixtures()
    fixture_odds = load_fixture_with_odds()
    matches = load_historical_matches()
    team_meta = load_team_meta()
    team_history = load_worldcup_team_history()
    players = load_players()
    live_matches = load_live_matches()
    live_events = load_live_events()
    wc_matches = load_worldcup_matches()
    wc_team_stats = load_worldcup_team_stats()
    wc_champions = load_worldcup_champions()
    wc_top4 = load_worldcup_top4()
    wc_head_to_head = load_worldcup_head_to_head()
    return (
        fixtures,
        fixture_odds,
        matches,
        team_meta,
        team_history,
        players,
        live_matches,
        live_events,
        wc_matches,
        wc_team_stats,
        wc_champions,
        wc_top4,
        wc_head_to_head,
    )


(
    fixtures_df,
    fixture_odds_df,
    matches_df,
    team_meta_df,
    team_history_df,
    players_df,
    live_matches_df,
    live_events_df,
    wc_matches_df,
    wc_team_stats_df,
    wc_champions_df,
    wc_top4_df,
    wc_head_to_head_df,
) = cached_data()

TEAM_FLAG_MAP = dict(zip(team_meta_df["team"], team_meta_df["flag_emoji"]))
TEAM_NAME_MAP = dict(zip(team_meta_df["team"], team_meta_df["team_zh"]))

inject_theme()

PAGE_OPTIONS = [
    "世界盃情報中心",
    "世界盃賽程表",
    "賽程頁",
    "單場分析頁",
    "市場機率分析",
    "模型回測頁",
    "冠軍機率預測",
    "小組出線機率分析",
    "晉級機率分析",
    "世界盃模擬器",
    "Elo 世界排名",
    "即時賽況",
    "球隊資料庫",
    "球員資料庫",
    "國家隊資料中心",
    "歷史世界盃數據分析",
    "國家隊世界盃戰績",
    "歷史交手分析",
    "免責聲明頁",
]

PAGE_GROUPS = {
    "首頁": [
        "世界盃情報中心",
    ],
    "賽程中心": [
        "世界盃賽程表",
        "即時世界盃中心",
        "賽果更新中心",
        "即時賽況",
    ],
    "預測中心": [
        "單場分析頁",
        "球員影響分析",
        "冠軍機率預測",
        "晉級機率分析",
        "世界盃模擬器",
        "AI 世界盃模擬器",
        "市場機率分析",
        "世足玩法教學",
        "賠率試算中心",
        "全玩法預測中心",
        "AI 賽事分析報告",
        "AI 串關分析",
        "冠軍路徑模擬",
        "對戰比較中心",
    ],
    "資料中心": [
        "Elo 世界排名",
        "xG 模型分析",
        "球隊資料庫",
        "球員資料庫",
        "國家隊資料中心",
        "歷史世界盃數據分析",
        "國家隊世界盃戰績",
        "歷史交手分析",
        "足球數據百科",
        "足球術語教學",
    ],
    "AI 工具": [
        "AI 世界盃分析師",
    ],
}

for group_name, pages in PAGE_GROUPS.items():
    if any("Elo" in item for item in pages) and "xG 模型分析" not in pages:
        pages.insert(1, "xG 模型分析")
        break

PAGE_GROUPS = {
    "首頁": [
        "首頁",
    ],
    "賽程中心": [
        "世界盃賽程表",
        "即時世界盃中心",
        "賽果更新中心",
        "即時賽況",
    ],
    "預測中心": [
        "單場分析",
        "冠軍機率",
        "晉級機率",
        "世界盃模擬器",
    ],
    "資料中心": [
        "Elo 世界排名",
        "球隊資料庫",
        "球員資料庫",
        "歷史世界盃戰績",
        "歷史交手分析",
    ],
    "AI 工具": [
        "串關分析",
        "對戰比較",
        "賠率試算",
    ],
    "學習中心": [
        "世界盃玩法教學",
        "足球術語教學",
    ],
}

PAGE_ROUTE_ALIASES = {
    "首頁": "世界盃情報中心",
    "單場分析": "單場分析頁",
    "冠軍機率": "冠軍機率預測",
    "晉級機率": "晉級機率分析",
    "串關分析": "AI 串關分析",
    "對戰比較": "對戰比較中心",
    "賠率試算": "賠率試算中心",
    "世界盃玩法教學": "世足玩法教學",
    "歷史世界盃戰績": "歷史世界盃數據分析",
}

with st.sidebar:
    components.html(
        """
        <style>
          html, body { margin: 0; padding: 0; background: transparent; }
        </style>
        <script>
        (() => {
          const doc = window.parent && window.parent.document;
          if (!doc) return;

          const styleId = "v26-mobile-sidebar-style";
          if (!doc.getElementById(styleId)) {
            const style = doc.createElement("style");
            style.id = styleId;
            style.textContent = `
              .v26-mobile-sidebar-close {
                display: none;
              }
              @media (max-width: 768px) {
                .v26-mobile-sidebar-close {
                  display: block;
                  position: fixed;
                  z-index: 2147483647;
                  left: 12px;
                  top: 60px;
                  width: min(76vw, 292px);
                  min-height: 48px;
                  border-radius: 10px;
                  border: 1px solid rgba(214, 178, 94, 0.58);
                  background: linear-gradient(135deg, #d6b25e, #f3d98b);
                  color: #06101f;
                  box-shadow: 0 12px 28px rgba(0, 0, 0, 0.36);
                  font: 900 15px/1.2 "Noto Sans TC", "Segoe UI", sans-serif;
                  cursor: pointer;
                }
                body.v26-sidebar-force-closed [data-testid="stSidebar"] {
                  transform: translateX(-110%) !important;
                  left: 0 !important;
                  pointer-events: none !important;
                }
                body.v26-sidebar-force-closed [data-testid="stAppViewContainer"],
                body.v26-sidebar-force-closed [data-testid="stAppViewContainer"] > .main {
                  margin-left: 0 !important;
                  width: 100vw !important;
                  max-width: 100vw !important;
                }
                body.v26-sidebar-force-closed .main .block-container {
                  width: 100% !important;
                  max-width: 100% !important;
                }
                body.v26-sidebar-force-closed .v26-mobile-sidebar-close {
                  display: none !important;
                }
              }
            `;
            doc.head.appendChild(style);
          }

          let button = doc.getElementById("v26-mobile-sidebar-close");
          if (!button) {
            button = doc.createElement("button");
            button.id = "v26-mobile-sidebar-close";
            button.className = "v26-mobile-sidebar-close";
            button.type = "button";
            button.textContent = "關閉選單 / 收合側欄";
            doc.body.appendChild(button);
          }

          const findNativeClose = () => {
            const selectors = [
              'button[aria-label="Close sidebar"]',
              'button[title="Close sidebar"]',
              'button[aria-label="Collapse sidebar"]',
              'button[title="Collapse sidebar"]',
              '[data-testid="stSidebarCollapseButton"] button'
            ];
            for (const selector of selectors) {
              const target = doc.querySelector(selector);
              if (target) return target;
            }
            return Array.from(doc.querySelectorAll("button")).find((candidate) => {
              const label = `${candidate.getAttribute("aria-label") || ""} ${candidate.title || ""} ${candidate.innerText || ""}`.toLowerCase();
              return label.includes("close sidebar")
                || label.includes("collapse sidebar")
                || label.includes("keyboard_double_arrow_left")
                || label.includes("chevron_left");
            });
          };

          const sidebarStillOpen = () => {
            const sidebar = doc.querySelector('[data-testid="stSidebar"]');
            if (!sidebar) return false;
            const rect = sidebar.getBoundingClientRect();
            return rect.width > 0 && rect.right > 56 && rect.left > -80;
          };

          const syncButton = () => {
            if (window.parent.innerWidth > 768 || doc.body.classList.contains("v26-sidebar-force-closed")) {
              button.style.display = "none";
              return;
            }
            button.style.display = "block";
          };

          const closeSidebar = () => {
            doc.body.classList.remove("v26-sidebar-force-closed");
            const nativeClose = findNativeClose();
            if (nativeClose) {
              nativeClose.click();
            }
            doc.body.classList.add("v26-sidebar-force-closed");
            window.setTimeout(() => {
              if (sidebarStillOpen()) {
                doc.body.classList.add("v26-sidebar-force-closed");
              }
              syncButton();
            }, 240);
          };

          const inlineClose = `
            (function(){
              document.body.classList.add('v26-sidebar-force-closed');
              const selectors = [
                'button[aria-label="Close sidebar"]',
                'button[title="Close sidebar"]',
                'button[aria-label="Collapse sidebar"]',
                'button[title="Collapse sidebar"]',
                '[data-testid="stSidebarCollapseButton"] button'
              ];
              let target = selectors.map(selector => document.querySelector(selector)).find(Boolean);
              if (!target) {
                target = Array.from(document.querySelectorAll('button')).find(button => {
                  const label = ((button.getAttribute('aria-label') || '') + ' ' + (button.title || '') + ' ' + (button.innerText || '')).toLowerCase();
                  return label.includes('close sidebar')
                    || label.includes('collapse sidebar')
                    || label.includes('keyboard_double_arrow_left')
                    || label.includes('chevron_left');
                });
              }
              if (target) target.click();
            })();
          `;
          button.setAttribute("onclick", inlineClose);
          button.addEventListener("click", closeSidebar);
          syncButton();
          window.setTimeout(syncButton, 600);
          window.setInterval(syncButton, 1400);

          if (!doc.body.dataset.v26SidebarListener) {
            doc.body.dataset.v26SidebarListener = "1";
            doc.addEventListener("click", (event) => {
              const target = event.target && event.target.closest ? event.target.closest("button") : null;
              if (!target) return;
              const label = `${target.getAttribute("aria-label") || ""} ${target.title || ""} ${target.innerText || ""}`.toLowerCase();
              if (label.includes("open sidebar")
                || label.includes("expand sidebar")
                || label.includes("keyboard_double_arrow_right")
                || label.includes("chevron_right")) {
                doc.body.classList.remove("v26-sidebar-force-closed");
                window.setTimeout(syncButton, 280);
              }
            }, true);
          }
        })();
        </script>
        """,
        height=0,
    )

selected_group = st.sidebar.selectbox("功能分類", list(PAGE_GROUPS.keys()))
page = st.sidebar.radio("頁面", PAGE_GROUPS[selected_group])
page = PAGE_ROUTE_ALIASES.get(page, page)
st.sidebar.divider()
st.sidebar.caption("世界盃資料平台：賽程、預測、球隊、球員、即時足球資料")


@st.cache_data(show_spinner=False)
def cached_tournament_simulation(
    fixtures: pd.DataFrame,
    team_meta: pd.DataFrame,
    worldcup_team_stats: pd.DataFrame,
    recent_matches: pd.DataFrame,
    simulations: int,
) -> pd.DataFrame:
    return run_tournament_simulation(
        fixtures,
        team_meta,
        worldcup_team_stats,
        recent_matches,
        simulations=simulations,
    )


@st.cache_data(show_spinner=False)
def cached_dynamic_tournament_simulation(
    fixtures: pd.DataFrame,
    team_meta: pd.DataFrame,
    worldcup_team_stats: pd.DataFrame,
    recent_matches: pd.DataFrame,
    results: pd.DataFrame,
    simulations: int,
) -> pd.DataFrame:
    return rerun_dynamic_simulation(
        fixtures,
        team_meta,
        worldcup_team_stats,
        recent_matches,
        results,
        simulations=simulations,
    )


V7_SIMULATIONS = 10000


def v7_simulation() -> pd.DataFrame:
    results = load_match_results()
    if not results.empty:
        return cached_dynamic_tournament_simulation(
            fixture_odds_df,
            team_meta_df,
            wc_team_stats_df,
            matches_df,
            results,
            V7_SIMULATIONS,
        )
    return cached_tournament_simulation(
        fixture_odds_df,
        team_meta_df,
        wc_team_stats_df,
        matches_df,
        V7_SIMULATIONS,
    )


@st.cache_data(ttl=60, show_spinner=False)
def cached_v29_live_matches() -> object:
    return fetch_live_matches(st.secrets)


@st.cache_data(ttl=120, show_spinner=False)
def cached_v29_group_standings() -> object:
    return fetch_group_standings(st.secrets)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_v29_team_stats() -> object:
    return fetch_team_stats(st.secrets)


@st.cache_data(ttl=21600, show_spinner=False)
def cached_v29_player_stats() -> object:
    return fetch_player_stats(st.secrets)


def selected_fixture(label: str = "選擇比賽") -> pd.Series:
    labels = {
        f"{taipei_time_text(row.datetime_taipei)} | {team_name(row.home_team)} vs {team_name(row.away_team)}": row.match_id
        for row in fixture_odds_df.itertuples()
    }
    selected_label = st.selectbox(label, list(labels.keys()))
    match_id = labels[selected_label]
    return fixture_odds_df[fixture_odds_df["match_id"] == match_id].iloc[0]


def render_ai_match_report(
    row: pd.Series,
    prediction,
    market_table: pd.DataFrame | None = None,
    confidence_result: ConfidenceResult | None = None,
) -> None:
    st.subheader("AI 賽事分析報告")
    try:
        if confidence_result is None:
            confidence_result = confidence_for_fixture(row, prediction, market_table)
        lines = generate_match_report(
            row,
            prediction,
            matches_df,
            team_meta_df,
            market_table=market_table,
            confidence_result=confidence_result,
        )
    except Exception:
        lines = [
            "目前資料不足，僅提供基礎分析。",
            "本場仍可參考比分預測、勝平負機率與歷史資料，但 AI 文字報告暫以保守描述呈現。",
            "本報告僅供機率分析，不構成下注建議。",
        ]
    st.markdown(
        "<div class='display-card'>"
        + "".join(f"<div class='card-note'>{html.escape(line)}</div>" for line in lines)
        + "</div>",
        unsafe_allow_html=True,
    )


def render_pre_match_analysis_card(row: pd.Series, prediction) -> None:
    st.subheader("AI 賽前對戰分析")
    try:
        shots = prepare_xg_data(pd.read_csv("data/xg_shots.csv"))
    except Exception:
        shots = pd.DataFrame()

    try:
        player_source = player_database(players_df, team_meta_df)
    except Exception:
        player_source = players_df.copy()

    try:
        analysis = build_pre_match_analysis(
            row,
            prediction,
            team_meta=team_meta_df,
            matches=matches_df,
            history=team_history_df,
            players=player_source,
            shots=shots,
            team_label_func=team_name,
        )
    except Exception:
        st.info("目前資料不足，僅提供基礎分析。")
        return

    summary = analysis["summary"]
    home_label = analysis["home_label"]
    away_label = analysis["away_label"]
    home_goals, away_goals = summary["predicted_score"]

    cols = st.columns(4)
    with cols[0]:
        display_card("主隊勝率", format_percent(float(summary["home_win_probability"])), home_label)
    with cols[1]:
        display_card("平手機率", format_percent(float(summary["draw_probability"])), "模型推估")
    with cols[2]:
        display_card("客隊勝率", format_percent(float(summary["away_win_probability"])), away_label)
    with cols[3]:
        display_card("分析信心", f"{float(summary['analysis_confidence']):.1f}%", f"風險：{summary['risk_level']}")

    st.markdown(
        f"""
        <div class="display-card">
          <div class="card-label">預測比分</div>
          <div class="card-value">{html.escape(home_label)} {home_goals}：{away_goals} {html.escape(away_label)}</div>
          <div class="card-note">AI 預測結果：{html.escape(home_label)}勝率 {float(summary["home_win_probability"]) * 100:.1f}%、
          {html.escape(away_label)}勝率 {float(summary["away_win_probability"]) * 100:.1f}%、平局 {float(summary["draw_probability"]) * 100:.1f}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    comparison = analysis["comparison"].copy()
    st.dataframe(comparison, use_container_width=True, hide_index=True)

    if analysis["is_estimated"]:
        notes = "、".join(analysis["notes"][:3]) if analysis["notes"] else "部分資料使用 fallback"
        st.caption(f"部分資料推估：{notes}")

    st.markdown(
        "<div class='display-card'>"
        + "".join(f"<div class='card-note'>• {html.escape(line)}</div>" for line in analysis["lines"])
        + "<div class='card-note'><strong>本分析由模型根據現有資料推估，僅供參考，非保證賽果。</strong></div>"
        + "</div>",
        unsafe_allow_html=True,
    )


def _parlay_candidates() -> pd.DataFrame:
    def predict_for(home: str, away: str):
        return predict_match(matches_df, home, away, wc_team_stats_df)

    return build_match_candidates(
        fixture_odds_df,
        market_probability_table,
        predict_for,
        team_name_func=team_name,
    )


def _format_parlay_table(table: pd.DataFrame) -> pd.DataFrame:
    if table is None or table.empty:
        return pd.DataFrame()
    display = table.copy()
    if "單場機率" in display.columns:
        display["單場機率"] = pd.to_numeric(display["單場機率"], errors="coerce").fillna(0).map(format_percent)
    if "預估命中率" in display.columns:
        display["預估命中率"] = pd.to_numeric(display["預估命中率"], errors="coerce").fillna(0).map(format_percent)
    for column in ["單場賠率", "總賠率"]:
        if column in display.columns:
            display[column] = pd.to_numeric(display[column], errors="coerce").fillna(1.01).map(lambda value: f"{value:.2f}")
    return display


def render_ai_parlay_analysis(limit: int = 6) -> None:
    st.subheader("AI 串關分析")
    st.caption("以下為機率候選組合，僅供機率分析，不構成下注建議。串關場數越多，整體命中率通常越低。")
    candidates = _parlay_candidates()
    if candidates.empty:
        st.info("目前資料不足，暫無可整理的串關候選組合。")
        return

    st.markdown("**單場候選池**")
    st.dataframe(_format_parlay_table(candidates.head(limit)), use_container_width=True, hide_index=True)

    for size in [2, 3]:
        table = build_parlay_combinations(candidates, size=size, max_rows=limit)
        st.markdown(f"**{size} 串 1 候選組合**")
        if table.empty:
            st.info(f"目前資料不足，暫無 {size} 串 1 候選組合。")
            continue
        st.dataframe(_format_parlay_table(table), use_container_width=True, hide_index=True)
        st.caption(parlay_summary_text(table))


def fixtures_with_flags(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()
    output["taiwan_time"] = output["datetime_taipei"].map(taipei_time_text)
    output["matchup_display"] = output.apply(
        lambda row: f"{team_name(row['home_team'])} vs {team_name(row['away_team'])}",
        axis=1,
    )
    return output.rename(
        columns={
            "match_id": "賽事 ID",
            "taiwan_time": "台灣時間",
            "stage": "階段",
            "matchup_display": "對戰",
            "venue": "場地",
            "home_odds": "主勝賠率",
            "draw_odds": "和局賠率",
            "away_odds": "客勝賠率",
        }
    )[
        ["賽事 ID", "台灣時間", "階段", "對戰", "場地", "主勝賠率", "和局賠率", "客勝賠率"]
    ]


def team_history_row(team: str) -> dict:
    row = team_history_df[team_history_df["team"] == team]
    if row.empty:
        return {
            "team": team,
            "tournaments_played": 0,
            "best_finish": "No data",
            "wins": 0,
            "draws": 0,
            "losses": 0,
            "goals_for": 0,
            "goals_against": 0,
            "win_rate": 0.0,
        }
    return row.iloc[0].to_dict()


def render_worldcup_history_block(home_team: str, away_team: str) -> None:
    st.subheader("歷史戰績")
    history = pd.DataFrame([team_history_row(home_team), team_history_row(away_team)])
    history["team"] = history["team"].map(team_name)
    history["win_rate"] = history["win_rate"].map(format_percent)
    st.dataframe(
        history.rename(
            columns={
                "team": "球隊",
                "tournaments_played": "參賽屆數",
                "best_finish": "最佳成績",
                "wins": "勝場",
                "draws": "平手",
                "losses": "敗場",
                "goals_for": "進球",
                "goals_against": "失球",
                "win_rate": "勝率",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )


def key_players_for(team: str) -> pd.DataFrame:
    data = player_database(players_df, team_meta_df)
    team_players = data[data["team"] == team].copy()
    sort_columns = [
        column
        for column in ["recent_form_rating", "goal_rate", "national_goals", "national_caps"]
        if column in team_players.columns
    ]
    if sort_columns:
        return team_players.sort_values(sort_columns, ascending=False).head(5)
    return team_players.head(5)


def h2h_summary(home_team: str, away_team: str) -> dict:
    direct = wc_head_to_head_df[
        (wc_head_to_head_df["team_a"] == home_team)
        & (wc_head_to_head_df["team_b"] == away_team)
    ]
    reverse = wc_head_to_head_df[
        (wc_head_to_head_df["team_a"] == away_team)
        & (wc_head_to_head_df["team_b"] == home_team)
    ]
    if not direct.empty:
        row = direct.iloc[0]
        return {
            "兩隊": f"{team_name(home_team)} vs {team_name(away_team)}",
            "歷史交手次數": int(row["matches"]),
            "主隊勝場": int(row["team_a_wins"]),
            "平手": int(row["draws"]),
            "客隊勝場": int(row["team_b_wins"]),
            "最近一次交手年份": int(row["last_meeting_year"]),
        }
    if not reverse.empty:
        row = reverse.iloc[0]
        return {
            "兩隊": f"{team_name(home_team)} vs {team_name(away_team)}",
            "歷史交手次數": int(row["matches"]),
            "主隊勝場": int(row["team_b_wins"]),
            "平手": int(row["draws"]),
            "客隊勝場": int(row["team_a_wins"]),
            "最近一次交手年份": int(row["last_meeting_year"]),
        }
    return {
        "兩隊": f"{team_name(home_team)} vs {team_name(away_team)}",
        "歷史交手次數": 0,
        "主隊勝場": 0,
        "平手": 0,
        "客隊勝場": 0,
        "最近一次交手年份": "無 2002~2022 世界盃交手資料",
    }


def render_h2h_summary_block(home_team: str, away_team: str) -> None:
    st.subheader("歷史交手摘要")
    st.dataframe(pd.DataFrame([h2h_summary(home_team, away_team)]), use_container_width=True, hide_index=True)


def _v26_best_finish_from_stats(summary: dict) -> str:
    if not summary.get("has_data", False):
        return "資料待補"
    titles = pd.to_numeric(summary.get("titles"), errors="coerce")
    top4 = pd.to_numeric(summary.get("top4_finishes"), errors="coerce")
    if pd.notna(titles) and titles > 0:
        return "Champion"
    if pd.notna(top4) and top4 > 0:
        return "Top 4"
    return "資料待補"


def team_history_row(team: str) -> dict:
    summary = team_summary(wc_team_stats_df, team)
    canonical = resolve_team_name(team)
    history_row = find_team_row(team_history_df, canonical)
    best_finish = "資料待補"
    if history_row is not None and "best_finish" in history_row.index:
        best_finish = history_row.get("best_finish") or "資料待補"
    elif summary.get("has_data", False):
        best_finish = _v26_best_finish_from_stats(summary)

    if not summary.get("has_data", False):
        return {
            "team": canonical,
            "tournaments_played": "資料待補",
            "best_finish": "資料待補",
            "wins": "資料待補",
            "draws": "資料待補",
            "losses": "資料待補",
            "goals_for": "資料待補",
            "goals_against": "資料待補",
            "win_rate": 0.0,
            "has_data": False,
        }

    return {
        "team": canonical,
        "tournaments_played": int(pd.to_numeric(summary.get("tournaments_played"), errors="coerce") or 0),
        "best_finish": best_finish,
        "wins": int(pd.to_numeric(summary.get("wins"), errors="coerce") or 0),
        "draws": int(pd.to_numeric(summary.get("draws"), errors="coerce") or 0),
        "losses": int(pd.to_numeric(summary.get("losses"), errors="coerce") or 0),
        "goals_for": int(pd.to_numeric(summary.get("goals_for"), errors="coerce") or 0),
        "goals_against": int(pd.to_numeric(summary.get("goals_against"), errors="coerce") or 0),
        "win_rate": float(pd.to_numeric(summary.get("win_rate"), errors="coerce") or 0),
        "has_data": True,
    }


def render_worldcup_history_block(home_team: str, away_team: str) -> None:
    st.subheader("歷史世界盃戰績")
    history = pd.DataFrame([team_history_row(home_team), team_history_row(away_team)])
    history["team"] = history["team"].map(lambda team: team_name(team))
    history["win_rate"] = history.apply(
        lambda row: format_percent(row["win_rate"]) if row.get("has_data", False) else "資料待補",
        axis=1,
    )
    display = history[
        [
            "team",
            "tournaments_played",
            "best_finish",
            "wins",
            "draws",
            "losses",
            "goals_for",
            "goals_against",
            "win_rate",
        ]
    ].rename(
        columns={
            "team": "球隊",
            "tournaments_played": "參賽屆數",
            "best_finish": "最佳成績",
            "wins": "勝場",
            "draws": "平手",
            "losses": "敗場",
            "goals_for": "進球",
            "goals_against": "失球",
            "win_rate": "勝率",
        }
    )
    st.dataframe(display, use_container_width=True, hide_index=True)


def h2h_summary(home_team: str, away_team: str) -> dict:
    home = resolve_team_name(home_team)
    away = resolve_team_name(away_team)
    record = head_to_head_record(wc_head_to_head_df, home, away)
    if record.empty or not bool(record.iloc[0].get("has_data", False)):
        return {
            "對戰": f"{team_name(home_team)} vs {team_name(away_team)}",
            "世界盃交手次數": 0,
            "主隊勝場": 0,
            "平手": 0,
            "客隊勝場": 0,
            "最近一次交手年份": "1930-2022 世界盃無交手紀錄",
        }

    row = record.iloc[0]
    team_a = resolve_team_name(row["team_a"])
    if team_a == home:
        home_wins = int(row["team_a_wins"])
        away_wins = int(row["team_b_wins"])
    else:
        home_wins = int(row["team_b_wins"])
        away_wins = int(row["team_a_wins"])
    return {
        "對戰": f"{team_name(home_team)} vs {team_name(away_team)}",
        "世界盃交手次數": int(row["matches"]),
        "主隊勝場": home_wins,
        "平手": int(row["draws"]),
        "客隊勝場": away_wins,
        "最近一次交手年份": int(row["last_meeting_year"]),
    }


def _history_number(row: pd.Series, column: str, fallback: float = 0) -> float:
    if row is None or column not in row.index:
        return fallback
    value = pd.to_numeric(row.get(column), errors="coerce")
    return fallback if pd.isna(value) else float(value)


def team_history_row(team: str) -> dict:
    canonical = resolve_team_name(team)
    history_row = find_team_row(team_history_df, canonical)
    if history_row is not None:
        return {
            "team": canonical,
            "tournaments_played": int(_history_number(history_row, "appearances", _history_number(history_row, "tournaments_played"))),
            "best_finish": history_row.get("best_finish") or "資料待補",
            "wins": int(_history_number(history_row, "wins")),
            "draws": int(_history_number(history_row, "draws")),
            "losses": int(_history_number(history_row, "losses")),
            "goals_for": int(_history_number(history_row, "goals_for")),
            "goals_against": int(_history_number(history_row, "goals_against")),
            "win_rate": _history_number(history_row, "win_rate"),
            "has_data": True,
        }

    summary = team_summary(wc_team_stats_df, canonical)
    if summary.get("has_data", False):
        return {
            "team": canonical,
            "tournaments_played": int(pd.to_numeric(summary.get("tournaments_played"), errors="coerce") or 0),
            "best_finish": _v26_best_finish_from_stats(summary),
            "wins": int(pd.to_numeric(summary.get("wins"), errors="coerce") or 0),
            "draws": int(pd.to_numeric(summary.get("draws"), errors="coerce") or 0),
            "losses": int(pd.to_numeric(summary.get("losses"), errors="coerce") or 0),
            "goals_for": int(pd.to_numeric(summary.get("goals_for"), errors="coerce") or 0),
            "goals_against": int(pd.to_numeric(summary.get("goals_against"), errors="coerce") or 0),
            "win_rate": float(pd.to_numeric(summary.get("win_rate"), errors="coerce") or 0),
            "has_data": True,
        }

    return {
        "team": canonical,
        "tournaments_played": "資料待補",
        "best_finish": "資料待補",
        "wins": "資料待補",
        "draws": "資料待補",
        "losses": "資料待補",
        "goals_for": "資料待補",
        "goals_against": "資料待補",
        "win_rate": 0.0,
        "has_data": False,
    }


def render_worldcup_history_block(home_team: str, away_team: str) -> None:
    st.subheader("歷史世界盃戰績")
    history = pd.DataFrame([team_history_row(home_team), team_history_row(away_team)])
    history["team"] = history["team"].map(lambda team: team_name(team))
    history["win_rate"] = history.apply(
        lambda row: format_percent(row["win_rate"]) if row.get("has_data", False) else "資料待補",
        axis=1,
    )
    display = history[
        [
            "team",
            "tournaments_played",
            "best_finish",
            "wins",
            "draws",
            "losses",
            "goals_for",
            "goals_against",
            "win_rate",
        ]
    ].rename(
        columns={
            "team": "球隊",
            "tournaments_played": "參賽屆數",
            "best_finish": "最佳成績",
            "wins": "勝場",
            "draws": "平手",
            "losses": "敗場",
            "goals_for": "進球",
            "goals_against": "失球",
            "win_rate": "勝率",
        }
    )
    st.dataframe(display, use_container_width=True, hide_index=True)


def render_key_players_block(home_team: str, away_team: str) -> None:
    st.subheader("關鍵球員")
    players = pd.concat([key_players_for(home_team), key_players_for(away_team)], ignore_index=True)
    st.caption(f"主隊關鍵球員：{team_name(home_team)} ｜ 客隊關鍵球員：{team_name(away_team)}")
    if players.empty:
        st.info("目前尚未匯入該球隊球員資料")
        return
    if "goal_rate" not in players.columns:
        safe_caps = pd.to_numeric(players.get("national_caps", 0), errors="coerce").replace(0, pd.NA)
        players["goal_rate"] = (
            pd.to_numeric(players.get("national_goals", 0), errors="coerce") / safe_caps
        ).fillna(0.0)
    if "recent_form_rating" not in players.columns:
        players["recent_form_rating"] = 0.5
    players["team_display"] = players["team"].map(team_name)
    players["goal_rate_display"] = players["goal_rate"].map(format_percent)
    st.dataframe(
        players[
            [
                "team_display",
                "player_name",
                "position",
                "national_goals",
                "national_caps",
                "goal_rate_display",
                "recent_form_rating",
            ]
        ].rename(
            columns={
                "team_display": "球隊",
                "player_name": "球員姓名",
                "position": "位置",
                "national_goals": "國家隊進球",
                "national_caps": "國家隊出賽",
                "goal_rate_display": "進球率",
                "recent_form_rating": "近況評分",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    chart = px.bar(
        players,
        x="player_name",
        y="goal_rate",
        color="team_display",
        barmode="group",
        labels={"player_name": "球員", "goal_rate": "進球率", "team_display": "球隊"},
        color_discrete_sequence=[GOLD, "#8aa0c3"],
    )
    chart.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color=INK)
    st.plotly_chart(chart, use_container_width=True)


def betting_context(row: pd.Series) -> tuple[str, str, str]:
    home_history = team_history_row(row["home_team"])
    away_history = team_history_row(row["away_team"])
    history_gap = float(home_history["win_rate"]) - float(away_history["win_rate"])
    if abs(history_gap) < 0.05:
        history_text = "歷史戰績接近，世界盃經驗差距不明顯。"
    else:
        leader = row["home_team"] if history_gap > 0 else row["away_team"]
        history_text = f"{flag(leader)} {leader} 的 2002~2022 世界盃勝率較高，歷史表現略占優。"

    home_players = key_players_for(row["home_team"])
    away_players = key_players_for(row["away_team"])
    home_rating = home_players["recent_form_rating"].mean() if not home_players.empty else 0
    away_rating = away_players["recent_form_rating"].mean() if not away_players.empty else 0
    if abs(home_rating - away_rating) < 0.25:
        player_text = "雙方關鍵球員近況評分接近，球員面影響偏中性。"
    else:
        leader = row["home_team"] if home_rating > away_rating else row["away_team"]
        player_text = f"{flag(leader)} {leader} 關鍵球員近況評分較佳，但僅作輔助說明。"

    recent_matches = matches_df[
        matches_df["home_team"].isin([row["home_team"], row["away_team"]])
        | matches_df["away_team"].isin([row["home_team"], row["away_team"]])
    ].tail(6)
    total_goals = int(recent_matches["home_goals"].sum() + recent_matches["away_goals"].sum())
    recent_text = f"近期樣本共 {len(recent_matches)} 場、總進球 {total_goals}，用於輔助理解攻守狀態。"
    return history_text, player_text, recent_text


def prediction_cards(row: pd.Series, compact: bool = False) -> None:
    prediction = predict_match(matches_df, row["home_team"], row["away_team"], wc_team_stats_df)
    signals = analyze_1x2(prediction, row["home_odds"], row["draw_odds"], row["away_odds"])
    best_signal = max(signals, key=lambda signal: signal.edge)
    market_table = market_probability_table(row, prediction)
    confidence_result = confidence_for_fixture(row, prediction, market_table)
    teams = matchup_text(row)

    st.markdown(
        f"""
        <div class="display-card score-card">
          <div class="score-teams">{html.escape(teams)}</div>
          <div class="score-value">{prediction.predicted_home_goals} : {prediction.predicted_away_goals}</div>
          <div class="score-note">預測比分｜{html.escape(fixture_time_text(row))} 台灣時間</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cols = st.columns(4)
    with cols[0]:
        display_card("主勝機率", format_percent(prediction.home_win_probability))
    with cols[1]:
        display_card("和局機率", format_percent(prediction.draw_probability))
    with cols[2]:
        display_card("客勝機率", format_percent(prediction.away_win_probability))
    with cols[3]:
        smart_confidence_card(confidence_result, compact=compact)

    st.markdown(
        f"""
        <div class="display-card">
          <div class="card-label">投注觀察</div>
          <div class="card-value">{html.escape(best_signal.market)} · {html.escape(best_signal.recommendation)}
            {risk_badge(best_signal.risk_level)}
          </div>
          <div class="card-note">優勢值：{format_percent(best_signal.edge)} · 賠率：{best_signal.odds:.2f}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if compact:
        st.caption("快速分析僅供資料參考，詳細內容請至單場分析頁與市場機率分析。")


def dashboard_page() -> None:
    metrics = backtest(matches_df, wc_team_stats_df, players_df)
    upcoming_count = len(fixtures_df)
    team_count = len(set(fixtures_df["home_team"]).union(set(fixtures_df["away_team"])))

    st.markdown(
        """
        <section class="hero">
          <div>
            <div class="hero-kicker">WORLD CUP PREDICTION INTELLIGENCE</div>
            <div class="timezone-badge">時區：台灣時間（UTC+8）</div>
            <div class="hero-title">世足智慧預測中心</div>
            <div class="hero-copy">
              深藍金色世界盃儀表板，整合賽程、比分預測、勝平負機率、信心分數、
              投注風險與 2002~2022 世界盃歷史資料，打造完整世界盃資料平台。
            </div>
          </div>
          <div class="hero-visual">
            <div class="trophy">🏆</div>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )
    disclaimer_box()

    cols = st.columns(4)
    with cols[0]:
        display_card("待分析賽事", str(upcoming_count), "世界盃賽程")
    with cols[1]:
        display_card("涵蓋隊伍", str(team_count), "展示用國家隊")
    with cols[2]:
        display_card("歷史世界盃場次", str(len(wc_matches_df)), "2002~2022")
    with cols[3]:
        display_card("回測命中率", format_percent(metrics["accuracy"]), "測試資料")

    st.subheader("近期賽程")
    st.dataframe(
        fixtures_with_flags(fixture_odds_df).drop(columns=["主勝賠率", "和局賠率", "客勝賠率"]),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("2002~2022 世界盃冠軍")
    st.dataframe(wc_champions_df, use_container_width=True, hide_index=True)

    goals = wc_matches_df.assign(total_goals=wc_matches_df["home_goals"] + wc_matches_df["away_goals"])
    chart = px.bar(
        goals.groupby("tournament_year", as_index=False)["total_goals"].sum(),
        x="tournament_year",
        y="total_goals",
        labels={"tournament_year": "年份", "total_goals": "總進球"},
        color_discrete_sequence=[GOLD],
    )
    chart.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color=INK)
    st.plotly_chart(chart, use_container_width=True)


def fixtures_page() -> None:
    page_header("世界盃賽程表", "2026 世界盃賽程、台灣時間、階段篩選與球隊搜尋")
    st.caption("若官方完整分組或淘汰賽對戰尚未齊全，本頁使用本地展示資料／模擬資料補足頁面展示。")
    cols = st.columns([1, 1.4])
    stage = cols[0].selectbox("篩選階段", ["全部"] + sorted(fixtures_df["stage"].unique().tolist()))
    keyword = cols[1].text_input("搜尋球隊", "")
    filtered = fixture_odds_df if stage == "全部" else fixture_odds_df[fixture_odds_df["stage"] == stage]
    if keyword.strip():
        term = keyword.strip()
        filtered = filtered[
            filtered["home_team"].str.contains(term, case=False, na=False)
            | filtered["away_team"].str.contains(term, case=False, na=False)
        ]
    st.dataframe(fixtures_with_flags(filtered), use_container_width=True, hide_index=True)
    st.subheader("快速分析")
    row = selected_fixture("選擇要分析的比賽")
    prediction_cards(row, compact=True)


def match_results_update_page() -> None:
    page_header("賽果更新中心", "已完成賽事、小組積分、Elo 更新與重新預測")
    st.caption("本頁可用本地賽果或手動輸入比分測試後續機率變動；資料不足時使用 fallback，不影響既有賽程資料。")
    base_results = load_match_results()
    manual_results = st.session_state.get("manual_match_results", pd.DataFrame())
    results = combine_results(base_results, manual_results)

    if results.empty:
        st.info("目前尚未匯入已完成賽事。可在下方手動輸入賽果測試重新計算。")
    else:
        completed_display = results.copy()
        completed_display["比分"] = completed_display["home_score"].astype(str) + " : " + completed_display["away_score"].astype(str)
        st.subheader("已完成賽事")
        st.dataframe(
            completed_display[["match_id", "date", "group", "home_team", "away_team", "比分", "status"]].rename(
                columns={
                    "match_id": "比賽 ID",
                    "date": "日期",
                    "group": "小組",
                    "home_team": "主隊",
                    "away_team": "客隊",
                    "status": "狀態",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("小組積分表")
        standings = group_standings(results)
        if standings.empty:
            st.info("目前小組積分資料不足。")
        else:
            st.dataframe(
                standings.rename(
                    columns={
                        "group": "小組",
                        "team": "球隊",
                        "played": "賽",
                        "wins": "勝",
                        "draws": "和",
                        "losses": "敗",
                        "gf": "進球",
                        "ga": "失球",
                        "gd": "淨勝球",
                        "points": "積分",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )

    st.subheader("手動輸入賽果測試")
    labels = {
        f"{taipei_time_text(row.datetime_taipei)} | {team_name(row.home_team)} vs {team_name(row.away_team)}": row.match_id
        for row in fixture_odds_df.itertuples()
    }
    selected_label = st.selectbox("選擇比賽", list(labels.keys()), key="dynamic_result_match")
    selected_match_id = labels[selected_label]
    row = fixture_odds_df[fixture_odds_df["match_id"] == selected_match_id].iloc[0]
    score_cols = st.columns(2)
    home_score = score_cols[0].number_input(f"{team_name(row['home_team'])} 比分", min_value=0, max_value=12, value=2, step=1)
    away_score = score_cols[1].number_input(f"{team_name(row['away_team'])} 比分", min_value=0, max_value=12, value=0, step=1)

    if st.button("重新計算機率", type="primary", use_container_width=True):
        manual_row = pd.DataFrame(
            [
                {
                    "match_id": str(row["match_id"]),
                    "date": str(row.get("date", ""))[:10],
                    "group": str(row.get("stage", "未分組")),
                    "home_team": row["home_team"],
                    "away_team": row["away_team"],
                    "home_score": int(home_score),
                    "away_score": int(away_score),
                    "status": "Completed",
                }
            ]
        )
        st.session_state["manual_match_results"] = combine_results(base_results, manual_results, manual_row)
        st.success("已套用手動賽果，並重新計算下方機率。")

    current_results = combine_results(base_results, st.session_state.get("manual_match_results", pd.DataFrame()))
    if current_results.empty:
        return

    updated_meta = update_elo(team_meta_df, current_results)
    elo_view = updated_meta.copy()
    elo_view["team_display"] = elo_view.get("flag_emoji", "").astype(str) + " " + elo_view.get("team_zh", elo_view["team"]).astype(str)
    changed_elo = elo_view[elo_view["elo_change"].abs() > 0.01].sort_values("elo_change", ascending=False)
    st.subheader("Elo 更新")
    if changed_elo.empty:
        st.info("目前沒有可顯示的 Elo 變化。")
    else:
        st.dataframe(
            changed_elo[["team_display", "original_elo", "updated_elo", "elo_change"]].rename(
                columns={"team_display": "球隊", "original_elo": "原始 Elo", "updated_elo": "更新後 Elo", "elo_change": "Elo 變化"}
            ),
            use_container_width=True,
            hide_index=True,
        )

    with st.spinner("重新執行 Monte Carlo 模擬..."):
        before = cached_tournament_simulation(fixture_odds_df, team_meta_df, wc_team_stats_df, matches_df, 1000)
        after = cached_dynamic_tournament_simulation(fixture_odds_df, team_meta_df, wc_team_stats_df, matches_df, current_results, 1000)
    delta = probability_delta(before, after)
    st.subheader("更新前 / 更新後機率差異")
    if delta.empty:
        st.info("目前機率差異資料不足。")
        return
    focus_teams = set(current_results["home_team"]).union(set(current_results["away_team"]))
    view = delta[delta["team"].isin(focus_teams)].copy()
    if view.empty:
        view = delta.head(12).copy()
    display_cols = [
        "team_display",
        "group_qualified_probability_before",
        "group_qualified_probability_after",
        "group_qualified_probability_delta",
        "round_16_probability_after",
        "round_8_probability_after",
        "semi_final_probability_after",
        "final_probability_after",
        "champion_probability_after",
    ]
    display_cols = [column for column in display_cols if column in view.columns]
    display = view[display_cols].copy()
    for column in display.columns:
        if column != "team_display":
            display[column] = pd.to_numeric(display[column], errors="coerce").fillna(0).map(format_percent)
    st.dataframe(
        display.rename(
            columns={
                "team_display": "球隊",
                "group_qualified_probability_before": "原出線率",
                "group_qualified_probability_after": "更新後出線率",
                "group_qualified_probability_delta": "出線率變化",
                "round_16_probability_after": "16 強率",
                "round_8_probability_after": "8 強率",
                "semi_final_probability_after": "4 強率",
                "final_probability_after": "決賽率",
                "champion_probability_after": "冠軍率",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )


def realtime_worldcup_center_page() -> None:
    page_header("即時世界盃中心", "賽果輸入、動態 Elo、小組積分與機率重新計算")
    st.caption("本頁使用本地賽果與手動輸入資料模擬即時世界盃模式；資料不足時自動使用 fallback。")
    base_results = load_match_results()
    manual_results = st.session_state.get("realtime_worldcup_manual_results", pd.DataFrame())
    results = combine_results(base_results, manual_results)

    cols = st.columns(4)
    with cols[0]:
        display_card("已完成賽事", str(len(results)), "match_results.csv + 手動輸入")
    with cols[1]:
        display_card("小組數", str(results["group"].nunique() if not results.empty and "group" in results.columns else 0), "動態積分")
    with cols[2]:
        display_card("模擬次數", "1000", "快速重算")
    with cols[3]:
        display_card("資料狀態", "Fallback Ready", "不因缺資料中斷")

    st.subheader("賽果輸入器")
    labels = {
        f"{taipei_time_text(row.datetime_taipei)} | {team_name(row.home_team)} vs {team_name(row.away_team)}": row.match_id
        for row in fixture_odds_df.itertuples()
    }
    selected_label = st.selectbox("選擇比賽", list(labels.keys()), key="realtime_worldcup_match")
    selected_match_id = labels[selected_label]
    row = fixture_odds_df[fixture_odds_df["match_id"] == selected_match_id].iloc[0]
    score_cols = st.columns(2)
    home_score = score_cols[0].number_input(f"{team_name(row['home_team'])} 進球", min_value=0, max_value=12, value=1, step=1)
    away_score = score_cols[1].number_input(f"{team_name(row['away_team'])} 進球", min_value=0, max_value=12, value=1, step=1)
    if st.button("套用賽果並重新計算", type="primary", use_container_width=True):
        st.session_state["realtime_worldcup_manual_results"] = append_manual_result(
            results,
            row,
            int(home_score),
            int(away_score),
        )
        st.success("已套用賽果，請查看下方更新後積分與機率變化。")
        results = st.session_state["realtime_worldcup_manual_results"]

    left, right = st.columns([1.05, 0.95])
    with left:
        st.subheader("目前已完成賽事")
        if results.empty:
            st.info("目前尚未有完成賽事。")
        else:
            result_view = results.copy()
            result_view["score"] = result_view["home_score"].astype(str) + " : " + result_view["away_score"].astype(str)
            st.dataframe(
                result_view[["match_id", "date", "group", "home_team", "away_team", "score", "status"]].rename(
                    columns={
                        "match_id": "比賽 ID",
                        "date": "日期",
                        "group": "小組",
                        "home_team": "主隊",
                        "away_team": "客隊",
                        "score": "比分",
                        "status": "狀態",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )
    with right:
        st.subheader("小組積分表")
        standings = build_group_standings(results)
        if standings.empty:
            st.info("目前小組積分資料不足。")
        else:
            st.dataframe(
                standings.rename(
                    columns={
                        "group": "小組",
                        "team": "球隊",
                        "played": "賽",
                        "wins": "勝",
                        "draws": "和",
                        "losses": "敗",
                        "gf": "進球",
                        "ga": "失球",
                        "gd": "淨勝球",
                        "points": "積分",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )

    st.subheader("動態 Elo 更新")
    dynamic_elo = build_dynamic_elo(team_meta_df, results)
    changed = dynamic_elo[dynamic_elo["elo_change"].abs() > 0.01].copy()
    if changed.empty:
        st.info("目前沒有可顯示的 Elo 變化。")
    else:
        changed["team_display"] = changed.get("flag_emoji", "").astype(str) + " " + changed.get("team_zh", changed["team"]).astype(str)
        st.dataframe(
            changed.sort_values("elo_change", ascending=False)[["team_display", "original_elo", "updated_elo", "elo_change"]].rename(
                columns={
                    "team_display": "球隊",
                    "original_elo": "原始 Elo",
                    "updated_elo": "更新後 Elo",
                    "elo_change": "Elo 變化",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("機率變化追蹤")
    if results.empty:
        st.info("請先匯入或輸入賽果。")
        return
    with st.spinner("重新計算機率..."):
        _, after, changes = recalculate_probabilities(
            fixture_odds_df,
            team_meta_df,
            wc_team_stats_df,
            matches_df,
            results,
            simulations=1000,
        )
    summary = probability_change_summary(changes)
    if summary.empty:
        st.info("目前機率變化資料不足。")
        return
    display_cols = [
        "team_display",
        "group_qualified_probability_before",
        "group_qualified_probability_after",
        "group_qualified_probability_delta",
        "round_16_probability_after",
        "round_8_probability_after",
        "semi_final_probability_after",
        "final_probability_after",
        "champion_probability_after",
    ]
    display_cols = [column for column in display_cols if column in summary.columns]
    view = summary[display_cols].copy()
    for column in view.columns:
        if column != "team_display":
            view[column] = pd.to_numeric(view[column], errors="coerce").fillna(0).map(format_percent)
    st.dataframe(
        view.rename(
            columns={
                "team_display": "球隊",
                "group_qualified_probability_before": "原出線率",
                "group_qualified_probability_after": "更新後出線率",
                "group_qualified_probability_delta": "出線率變化",
                "round_16_probability_after": "16 強率",
                "round_8_probability_after": "8 強率",
                "semi_final_probability_after": "4 強率",
                "final_probability_after": "決賽率",
                "champion_probability_after": "冠軍率",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )
    top = after.sort_values("champion_probability", ascending=False).head(10).copy()
    top["label"] = top["champion_probability"].map(format_percent)
    chart = px.bar(top.sort_values("champion_probability"), x="champion_probability", y="team_display", orientation="h", text="label", color="champion_probability", color_continuous_scale=["#415a77", GOLD_LIGHT])
    chart.update_xaxes(tickformat=".0%")
    chart.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color=INK, coloraxis_showscale=False)
    st.plotly_chart(chart, use_container_width=True)


def realtime_worldcup_center_page() -> None:
    page_header("即時世界盃中心", "賽程比分、小組積分榜與晉級情境分析")
    st.caption("資料來源：data/match_results.csv。scheduled / live 不計入正式積分，只有 finished 且有比分的賽事會進入積分榜。")

    results = load_realtime_match_results()
    scoreboard = scoreboard_table(results)
    standings = calculate_group_standings(results)
    scenarios = qualification_scenarios(standings)

    finished_count = int((results["status"] == "finished").sum()) if not results.empty else 0
    live_count = int((results["status"] == "live").sum()) if not results.empty else 0
    scheduled_count = int((results["status"] == "scheduled").sum()) if not results.empty else 0

    cols = st.columns(4)
    with cols[0]:
        display_card("賽事總數", str(len(results)), "match_results.csv")
    with cols[1]:
        display_card("已結束", str(finished_count), "計入正式積分")
    with cols[2]:
        display_card("進行中", str(live_count), "暫不計入積分")
    with cols[3]:
        display_card("未開始", str(scheduled_count), "顯示待開賽")

    st.subheader("今日 / 近期賽程")
    if scoreboard.empty:
        st.info("目前沒有可顯示的賽程比分資料。")
    else:
        schedule_view = scoreboard.rename(
            columns={
                "match_id": "比賽 ID",
                "date_display": "比賽日期",
                "group": "小組",
                "home_team": "主隊",
                "away_team": "客隊",
                "score": "比分",
                "status_label": "比賽狀態",
            }
        )
        st.dataframe(schedule_view, use_container_width=True, hide_index=True)

    st.subheader("小組積分榜")
    if standings.empty:
        st.info("目前沒有足夠資料計算小組積分榜。")
    else:
        for group_name, group_table in standings.groupby("group", sort=True):
            st.markdown(f"**{group_name}**")
            view = group_table.rename(
                columns={
                    "group": "小組",
                    "rank": "排名",
                    "team": "球隊",
                    "played": "場次",
                    "wins": "勝",
                    "draws": "平",
                    "losses": "敗",
                    "goals_for": "進球",
                    "goals_against": "失球",
                    "goal_difference": "淨勝球",
                    "points": "積分",
                }
            )
            st.dataframe(view, use_container_width=True, hide_index=True)

    st.subheader("晉級情境分析")
    if scenarios.empty:
        st.info("目前沒有足夠資料分析晉級情境。")
    else:
        for group_name, group_table in scenarios.groupby("group", sort=True):
            st.markdown(f"**{group_name}**")
            view = group_table.rename(
                columns={
                    "group": "小組",
                    "rank": "目前排名",
                    "team": "球隊",
                    "status": "出線狀態",
                }
            )
            st.dataframe(view, use_container_width=True, hide_index=True)

    st.info("排序規則：積分高者優先，其次為淨勝球、進球數，最後依隊名字母順序。")


def realtime_worldcup_center_page() -> None:
    page_header("即時世界盃中心", "Live API / Mock 資料、即時比分、小組積分與晉級情境")
    live_result = cached_v29_live_matches()
    standings_result = cached_v29_group_standings()
    matches = live_result.data.copy()
    standings = standings_result.data.copy()
    scenarios = qualification_scenarios(standings)
    scoreboard = scoreboard_table(matches)

    status_counts = matches["status"].value_counts().to_dict() if not matches.empty else {}
    source_badge = "Live API" if live_result.source_mode == "Live API" else "Mock"
    if live_result.fallback_used:
        st.warning("目前使用備援資料")
    else:
        st.success(f"資料來源模式：{source_badge}")
    st.caption(f"最後更新時間：{live_result.updated_at}｜Provider：{live_result.provider}｜{live_result.message}")

    cols = st.columns(5)
    with cols[0]:
        display_card("總場次", str(len(matches)), "即時資料")
    with cols[1]:
        display_card("進行中", str(status_counts.get("live", 0)), "Live")
    with cols[2]:
        display_card("中場", str(status_counts.get("halftime", 0)), "Halftime")
    with cols[3]:
        display_card("已結束", str(status_counts.get("finished", 0)), "Finished")
    with cols[4]:
        display_card("未開始", str(status_counts.get("scheduled", 0)), "Scheduled")

    st.subheader("A. 今日賽事")
    if scoreboard.empty:
        st.info("目前沒有可顯示的即時賽程。")
    else:
        today_view = scoreboard.rename(
            columns={
                "match_id": "比賽 ID",
                "time_display": "比賽時間",
                "group": "小組",
                "home_team": "主隊",
                "away_team": "客隊",
                "home_score": "主隊比分",
                "away_score": "客隊比分",
                "status_label": "狀態",
                "minute": "比賽分鐘",
                "venue": "場地",
            }
        )[
            ["比賽 ID", "比賽時間", "小組", "主隊", "客隊", "主隊比分", "客隊比分", "狀態", "比賽分鐘", "場地"]
        ]
        st.dataframe(today_view, use_container_width=True, hide_index=True)

    st.subheader("B. 進行中賽事")
    live_view = scoreboard[scoreboard["status_label"].isin(["進行中", "中場"])].copy()
    if live_view.empty:
        st.info("目前沒有進行中賽事。")
    else:
        st.dataframe(
            live_view.rename(
                columns={
                    "time_display": "比賽時間",
                    "group": "小組",
                    "home_team": "主隊",
                    "away_team": "客隊",
                    "score": "比分",
                    "status_label": "狀態",
                    "minute": "比賽分鐘",
                    "venue": "場地",
                }
            )[["比賽時間", "小組", "主隊", "比分", "客隊", "狀態", "比賽分鐘", "場地"]],
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("C. 最新完賽結果")
    finished_view = scoreboard[scoreboard["status_label"].eq("已結束")].copy()
    if finished_view.empty:
        st.info("目前沒有完賽結果。")
    else:
        st.dataframe(
            finished_view.rename(
                columns={
                    "time_display": "比賽時間",
                    "group": "小組",
                    "home_team": "主隊",
                    "away_team": "客隊",
                    "score": "比分",
                    "venue": "場地",
                }
            )[["比賽時間", "小組", "主隊", "比分", "客隊", "場地"]],
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("D. 小組積分榜")
    if standings.empty:
        st.info("目前沒有足夠資料計算小組積分榜。")
    else:
        for group_name, group_table in standings.groupby("group", sort=True):
            st.markdown(f"**{group_name}**")
            st.dataframe(
                group_table.rename(
                    columns={
                        "rank": "排名",
                        "team": "球隊",
                        "played": "場次",
                        "wins": "勝",
                        "draws": "平",
                        "losses": "敗",
                        "goals_for": "進球",
                        "goals_against": "失球",
                        "goal_difference": "淨勝球",
                        "points": "積分",
                    }
                )[["排名", "球隊", "場次", "勝", "平", "敗", "進球", "失球", "淨勝球", "積分"]],
                use_container_width=True,
                hide_index=True,
            )

    st.subheader("E. 即時晉級情境")
    if scenarios.empty:
        st.info("目前沒有足夠資料分析晉級情境。")
    else:
        for group_name, group_table in scenarios.groupby("group", sort=True):
            st.markdown(f"**{group_name}**")
            st.dataframe(
                group_table.rename(
                    columns={
                        "rank": "目前排名",
                        "team": "球隊",
                        "status": "出線狀態",
                    }
                )[["目前排名", "球隊", "出線狀態"]],
                use_container_width=True,
                hide_index=True,
            )

    st.subheader("F. 最後更新資訊")
    st.info(
        "快取設定：live match 60 秒，小組積分 120 秒，歷史/隊伍資料 1 小時，球員資料 6 小時。"
        "若 API key 缺失、rate limit、timeout、欄位缺失或網路錯誤，會自動使用備援資料。"
    )


def render_team_history_comparison(home_team: str, away_team: str) -> None:
    comparison = pd.DataFrame(
        [team_summary(wc_team_stats_df, home_team), team_summary(wc_team_stats_df, away_team)]
    )
    columns = [
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
        "performance_score",
    ]
    st.dataframe(
        comparison[columns].rename(
            columns={
                "team": "國家",
                "tournaments_played": "參賽屆數",
                "matches": "場次",
                "wins": "勝",
                "draws": "和",
                "losses": "敗",
                "goals_for": "進球",
                "goals_against": "失球",
                "goal_difference": "淨勝球",
                "win_rate": "勝率",
                "titles": "冠軍",
                "top4_finishes": "四強次數",
                "performance_score": "歷史表現分數",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )


def rule_based_match_analysis(row: pd.Series, prediction) -> list[str]:
    home = row["home_team"]
    away = row["away_team"]
    home_strength = team_strength(matches_df, home)
    away_strength = team_strength(matches_df, away)
    elo_gap = home_strength["elo"] - away_strength["elo"]
    form_gap = home_strength["form"] - away_strength["form"]
    h2h = head_to_head_record(wc_head_to_head_df, home, away)
    player_source = player_database(players_df, team_meta_df)
    home_players = player_source[player_source["team"] == home]
    away_players = player_source[player_source["team"] == away]
    home_goal_rate = home_players["goal_rate"].mean() if (not home_players.empty and "goal_rate" in home_players.columns) else 0
    away_goal_rate = away_players["goal_rate"].mean() if (not away_players.empty and "goal_rate" in away_players.columns) else 0

    lines = []
    lines.append(
        f"勝率預測：{team_name(home)} {format_percent(prediction.home_win_probability)}、平手 {format_percent(prediction.draw_probability)}、{team_name(away)} {format_percent(prediction.away_win_probability)}。"
    )
    lines.append(f"可能比分：模型預測 {prediction.predicted_home_goals} : {prediction.predicted_away_goals}，預期進球約 {prediction.expected_home_goals} : {prediction.expected_away_goals}。")

    if abs(elo_gap) >= 90:
        leader = team_name(home if elo_gap > 0 else away)
        lines.append(f"Elo 差距達 {abs(elo_gap):.0f} 分，{leader} 在整體實力評分上優勢較明顯。")
    else:
        lines.append("雙方 Elo 差距不大，模型判斷比賽可能更接近，平手或小比分機率需要特別注意。")

    if abs(form_gap) < 0.12:
        lines.append("近期狀態接近，節奏可能偏膠著，單一進球或定位球會放大比賽影響。")
    else:
        form_team = team_name(home if form_gap > 0 else away)
        lines.append(f"{form_team} 近期狀態略佳，模型會給予較高的臨場表現權重。")

    h2h_row = h2h.iloc[0] if isinstance(h2h, pd.DataFrame) and not h2h.empty else None
    if h2h_row is not None and int(h2h_row.get("matches", 0)) > 0:
        lines.append(
            f"歷史交手共有 {int(h2h_row.get('matches', 0))} 場，雙方勝場分別為 {int(h2h_row.get('team_a_wins', 0))} 與 {int(h2h_row.get('team_b_wins', 0))}，平手 {int(h2h_row.get('draws', 0))} 場。"
        )
    else:
        lines.append("目前 2002～2022 世界盃資料中缺少足夠直接交手紀錄，因此歷史交手不作為主要判斷。")

    if max(home_goal_rate, away_goal_rate) > 0:
        player_edge = team_name(home if home_goal_rate >= away_goal_rate else away)
        lines.append(f"關鍵球員平均進球率方面，{player_edge} 略佔優勢，但此特徵僅作展示輔助，不直接重寫主模型。")

    top_probability = max(prediction.home_win_probability, prediction.draw_probability, prediction.away_win_probability)
    if top_probability < 0.42:
        lines.append("比賽風險：勝平負機率分布較分散，模型信心偏保守，適合視為高不確定性場次。")
    else:
        lines.append(f"比賽風險：模型最高方向機率約 {format_percent(top_probability)}，仍需搭配信心分數與風險等級一起解讀。")
    lines.append("風險提醒：本網站不提供下注功能，所有預測與文字分析僅供資料分析參考，不保證命中或獲利。")
    return lines


def match_analysis_page() -> None:
    page_header("單場分析頁", "大型比分卡、勝平負機率、信心分數進度條與世界盃歷史表現")
    disclaimer_box()
    row = selected_fixture()
    prediction = predict_match(matches_df, row["home_team"], row["away_team"], wc_team_stats_df)
    market_table = market_probability_table(row, prediction)
    prediction_cards(row)

    st.subheader("AI 賽事分析文字")
    for line in rule_based_match_analysis(row, prediction):
        st.markdown(f"- {line}")
    st.caption("規則式繁體中文分析，未串接 OpenAI API；資料不足時使用本地展示資料與保守描述。")

    probabilities = prediction_to_frame(prediction)
    st.dataframe(
        probabilities.assign(機率=probabilities["機率"].map(format_percent)),
        use_container_width=True,
        hide_index=True,
    )
    chart = px.pie(
        probabilities,
        values="機率",
        names="結果",
        title="勝平負機率分布",
        color_discrete_sequence=[GOLD, "#6d7fa0", "#c8d3e6"],
    )
    chart.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color=INK)
    st.plotly_chart(chart, use_container_width=True)

    st.subheader("世界盃歷史表現比較")
    render_team_history_comparison(row["home_team"], row["away_team"])

    render_worldcup_history_block(row["home_team"], row["away_team"])
    render_h2h_summary_block(row["home_team"], row["away_team"])
    render_key_players_block(row["home_team"], row["away_team"])

    st.subheader("兩隊近期歷史資料")
    team_matches = matches_df[
        matches_df["home_team"].isin([row["home_team"], row["away_team"]])
        | matches_df["away_team"].isin([row["home_team"], row["away_team"]])
    ].tail(8)
    st.dataframe(team_matches, use_container_width=True, hide_index=True)


def betting_page() -> None:
    page_header("市場機率分析", "比較模型機率與市場隱含機率，並以風險色塊呈現分析等級")
    disclaimer_box()
    row = selected_fixture()
    prediction = predict_match(matches_df, row["home_team"], row["away_team"], wc_team_stats_df)
    signals = analyze_1x2(prediction, row["home_odds"], row["draw_odds"], row["away_odds"])

    st.subheader(f"{fixture_time_text(row)} 台灣時間｜{matchup_text(row)}")
    st.dataframe(signal_dataframe(signals), use_container_width=True, hide_index=True)

    best_signal = max(signals, key=lambda signal: signal.edge)
    st.markdown(
        f"""
        <div class="display-card">
          <div class="card-label">目前最佳觀察方向</div>
          <div class="card-value">{html.escape(best_signal.market)} · {html.escape(best_signal.recommendation)}
            {risk_badge(best_signal.risk_level)}
          </div>
          <div class="card-note">此資訊僅供資料分析參考，不構成下注指示。</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("輔助分析說明")
    history_text, player_text, recent_text = betting_context(row)
    st.markdown(
        f"""
        <div class="display-card">
          <div class="card-label">歷史戰績影響</div>
          <div class="card-note">{html.escape(history_text)}</div>
        </div>
        <div class="display-card">
          <div class="card-label">關鍵球員影響</div>
          <div class="card-note">{html.escape(player_text)}</div>
        </div>
        <div class="display-card">
          <div class="card-label">近期狀態影響</div>
          <div class="card-note">{html.escape(recent_text)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    comparison = pd.DataFrame(
        [{"市場": s.market, "類型": "模型機率", "機率": s.model_probability} for s in signals]
        + [{"市場": s.market, "類型": "隱含機率", "機率": s.implied_probability} for s in signals]
    )
    chart = px.bar(
        comparison,
        x="市場",
        y="機率",
        color="類型",
        barmode="group",
        color_discrete_sequence=[GOLD, "#8aa0c3"],
    )
    chart.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color=INK)
    st.plotly_chart(chart, use_container_width=True)


def model_backtest_page() -> None:
    page_header("模型回測頁", "以信心分層呈現模型回測結果，提升預測可信度與可解釋性")
    disclaimer_box()
    metrics = backtest(matches_df, wc_team_stats_df, players_df)

    cols = st.columns(4)
    with cols[0]:
        display_card("回測場次", str(metrics["matches"]), "測試資料")
    with cols[1]:
        display_card("目前回測命中率", format_percent(metrics["accuracy"]), "勝平負方向")
    with cols[2]:
        display_card("ROI", format_percent(metrics["simulated_roi"]), "固定示範賠率")
    with cols[3]:
        confidence_card(metrics["average_confidence"])

    st.subheader("信心級別命中率")
    cols = st.columns(3)
    with cols[0]:
        display_card(
            "高信心場次命中率",
            format_percent(metrics["high_confidence_accuracy"]),
            f"{metrics['high_confidence_matches']} 場",
        )
    with cols[1]:
        display_card(
            "中信心場次命中率",
            format_percent(metrics["mid_confidence_accuracy"]),
            f"{metrics['mid_confidence_matches']} 場",
        )
    with cols[2]:
        display_card(
            "低信心場次命中率",
            format_percent(metrics["low_confidence_accuracy"]),
            f"{metrics['low_confidence_matches']} 場",
        )

    bucket_df = pd.DataFrame(metrics["bucket_rows"])
    if not bucket_df.empty:
        bucket_df["confidence"] = bucket_df["confidence"].map(format_percent)
        bucket_df["player_goal_feature_home"] = bucket_df["player_goal_feature_home"].map(format_percent)
        bucket_df["player_goal_feature_away"] = bucket_df["player_goal_feature_away"].map(format_percent)
        st.dataframe(
            bucket_df.rename(
                columns={
                    "match": "比賽",
                    "prediction": "預測",
                    "actual": "實際",
                    "correct": "是否命中",
                    "confidence": "輔助信心分數",
                    "confidence_bucket": "信心級別",
                    "player_goal_feature_home": "主隊關鍵球員進球率",
                    "player_goal_feature_away": "客隊關鍵球員進球率",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    st.caption(
        "高信心篩選使用模型信心分數加上關鍵球員進球率的輔助特徵；"
        "此設定不改變原始比分預測，只用於回測分層與展示說明。"
    )


def champion_probability_page() -> None:
    page_header("冠軍機率預測", "以 Elo、Poisson、近期狀態、歷史世界盃表現與 Monte Carlo 模擬估算 2026 奪冠機率")
    st.info(
        "本頁透過 Elo Rating、Poisson 進球模型與 Monte Carlo 模擬，估算各隊在不同階段的晉級機率。"
        "結果僅供資料分析參考，不代表實際賽果。"
    )
    disclaimer_box()

    simulations = 1000
    with st.spinner("正在執行 1000 次 Monte Carlo 模擬..."):
        simulation_df = cached_tournament_simulation(
            fixtures_df,
            team_meta_df,
            wc_team_stats_df,
            matches_df,
            simulations,
        )

    top10 = simulation_df.head(10).copy()
    champion = top10.iloc[0]

    cols = st.columns(4)
    with cols[0]:
        display_card("模擬次數", f"{simulations:,}", "Monte Carlo")
    with cols[1]:
        display_card("模擬隊伍", str(len(simulation_df)), "2026 參賽隊伍")
    with cols[2]:
        display_card("最高冠軍機率", format_percent(float(champion["champion_probability"])), champion["team_display"])
    with cols[3]:
        display_card("模型類型", "可解釋 AI", "Elo + Poisson")

    st.subheader("奪冠機率排行榜 Top 10")
    chart_df = top10.sort_values("champion_probability", ascending=True).copy()
    chart_df["champion_label"] = chart_df["champion_probability"].map(format_percent)
    chart = px.bar(
        chart_df,
        x="champion_probability",
        y="team_display",
        orientation="h",
        text="champion_label",
        labels={"champion_probability": "模擬冠軍機率", "team_display": "球隊"},
        color="champion_probability",
        color_continuous_scale=["#415a77", GOLD_LIGHT],
    )
    chart.update_traces(textposition="outside")
    chart.update_xaxes(tickformat=".0%")
    chart.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color=INK,
        coloraxis_showscale=False,
        margin=dict(l=10, r=40, t=10, b=10),
    )
    st.plotly_chart(chart, use_container_width=True)

    ranking = top10[
        [
            "team_display",
            "elo",
            "history_score",
            "recent_form",
            "champion_probability",
        ]
    ].copy()
    ranking["history_score"] = ranking["history_score"].map(format_percent)
    ranking["recent_form"] = ranking["recent_form"].map(format_percent)
    ranking["champion_probability"] = ranking["champion_probability"].map(format_percent)
    st.dataframe(
        ranking.rename(
            columns={
                "team_display": "球隊",
                "elo": "Elo 分數",
                "history_score": "歷史世界盃表現",
                "recent_form": "近期狀態",
                "champion_probability": "模擬冠軍機率",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("淘汰賽晉級機率")
    stage_columns = [
        "group_qualified_probability",
        "round_16_probability",
        "round_8_probability",
        "semi_final_probability",
        "final_probability",
        "champion_probability",
    ]
    stage_labels = {
        "group_qualified_probability": "小組出線機率",
        "round_16_probability": "16 強機率",
        "round_8_probability": "8 強機率",
        "semi_final_probability": "4 強機率",
        "final_probability": "決賽機率",
        "champion_probability": "冠軍機率",
    }
    progression = simulation_df[["team_display", "elo", *stage_columns]].copy()
    display_progression = progression.copy()
    for column in stage_columns:
        display_progression[column] = display_progression[column].map(format_percent)
    st.dataframe(
        display_progression.rename(
            columns={
                "team_display": "球隊",
                "elo": "Elo 分數",
                **stage_labels,
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    stage_chart = top10[["team_display", *stage_columns]].melt(
        id_vars="team_display",
        var_name="stage",
        value_name="probability",
    )
    stage_chart["stage"] = stage_chart["stage"].map(stage_labels)
    line = px.line(
        stage_chart,
        x="stage",
        y="probability",
        color="team_display",
        markers=True,
        labels={"stage": "階段", "probability": "晉級機率", "team_display": "球隊"},
    )
    line.update_yaxes(tickformat=".0%")
    line.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color=INK,
        legend_title_text="Top 10 球隊",
    )
    st.plotly_chart(line, use_container_width=True)


def elo_ranking_page() -> None:
    page_header("Elo 世界排名", "以本地 team_meta.csv 的 Elo Rating 建立 2026 世界盃參賽隊伍排名")
    st.info(
        "Elo Rating 用於衡量球隊相對強度。本頁排名資料使用本地與更新資料，"
        "可作為單場預測與冠軍機率模擬的輔助參考，不代表官方 FIFA 排名。"
    )

    rankings = build_elo_rankings(team_meta_df)
    top10 = rankings.head(10).copy()

    cols = st.columns(4)
    with cols[0]:
        display_card("排名隊伍", str(len(rankings)), "2026 參賽隊伍")
    with cols[1]:
        leader = rankings.iloc[0]
        display_card("Elo 第一名", leader["team_display"], f"{int(leader['elo'])} 分")
    with cols[2]:
        display_card("平均 Elo", f"{rankings['elo'].mean():.0f}", "展示資料")
    with cols[3]:
        display_card("資料來源", "team_meta.csv", "本地 Elo 欄位")

    st.subheader("前 10 名排行榜")
    chart_df = top10.sort_values("elo", ascending=True)
    chart = px.bar(
        chart_df,
        x="elo",
        y="team_display",
        orientation="h",
        text="elo",
        color="elo",
        color_continuous_scale=["#415a77", GOLD_LIGHT],
        labels={"elo": "Elo 分數", "team_display": "球隊"},
    )
    chart.update_traces(textposition="outside")
    chart.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color=INK,
        coloraxis_showscale=False,
        margin=dict(l=10, r=40, t=10, b=10),
    )
    st.plotly_chart(chart, use_container_width=True)

    st.subheader("完整 Elo 排名表")
    sort_mode = st.selectbox("排序方式", ["Elo 高到低", "Elo 低到高"])
    table = rankings.sort_values("elo", ascending=sort_mode == "Elo 低到高").copy()
    st.dataframe(
        table[
            [
                "rank",
                "team_display",
                "country_code",
                "elo",
                "elo_tier",
                "fifa_ranking",
            ]
        ].rename(
            columns={
                "rank": "排名",
                "team_display": "球隊",
                "country_code": "國碼",
                "elo": "Elo 分數",
                "elo_tier": "級距",
                "fifa_ranking": "FIFA 排名",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )


def advancement_probability_page() -> None:
    page_header("晉級機率分析", "選擇任一參賽國家，查看小組出線到奪冠的階段機率")
    st.info(
        "本頁重用 V2 冠軍機率模擬系統，透過 Elo Rating、Poisson 進球模型、近期狀態、"
        "世界盃歷史表現與 1000 次 Monte Carlo 模擬，估算各階段晉級機率。"
        "結果僅供資料分析參考，不代表實際賽果。"
    )

    simulations = 1000
    simulation_df = cached_tournament_simulation(
        fixtures_df,
        team_meta_df,
        wc_team_stats_df,
        matches_df,
        simulations,
    )

    team_options = simulation_df.sort_values("team_zh")["team_display"].tolist()
    selected_display = st.selectbox("選擇國家隊", team_options)
    selected_row = simulation_df[simulation_df["team_display"] == selected_display].iloc[0]

    stage_rows = [
        ("小組出線機率", selected_row["group_qualified_probability"]),
        ("16 強機率", selected_row["round_16_probability"]),
        ("8 強機率", selected_row["round_8_probability"]),
        ("4 強機率", selected_row["semi_final_probability"]),
        ("決賽機率", selected_row["final_probability"]),
        ("奪冠機率", selected_row["champion_probability"]),
    ]
    probability_df = pd.DataFrame(stage_rows, columns=["階段", "機率"])
    probability_df["百分比"] = probability_df["機率"].map(format_percent)

    cols = st.columns(4)
    with cols[0]:
        display_card("選擇球隊", selected_row["team_display"], f"Elo {int(selected_row['elo'])}")
    with cols[1]:
        display_card("小組出線", format_percent(float(selected_row["group_qualified_probability"])), "Monte Carlo")
    with cols[2]:
        display_card("決賽機率", format_percent(float(selected_row["final_probability"])), "淘汰賽模擬")
    with cols[3]:
        display_card("奪冠機率", format_percent(float(selected_row["champion_probability"])), f"{simulations:,} 次模擬")

    st.subheader("階段晉級機率")
    chart = px.bar(
        probability_df,
        x="階段",
        y="機率",
        text="百分比",
        color="機率",
        color_continuous_scale=["#415a77", GOLD_LIGHT],
        labels={"機率": "晉級機率"},
    )
    chart.update_traces(textposition="outside")
    chart.update_yaxes(tickformat=".0%", range=[0, 1])
    chart.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color=INK,
        coloraxis_showscale=False,
        margin=dict(l=10, r=30, t=10, b=10),
    )
    st.plotly_chart(chart, use_container_width=True)

    st.dataframe(
        probability_df[["階段", "百分比"]].rename(columns={"百分比": "機率"}),
        use_container_width=True,
        hide_index=True,
    )

    st.caption("晉級機率使用 Elo、近期狀態與歷史世界盃資料估算，僅供資料分析參考。")


def presentation_mode_page() -> None:
    dashboard_page()
    return
    sim_df = v7_simulation()
    rankings = build_elo_rankings(team_meta_df).head(20)
    player_db = player_database(players_df, team_meta_df)
    live_result = cached_v29_live_matches()
    live_matches = live_result.data
    live_source = live_result.source_mode

    cols = st.columns(4)
    with cols[0]:
        display_card("球員資料", f"{len(player_db):,}", f"{player_db['team'].nunique()} 隊")
    with cols[1]:
        display_card("模擬次數", f"{V7_SIMULATIONS:,}", "Monte Carlo")
    with cols[2]:
        display_card("即時資料來源", live_source, "API/Fallback")
    with cols[3]:
        display_card("最高奪冠率", format_percent(float(sim_df.iloc[0]["champion_probability"])), sim_df.iloc[0]["team_display"])

    left, right = st.columns(2)
    with left:
        st.subheader("世界排名 TOP20")
        rank_chart = px.bar(rankings.sort_values("elo", ascending=True), x="elo", y="team_display", orientation="h", color="elo", color_continuous_scale=["#415a77", GOLD_LIGHT])
        rank_chart.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color=INK, coloraxis_showscale=False)
        st.plotly_chart(rank_chart, use_container_width=True)
    with right:
        st.subheader("奪冠機率 TOP20")
        champ = sim_df.head(20).sort_values("champion_probability", ascending=True)
        champ_chart = px.bar(champ, x="champion_probability", y="team_display", orientation="h", color="champion_probability", color_continuous_scale=["#415a77", GOLD_LIGHT])
        champ_chart.update_xaxes(tickformat=".0%")
        champ_chart.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color=INK, coloraxis_showscale=False)
        st.plotly_chart(champ_chart, use_container_width=True)

    st.subheader("即時賽況")
    if live_matches.empty:
        st.info("目前沒有即時賽況。")
    else:
        live_table = live_matches[["scheduled_time", "home_team", "home_score", "away_score", "away_team", "status"]].copy().head(6)
        live_table["scheduled_time"] = live_table["scheduled_time"].map(live_time_text)
        st.dataframe(live_table.rename(columns={"scheduled_time": "時間", "home_team": "主隊", "home_score": "主分", "away_score": "客分", "away_team": "客隊", "status": "狀態"}), use_container_width=True, hide_index=True)

    st.subheader("國家隊比較")
    compare = sim_df.head(12)[["team_display", "elo", "group_qualified_probability", "round_8_probability", "champion_probability"]].copy()
    compare_heat = compare.set_index("team_display").rename(columns={"elo": "Elo", "group_qualified_probability": "小組出線率", "round_8_probability": "八強率", "champion_probability": "奪冠率"})
    heatmap = px.imshow(compare_heat, aspect="auto", color_continuous_scale=["#071426", GOLD_LIGHT])
    heatmap.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color=INK)
    st.plotly_chart(heatmap, use_container_width=True)


def load_v8_player_database() -> pd.DataFrame:
    try:
        return pd.read_csv("data/player_database.csv")
    except Exception:
        fallback = player_database(players_df, team_meta_df).copy()
        fallback["assists"] = 0
        fallback["appearances"] = fallback["national_caps"]
        fallback["recent_form"] = fallback.get("recent_form_rating", 7.0)
        fallback["is_key_player"] = fallback.groupby("team")["recent_form"].rank(method="first", ascending=False).le(3)
        fallback["data_note"] = "展示資料／模擬資料：fallback from players_2026.csv"
        return fallback.rename(columns={"national_goals": "national_goals"})


def load_v8_team_database() -> pd.DataFrame:
    try:
        return pd.read_csv("data/team_database.csv")
    except Exception:
        fallback = team_meta_df.copy()
        fallback["confederation"] = fallback.get("confederation", "展示資料")
        fallback["recent_form"] = 0.5
        fallback["best_finish"] = "待資料補齊"
        fallback["star_players"] = "待資料補齊"
        fallback["tactical_style"] = "展示資料"
        fallback["data_note"] = "展示資料／模擬資料：fallback from team_meta.csv"
        return fallback


def dashboard_page() -> None:
    st.markdown(
        """
        <div class="hero">
          <div>
            <div class="hero-kicker">WORLD CUP PREDICTION CENTER · V8</div>
            <div class="hero-title">世界盃智慧預測中心</div>
            <div class="hero-copy">
              整合賽程、單場比分、冠軍機率、晉級機率、即時實況摘要與 Monte Carlo 模擬，
              以深色科技風呈現世界盃資料重點。
            </div>
          </div>
          <div class="hero-visual"><div class="trophy">🏆</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    disclaimer_box()

    sim_df = v7_simulation()
    champion = sim_df.iloc[0]
    row = fixture_odds_df.iloc[0]
    prediction = predict_match(matches_df, row["home_team"], row["away_team"], wc_team_stats_df)
    live_result = cached_v29_live_matches()
    live_matches = live_result.data
    live_source = live_result.source_mode

    cols = st.columns(4)
    with cols[0]:
        display_card("今日／近期賽程", f"{len(fixture_odds_df.head(6))} 場", "台灣時間 UTC+8")
    with cols[1]:
        display_card("預測比分", f"{prediction.predicted_home_goals} : {prediction.predicted_away_goals}", matchup_text(row))
    with cols[2]:
        display_card("冠軍機率最高", format_percent(float(champion["champion_probability"])), champion["team_display"])
    with cols[3]:
        display_card("Monte Carlo", f"{V7_SIMULATIONS:,}", "展示資料／模擬資料")

    left, right = st.columns([1.05, 0.95])
    with left:
        st.subheader("今日／近期賽程")
        upcoming = fixture_odds_df.head(5).copy()
        st.dataframe(fixtures_with_flags(upcoming), use_container_width=True, hide_index=True)
    with right:
        st.subheader("冠軍機率 Top 5")
        top5 = sim_df.head(5).sort_values("champion_probability", ascending=True).copy()
        top5["label"] = top5["champion_probability"].map(format_percent)
        chart = px.bar(
            top5,
            x="champion_probability",
            y="team_display",
            orientation="h",
            text="label",
            color="champion_probability",
            color_continuous_scale=["#415a77", GOLD_LIGHT],
        )
        chart.update_xaxes(tickformat=".0%")
        chart.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color=INK, coloraxis_showscale=False, margin=dict(l=8, r=35, t=8, b=8))
        st.plotly_chart(chart, use_container_width=True)

    cols = st.columns(3)
    with cols[0]:
        display_card("晉級機率摘要", format_percent(float(champion["group_qualified_probability"])), f"{champion['team_display']} 小組出線率")
    with cols[1]:
        display_card("即時實況摘要", live_source, "API-Football 或 Fallback Dataset")
    with cols[2]:
        display_card("模型說明", "Elo + Poisson", "Monte Carlo 模擬與歷史權重")


def player_database_page() -> None:
    page_header("球員資料庫", "V8：球員搜尋、國家/位置篩選、關鍵球員 Top 10")
    players = load_v8_player_database()
    st.caption("資料來源：展示資料／模擬資料會於 data_note 欄位標示；缺資料時不讓頁面報錯。")

    teams = ["全部"] + sorted(players["team_zh"].dropna().unique().tolist())
    positions = ["全部"] + sorted(players["position"].dropna().unique().tolist())
    cols = st.columns([1, 1, 1.4])
    selected_team = cols[0].selectbox("國家隊", teams)
    selected_position = cols[1].selectbox("位置", positions)
    keyword = cols[2].text_input("搜尋球員", "")

    filtered = players.copy()
    if selected_team != "全部":
        filtered = filtered[filtered["team_zh"] == selected_team]
    if selected_position != "全部":
        filtered = filtered[filtered["position"] == selected_position]
    if keyword.strip():
        filtered = filtered[filtered["player_name"].str.contains(keyword.strip(), case=False, na=False)]

    key_players = players.sort_values(["is_key_player", "recent_form", "national_goals"], ascending=False).head(10).copy()
    st.subheader("關鍵球員 Top 10")
    key_chart = px.bar(
        key_players.sort_values("recent_form"),
        x="recent_form",
        y="player_name",
        color="recent_form",
        orientation="h",
        hover_data=["team_zh", "position", "national_goals", "assists"],
        color_continuous_scale=["#415a77", GOLD_LIGHT],
    )
    key_chart.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color=INK, coloraxis_showscale=False)
    st.plotly_chart(key_chart, use_container_width=True)

    table = filtered[["player_name", "team_zh", "position", "age", "national_goals", "assists", "appearances", "recent_form", "is_key_player", "data_note"]].copy()
    st.dataframe(
        table.rename(columns={
            "player_name": "球員姓名",
            "team_zh": "國家隊",
            "position": "位置",
            "age": "年齡",
            "national_goals": "進球數",
            "assists": "助攻數",
            "appearances": "出場數",
            "recent_form": "近期狀態",
            "is_key_player": "關鍵球員標記",
            "data_note": "資料說明",
        }),
        use_container_width=True,
        hide_index=True,
    )


def team_database_page() -> None:
    page_header("球隊資料庫", "V8：洲別篩選、國家搜尋、Elo Top 10 與各洲隊伍數")
    teams = load_v8_team_database()
    st.caption("資料來源：展示資料／模擬資料會於 data_note 欄位標示；戰術風格與近期狀態為本地推估。")

    confeds = ["全部"] + sorted(teams["confederation"].dropna().unique().tolist())
    cols = st.columns([1, 1.4])
    selected_confed = cols[0].selectbox("洲別", confeds)
    keyword = cols[1].text_input("搜尋國家", "")

    filtered = teams.copy()
    if selected_confed != "全部":
        filtered = filtered[filtered["confederation"] == selected_confed]
    if keyword.strip():
        filtered = filtered[
            filtered["team"].str.contains(keyword.strip(), case=False, na=False)
            | filtered["team_zh"].str.contains(keyword.strip(), case=False, na=False)
        ]

    left, right = st.columns(2)
    with left:
        st.subheader("Elo Top 10")
        top10 = teams.sort_values("elo", ascending=False).head(10).sort_values("elo")
        chart = px.bar(top10, x="elo", y="team_zh", orientation="h", color="elo", color_continuous_scale=["#415a77", GOLD_LIGHT])
        chart.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color=INK, coloraxis_showscale=False)
        st.plotly_chart(chart, use_container_width=True)
    with right:
        st.subheader("各洲參賽隊伍數")
        confed_counts = teams.groupby("confederation", as_index=False)["team"].count()
        pie = px.pie(confed_counts, values="team", names="confederation", hole=0.42, color_discrete_sequence=[GOLD, "#8aa0c3", "#28a745", "#dc3545", "#6f42c1"])
        pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color=INK)
        st.plotly_chart(pie, use_container_width=True)

    st.dataframe(
        filtered[["flag_emoji", "team_zh", "confederation", "fifa_ranking", "elo", "recent_form", "best_finish", "star_players", "tactical_style", "data_note"]].rename(
            columns={
                "flag_emoji": "國旗",
                "team_zh": "國家",
                "confederation": "洲別",
                "fifa_ranking": "FIFA 排名",
                "elo": "Elo 分數",
                "recent_form": "近期狀態",
                "best_finish": "世界盃最佳成績",
                "star_players": "主要球星",
                "tactical_style": "戰術風格",
                "data_note": "資料說明",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )


def presentation_mode_page() -> None:
    dashboard_page()
    return
    cards = [
        ("平台名稱", "世界盃情報中心", "以資料分析與可解釋模型預測 2026 世界盃。"),
        ("產品目標", "把賽程、球隊、球員與模型整合", "讓使用者快速理解比賽風險與預測依據。"),
        ("系統功能架構", "賽程中心／預測中心／資料中心", "用分類側邊欄降低操作複雜度。"),
        ("使用資料", "本地 CSV + API-Football fallback", "所有展示資料／模擬資料皆清楚標示。"),
        ("預測模型說明", "Elo + Poisson + 歷史權重", "保留可解釋性，不使用黑盒深度學習。"),
        ("Monte Carlo 模擬", "1000 / 5000 / 10000 次", "輸出小組出線到奪冠機率。"),
        ("系統限制", "不保證賽果、不提供下注", "API 無資料時使用 fallback，避免頁面壞掉。"),
        ("未來發展", "串接更多官方/商業資料源", "補足身價、慣用腳、即時事件與球員進階數據。"),
        ("操作流程", "首頁 → 單場分析 → 模擬器 → 資料庫", "適合快速掌握賽程、預測、模擬與資料細節。"),
    ]
    for index in range(0, len(cards), 3):
        cols = st.columns(3)
        for col, (title, value, note) in zip(cols, cards[index:index + 3]):
            with col:
                display_card(title, value, note)

    sim_df = v7_simulation()
    st.subheader("展示亮點：冠軍機率 Top 5")
    top5 = sim_df.head(5).sort_values("champion_probability", ascending=True).copy()
    top5["label"] = top5["champion_probability"].map(format_percent)
    chart = px.bar(top5, x="champion_probability", y="team_display", orientation="h", text="label", color="champion_probability", color_continuous_scale=["#415a77", GOLD_LIGHT])
    chart.update_xaxes(tickformat=".0%")
    chart.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color=INK, coloraxis_showscale=False)
    st.plotly_chart(chart, use_container_width=True)


def live_matches_page() -> None:
    page_header("即時實況", "使用 Streamlit Secrets 串接 API-Football；未設定金鑰或 API 無資料時自動使用展示資料。")
    refresh_seconds = 45
    st.markdown(f"<meta http-equiv='refresh' content='{refresh_seconds}'>", unsafe_allow_html=True)

    source_mode = st.radio(
        "資料來源模式",
        ["世界盃展示資料", "真實足球即時資料"],
        horizontal=True,
        index=1,
    )
    live_mode = "real" if source_mode == "真實足球即時資料" else "demo"
    target_date = pd.Timestamp.now(tz="Asia/Taipei").date()
    live_matches, live_events, data_source, _live_debug = load_live_matches_with_debug(
        st.secrets,
        live_matches_df,
        live_events_df,
        target_date=target_date,
        mode=live_mode,
    )

    is_mock_source = "mock" in str(data_source).lower() or "展示" in str(data_source)
    if is_mock_source:
        st.warning("🟡 展示資料")
    else:
        st.success("🟢 真實 API 即時資料")
    st.caption(f"時區：台灣時間（UTC+8）｜每 {refresh_seconds} 秒自動刷新")

    if live_matches.empty:
        st.info("目前沒有可顯示的即時賽況資料。")
        return

    live_matches = live_matches.copy()
    live_matches["live_match_id"] = live_matches["live_match_id"].astype(str)
    live_events = live_events.copy()
    if not live_events.empty and "live_match_id" in live_events.columns:
        live_events["live_match_id"] = live_events["live_match_id"].astype(str)

    def option_label(row: pd.Series) -> str:
        return (
            f"{live_time_text(row.get('scheduled_time'))} | "
            f"{team_name(row.get('home_team', 'TBD'))} "
            f"{row.get('home_score', 0)}-{row.get('away_score', 0)} "
            f"{team_name(row.get('away_team', 'TBD'))}"
        )

    labels = {option_label(row): row["live_match_id"] for _, row in live_matches.iterrows()}
    selected = st.selectbox("選擇比賽", list(labels.keys()))
    live_match_id = labels[selected]
    row = live_matches[live_matches["live_match_id"] == live_match_id].iloc[0]
    scheduled_time = live_time_text(row.get("scheduled_time"))
    venue = row.get("venue", "未提供")
    league_name = row.get("league_name", "未提供")

    def stat_text(value, suffix: str = "") -> str:
        if value is None or pd.isna(value):
            return "--"
        return f"{int(value)}{suffix}"

    st.markdown(
        f"""
        <div class="display-card score-card">
          <div class="score-teams">{html.escape(team_name(row.get('home_team', 'TBD')))} vs {html.escape(team_name(row.get('away_team', 'TBD')))}</div>
          <div class="score-value">{row.get('home_score', 0)} : {row.get('away_score', 0)}</div>
          <div class="score-note">聯賽：{html.escape(str(league_name))} ｜ 比賽時間：{html.escape(scheduled_time)} ｜ 狀態：{html.escape(str(row.get('status', '未提供')))} ｜ 分鐘：{int(row.get('minute', 0) or 0)}' ｜ 場地：{html.escape(str(venue))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    stat_cols = st.columns(3)
    with stat_cols[0]:
        display_card("射門數", f"{stat_text(row.get('home_shots'))} : {stat_text(row.get('away_shots'))}")
    with stat_cols[1]:
        display_card("控球率", f"{stat_text(row.get('home_possession'), '%')} : {stat_text(row.get('away_possession'), '%')}")
    with stat_cols[2]:
        display_card("角球", f"{stat_text(row.get('home_corners'))} : {stat_text(row.get('away_corners'))}")

    st.subheader("比賽事件")
    if live_events.empty or "live_match_id" not in live_events.columns:
        events = pd.DataFrame()
    else:
        events = live_events[live_events["live_match_id"] == live_match_id].copy()

    if events.empty:
        st.info("目前沒有進球、黃牌、紅牌或換人事件資料。")
    else:
        events["事件"] = events["event_type"].map(live_event_label)
        events["球隊"] = events["team"].map(lambda team: f"{flag(team)} {team_name(team, with_flag=False)}")
        event_columns = ["minute", "事件", "球隊", "player", "detail"]
        available_event_columns = [column for column in event_columns if column in events.columns]
        st.dataframe(
            events[available_event_columns].rename(
                columns={
                    "minute": "時間",
                    "player": "球員",
                    "detail": "說明",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("技術統計")
    stats = pd.DataFrame(
        [
            {"指標": "射門數", row["home_team"]: row.get("home_shots", 0), row["away_team"]: row.get("away_shots", 0)},
            {"指標": "控球率", row["home_team"]: row.get("home_possession", 50), row["away_team"]: row.get("away_possession", 50)},
            {"指標": "角球", row["home_team"]: row.get("home_corners", 0), row["away_team"]: row.get("away_corners", 0)},
        ]
    )
    stats_long = stats.melt(id_vars="指標", var_name="球隊", value_name="數值")
    chart = px.bar(
        stats_long,
        x="指標",
        y="數值",
        color="球隊",
        barmode="group",
        color_discrete_sequence=[GOLD, "#8aa0c3"],
    )
    chart.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color=INK)
    st.plotly_chart(chart, use_container_width=True)


def live_matches_page() -> None:
    page_header("即時賽況", "Live API / Mock 即時比分與比賽狀態")
    refresh_seconds = 60
    st.markdown(f"<meta http-equiv='refresh' content='{refresh_seconds}'>", unsafe_allow_html=True)

    live_result = cached_v29_live_matches()
    matches = live_result.data.copy()
    scoreboard = scoreboard_table(matches)

    if live_result.fallback_used:
        st.warning("目前使用備援資料")
    else:
        st.success(f"資料來源模式：{live_result.source_mode}")
    st.caption(f"最後更新時間：{live_result.updated_at}｜Provider：{live_result.provider}｜{live_result.message}")

    if scoreboard.empty:
        st.info("目前沒有可顯示的即時賽事資料。")
        return

    def match_option(row: pd.Series) -> str:
        minute = row.get("minute", "")
        minute_text = f"｜{int(minute)}'" if pd.notna(minute) and float(minute or 0) > 0 else ""
        return (
            f"{row.get('time_display', 'N/A')}｜"
            f"{row.get('home_team', 'TBD')} {row.get('score', '待開賽')} {row.get('away_team', 'TBD')}"
            f"｜{row.get('status_label', '資料待補')}{minute_text}"
        )

    options = {match_option(row): row.get("match_id") for _, row in scoreboard.iterrows()}
    selected = st.selectbox("選擇比賽", list(options.keys()))
    selected_id = options[selected]
    row = scoreboard[scoreboard["match_id"] == selected_id].iloc[0]

    st.markdown(
        f"""
        <div class="display-card score-card">
          <div class="score-teams">{html.escape(str(row.get('home_team', 'TBD')))} vs {html.escape(str(row.get('away_team', 'TBD')))}</div>
          <div class="score-value">{html.escape(str(row.get('score', '待開賽')))}</div>
          <div class="score-note">
            {html.escape(str(row.get('group', '資料待補')))}｜{html.escape(str(row.get('status_label', '資料待補')))}
            ｜分鐘：{html.escape(str(int(row.get('minute', 0) or 0)))}
            ｜時間：{html.escape(str(row.get('time_display', 'N/A')))}
            ｜場地：{html.escape(str(row.get('venue', '資料待補')))}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    table = scoreboard.rename(
        columns={
            "match_id": "比賽 ID",
            "time_display": "比賽時間",
            "group": "小組",
            "home_team": "主隊",
            "away_team": "客隊",
            "score": "比分",
            "status_label": "狀態",
            "minute": "比賽分鐘",
            "venue": "場地",
        }
    )
    st.dataframe(
        table[["比賽 ID", "比賽時間", "小組", "主隊", "比分", "客隊", "狀態", "比賽分鐘", "場地"]],
        use_container_width=True,
        hide_index=True,
    )


def _player_center_data() -> pd.DataFrame:
    data = player_database(players_df, team_meta_df).copy()
    defaults = {
        "player_name": "N/A",
        "team_zh": "N/A",
        "position": "N/A",
        "age": 0,
        "height_cm": 0,
        "club": "N/A",
        "national_caps": 0,
        "national_goals": 0,
        "assists": 0,
        "market_value_eur_m": 0.0,
        "fifa_ranking": 0,
        "flag_emoji": "",
        "goal_rate": 0.0,
        "assist_rate": 0.0,
        "recent_form_rating": 0.5,
    }
    for column, default in defaults.items():
        if column not in data.columns:
            data[column] = default

    numeric_columns = [
        "age",
        "height_cm",
        "national_caps",
        "national_goals",
        "assists",
        "market_value_eur_m",
        "fifa_ranking",
        "goal_rate",
        "assist_rate",
        "recent_form_rating",
    ]
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce").fillna(defaults.get(column, 0))

    safe_caps = pd.to_numeric(data["national_caps"], errors="coerce").replace(0, pd.NA)
    if (data["goal_rate"].fillna(0) == 0).all():
        data["goal_rate"] = (
            pd.to_numeric(data["national_goals"], errors="coerce") / safe_caps
        ).fillna(0.0)
    if (data["assist_rate"].fillna(0) == 0).all():
        data["assist_rate"] = (
            pd.to_numeric(data["assists"], errors="coerce") / safe_caps
        ).fillna(0.0)

    def clean_display_text(value) -> str:
        text = str(value or "").strip()
        if not text or text.lower() == "nan":
            return "N/A"
        mojibake_markers = ["嚙", "蝛", "蝳", "鞈", "�"]
        if any(marker in text for marker in mojibake_markers) or set(text) == {"?"}:
            return "N/A"
        return text

    for column in ["player_name", "team_zh", "position", "club"]:
        data[column] = data[column].map(clean_display_text)
    data["team_label"] = data["flag_emoji"].fillna("") + " " + data["team_zh"].fillna(data["team"])
    data["player_score"] = (
        data["recent_form_rating"] * 2
        + data["goal_rate"] * 10
        + data["assist_rate"] * 5
        + data["national_goals"] * 0.15
        + data["national_caps"] * 0.03
        + data["market_value_eur_m"] * 0.02
    )
    return data


def _render_team_key_players(player_data: pd.DataFrame, team: str) -> None:
    team_rows = player_data[player_data["team"] == team].copy()
    if team_rows.empty:
        st.info("目前尚未匯入該球隊球員資料")
        return
    team_rows = team_rows.sort_values("player_score", ascending=False).head(5)
    first = team_rows.iloc[0]
    st.markdown(
        f"""
        <div class="display-card">
          <div class="card-label">隊徽 / 國旗</div>
          <div class="card-value">{html.escape(str(first.get('flag_emoji', '')))} {html.escape(str(first.get('team_zh', team)))}</div>
          <div class="card-note">核心球員 Top5</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.dataframe(
        team_rows[
            [
                "player_name",
                "position",
                "age",
                "club",
                "national_caps",
                "national_goals",
                "assists",
                "market_value_eur_m",
                "fifa_ranking",
            ]
        ].rename(
            columns={
                "player_name": "姓名",
                "position": "位置",
                "age": "年齡",
                "club": "所屬俱樂部",
                "national_caps": "出場數",
                "national_goals": "進球",
                "assists": "助攻",
                "market_value_eur_m": "身價（百萬歐元）",
                "fifa_ranking": "國家隊排名",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )


def player_database_page() -> None:
    page_header("球員資料中心", "世界盃球員資料、搜尋篩選與核心球員 Top5")
    data = _player_center_data()
    if data.empty:
        st.info("目前尚未匯入該球隊球員資料")
        return

    teams = ["全部"] + sorted(data["team_zh"].dropna().unique().tolist())
    positions = ["全部"] + sorted(data["position"].dropna().unique().tolist())
    cols = st.columns([1.1, 1.0, 1.4])
    selected_team = cols[0].selectbox("國家篩選", teams)
    selected_position = cols[1].selectbox("位置篩選", positions)
    keyword = cols[2].text_input("搜尋球員", "")

    filtered = data.copy()
    if selected_team != "全部":
        filtered = filtered[filtered["team_zh"] == selected_team]
    if selected_position != "全部":
        filtered = filtered[filtered["position"] == selected_position]
    if keyword.strip():
        filtered = filtered[filtered["player_name"].str.contains(keyword.strip(), case=False, na=False)]

    metric_cols = st.columns(3)
    with metric_cols[0]:
        display_card("球員數", f"{len(filtered):,}", "目前篩選結果")
    with metric_cols[1]:
        display_card("國家隊數", f"{filtered['team'].nunique():,}", "涵蓋隊伍")
    with metric_cols[2]:
        avg_age = filtered["age"].mean() if not filtered.empty else 0
        display_card("平均年齡", f"{avg_age:.1f}", "歲")

    st.subheader("球員資料表")
    if filtered.empty:
        st.info("目前尚未匯入該球隊球員資料")
    else:
        table = filtered[
            [
                "player_name",
                "team_label",
                "position",
                "age",
                "height_cm",
                "club",
                "national_caps",
                "national_goals",
                "assists",
                "market_value_eur_m",
                "fifa_ranking",
            ]
        ].copy()
        table["height_cm"] = pd.to_numeric(table["height_cm"], errors="coerce").apply(
            lambda value: "N/A" if pd.isna(value) or value <= 0 else f"{int(value)} cm"
        )
        table["market_value_eur_m"] = pd.to_numeric(table["market_value_eur_m"], errors="coerce").apply(
            lambda value: "N/A" if pd.isna(value) or value <= 0 else f"{value:.1f}"
        )
        st.dataframe(
            table.rename(
                columns={
                    "player_name": "姓名",
                    "team_label": "國家",
                    "position": "位置",
                    "age": "年齡",
                    "height_cm": "身高",
                    "club": "所屬俱樂部",
                    "national_caps": "出場數",
                    "national_goals": "進球",
                    "assists": "助攻",
                    "market_value_eur_m": "身價（百萬歐元）",
                    "fifa_ranking": "國家隊排名",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("關鍵球員")
    team_options = sorted(data["team"].dropna().unique().tolist())
    default_teams = [team for team in ["Argentina", "France", "Brazil", "Spain"] if team in team_options]
    selected_key_teams = st.multiselect(
        "選擇國家隊",
        options=team_options,
        default=default_teams or team_options[:2],
    )
    if not selected_key_teams:
        st.info("目前尚未匯入該球隊球員資料")
        return
    for team in selected_key_teams:
        _render_team_key_players(data, team)


def elo_ranking_page() -> None:
    page_header("Elo 世界排名", "依本地近期賽果自動更新 Elo，呈現原始分數、更新後分數與最近 10 場戰績")
    st.caption(
        "預測權重參考："
        f"Elo {PREDICTION_WEIGHTS['elo']:.0%}、"
        f"近期狀態 {PREDICTION_WEIGHTS['recent_form']:.0%}、"
        f"歷史成績 {PREDICTION_WEIGHTS['worldcup_history']:.0%}。"
    )
    try:
        elo_results = pd.read_csv("data/elo_match_results.csv")
    except Exception:
        elo_results = pd.DataFrame()
    rankings = elo_ranking_with_updates(elo_results, team_meta_df)
    if rankings.empty:
        st.info("目前尚未匯入 Elo 更新資料")
        return

    rankings.insert(0, "排名", range(1, len(rankings) + 1))
    top10 = rankings.head(10).sort_values("updated_elo", ascending=True)
    chart = px.bar(
        top10,
        x="updated_elo",
        y="team_zh",
        orientation="h",
        color="elo_change",
        color_continuous_scale=["#8aa0c3", GOLD_LIGHT],
        labels={"updated_elo": "更新後 Elo", "team_zh": "國家", "elo_change": "Elo 變化"},
    )
    chart.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color=INK)
    st.plotly_chart(chart, use_container_width=True)

    st.dataframe(
        rankings[
            [
                "排名",
                "flag_emoji",
                "team_zh",
                "original_elo",
                "updated_elo",
                "elo_change",
                "recent_10",
            ]
        ].rename(
            columns={
                "flag_emoji": "國旗",
                "team_zh": "國家",
                "original_elo": "原始 Elo",
                "updated_elo": "更新後 Elo",
                "elo_change": "Elo 變化",
                "recent_10": "最近 10 場戰績",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )
    st.caption("預測權重：Elo 50%、近期狀態 30%、歷史成績 20%。")


def _player_impact_data() -> pd.DataFrame:
    try:
        data = pd.read_csv("data/player_impact.csv")
    except Exception:
        base = player_database(players_df, team_meta_df).rename(
            columns={
                "player_name": "player",
                "national_caps": "appearances",
                "national_goals": "goals",
            }
        )
        data = base
    return data


def _render_player_impact(home_team: str, away_team: str) -> None:
    st.subheader("球員影響模型")
    impact_data = _player_impact_data()
    home_top = team_impact(impact_data, home_team)
    away_top = team_impact(impact_data, away_team)
    home_adjust = win_probability_adjustment(impact_data, home_team)
    away_adjust = win_probability_adjustment(impact_data, away_team)

    cols = st.columns(2)
    for col, team, rows, adjustment in [
        (cols[0], home_team, home_top, home_adjust),
        (cols[1], away_team, away_top, away_adjust),
    ]:
        with col:
            display_card(team_name(team), f"{adjustment:+.1%}", "對勝率的輔助影響")
            if rows.empty:
                st.info("目前尚未匯入該隊球員資料")
            else:
                columns = ["player", "position", "appearances", "goals", "assists", "impact_score", "is_available"]
                if "availability_note" in rows.columns:
                    columns.append("availability_note")
                table = rows[columns].copy()
                st.dataframe(
                    table.rename(
                        columns={
                            "player": "球員",
                            "position": "位置",
                            "appearances": "出場數",
                            "goals": "進球",
                            "assists": "助攻",
                            "impact_score": "球員影響分數",
                            "is_available": "可出賽",
                            "availability_note": "影響說明",
                        }
                    ),
                    use_container_width=True,
                    hide_index=True,
                )


def match_analysis_page() -> None:
    page_header("單場分析", "比分預測、勝平負機率、歷史資料與球員影響模型")
    disclaimer_box()
    row = selected_fixture()
    prediction = predict_match(matches_df, row["home_team"], row["away_team"], wc_team_stats_df)
    market_table = market_probability_table(row, prediction)
    prediction_cards(row)

    probabilities = prediction_to_frame(prediction)
    display_df = probabilities.copy()
    probability_column = display_df.columns[1]
    display_df[probability_column] = (
        pd.to_numeric(display_df[probability_column], errors="coerce")
        .fillna(0)
        .map(format_percent)
    )
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
    )

    render_pre_match_analysis_card(row, prediction)
    render_worldcup_history_block(row["home_team"], row["away_team"])
    render_h2h_summary_block(row["home_team"], row["away_team"])
    render_key_players_block(row["home_team"], row["away_team"])
    _render_player_impact(row["home_team"], row["away_team"])
    render_ai_match_report(row, prediction, market_table)


def player_impact_analysis_page() -> None:
    page_header("球員影響分析", "球員評分、傷病模擬、缺陣影響與勝率變化")
    st.caption("本頁使用本地球員資料與 player_status.csv fallback；缺資料時會以保守假設顯示。")
    players = normalize_players(load_player_pool())
    if players.empty:
        st.info("目前尚未匯入球員資料。")
        return

    row = selected_fixture("選擇比賽")
    home_team = row["home_team"]
    away_team = row["away_team"]
    prediction = predict_match(matches_df, home_team, away_team, wc_team_stats_df)
    sim_df = v7_simulation()

    st.subheader(f"{fixture_time_text(row)} ｜ {matchup_text(row)}")
    cols = st.columns(3)
    with cols[0]:
        display_card("原主勝率", format_percent(float(prediction.home_win_probability)), team_name(home_team))
    with cols[1]:
        display_card("原和局率", format_percent(float(prediction.draw_probability)), "Poisson + Elo")
    with cols[2]:
        display_card("原客勝率", format_percent(float(prediction.away_win_probability)), team_name(away_team))

    home_all_players = team_player_impact(players, home_team, top_n=60)
    away_all_players = team_player_impact(players, away_team, top_n=60)
    home_players = home_all_players.head(10).copy()
    away_players = away_all_players.head(10).copy()
    left, right = st.columns(2)
    with left:
        st.subheader(f"{team_name(home_team)} 核心球員")
        if home_all_players.empty:
            st.info("目前尚未匯入該隊球員資料")
            home_out = []
        else:
            home_out = st.multiselect(
                "模擬主隊缺陣",
                home_all_players["player_name"].tolist(),
                default=home_all_players.loc[~home_all_players["is_available"], "player_name"].tolist()[:3],
                key="home_player_absence",
            )
            st.dataframe(
                home_players[["player_name", "position", "club", "appearances", "goals", "assists", "recent_form", "impact_score", "status"]].rename(
                    columns={
                        "player_name": "球員",
                        "position": "位置",
                        "club": "俱樂部",
                        "appearances": "出場",
                        "goals": "進球",
                        "assists": "助攻",
                        "recent_form": "近期狀態",
                        "impact_score": "影響分數",
                        "status": "狀態",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )
    with right:
        st.subheader(f"{team_name(away_team)} 核心球員")
        if away_all_players.empty:
            st.info("目前尚未匯入該隊球員資料")
            away_out = []
        else:
            away_out = st.multiselect(
                "模擬客隊缺陣",
                away_all_players["player_name"].tolist(),
                default=away_all_players.loc[~away_all_players["is_available"], "player_name"].tolist()[:3],
                key="away_player_absence",
            )
            st.dataframe(
                away_players[["player_name", "position", "club", "appearances", "goals", "assists", "recent_form", "impact_score", "status"]].rename(
                    columns={
                        "player_name": "球員",
                        "position": "位置",
                        "club": "俱樂部",
                        "appearances": "出場",
                        "goals": "進球",
                        "assists": "助攻",
                        "recent_form": "近期狀態",
                        "impact_score": "影響分數",
                        "status": "狀態",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )

    home_champion = sim_df.loc[sim_df["team"].eq(home_team), "champion_probability"]
    away_champion = sim_df.loc[sim_df["team"].eq(away_team), "champion_probability"]
    home_champion_base = float(home_champion.iloc[0]) if not home_champion.empty else 0.0
    away_champion_base = float(away_champion.iloc[0]) if not away_champion.empty else 0.0
    impact = match_player_impact(
        players,
        home_team,
        away_team,
        home_out,
        away_out,
        prediction=prediction,
        home_champion_probability=home_champion_base,
        away_champion_probability=away_champion_base,
    )
    base_values = impact["base"]
    adjusted_values = impact["adjusted"]
    adjusted_home_win = float(adjusted_values["home_win_probability"])
    adjusted_draw = float(adjusted_values["draw_probability"])
    adjusted_away_win = float(adjusted_values["away_win_probability"])
    adjusted_home_xg = float(adjusted_values["home_xg"])
    adjusted_away_xg = float(adjusted_values["away_xg"])

    st.subheader("缺陣影響摘要")
    metric_cols = st.columns(4)
    with metric_cols[0]:
        display_card(
            "主勝率",
            f"{format_percent(float(base_values['home_win_probability']))} → {format_percent(adjusted_home_win)}",
            f"差異 {format_percent(float(impact['home_win_probability_change']))}",
        )
    with metric_cols[1]:
        display_card(
            "客勝率",
            f"{format_percent(float(base_values['away_win_probability']))} → {format_percent(adjusted_away_win)}",
            f"差異 {format_percent(float(impact['away_win_probability_change']))}",
        )
    with metric_cols[2]:
        display_card(
            "主隊 xG",
            f"{float(base_values['home_xg']):.2f} → {adjusted_home_xg:.2f}",
            f"差異 {float(impact['home_xg_change']):+.2f}",
        )
    with metric_cols[3]:
        display_card(
            "客隊 xG",
            f"{float(base_values['away_xg']):.2f} → {adjusted_away_xg:.2f}",
            f"差異 {float(impact['away_xg_change']):+.2f}",
        )

    score_cols = st.columns(3)
    with score_cols[0]:
        display_card("預測比分", str(impact["predicted_score_change"]), "Poisson 重新計算")
    with score_cols[1]:
        display_card(
            "和局率",
            f"{format_percent(float(base_values['draw_probability']))} → {format_percent(adjusted_draw)}",
            f"差異 {format_percent(float(impact['draw_probability_change']))}",
        )
    with score_cols[2]:
        total_loss = float(impact["home"]["impact_loss"]) + float(impact["away"]["impact_loss"])
        display_card("總影響損失", f"{total_loss:.1f}", "依 impact_score 加總")

    champion_cols = st.columns(2)
    with champion_cols[0]:
        display_card(
            "主隊冠軍率",
            f"{format_percent(home_champion_base)} → {format_percent(max(0, home_champion_base + float(impact['home']['champion_probability_delta'])))}",
            f"差異 {format_percent(float(impact['home']['champion_probability_delta']))}",
        )
    with champion_cols[1]:
        display_card(
            "客隊冠軍率",
            f"{format_percent(away_champion_base)} → {format_percent(max(0, away_champion_base + float(impact['away']['champion_probability_delta'])))}",
            f"差異 {format_percent(float(impact['away']['champion_probability_delta']))}",
        )

    scenario_df = pd.DataFrame(
        [
            {
                "球隊": team_name(home_team),
                "原勝率": base_values["home_win_probability"],
                "調整後勝率": adjusted_home_win,
                "勝率變化": impact["home_win_probability_change"],
                "原 xG": base_values["home_xg"],
                "調整後 xG": adjusted_home_xg,
                "xG 變化": impact["home_xg_change"],
                "冠軍率變化": impact["home"]["champion_probability_delta"],
            },
            {
                "球隊": team_name(away_team),
                "原勝率": base_values["away_win_probability"],
                "調整後勝率": adjusted_away_win,
                "勝率變化": impact["away_win_probability_change"],
                "原 xG": base_values["away_xg"],
                "調整後 xG": adjusted_away_xg,
                "xG 變化": impact["away_xg_change"],
                "冠軍率變化": impact["away"]["champion_probability_delta"],
            },
        ]
    )
    for column in ["原勝率", "調整後勝率", "勝率變化", "冠軍率變化"]:
        scenario_df[column] = pd.to_numeric(scenario_df[column], errors="coerce").fillna(0).map(format_percent)
    st.dataframe(scenario_df, use_container_width=True, hide_index=True)

    ranking_df = match_absence_ranking(players, home_team, away_team, prediction=prediction, top_n=10)
    if not ranking_df.empty:
        st.subheader("球員缺陣影響排行 Top 10")
        ranking_display = ranking_df.rename(
            columns={
                "player_name": "球員",
                "team": "國家",
                "position": "位置",
                "impact_score": "影響分數",
                "win_probability_loss": "勝率損失",
            }
        )
        ranking_display["國家"] = ranking_display["國家"].map(team_name)
        ranking_display["勝率損失"] = pd.to_numeric(ranking_display["勝率損失"], errors="coerce").fillna(0).map(format_percent)
        st.dataframe(ranking_display, use_container_width=True, hide_index=True)
        ranking_chart = px.bar(
            ranking_df.sort_values("win_probability_loss", ascending=True),
            x="win_probability_loss",
            y="player_name",
            color="impact_score",
            orientation="h",
            labels={"win_probability_loss": "勝率損失", "player_name": "球員", "impact_score": "影響分數"},
            color_continuous_scale=["#415a77", GOLD_LIGHT],
        )
        ranking_chart.update_xaxes(tickformat=".1%")
        ranking_chart.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color=INK)
        st.plotly_chart(ranking_chart, use_container_width=True)

    try:
        log_df = pd.read_csv("data/player_impact_log.csv")
    except Exception:
        log_df = pd.DataFrame()
    if not log_df.empty:
        st.subheader("球員影響紀錄")
        st.dataframe(log_df, use_container_width=True, hide_index=True)


def worldcup_simulator_page() -> None:
    page_header("世界盃模擬器", "Monte Carlo 高次數模擬：小組出線、淘汰賽晉級與冠軍率")
    simulations = st.selectbox("模擬次數", [1000, 10000, 50000], index=0)
    run_clicked = st.button("開始模擬", type="primary")
    if not run_clicked:
        st.info("請選擇模擬次數後按下「開始模擬」。")
        return

    with st.spinner("正在執行 Monte Carlo 模擬..."):
        sim_df = run_worldcup_monte_carlo(
            fixture_odds_df,
            team_meta_df,
            wc_team_stats_df,
            matches_df,
            simulations=int(simulations),
        )
    if "round_32_probability" not in sim_df.columns:
        sim_df["round_32_probability"] = sim_df.get("group_qualified_probability", 0)
    sim_df = sim_df.sort_values("champion_probability", ascending=False).reset_index(drop=True)

    display_card("模擬次數", f"{int(simulations):,}", "Monte Carlo")
    st.subheader("冠軍率 Top20")
    top20 = sim_df.head(20).sort_values("champion_probability", ascending=True).copy()
    top20["label"] = top20["champion_probability"].map(format_percent)
    chart = px.bar(
        top20,
        x="champion_probability",
        y="team_display",
        text="label",
        orientation="h",
        color="champion_probability",
        color_continuous_scale=["#415a77", GOLD_LIGHT],
        labels={"champion_probability": "冠軍率", "team_display": "球隊"},
    )
    chart.update_xaxes(tickformat=".0%")
    chart.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color=INK, coloraxis_showscale=False)
    st.plotly_chart(chart, use_container_width=True)

    table = sim_df[
        [
            "team_display",
            "group_qualified_probability",
            "round_32_probability",
            "round_16_probability",
            "round_8_probability",
            "semi_final_probability",
            "final_probability",
            "champion_probability",
        ]
    ].copy()
    for column in table.columns[1:]:
        table[column] = table[column].map(format_percent)
    st.dataframe(
        table.rename(
            columns={
                "team_display": "球隊",
                "group_qualified_probability": "小組出線率",
                "round_32_probability": "32 強率",
                "round_16_probability": "16 強率",
                "round_8_probability": "8 強率",
                "semi_final_probability": "4 強率",
                "final_probability": "決賽率",
                "champion_probability": "冠軍率",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )
    st.caption("模型使用 Elo、近期狀態、歷史世界盃表現與 Poisson 進球分布；結果僅供資料分析參考，不保證準確。")


def ai_worldcup_simulator_page() -> None:
    page_header("AI 世界盃模擬器", "Monte Carlo 模擬、決賽組合、黑馬榜與洲別冠軍機率")
    st.caption("本頁使用可解釋的 Elo + Poisson + Monte Carlo 模擬；結果僅供資料分析。")
    simulations = st.selectbox("模擬次數", [1000, 5000, 10000], index=0, key="ai_mc_runs")
    run_clicked = st.button("開始 AI 模擬", type="primary", use_container_width=True)
    if not run_clicked:
        st.info("請選擇模擬次數後開始模擬。")
        return

    with st.spinner("正在執行 AI Monte Carlo 模擬..."):
        sim_df = run_ai_monte_carlo(
            fixture_odds_df,
            team_meta_df,
            wc_team_stats_df,
            matches_df,
            simulations=int(simulations),
        )
    if sim_df.empty:
        st.info("目前模擬資料不足。")
        return

    champion = sim_df.sort_values("champion_probability", ascending=False).iloc[0]
    cols = st.columns(4)
    with cols[0]:
        display_card("模擬次數", f"{int(simulations):,}", "Monte Carlo")
    with cols[1]:
        display_card("冠軍率最高", format_percent(float(champion["champion_probability"])), str(champion["team_display"]))
    with cols[2]:
        display_card("平均出線率", format_percent(float(sim_df["group_qualified_probability"].mean())), "48 隊")
    with cols[3]:
        display_card("資料模式", "Fallback Ready", "不足時保守計算")

    st.subheader("冠軍率 Top20")
    top20 = sim_df.sort_values("champion_probability", ascending=False).head(20).copy()
    top20_chart = top20.sort_values("champion_probability", ascending=True).copy()
    top20_chart["label"] = top20_chart["champion_probability"].map(format_percent)
    chart = px.bar(
        top20_chart,
        x="champion_probability",
        y="team_display",
        orientation="h",
        text="label",
        color="champion_probability",
        color_continuous_scale=["#415a77", GOLD_LIGHT],
    )
    chart.update_xaxes(tickformat=".0%")
    chart.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color=INK, coloraxis_showscale=False)
    st.plotly_chart(chart, use_container_width=True)

    stage_table = ai_stage_probability_table(sim_df).sort_values("champion_probability", ascending=False).copy()
    for column in stage_table.columns:
        if column != "team_display":
            stage_table[column] = pd.to_numeric(stage_table[column], errors="coerce").fillna(0).map(format_percent)
    st.subheader("晉級機率總表")
    st.dataframe(
        stage_table.rename(
            columns={
                "team_display": "球隊",
                "group_qualified_probability": "小組出線率",
                "round_16_probability": "16強率",
                "round_8_probability": "8強率",
                "semi_final_probability": "4強率",
                "final_probability": "決賽率",
                "champion_probability": "奪冠率",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    left, right = st.columns(2)
    with left:
        st.subheader("最常見決賽組合")
        finals = common_final_combinations(sim_df)
        if finals.empty:
            st.info("決賽組合資料不足。")
        else:
            finals["estimated_probability"] = finals["estimated_probability"].map(format_percent)
            st.dataframe(finals.rename(columns={"final_combo": "決賽組合", "estimated_probability": "估算機率"}), use_container_width=True, hide_index=True)
    with right:
        st.subheader("黑馬排行榜")
        dark = dark_horse_ranking(sim_df)
        if dark.empty:
            st.info("黑馬資料不足。")
        else:
            dark_view = dark[["team_display", "elo", "champion_probability", "dark_horse_score"]].copy()
            dark_view["champion_probability"] = dark_view["champion_probability"].map(format_percent)
            dark_view["dark_horse_score"] = pd.to_numeric(dark_view["dark_horse_score"], errors="coerce").fillna(0).round(4)
            st.dataframe(dark_view.rename(columns={"team_display": "球隊", "elo": "Elo", "champion_probability": "冠軍率", "dark_horse_score": "黑馬分數"}), use_container_width=True, hide_index=True)

    st.subheader("洲別冠軍機率")
    confed = continent_champion_probabilities(sim_df, team_meta_df)
    missing_teams = []
    if not confed.empty and "missing_teams" in confed.columns:
        missing_text = " ".join(confed["missing_teams"].dropna().astype(str).tolist()).strip()
        missing_teams = [team.strip() for team in missing_text.split(",") if team.strip()]
    confed_chart = confed[
        confed["confederation"].isin(["UEFA", "CONMEBOL", "AFC", "CAF", "CONCACAF", "OFC"])
    ].copy() if not confed.empty and "confederation" in confed.columns else pd.DataFrame()
    confed_chart["champion_probability"] = pd.to_numeric(confed_chart.get("champion_probability", 0), errors="coerce").fillna(0)
    if confed_chart.empty or confed_chart["champion_probability"].sum() <= 0:
        st.info("洲別資料不足。")
        if missing_teams:
            st.warning("資料待補：" + "、".join(missing_teams[:20]))
    else:
        pie = px.pie(confed_chart, names="confederation", values="champion_probability", color_discrete_sequence=[GOLD, "#8aa0c3", "#2dd4bf", "#f97316", "#a78bfa", "#ef4444"])
        pie.update_traces(textinfo="label+percent")
        pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color=INK)
        st.plotly_chart(pie, use_container_width=True)
        if missing_teams:
            st.warning("資料待補：" + "、".join(missing_teams[:20]))


def champion_path_page() -> None:
    page_header("冠軍路徑模擬", "選擇國家隊，查看各階段晉級率、可能淘汰賽路徑與最大阻礙")
    st.caption("本頁使用既有 Monte Carlo 模擬結果，僅供機率分析與資料參考。")
    sim_df = v7_simulation().copy()
    if sim_df.empty:
        st.info("目前資料不足，暫無冠軍路徑模擬結果。")
        return
    if "round_32_probability" not in sim_df.columns:
        sim_df["round_32_probability"] = sim_df.get("group_qualified_probability", 0)
    sim_df = sim_df.sort_values("team_display")
    selected_display = st.selectbox("選擇國家隊", sim_df["team_display"].tolist())
    team_row = sim_df[sim_df["team_display"] == selected_display].iloc[0]

    cols = st.columns(4)
    with cols[0]:
        display_card("選擇球隊", str(team_row["team_display"]), f"Elo {int(float(team_row.get('elo', 0) or 0))}")
    with cols[1]:
        display_card("小組出線率", format_percent(float(team_row.get("group_qualified_probability", 0))), "Monte Carlo")
    with cols[2]:
        display_card("決賽率", format_percent(float(team_row.get("final_probability", 0))), "淘汰賽路徑")
    with cols[3]:
        display_card("冠軍率", format_percent(float(team_row.get("champion_probability", 0))), "最終模擬")

    stage_df = stage_probability_table(team_row)
    stage_df["百分比"] = stage_df["機率"].map(format_percent)
    st.subheader("階段晉級機率")
    for _, item in stage_df.iterrows():
        st.progress(float(item["機率"]), text=f"{item['階段']}：{item['百分比']}")
    chart = px.line(
        stage_df,
        x="階段",
        y="機率",
        markers=True,
        text="百分比",
        labels={"機率": "晉級機率"},
    )
    chart.update_yaxes(tickformat=".0%")
    chart.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color=INK)
    st.plotly_chart(chart, use_container_width=True)

    st.subheader("可能淘汰賽路徑")
    path_df = likely_knockout_path(team_row, sim_df)
    if path_df.empty:
        st.info("目前資料不足，暫無可能路徑。")
    else:
        display_path = path_df.copy()
        display_path["對手冠軍率"] = pd.to_numeric(display_path["對手冠軍率"], errors="coerce").fillna(0).map(format_percent)
        st.dataframe(display_path, use_container_width=True, hide_index=True)

    st.subheader("AI 路徑分析")
    st.markdown(f"<div class='display-card'><div class='card-note'>{html.escape(champion_path_text(team_row, sim_df))}</div></div>", unsafe_allow_html=True)


def team_comparison_page() -> None:
    page_header("對戰比較中心", "選擇兩支球隊，比較 Elo、FIFA 排名、xG、晉級率與冠軍率")
    teams = sorted(team_meta_df["team"].dropna().astype(str).unique().tolist())
    if len(teams) < 2:
        st.info("目前資料不足，暫無法進行對戰比較。")
        return
    cols = st.columns(2)
    with cols[0]:
        home = st.selectbox("球隊 A", teams, format_func=team_name, index=0)
    with cols[1]:
        default_index = 1 if len(teams) > 1 else 0
        away = st.selectbox("球隊 B", teams, format_func=team_name, index=default_index)
    if home == away:
        st.info("請選擇兩支不同球隊。")
        return

    sim_df = v7_simulation().copy()
    try:
        shots_df = prepare_xg_data(pd.read_csv("data/xg_shots.csv"))
    except Exception:
        shots_df = pd.DataFrame()
    try:
        players_df = pd.read_csv("data/player_database.csv")
    except Exception:
        players_df = pd.DataFrame()

    home_profile = team_profile(home, team_meta_df, players_df, sim_df, wc_team_stats_df, shots_df)
    away_profile = team_profile(away, team_meta_df, players_df, sim_df, wc_team_stats_df, shots_df)
    home_name = team_name(home)
    away_name = team_name(away)

    prediction = predict_match(matches_df, home, away, wc_team_stats_df)
    metric_cols = st.columns(3)
    with metric_cols[0]:
        display_card("主隊勝率", format_percent(float(prediction.home_win_probability)), home_name)
    with metric_cols[1]:
        display_card("和局機率", format_percent(float(prediction.draw_probability)), "Poisson + Elo")
    with metric_cols[2]:
        display_card("客隊勝率", format_percent(float(prediction.away_win_probability)), away_name)

    st.subheader("比較表")
    table = comparison_table(home_profile, away_profile, home_name, away_name)
    display = table.copy()
    for column in [home_name, away_name]:
        display[column] = display.apply(
            lambda row: format_percent(float(row[column])) if row["指標"] in ["近期狀態", "冠軍率", "小組出線率"] else row[column],
            axis=1,
        )
    display = display.astype(str)
    st.dataframe(display, use_container_width=True, hide_index=True)

    st.subheader("雷達圖")
    home_radar = radar_values(home_profile)
    away_radar = radar_values(away_profile)
    categories = list(home_radar.keys())
    radar_df = pd.DataFrame(
        {
            "指標": categories + categories,
            "分數": [home_radar[item] for item in categories] + [away_radar[item] for item in categories],
            "球隊": [home_name] * len(categories) + [away_name] * len(categories),
        }
    )
    chart = px.line_polar(radar_df, r="分數", theta="指標", color="球隊", line_close=True, range_r=[0, 1])
    chart.update_traces(fill="toself")
    chart.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color=INK)
    st.plotly_chart(chart, use_container_width=True)

    st.subheader("AI 對戰分析")
    text = comparison_text(home_name, home_profile, away_name, away_profile)
    st.markdown(f"<div class='display-card'><div class='card-note'>{html.escape(text)}</div></div>", unsafe_allow_html=True)


def ai_worldcup_analyst_page() -> None:
    page_header("AI 世界盃分析師", "以本地賽程、Elo、xG、球員與模擬資料回答常見世界盃問題")
    examples = [
        "法國奪冠機率多少？",
        "巴西跟阿根廷誰比較強？",
        "今天有哪些比賽？",
        "哪隊 Elo 最高？",
        "哪隊 xG 表現最好？",
        "哪場比賽最膠著？",
        "哪支球隊冠軍路徑最難？",
    ]
    if "ai_question" not in st.session_state:
        st.session_state["ai_question"] = examples[0]

    st.caption("規則式問答，未串接外部 AI API；回答僅供資料分析參考。")
    button_cols = st.columns(2)
    for idx, example in enumerate(examples):
        with button_cols[idx % 2]:
            if st.button(example, key=f"ai_example_{idx}", use_container_width=True):
                st.session_state["ai_question"] = example

    question = st.text_input("輸入問題", key="ai_question")
    try:
        shots_df = prepare_xg_data(pd.read_csv("data/xg_shots.csv"))
    except Exception:
        shots_df = pd.DataFrame()
    try:
        players_df = pd.read_csv("data/player_database.csv")
    except Exception:
        players_df = pd.DataFrame()

    answer = answer_question(
        question,
        fixture_odds_df,
        team_meta_df,
        v7_simulation(),
        shots_df,
        players_df,
        matches_df,
    )
    st.subheader("回答")
    st.markdown(f"<div class='display-card'><div class='card-note'>{html.escape(answer)}</div></div>", unsafe_allow_html=True)


def betting_page() -> None:
    page_header("市場機率分析", "將模型機率與賠率隱含機率融合，提供風險參考")
    st.caption("融合公式：模型機率 70% + 市場隱含機率 30%。市場隱含機率已正規化以降低 bookmaker margin 影響。")
    disclaimer_box()
    row = selected_fixture()
    prediction = predict_match(matches_df, row["home_team"], row["away_team"], wc_team_stats_df)
    table = market_probability_table(row, prediction)
    confidence_result = confidence_for_fixture(row, prediction, table)

    st.subheader(f"{fixture_time_text(row)} ｜ {matchup_text(row)}")
    smart_confidence_card(confidence_result)
    display = table.copy()
    for column in ["model_probability", "market_probability", "fused_probability"]:
        display[column] = (
            pd.to_numeric(display[column], errors="coerce")
            .fillna(0)
            .map(format_percent)
        )
    st.dataframe(
        display.rename(
            columns={
                "market": "結果",
                "model_probability": "模型機率",
                "market_probability": "市場機率",
                "fused_probability": "融合後機率",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )
    chart_data = table.melt(id_vars="market", var_name="type", value_name="probability")
    chart = px.bar(
        chart_data,
        x="market",
        y="probability",
        color="type",
        barmode="group",
        labels={"market": "結果", "probability": "機率", "type": "來源"},
        color_discrete_sequence=[GOLD, "#8aa0c3", "#28a745"],
    )
    chart.update_yaxes(tickformat=".0%")
    chart.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color=INK)
    st.plotly_chart(chart, use_container_width=True)
    render_ai_match_report(row, prediction, table, confidence_result)
    render_ai_parlay_analysis(limit=5)
    st.warning("風險提醒：本頁僅做市場機率與模型機率比較，不提供下注功能，不保證賽果或獲利。")


def football_betting_guide_page() -> None:
    page_header("世足玩法教學", "用資料分析角度理解常見足球玩法與風險")
    st.warning("風險提醒：以下內容僅為玩法與機率概念說明，不鼓勵下注，不保證任何結果。")

    topics = [
        {
            "title": "不讓分（勝平負）",
            "body": "預測 90 分鐘正規時間的主勝、和局或客勝。這是最直覺的足球結果市場。",
            "example": "阿根廷 vs 法國：主勝 2.10、和局 3.30、客勝 3.40",
            "risk": "足球和局機率高，熱門隊不一定能在正規時間勝出。",
        },
        {
            "title": "讓分盤",
            "body": "強隊先被扣分或弱隊先加分，再判斷投注結果。常用來平衡雙方實力差距。",
            "example": "巴西 -1：巴西需贏 2 球以上才算過盤。",
            "risk": "即使強隊贏球，也可能因贏不夠多而未過盤。",
        },
        {
            "title": "大小球",
            "body": "預測兩隊總進球數高於或低於指定門檻，例如 2.5 球。",
            "example": "大 2.5：比賽總進球 3 球以上成立。",
            "risk": "淘汰賽或關鍵戰常偏保守，進球數波動較大。",
        },
        {
            "title": "半全場",
            "body": "同時預測半場結果與全場結果，例如半場和局、全場主勝。",
            "example": "半場和 / 全場主：上半場平手，終場主隊勝。",
            "risk": "需要同時猜中兩個階段，難度明顯高於單一勝平負。",
        },
        {
            "title": "串關",
            "body": "把多場賽事的賠率相乘，全部命中才成立。",
            "example": "2.00 × 1.80 × 1.60 = 總賠率 5.76。",
            "risk": "總賠率提高，但任一場失準就會使整組失敗。",
        },
    ]

    for item in topics:
        st.markdown(
            f"""
            <div class="display-card">
              <div class="card-label">{html.escape(item['title'])}</div>
              <div class="card-value" style="font-size:1.05rem;line-height:1.65;">{html.escape(item['body'])}</div>
              <div class="card-note">範例：{html.escape(item['example'])}</div>
              <div class="risk-tag risk-medium">風險：{html.escape(item['risk'])}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    example_df = pd.DataFrame(
        [
            {"玩法": "不讓分", "判斷方式": "主勝 / 和局 / 客勝", "常見用途": "基本賽果判斷"},
            {"玩法": "讓分盤", "判斷方式": "套用讓分後比較", "常見用途": "強弱差距較大時"},
            {"玩法": "大小球", "判斷方式": "總進球高於或低於門檻", "常見用途": "預估比賽節奏"},
            {"玩法": "半全場", "判斷方式": "半場結果 + 全場結果", "常見用途": "高賠率情境"},
            {"玩法": "串關", "判斷方式": "多場全部命中", "常見用途": "小額高倍率試算"},
        ]
    )
    st.subheader("玩法比較表")
    st.dataframe(example_df, use_container_width=True, hide_index=True)

    risk_df = pd.DataFrame(
        [
            {"玩法": "不讓分", "相對難度": 2},
            {"玩法": "讓分盤", "相對難度": 3},
            {"玩法": "大小球", "相對難度": 3},
            {"玩法": "半全場", "相對難度": 5},
            {"玩法": "串關", "相對難度": 5},
        ]
    )
    chart = px.bar(risk_df, x="玩法", y="相對難度", color="相對難度", color_continuous_scale=["#415a77", GOLD_LIGHT])
    chart.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color=INK, coloraxis_showscale=False)
    st.plotly_chart(chart, use_container_width=True)


def odds_calculator_page() -> None:
    page_header("賠率試算中心", "依比賽自動帶入模型機率、市場機率與玩法賠率試算")
    st.warning(
        "本頁僅供機率分析與賠率試算，不構成下注建議。賠率越高不代表越值得，"
        "串關場數越多，中獎機率通常越低，請理性看待所有模型結果。"
    )

    def clamp_probability(value: float) -> float:
        return float(max(0.01, min(0.95, value)))

    def estimated_odds(probability: float) -> float:
        return round(max(1.01, min(15.0, 1.04 / clamp_probability(probability))), 2)

    def risk_level(probability: float) -> str:
        if probability >= 0.58:
            return "低"
        if probability >= 0.40:
            return "中"
        return "高"

    def risk_class(level: str) -> str:
        return {"低": "risk-low", "中": "risk-medium", "高": "risk-high"}.get(level, "risk-medium")

    def poisson_score_grid(home_xg: float, away_xg: float, max_goals: int = 7) -> list[dict]:
        rows = []
        for home_goals in range(max_goals + 1):
            for away_goals in range(max_goals + 1):
                probability = poisson_probability(home_xg, home_goals) * poisson_probability(away_xg, away_goals)
                rows.append({"home_goals": home_goals, "away_goals": away_goals, "probability": probability})
        total = sum(row["probability"] for row in rows) or 1
        for row in rows:
            row["probability"] = row["probability"] / total
        return rows

    def display_play_card(play: dict, key: str) -> None:
        model_text = format_percent(float(play.get("model_probability", 0)))
        market_value = play.get("market_probability")
        market_text = "N/A" if market_value == "N/A" else format_percent(float(market_value))
        fused_text = format_percent(float(play.get("fused_probability", play.get("model_probability", 0))))
        odds_text = f"{float(play.get('odds', 1.01)):.2f}"
        level = str(play.get("risk", "中"))
        st.markdown(
            f"""
            <div class="display-card">
              <div class="card-label">{html.escape(str(play.get('category', '玩法')))}</div>
              <div class="card-value" style="font-size:1.2rem;">{html.escape(str(play.get('name', 'N/A')))}</div>
              <div class="card-note">模型機率：{model_text}｜市場機率：{market_text}｜融合機率：{fused_text}</div>
              <div class="card-note">賠率：{odds_text}</div>
              <div class="risk-tag {risk_class(level)}">風險：{html.escape(level)}</div>
              <div class="card-note">{html.escape(str(play.get('note', '僅供機率分析參考。')))}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("套用此玩法到試算器", key=f"apply_{key}", use_container_width=True):
            st.session_state["applied_single_odds"] = float(play.get("odds", 1.01))
            st.session_state["applied_play_name"] = str(play.get("name", "自選玩法"))
            st.session_state["single_odds"] = float(play.get("odds", 1.01))
            st.session_state["parlay_odds_0"] = float(play.get("odds", 1.01))
            st.success(f"已套用：{play.get('name', '自選玩法')}，賠率 {odds_text}")

    st.subheader("選擇比賽")
    row = selected_fixture("選擇比賽")
    prediction = predict_match(matches_df, row["home_team"], row["away_team"], wc_team_stats_df)
    market_table = market_probability_table(row, prediction)
    confidence_result = confidence_for_fixture(row, prediction, market_table)
    score_grid = poisson_score_grid(prediction.expected_home_goals, prediction.expected_away_goals)
    total_goal_estimate = prediction.expected_home_goals + prediction.expected_away_goals

    cols = st.columns(4)
    with cols[0]:
        display_card("比賽時間", fixture_time_text(row), "台灣時間 UTC+8")
    with cols[1]:
        display_card("主隊", team_name(row["home_team"]), str(row.get("venue", "N/A")))
    with cols[2]:
        display_card("客隊", team_name(row["away_team"]), "場地如賽程資料")
    with cols[3]:
        display_card("模型預測比分", f"{prediction.predicted_home_goals} : {prediction.predicted_away_goals}", "Poisson")

    prob_cols = st.columns(3)
    with prob_cols[0]:
        display_card("主勝機率", format_percent(prediction.home_win_probability), f"賠率 {float(row['home_odds']):.2f}")
    with prob_cols[1]:
        display_card("和局機率", format_percent(prediction.draw_probability), f"賠率 {float(row['draw_odds']):.2f}")
    with prob_cols[2]:
        display_card("客勝機率", format_percent(prediction.away_win_probability), f"賠率 {float(row['away_odds']):.2f}")

    smart_confidence_card(confidence_result)
    render_ai_match_report(row, prediction, market_table, confidence_result)

    st.subheader("不讓分勝平負")
    odds_map = {"主勝": row["home_odds"], "和局": row["draw_odds"], "客勝": row["away_odds"]}
    one_x_two = []
    for _, item in market_table.iterrows():
        play = {
            "category": "不讓分",
            "name": str(item["market"]),
            "model_probability": float(item["model_probability"]),
            "market_probability": float(item["market_probability"]),
            "fused_probability": float(item["fused_probability"]),
            "odds": float(odds_map.get(str(item["market"]), estimated_odds(float(item["fused_probability"])))),
            "risk": risk_level(float(item["fused_probability"])),
            "note": "比較模型與市場定價後的勝平負機率。",
        }
        one_x_two.append(play)
    st.dataframe(
        pd.DataFrame(one_x_two)[["name", "odds", "model_probability", "market_probability", "fused_probability", "risk"]]
        .rename(
            columns={
                "name": "結果",
                "odds": "對應賠率",
                "model_probability": "模型機率",
                "market_probability": "市場機率",
                "fused_probability": "融合後機率",
                "risk": "風險等級",
            }
        )
        .assign(
            模型機率=lambda df: df["模型機率"].map(format_percent),
            市場機率=lambda df: df["市場機率"].map(format_percent),
            融合後機率=lambda df: df["融合後機率"].map(format_percent),
        ),
        use_container_width=True,
        hide_index=True,
    )
    for index, play in enumerate(one_x_two):
        display_play_card(play, f"one_x_two_{index}")

    st.subheader("讓分盤")
    handicap_specs = [
        ("主隊 -1", lambda s: s["home_goals"] - 1 > s["away_goals"]),
        ("主隊 +1", lambda s: s["home_goals"] + 1 > s["away_goals"]),
        ("客隊 -1", lambda s: s["away_goals"] - 1 > s["home_goals"]),
        ("客隊 +1", lambda s: s["away_goals"] + 1 > s["home_goals"]),
    ]
    handicap_rows = []
    for name, condition in handicap_specs:
        probability = sum(score["probability"] for score in score_grid if condition(score))
        probability = clamp_probability(probability)
        handicap_rows.append(
            {
                "category": "讓分盤",
                "name": name,
                "model_probability": probability,
                "market_probability": "N/A",
                "fused_probability": probability,
                "odds": estimated_odds(probability),
                "risk": risk_level(probability),
                "note": "以模型比分分布估算讓分後過盤機率。",
            }
        )
    st.dataframe(
        pd.DataFrame(handicap_rows)[["name", "model_probability", "odds", "risk"]]
        .rename(columns={"name": "盤口", "model_probability": "過盤機率", "odds": "對應賠率", "risk": "風險等級"})
        .assign(過盤機率=lambda df: df["過盤機率"].map(format_percent)),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("大小球")
    total_rows = []
    for line in [1.5, 2.5, 3.5]:
        over_probability = clamp_probability(sum(score["probability"] for score in score_grid if score["home_goals"] + score["away_goals"] > line))
        under_probability = clamp_probability(1 - over_probability)
        total_rows.append(
            {
                "盤口": line,
                "大球機率": over_probability,
                "小球機率": under_probability,
                "大球賠率": estimated_odds(over_probability),
                "小球賠率": estimated_odds(under_probability),
                "模型預估總進球": round(total_goal_estimate, 2),
            }
        )
    total_df = pd.DataFrame(total_rows)
    st.dataframe(
        total_df.assign(
            大球機率=lambda df: df["大球機率"].map(format_percent),
            小球機率=lambda df: df["小球機率"].map(format_percent),
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("半全場")
    half_home_xg = max(0.1, prediction.expected_home_goals * 0.45)
    half_away_xg = max(0.1, prediction.expected_away_goals * 0.45)
    half_grid = poisson_score_grid(half_home_xg, half_away_xg, 5)
    half_probs = {
        "1": sum(item["probability"] for item in half_grid if item["home_goals"] > item["away_goals"]),
        "X": sum(item["probability"] for item in half_grid if item["home_goals"] == item["away_goals"]),
        "2": sum(item["probability"] for item in half_grid if item["home_goals"] < item["away_goals"]),
    }
    full_probs = {"1": prediction.home_win_probability, "X": prediction.draw_probability, "2": prediction.away_win_probability}
    half_full_rows = []
    for half in ["1", "X", "2"]:
        for full in ["1", "X", "2"]:
            probability = clamp_probability(float(half_probs[half]) * float(full_probs[full]))
            half_full_rows.append(
                {
                    "組合": f"{half}/{full}",
                    "機率": probability,
                    "估算賠率": estimated_odds(probability),
                    "風險等級": risk_level(probability),
                }
            )
    st.dataframe(
        pd.DataFrame(half_full_rows).assign(機率=lambda df: df["機率"].map(format_percent)),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("玩法卡片")
    card_candidates = []
    card_candidates.extend(one_x_two)
    card_candidates.extend(sorted(handicap_rows, key=lambda item: item["fused_probability"], reverse=True)[:2])
    for _, total_item in total_df.head(3).iterrows():
        probability = float(total_item["大球機率"])
        card_candidates.append(
            {
                "category": "大小球",
                "name": f"大 {total_item['盤口']}",
                "model_probability": probability,
                "market_probability": "N/A",
                "fused_probability": probability,
                "odds": float(total_item["大球賠率"]),
                "risk": risk_level(probability),
                "note": f"模型預估總進球 {total_goal_estimate:.2f}。",
            }
        )
    for index, play in enumerate(card_candidates):
        display_play_card(play, f"play_card_{index}")

    st.subheader("串關候選")
    candidate_rows = []
    for _, fixture in fixture_odds_df.head(24).iterrows():
        candidate_prediction = predict_match(matches_df, fixture["home_team"], fixture["away_team"], wc_team_stats_df)
        candidate_market = market_probability_table(fixture, candidate_prediction)
        best = candidate_market.sort_values("fused_probability", ascending=False).iloc[0]
        name = str(best["market"])
        odds_value = {"主勝": fixture["home_odds"], "和局": fixture["draw_odds"], "客勝": fixture["away_odds"]}.get(name, estimated_odds(float(best["fused_probability"])))
        level = risk_level(float(best["fused_probability"]))
        candidate_rows.append(
            {
                "比賽": f"{team_name(fixture['home_team'])} vs {team_name(fixture['away_team'])}",
                "候選方向": name,
                "融合機率": float(best["fused_probability"]),
                "賠率": round(float(odds_value), 2),
                "風險": level,
            }
        )
    candidate_df = pd.DataFrame(candidate_rows)
    for level in ["低", "中", "高"]:
        st.markdown(f"**{level}風險候選**")
        section = candidate_df[candidate_df["風險"] == level].head(5).copy()
        if section.empty:
            st.info("目前沒有符合此風險級別的候選資料。")
        else:
            st.dataframe(
                section.assign(融合機率=lambda df: df["融合機率"].map(format_percent)),
                use_container_width=True,
                hide_index=True,
            )

    st.subheader("單場賠率試算")
    applied_odds = float(st.session_state.get("applied_single_odds", 2.00))
    applied_name = st.session_state.get("applied_play_name", "手動輸入")
    st.caption(f"目前套用：{applied_name}｜可自行調整金額與賠率。")
    cols = st.columns(2)
    stake = cols[0].number_input("投注金額", min_value=0.0, value=1000.0, step=100.0, key="single_stake")
    if "single_odds" not in st.session_state:
        st.session_state["single_odds"] = applied_odds
    odds = cols[1].number_input("賠率", min_value=1.01, step=0.01, key="single_odds")
    payout = stake * odds
    profit = payout - stake
    result_cols = st.columns(2)
    with result_cols[0]:
        display_card("可得彩金", f"{payout:,.0f}", "投注金額 × 賠率")
    with result_cols[1]:
        display_card("實際獲利", f"{profit:,.0f}", "可得彩金 - 投注金額")

    st.subheader("串關試算器")
    parlay_stake = st.number_input("串關投注金額", min_value=0.0, value=500.0, step=100.0, key="parlay_stake")
    match_count = st.slider("串關場數", min_value=2, max_value=5, value=3)
    odds_values = []
    input_cols = st.columns(match_count)
    for index in range(match_count):
        default_value = applied_odds if index == 0 else 1.80
        if f"parlay_odds_{index}" not in st.session_state:
            st.session_state[f"parlay_odds_{index}"] = default_value
        odds_values.append(
            input_cols[index].number_input(
                f"賠率 {index + 1}",
                min_value=1.01,
                step=0.01,
                key=f"parlay_odds_{index}",
            )
        )

    total_odds = 1.0
    for value in odds_values:
        total_odds *= value
    parlay_payout = parlay_stake * total_odds
    parlay_profit = parlay_payout - parlay_stake

    cols = st.columns(3)
    with cols[0]:
        display_card("總賠率", f"{total_odds:.2f}", "各場賠率相乘")
    with cols[1]:
        display_card("可得彩金", f"{parlay_payout:,.0f}", "投注金額 × 總賠率")
    with cols[2]:
        display_card("實際獲利", f"{parlay_profit:,.0f}", "可得彩金 - 投注金額")

    table = pd.DataFrame(
        [{"場次": f"第 {idx + 1} 場", "賠率": f"{value:.2f}"} for idx, value in enumerate(odds_values)]
    )
    st.dataframe(table, use_container_width=True, hide_index=True)
    render_ai_parlay_analysis(limit=5)


def _market_card_grid(df: pd.DataFrame, key_prefix: str, max_items: int | None = None) -> None:
    if df is None or df.empty:
        st.info("目前資料不足，暫無此玩法機率。")
        return
    data = df.copy()
    data["probability"] = pd.to_numeric(data.get("probability", 0), errors="coerce").fillna(0)
    data["fused_probability"] = pd.to_numeric(data.get("fused_probability", data["probability"]), errors="coerce").fillna(data["probability"])
    data["estimated_odds"] = pd.to_numeric(data.get("estimated_odds", 1.01), errors="coerce").fillna(1.01)
    if max_items:
        data = data.head(max_items)
    for market_name, market_df in data.groupby("market_name", sort=False):
        with st.expander(str(market_name), expanded=True):
            rows = market_df.to_dict("records")
            for start in range(0, len(rows), 3):
                cols = st.columns(min(3, len(rows) - start))
                for offset, item in enumerate(rows[start:start + 3]):
                    with cols[offset]:
                        market_odds = item.get("market_odds", "")
                        market_text = f"市場賠率：{float(market_odds):.2f}" if str(market_odds).strip() else "市場賠率：未提供"
                        st.markdown(
                            f"""
                            <div class="display-card">
                              <div class="card-label">{html.escape(str(item.get("market_name", "")))}</div>
                              <div class="card-value">{html.escape(str(item.get("option_name", "")))}</div>
                              <div class="card-note">模型機率：{format_percent(float(item["probability"]))}</div>
                              <div class="card-note">估算賠率：{float(item["estimated_odds"]):.2f}</div>
                              <div class="card-note">{html.escape(market_text)}</div>
                              <div class="card-note">融合機率：{format_percent(float(item["fused_probability"]))}</div>
                              <div class="card-note">風險：{risk_badge(str(item.get("risk_level", "中")))}</div>
                              <div class="card-note">{html.escape(str(item.get("explanation", "")))}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                        button_key = f"{key_prefix}_{item.get('market_id', '')}_{start}_{offset}"
                        if st.button("套用到試算器", key=button_key, use_container_width=True):
                            st.session_state["applied_single_odds"] = float(item["estimated_odds"])
                            st.session_state["applied_play_name"] = str(item.get("option_name", "自選玩法"))
                            st.session_state["single_odds"] = float(item["estimated_odds"])
                            st.session_state["parlay_odds_0"] = float(item["estimated_odds"])
                            st.success(f"已套用：{item.get('option_name', '自選玩法')}，估算賠率 {float(item['estimated_odds']):.2f}")


def _parlay_candidate_tabs(*frames: pd.DataFrame) -> None:
    data = pd.concat([frame for frame in frames if frame is not None and not frame.empty], ignore_index=True)
    if data.empty:
        st.info("目前資料不足，暫無串關候選。")
        return
    data["fused_probability"] = pd.to_numeric(data.get("fused_probability", data.get("probability", 0)), errors="coerce").fillna(0)
    for label, levels in [("低風險候選", ["低"]), ("中風險候選", ["中"]), ("高風險候選", ["高"])]:
        st.subheader(label)
        subset = data[data["risk_level"].isin(levels)].sort_values("fused_probability", ascending=False).head(9)
        _market_card_grid(subset, f"parlay_{label}", max_items=9)


def all_market_prediction_page() -> None:
    page_header("全玩法預測中心", "以 Poisson、Elo、xG、市場機率與 Monte Carlo 推導足球常見玩法機率")
    st.warning("本頁僅供數據分析與機率學習，不提供下注建議。")
    row = selected_fixture("選擇比賽")
    prediction = predict_match(matches_df, row["home_team"], row["away_team"], wc_team_stats_df)
    market_table = market_probability_table(row, prediction)
    confidence_result = confidence_for_fixture(row, prediction, market_table)
    market_frames = bet_market_engine.build_match_markets(prediction, row)
    sim_df = v7_simulation().copy()
    if "round_32_probability" not in sim_df.columns and "group_qualified_probability" in sim_df.columns:
        sim_df["round_32_probability"] = sim_df["group_qualified_probability"]
    fixtures_for_group = fixture_odds_df.copy()
    if "group" not in fixtures_for_group.columns:
        fixtures_for_group["group"] = fixtures_for_group.get("stage", "Group TBD")

    cols = st.columns(4)
    with cols[0]:
        display_card("比賽時間", fixture_time_text(row), "台灣時間 UTC+8")
    with cols[1]:
        display_card("主隊", team_name(row["home_team"]), str(row.get("venue", row.get("stadium", ""))))
    with cols[2]:
        display_card("客隊", team_name(row["away_team"]), "賽程資料")
    with cols[3]:
        display_card("預測比分", f"{prediction.predicted_home_goals} : {prediction.predicted_away_goals}", "Poisson")

    prob_cols = st.columns(3)
    with prob_cols[0]:
        display_card("主勝", format_percent(prediction.home_win_probability))
    with prob_cols[1]:
        display_card("和局", format_percent(prediction.draw_probability))
    with prob_cols[2]:
        display_card("客勝", format_percent(prediction.away_win_probability))
    smart_confidence_card(confidence_result, compact=True)

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["單場核心", "進球玩法", "半場玩法", "冠軍與小組", "串關候選"])
    with tab1:
        _market_card_grid(
            pd.concat(
                [
                    market_frames["1x2"],
                    market_frames["handicap"],
                    market_frames["over_under"],
                    market_frames["both_teams_score"],
                ],
                ignore_index=True,
            ),
            "core",
        )
    with tab2:
        _market_card_grid(
            pd.concat(
                [
                    market_frames["first_goal"],
                    market_frames["correct_score"],
                    market_frames["team_goals"],
                    market_frames["total_goals_range"],
                    market_frames["odd_even"],
                ],
                ignore_index=True,
            ),
            "goals",
        )
    with tab3:
        _market_card_grid(
            pd.concat(
                [
                    market_frames["half_time_1x2"],
                    market_frames["half_full_time"],
                    market_frames["half_over_under"],
                    market_frames["half_team_goals"],
                    market_frames["half_correct_score"],
                ],
                ignore_index=True,
            ),
            "half",
        )
    with tab4:
        champion_df = bet_market_engine.predict_champion(sim_df).head(20)
        continent_df = bet_market_engine.predict_continent_winner(sim_df, team_meta_df)
        group_df = bet_market_engine.predict_group_winner(fixtures_for_group, sim_df)
        stage_df = bet_market_engine.predict_team_reaches_stage(sim_df).head(80)
        _market_card_grid(champion_df, "champion", max_items=20)
        _market_card_grid(continent_df, "continent")
        _market_card_grid(group_df, "group", max_items=48)
        _market_card_grid(stage_df, "stage", max_items=80)
    with tab5:
        _parlay_candidate_tabs(
            market_frames["1x2"],
            market_frames["over_under"],
            market_frames["both_teams_score"],
            market_frames["handicap"],
        )


def ai_analysis_center_page() -> None:
    page_header("AI 分析中心", "整合 AI 賽事分析與 AI 世界盃模擬")
    st.caption("此頁整合原本分散的 AI 分析功能，降低 Sidebar 選單長度；原功能仍保留在內部。")
    mode = st.radio(
        "選擇分析功能",
        ["AI 賽事分析", "AI 世界盃模擬"],
        horizontal=True,
    )
    if mode == "AI 賽事分析":
        ai_match_report_page()
    else:
        ai_worldcup_simulator_page()


def ai_match_report_page() -> None:
    page_header("AI 賽事分析報告", "規則式整合勝率、Elo、近期狀態、xG、市場機率與比分傾向")
    st.warning("本頁僅供機率分析，不構成下注建議。")
    row = selected_fixture()
    prediction = predict_match(matches_df, row["home_team"], row["away_team"], wc_team_stats_df)
    market_table = market_probability_table(row, prediction)
    prediction_cards(row)
    render_ai_match_report(row, prediction, market_table)


def ai_parlay_analysis_page() -> None:
    page_header("AI 串關分析", "以模型機率與市場機率整理 2 串 1、3 串 1 候選組合")
    st.warning("本頁僅供機率分析，不構成下注建議；賠率越高不代表越值得。")
    render_ai_parlay_analysis(limit=10)


def xg_model_page() -> None:
    page_header("xG 模型分析", "以射門距離、角度、身體部位與機會品質估算預期進球")
    try:
        shots = prepare_xg_data(pd.read_csv("data/xg_shots.csv"))
    except Exception:
        shots = pd.DataFrame()
    if shots.empty:
        st.info("目前尚未匯入 xG 射門資料")
        return

    match_ids = shots["match_id"].dropna().unique().tolist()
    selected_match = st.selectbox("選擇比賽", match_ids)
    match_shots = shots[shots["match_id"] == selected_match].copy()
    summary = xg_match_summary(shots, selected_match)
    if summary.empty:
        st.info("目前尚未匯入 xG 射門資料")
        return

    total_goals = float(summary["goals"].sum())
    total_xg = float(summary["xg"].sum())
    summary = summary.copy()
    summary["xga"] = (total_xg - summary["xg"]).round(2)
    summary["goals_against"] = total_goals - summary["goals"]
    summary["attack_efficiency"] = (summary["goals"] / summary["xg"].replace(0, pd.NA)).fillna(0).round(2)
    summary["defense_efficiency"] = (summary["goals_against"] / summary["xga"].replace(0, pd.NA)).fillna(0).round(2)

    metric_cols = st.columns(len(summary))
    for col, (_, row) in zip(metric_cols, summary.iterrows()):
        with col:
            display_card(str(row["team"]), f"xG {row['xg']:.2f}", f"xGA {row['xga']:.2f}")
            display_card("攻擊效率", f"{row['attack_efficiency']:.2f}", "進球 / xG")
            display_card("防守效率", f"{row['defense_efficiency']:.2f}", "失球 / xGA，越低越好")

    st.subheader("xG vs 實際進球")
    comparison = summary.melt(id_vars="team", value_vars=["xg", "xga", "goals"], var_name="指標", value_name="數值")
    bar = px.bar(comparison, x="team", y="數值", color="指標", barmode="group", color_discrete_sequence=[GOLD, "#8aa0c3", "#28a745"])
    bar.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color=INK)
    st.plotly_chart(bar, use_container_width=True)

    st.subheader("Top10 射門 xG 排名")
    top10 = match_shots.sort_values("xg_value", ascending=False).head(10).copy()
    top10_chart = px.bar(
        top10.sort_values("xg_value", ascending=True),
        x="xg_value",
        y="player",
        color="team",
        orientation="h",
        text="xg_value",
        labels={"xg_value": "xG", "player": "球員", "team": "球隊"},
    )
    top10_chart.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color=INK)
    st.plotly_chart(top10_chart, use_container_width=True)

    st.subheader("xG 累積走勢")
    match_shots["cumulative_xg"] = match_shots.groupby("team")["xg_value"].cumsum()
    line = px.line(match_shots, x="minute", y="cumulative_xg", color="team", markers=True, labels={"minute": "分鐘", "cumulative_xg": "累積 xG"})
    line.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color=INK)
    st.plotly_chart(line, use_container_width=True)

    st.subheader("球員射門 xG 表")
    shot_table = match_shots.sort_values("xg_value", ascending=False)[["minute", "team", "player", "body_part", "situation", "is_goal", "xg_value"]].rename(
        columns={
            "minute": "分鐘",
            "team": "球隊",
            "player": "球員",
            "body_part": "身體部位",
            "situation": "射門情境",
            "is_goal": "是否進球",
            "xg_value": "xG",
        }
    )
    st.dataframe(shot_table, use_container_width=True, hide_index=True)

    st.subheader("xG 分析")
    for line_text in xg_analysis_text(summary):
        st.markdown(f"- {line_text}")
    st.caption(
        "預測權重參考："
        f"Elo {PREDICTION_WEIGHTS_XG['elo']:.0%}、"
        f"近期狀態 {PREDICTION_WEIGHTS_XG['recent_form']:.0%}、"
        f"歷史成績 {PREDICTION_WEIGHTS_XG['worldcup_history']:.0%}、"
        f"xG 表現 {PREDICTION_WEIGHTS_XG['xg']:.0%}。"
    )


def _clean_table_for_display(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()
    for column in output.columns:
        if output[column].dtype == object:
            output[column] = output[column].fillna("N/A").astype(str).replace(
                {"nan": "N/A", "None": "N/A", "": "N/A"}
            )
    return output.fillna("N/A")


def live_event_label(event_type: str) -> str:
    labels = {
        "Goal": "進球",
        "Yellow Card": "黃牌",
        "Red Card": "紅牌",
        "Substitution": "換人",
    }
    return labels.get(str(event_type), str(event_type) if str(event_type) else "事件")


def national_team_center_page() -> None:
    page_header("國家隊資料中心", "整合世界排名、Elo、球員名單與晉級機率")
    teams = sorted(team_meta_df["team"].dropna().unique().tolist())
    selected = st.selectbox("選擇國家隊", teams, format_func=lambda team: team_name(team))
    meta = team_meta_df[team_meta_df["team"] == selected].iloc[0].to_dict()
    players = player_database(players_df, team_meta_df)
    squad = players[players["team"] == selected].copy()
    sim = v7_simulation()
    sim_row = sim[sim["team"] == selected]

    cols = st.columns(4)
    with cols[0]:
        display_card("FIFA 排名", str(int(meta.get("fifa_ranking", 0) or 0)), team_name(selected))
    with cols[1]:
        display_card("Elo Rating", str(int(meta.get("elo", 0) or 0)), meta.get("confederation", "N/A"))
    with cols[2]:
        avg_age = squad["age"].mean() if not squad.empty and "age" in squad.columns else 0
        display_card("平均年齡", f"{avg_age:.1f}" if avg_age else "N/A", "球員資料")
    with cols[3]:
        champion_prob = float(sim_row["champion_probability"].iloc[0]) if not sim_row.empty else 0
        display_card("奪冠率", format_percent(champion_prob), "Monte Carlo")

    if not sim_row.empty:
        stage_cols = [
            "group_qualified_probability",
            "round_16_probability",
            "round_8_probability",
            "semi_final_probability",
            "final_probability",
            "champion_probability",
        ]
        stage_labels = ["小組出線率", "16強率", "8強率", "4強率", "決賽率", "冠軍率"]
        stage_df = pd.DataFrame(
            {"階段": stage_labels, "機率": [float(sim_row[col].iloc[0]) for col in stage_cols]}
        )
        stage_df["百分比"] = stage_df["機率"].map(format_percent)
        chart = px.bar(stage_df, x="階段", y="機率", text="百分比", color="機率", color_continuous_scale=["#415a77", GOLD_LIGHT])
        chart.update_yaxes(tickformat=".0%", range=[0, 1])
        chart.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color=INK, coloraxis_showscale=False)
        st.plotly_chart(chart, use_container_width=True)

    st.subheader("球員名單")
    if squad.empty:
        st.info("目前尚未匯入該隊球員資料")
    else:
        table = squad[["player_name", "position_zh", "age", "club", "national_caps", "national_goals"]].rename(
            columns={
                "player_name": "姓名",
                "position_zh": "位置",
                "age": "年齡",
                "club": "俱樂部",
                "national_caps": "出場數",
                "national_goals": "進球",
            }
        )
        st.dataframe(_clean_table_for_display(table), use_container_width=True, hide_index=True)


def worldcup_history_page() -> None:
    page_header("歷史世界盃數據分析", "2002～2022 世界盃賽果、四強與國家隊表現")
    cols = st.columns(4)
    with cols[0]:
        display_card("比賽場次", f"{len(wc_matches_df):,}", "2002～2022")
    with cols[1]:
        display_card("參賽國家", f"{wc_team_stats_df['team'].nunique():,}", "歷史資料")
    with cols[2]:
        goals = int(pd.to_numeric(wc_matches_df.get("home_goals", 0), errors="coerce").fillna(0).sum() + pd.to_numeric(wc_matches_df.get("away_goals", 0), errors="coerce").fillna(0).sum())
        display_card("總進球", f"{goals:,}", "歷屆賽果")
    with cols[3]:
        display_card("資料狀態", "本地資料", "UTF-8")

    st.subheader("歷屆四強")
    if wc_top4_df.empty:
        st.info("目前尚未匯入歷屆四強資料")
    else:
        st.dataframe(_clean_table_for_display(wc_top4_df), use_container_width=True, hide_index=True)

    st.subheader("勝率最高國家 Top 12")
    top_stats = top_team_stats(wc_team_stats_df)
    if top_stats.empty:
        st.info("目前尚未匯入國家隊歷史戰績")
    else:
        st.dataframe(_clean_table_for_display(top_stats), use_container_width=True, hide_index=True)


def team_record_page() -> None:
    page_header("國家隊世界盃戰績", "查詢 2002～2022 世界盃國家隊歷史戰績")
    teams = sorted(wc_team_stats_df["team"].dropna().unique().tolist())
    selected = st.selectbox("選擇國家隊", teams, format_func=lambda team: team_name(team))
    summary = team_summary(wc_team_stats_df, selected)

    cols = st.columns(4)
    with cols[0]:
        display_card("參賽屆數", str(int(summary.get("tournaments_played", 0) or 0)))
    with cols[1]:
        display_card("勝率", format_percent(float(summary.get("win_rate", 0) or 0)))
    with cols[2]:
        display_card("進球", str(int(summary.get("goals_for", 0) or 0)))
    with cols[3]:
        display_card("失球", str(int(summary.get("goals_against", 0) or 0)))

    table = pd.DataFrame([summary])
    st.dataframe(_clean_table_for_display(table), use_container_width=True, hide_index=True)


def head_to_head_page() -> None:
    page_header("歷史交手分析", "查詢兩隊 2002～2022 世界盃交手紀錄")
    teams = sorted(set(wc_matches_df["home_team"]).union(set(wc_matches_df["away_team"])))
    col1, col2 = st.columns(2)
    team_a = col1.selectbox("國家隊 A", teams, index=0, format_func=lambda team: team_name(team))
    team_b = col2.selectbox("國家隊 B", teams, index=1 if len(teams) > 1 else 0, format_func=lambda team: team_name(team))
    if team_a == team_b:
        st.info("請選擇兩支不同國家隊。")
        return
    record = head_to_head_record(wc_head_to_head_df, team_a, team_b)
    if record.empty:
        st.info("目前尚未匯入這兩隊的歷史交手資料")
    else:
        st.dataframe(_clean_table_for_display(record), use_container_width=True, hide_index=True)


def disclaimer_page() -> None:
    page_header("免責聲明頁", "世界盃資料平台使用邊界")
    st.warning("所有預測、機率、xG、市場機率與模擬結果僅供資料分析參考，不保證賽果或獲利。")
    st.markdown(
        """
        - 本網站不提供下注、金流、會員錢包或交易功能。
        - 即時資料可能來自 API 或本地 fallback，請以官方賽事資訊為準。
        - 使用者若依據本網站資訊做任何財務決策，需自行承擔風險。
        """
    )


def football_data_encyclopedia_page() -> None:
    page_header("足球數據百科", "用白話理解常見足球預測與分析指標")
    concepts = [
        {
            "name": "Elo Rating",
            "plain": "用分數表示球隊強度。擊敗強隊加分較多，輸給弱隊扣分較多。",
            "use": "衡量兩隊基礎實力差距。",
            "score": 5,
        },
        {
            "name": "xG",
            "plain": "預期進球，估算每次射門變成進球的機率。",
            "use": "判斷一隊創造機會的品質。",
            "score": 5,
        },
        {
            "name": "xGA",
            "plain": "預期失球，代表對手面對本隊時創造出的射門品質。",
            "use": "觀察防守是否容易給出高品質機會。",
            "score": 4,
        },
        {
            "name": "Poisson Model",
            "plain": "用平均進球數推估比分分布，常用在足球比分預測。",
            "use": "產生可能比分與勝平負機率。",
            "score": 4,
        },
        {
            "name": "Monte Carlo",
            "plain": "重複模擬很多次比賽或賽事，用結果比例估算機率。",
            "use": "估算小組出線、晉級、決賽與奪冠率。",
            "score": 5,
        },
        {
            "name": "Market Probability",
            "plain": "把賠率轉換成隱含機率，再和模型機率比較。",
            "use": "理解市場對比賽結果的定價。",
            "score": 4,
        },
    ]

    for item in concepts:
        st.markdown(
            f"""
            <div class="display-card">
              <div class="card-label">{html.escape(item['name'])}</div>
              <div class="card-value" style="font-size:1.05rem;line-height:1.65;">{html.escape(item['plain'])}</div>
              <div class="card-note">用途：{html.escape(item['use'])}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    concept_df = pd.DataFrame(concepts).rename(
        columns={"name": "指標", "plain": "白話說明", "use": "用途", "score": "重要度"}
    )
    st.subheader("指標速查表")
    st.dataframe(concept_df[["指標", "白話說明", "用途", "重要度"]], use_container_width=True, hide_index=True)

    chart = px.bar(concept_df, x="指標", y="重要度", color="重要度", color_continuous_scale=["#415a77", GOLD_LIGHT])
    chart.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color=INK, coloraxis_showscale=False)
    st.plotly_chart(chart, use_container_width=True)


def football_terms_page() -> None:
    page_header("足球術語教學", "快速理解看球與資料分析常見詞彙")
    terms = [
        {"術語": "越位", "說明": "進攻球員在傳球瞬間比倒數第二名防守球員更接近球門，並參與進攻。", "類型": "規則"},
        {"術語": "角球", "說明": "防守方最後觸球並讓球越過本方底線時，進攻方從角旗區開球。", "類型": "定位球"},
        {"術語": "自由球", "說明": "犯規後由對方在指定位置重新開球，分為直接與間接自由球。", "類型": "定位球"},
        {"術語": "PK", "說明": "禁區內犯規後的點球，射門距離短，進球機率通常較高。", "類型": "定位球"},
        {"術語": "傷停補時", "說明": "裁判因換人、受傷、VAR 等停頓補上的比賽時間。", "類型": "時間"},
        {"術語": "黃牌", "說明": "警告性處分，單場兩張黃牌會變成紅牌離場。", "類型": "判罰"},
        {"術語": "紅牌", "說明": "球員被罰下，球隊少打一人。", "類型": "判罰"},
        {"術語": "帽子戲法", "說明": "同一球員在一場比賽中進三球。", "類型": "表現"},
        {"術語": "Clean Sheet", "說明": "零封，球隊整場沒有失球。", "類型": "防守"},
        {"術語": "Expected Goals", "說明": "預期進球，也就是 xG，用射門品質估算進球機率。", "類型": "數據"},
    ]
    terms_df = pd.DataFrame(terms)
    for _, row in terms_df.iterrows():
        st.markdown(
            f"""
            <div class="display-card">
              <div class="card-label">{html.escape(row['類型'])}</div>
              <div class="card-value" style="font-size:1.15rem;">{html.escape(row['術語'])}</div>
              <div class="card-note">{html.escape(row['說明'])}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.subheader("術語速查表")
    st.dataframe(terms_df, use_container_width=True, hide_index=True)

    counts = terms_df.groupby("類型", as_index=False).size().rename(columns={"size": "數量"})
    pie = px.pie(counts, names="類型", values="數量", hole=0.45, color_discrete_sequence=[GOLD, "#8aa0c3", "#28a745", "#f3d98b", "#415a77"])
    pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color=INK)
    st.plotly_chart(pie, use_container_width=True)


def dashboard_page() -> None:
    st.markdown(
        """
        <div class="hero">
          <div>
            <div class="hero-kicker">WORLD CUP INTELLIGENCE PLATFORM · FINAL</div>
            <div class="hero-title">世界盃智慧預測平台</div>
            <div class="hero-copy">
              即時賽果、Elo、xG、球員影響、市場機率與 Monte Carlo 模擬整合成一個可解釋的世界盃情報儀表板。
            </div>
          </div>
          <div class="hero-visual"><div class="trophy">🏆</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    disclaimer_box()
    sim_df = v7_simulation()
    row = fixture_odds_df.iloc[0]
    prediction = predict_match(matches_df, row["home_team"], row["away_team"], wc_team_stats_df)
    champion = sim_df.iloc[0]
    live_result = cached_v29_live_matches()
    live_matches = live_result.data
    live_source = live_result.source_mode
    market_table = market_probability_table(row, prediction)
    xg_summary = pd.DataFrame()
    try:
        shots_df = prepare_xg_data(pd.read_csv("data/xg_shots.csv"))
        xg_summary = xg_match_summary(shots_df)
    except Exception:
        shots_df = pd.DataFrame()

    cols = st.columns(4)
    with cols[0]:
        display_card("今日焦點賽事", matchup_text(row), fixture_time_text(row))
    with cols[1]:
        display_card("預測比分", f"{prediction.predicted_home_goals} : {prediction.predicted_away_goals}", "Poisson + Elo")
    with cols[2]:
        display_card("奪冠熱門", format_percent(float(champion["champion_probability"])), str(champion["team_display"]))
    with cols[3]:
        display_card("即時資料", str(len(live_matches)) if live_matches is not None else "0", str(live_source))

    left, right = st.columns([1.05, 0.95])
    with left:
        st.subheader("今日／近期賽程")
        st.dataframe(fixtures_with_flags(fixture_odds_df.head(5)), use_container_width=True, hide_index=True)
    with right:
        st.subheader("奪冠熱門 Top10")
        top10 = sim_df.head(10).sort_values("champion_probability", ascending=True).copy()
        top10["label"] = top10["champion_probability"].map(format_percent)
        chart = px.bar(top10, x="champion_probability", y="team_display", orientation="h", text="label", color="champion_probability", color_continuous_scale=["#415a77", GOLD_LIGHT])
        chart.update_xaxes(tickformat=".0%")
        chart.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color=INK, coloraxis_showscale=False)
        st.plotly_chart(chart, use_container_width=True)

    st.subheader("即時世界盃總覽")
    overview_cols = st.columns(4)
    with overview_cols[0]:
        display_card("完成賽事", str(len(load_match_results())), "match_results.csv")
    with overview_cols[1]:
        display_card("小組出線均值", format_percent(float(sim_df["group_qualified_probability"].mean())), "Monte Carlo")
    with overview_cols[2]:
        display_card("決賽率最高", format_percent(float(sim_df.iloc[0]["final_probability"])), str(sim_df.iloc[0]["team_display"]))
    with overview_cols[3]:
        display_card("模擬版本", "V25", "智慧預測平台")

    st.subheader("晉級機率摘要")
    advance = sim_df.head(10)[["team_display", "group_qualified_probability", "round_16_probability", "round_8_probability", "semi_final_probability", "final_probability", "champion_probability"]].copy()
    for column in advance.columns[1:]:
        advance[column] = advance[column].map(format_percent)
    st.dataframe(
        advance.rename(
            columns={
                "team_display": "球隊",
                "group_qualified_probability": "小組出線率",
                "round_16_probability": "16強率",
                "round_8_probability": "8強率",
                "semi_final_probability": "4強率",
                "final_probability": "決賽率",
                "champion_probability": "奪冠率",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("AI 今日分析摘要")
    report_lines = generate_match_report(row, prediction, matches_df, team_meta_df, market_table=market_table)
    for line in report_lines[:4]:
        st.markdown(f"<div class='display-card'><div class='card-note'>{html.escape(line)}</div></div>", unsafe_allow_html=True)

    bottom_left, bottom_right = st.columns(2)
    with bottom_left:
        st.subheader("xG 模型摘要")
        if xg_summary.empty:
            st.info("目前 xG 資料不足，暫以 Poisson 預期進球作為參考。")
        else:
            xg_top = xg_summary.head(8).copy()
            st.dataframe(xg_top, use_container_width=True, hide_index=True)
    with bottom_right:
        st.subheader("市場機率摘要")
        market_display = market_table.copy()
        for column in ["model_probability", "market_probability", "fused_probability"]:
            market_display[column] = pd.to_numeric(market_display[column], errors="coerce").fillna(0).map(format_percent)
        st.dataframe(
            market_display.rename(
                columns={
                    "market": "結果",
                    "model_probability": "模型機率",
                    "market_probability": "市場機率",
                    "fused_probability": "融合機率",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )


if page == "世界盃情報中心":
    dashboard_page()
elif page == "世界盃賽程表":
    fixtures_page()
elif page == "即時世界盃中心":
    realtime_worldcup_center_page()
elif page == "賽果更新中心":
    match_results_update_page()
elif page == "賽程頁":
    fixtures_page()
elif page == "單場分析頁":
    match_analysis_page()
elif page == "球員影響分析":
    player_impact_analysis_page()
elif page == "市場機率分析":
    betting_page()
elif page == "投注分析頁":
    betting_page()
elif page == "世足玩法教學":
    football_betting_guide_page()
elif page == "賠率試算中心":
    odds_calculator_page()
elif page == "全玩法預測中心":
    all_market_prediction_page()
elif page == "AI 分析中心":
    ai_analysis_center_page()
elif page == "AI 賽事分析報告":
    ai_match_report_page()
elif page == "AI 串關分析":
    ai_parlay_analysis_page()
elif page == "模型回測頁":
    model_backtest_page()
elif page == "冠軍機率預測":
    champion_probability_page()
elif page == "小組出線機率分析":
    group_qualification_page()
elif page == "晉級機率分析":
    advancement_probability_page()
elif page == "世界盃模擬器":
    worldcup_simulator_page()
elif page == "AI 世界盃模擬器":
    ai_worldcup_simulator_page()
elif page == "冠軍路徑模擬":
    champion_path_page()
elif page == "對戰比較中心":
    team_comparison_page()
elif page == "AI 世界盃分析師":
    ai_worldcup_analyst_page()
elif page == "Elo 世界排名":
    elo_ranking_page()
elif page == "xG 模型分析":
    xg_model_page()
elif page == "即時賽況":
    live_matches_page()
elif page == "球隊資料庫":
    team_database_page()
elif page == "球員資料庫":
    player_database_page()
elif page == "國家隊資料中心":
    national_team_center_page()
elif page == "足球數據百科":
    football_data_encyclopedia_page()
elif page == "足球術語教學":
    football_terms_page()
elif page == "歷史世界盃數據分析":
    worldcup_history_page()
elif page == "國家隊世界盃戰績":
    team_record_page()
elif page == "歷史交手分析":
    head_to_head_page()
else:
    disclaimer_page()

footer()
