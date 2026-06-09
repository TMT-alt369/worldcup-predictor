import html

import pandas as pd
import plotly.express as px
import streamlit as st

from utils.live_api import (
    load_live_matches_from_secrets as load_live_data,
    load_live_matches_with_debug,
)
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
from worldcup_predictor.model import predict_match, prediction_to_frame, team_strength
from worldcup_predictor.players import player_database, squad_summary
from worldcup_predictor.tournament import run_tournament_simulation
from worldcup_predictor.ui import disclaimer_box, format_percent, signal_dataframe
from utils.simulation import run_worldcup_monte_carlo
from utils.elo_update import elo_ranking_with_updates
from utils.player_impact import team_impact, win_probability_adjustment
from utils.market_probability import market_probability_table
from utils.xg_model import prepare_xg_data, xg_match_summary, xg_analysis_text


st.set_page_config(page_title="世足智慧預測中心", page_icon="⚽", layout="wide")

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
            overflow: hidden;
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
    "首頁儀表板",
    "世界盃賽程表",
    "賽程頁",
    "單場分析頁",
    "投注分析頁",
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
    "首頁": ["首頁儀表板"],
    "賽程中心": ["世界盃賽程表", "即時賽況"],
    "預測中心": ["單場分析頁", "冠軍機率預測", "晉級機率分析", "世界盃模擬器", "投注分析頁"],
    "資料中心": ["Elo 世界排名", "球隊資料庫", "球員資料庫", "國家隊資料中心", "歷史世界盃數據分析", "國家隊世界盃戰績", "歷史交手分析"],
    "系統資訊": ["模型回測頁", "免責聲明頁"],
}

for group_name, pages in PAGE_GROUPS.items():
    if any("Elo" in item for item in pages) and "xG 模型分析" not in pages:
        pages.insert(1, "xG 模型分析")
        break

selected_group = st.sidebar.selectbox("功能分類", list(PAGE_GROUPS.keys()))
page = st.sidebar.radio("頁面", PAGE_GROUPS[selected_group])
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


def selected_fixture(label: str = "選擇比賽") -> pd.Series:
    labels = {
        f"{taipei_time_text(row.datetime_taipei)} | {team_name(row.home_team)} vs {team_name(row.away_team)}": row.match_id
        for row in fixture_odds_df.itertuples()
    }
    selected_label = st.selectbox(label, list(labels.keys()))
    match_id = labels[selected_label]
    return fixture_odds_df[fixture_odds_df["match_id"] == match_id].iloc[0]


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


def render_key_players_block(home_team: str, away_team: str) -> None:
    st.subheader("關鍵球員")
    players = pd.concat([key_players_for(home_team), key_players_for(away_team)], ignore_index=True)
    st.caption(f"主隊關鍵球員：{team_name(home_team)} ｜ 客隊關鍵球員：{team_name(away_team)}")
    if players.empty:
        st.info("目前尚未匯入該球隊球員資料")
        return
    if "goal_rate" not in players.columns:
        players["goal_rate"] = 0.0
    if "recent_form_rating" not in players.columns:
        players["recent_form_rating"] = 0.0
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
        confidence_card(prediction.confidence)

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
        st.caption("快速分析僅供展示與資料參考，詳細內容請至單場分析頁與投注分析頁。")


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
    page_header("投注分析頁", "比較模型機率與賠率隱含機率，並以風險色塊呈現建議等級")
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

    st.caption(
        "晉級機率來自同一份 1000 次 Monte Carlo 模擬結果；V4 僅新增單隊分析視角，"
        "沒有修改 V1 單場預測、V2 冠軍模擬或 V3 Elo 排名邏輯。"
    )


def live_event_label(event_type: str) -> str:
    icons = {
        "Goal": "⚽",
        "Yellow Card": "🟨",
        "Red Card": "🟥",
        "Substitution": "🔄",
    }
    return f"{icons.get(event_type, '•')} {event_type}"


def live_fixture_row(row: pd.Series) -> pd.Series | None:
    matches = fixture_odds_df[
        (fixture_odds_df["home_team"] == row["home_team"])
        & (fixture_odds_df["away_team"] == row["away_team"])
    ]
    if matches.empty:
        return None
    return matches.iloc[0]


def live_match_label(row: pd.Series) -> str:
    fixture = live_fixture_row(row)
    scheduled_time = fixture_time_text(fixture) if fixture is not None else "時間未定"
    return (
        f"{scheduled_time} | {team_name(row['home_team'])} "
        f"{row['home_score']}-{row['away_score']} {team_name(row['away_team'])}"
    )


def live_matches_page() -> None:
    page_header("即時賽況", "使用本地 mock 資料展示即時比分、事件與比賽數據，不含影音直播")
    refresh_seconds = st.selectbox("刷新頻率", [30, 45, 60], index=1)
    st.markdown(
        f"<meta http-equiv='refresh' content='{refresh_seconds}'>",
        unsafe_allow_html=True,
    )
    st.caption(f"展示頁會嘗試每 {refresh_seconds} 秒刷新一次；目前資料來源為本地 mock CSV。")
    st.warning("展示資料／模擬即時賽況：目前未串接真實即時 API，比分、事件、控球率與射門數皆為本地展示資料。")

    labels = {
        live_match_label(row): row["live_match_id"]
        for _, row in live_matches_df.iterrows()
    }
    selected = st.selectbox("選擇即時比賽", list(labels.keys()))
    live_match_id = labels[selected]
    row = live_matches_df[live_matches_df["live_match_id"] == live_match_id].iloc[0]
    fixture = live_fixture_row(row)
    scheduled_time = fixture_time_text(fixture) if fixture is not None else "時間未定"

    st.markdown(
        f"""
        <div class="display-card score-card">
          <div class="score-teams">{html.escape(team_name(row['home_team']))} vs {html.escape(team_name(row['away_team']))}</div>
          <div class="score-value">{row['home_score']} : {row['away_score']}</div>
          <div class="score-note">賽程時間：{html.escape(scheduled_time)} 台灣時間 · {html.escape(row['status'])} · {int(row['minute'])}' · 更新時間 {html.escape(str(row['updated_at']))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cols = st.columns(3)
    with cols[0]:
        display_card("射門", f"{row['home_shots']} : {row['away_shots']}")
    with cols[1]:
        display_card("控球率", f"{row['home_possession']}% : {row['away_possession']}%")
    with cols[2]:
        display_card("角球", f"{row['home_corners']} : {row['away_corners']}")

    st.subheader("比賽事件")
    events = live_events_df[live_events_df["live_match_id"] == live_match_id].copy()
    if events.empty:
        st.info("目前沒有事件資料。")
    else:
        events["事件"] = events["event_type"].map(live_event_label)
        events["球隊"] = events["team"].map(lambda team: f"{flag(team)} {team}")
        st.dataframe(
            events[["minute", "事件", "球隊", "player", "detail"]].rename(
                columns={
                    "minute": "時間",
                    "player": "球員",
                    "detail": "內容",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("技術統計")
    stats = pd.DataFrame(
        [
            {"項目": "射門", row["home_team"]: row["home_shots"], row["away_team"]: row["away_shots"]},
            {"項目": "控球率", row["home_team"]: row["home_possession"], row["away_team"]: row["away_possession"]},
            {"項目": "角球", row["home_team"]: row["home_corners"], row["away_team"]: row["away_corners"]},
        ]
    )
    stats_long = stats.melt(id_vars="項目", var_name="球隊", value_name="數值")
    chart = px.bar(
        stats_long,
        x="項目",
        y="數值",
        color="球隊",
        barmode="group",
        color_discrete_sequence=[GOLD, "#8aa0c3"],
    )
    chart.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color=INK)
    st.plotly_chart(chart, use_container_width=True)


def worldcup_history_page() -> None:
    page_header("歷史世界盃數據分析", "2002~2022 世界盃比賽結果、冠軍、四強與整體趨勢")
    cols = st.columns(4)
    with cols[0]:
        display_card("涵蓋屆數", str(wc_matches_df["tournament_year"].nunique()), "2002~2022")
    with cols[1]:
        display_card("比賽場次", str(len(wc_matches_df)), "完整賽果")
    with cols[2]:
        display_card("參賽國家", str(wc_team_stats_df["team"].nunique()), "國家隊統計")
    with cols[3]:
        total_goals = int(wc_matches_df["home_goals"].sum() + wc_matches_df["away_goals"].sum())
        display_card("總進球", str(total_goals), "歷史趨勢")

    st.subheader("歷屆四強")
    st.dataframe(wc_top4_df, use_container_width=True, hide_index=True)

    goals_by_year = wc_matches_df.assign(
        total_goals=wc_matches_df["home_goals"] + wc_matches_df["away_goals"]
    ).groupby("tournament_year", as_index=False)["total_goals"].sum()
    chart = px.line(
        goals_by_year,
        x="tournament_year",
        y="total_goals",
        markers=True,
        color_discrete_sequence=[GOLD_LIGHT],
    )
    chart.update_traces(line=dict(width=4), marker=dict(size=10))
    chart.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color=INK)
    st.plotly_chart(chart, use_container_width=True)

    st.subheader("勝率最高國家 Top 12")
    st.dataframe(top_team_stats(wc_team_stats_df), use_container_width=True, hide_index=True)


def team_record_page() -> None:
    page_header("國家隊世界盃戰績", "查看各國 2002~2022 世界盃勝率、進失球與四強成績")
    teams = wc_team_stats_df["team"].sort_values().tolist()
    selected_team = st.selectbox("選擇國家隊", teams)
    summary = team_summary(wc_team_stats_df, selected_team)

    st.subheader(f"{flag(selected_team)} {selected_team}")
    cols = st.columns(4)
    with cols[0]:
        display_card("參賽屆數", str(int(summary["tournaments_played"])))
    with cols[1]:
        display_card("勝率", format_percent(float(summary["win_rate"])))
    with cols[2]:
        display_card("冠軍", str(int(summary["titles"])))
    with cols[3]:
        display_card("四強次數", str(int(summary["top4_finishes"])))

    st.subheader("完整戰績")
    st.dataframe(pd.DataFrame([summary]), use_container_width=True, hide_index=True)

    st.subheader("該隊世界盃比賽")
    team_matches = wc_matches_df[
        (wc_matches_df["home_team"] == selected_team) | (wc_matches_df["away_team"] == selected_team)
    ]
    st.dataframe(team_matches, use_container_width=True, hide_index=True)


def head_to_head_page() -> None:
    page_header("歷史交手分析", "查詢 2002~2022 世界盃任兩隊交手紀錄")
    teams = sorted(set(wc_matches_df["home_team"]).union(set(wc_matches_df["away_team"])))
    col1, col2 = st.columns(2)
    team_a = col1.selectbox("國家隊 A", teams, index=teams.index("Argentina") if "Argentina" in teams else 0)
    team_b = col2.selectbox("國家隊 B", teams, index=teams.index("France") if "France" in teams else 1)

    if team_a == team_b:
        st.info("請選擇兩支不同國家隊。")
        return

    st.subheader(f"{flag(team_a)} {team_a} vs {flag(team_b)} {team_b}")
    st.dataframe(head_to_head_record(wc_head_to_head_df, team_a, team_b), use_container_width=True, hide_index=True)

    st.subheader("交手比賽明細")
    pair_matches = wc_matches_df[
        ((wc_matches_df["home_team"] == team_a) & (wc_matches_df["away_team"] == team_b))
        | ((wc_matches_df["home_team"] == team_b) & (wc_matches_df["away_team"] == team_a))
    ]
    if pair_matches.empty:
        st.info("2002~2022 世界盃沒有交手紀錄。")
    else:
        st.dataframe(pair_matches, use_container_width=True, hide_index=True)


def disclaimer_page() -> None:
    page_header("免責聲明頁", "本平台的分析邊界與投注風險說明")
    st.markdown(
        """
        ### 重要聲明

        本網站是世界盃資料分析平台，所有預測、機率、
        信心分數、風險分級與投注訊號都只供參考。

        ### 不保證事項

        - 不保證預測比分命中。
        - 不保證勝平負投注獲利。
        - 不提供下注、金流、會員錢包或任何實際交易功能。
        - 測試資料與歷史資料不代表即時官方資料。

        ### 使用者責任

        若使用者依據本網站資訊進行任何投注或財務決策，應自行承擔全部風險。
        建議將本網站視為資料分析練習與模型展示，而不是保證獲利工具。
        """
    )


V7_SIMULATIONS = 10000


def v7_simulation() -> pd.DataFrame:
    return cached_tournament_simulation(
        fixtures_df,
        team_meta_df,
        wc_team_stats_df,
        matches_df,
        V7_SIMULATIONS,
    )


def group_lookup() -> dict[str, str]:
    groups: dict[str, str] = {}
    group_fixtures = fixtures_df[fixtures_df["stage"].str.startswith("Group ", na=False)]
    for _, row in group_fixtures.iterrows():
        groups[row["home_team"]] = row["stage"]
        groups[row["away_team"]] = row["stage"]
    return groups


def v7_stage_columns() -> list[str]:
    return [
        "group_qualified_probability",
        "round_16_probability",
        "round_8_probability",
        "semi_final_probability",
        "final_probability",
        "champion_probability",
    ]


def v7_stage_labels() -> dict[str, str]:
    return {
        "group_qualified_probability": "小組出線率",
        "round_16_probability": "16強機率",
        "round_8_probability": "8強機率",
        "semi_final_probability": "4強機率",
        "final_probability": "決賽機率",
        "champion_probability": "奪冠機率",
    }


def champion_probability_page() -> None:
    page_header("冠軍機率預測", "V7 智慧預測中心：Elo Rating + Poisson + 10000 次 Monte Carlo Simulation")
    st.info(
        "本頁以可解釋模型估算 2026 世界盃奪冠機率，整合 Elo、近期狀態、世界盃歷史表現與 Poisson 進球模擬。"
        "結果僅供資料分析，不代表實際賽果。"
    )
    disclaimer_box()

    with st.spinner(f"執行 {V7_SIMULATIONS:,} 次 Monte Carlo 模擬..."):
        simulation_df = v7_simulation()

    top20 = simulation_df.head(20).copy()
    champion = top20.iloc[0]

    cols = st.columns(4)
    with cols[0]:
        display_card("模擬次數", f"{V7_SIMULATIONS:,}", "Monte Carlo")
    with cols[1]:
        display_card("參賽隊伍", str(len(simulation_df)), "2026 世界盃")
    with cols[2]:
        display_card("最高奪冠率", format_percent(float(champion["champion_probability"])), champion["team_display"])
    with cols[3]:
        display_card("模型架構", "Elo + Poisson", "可解釋 AI 模擬")

    st.subheader("奪冠機率排行榜 TOP20")
    chart_df = top20.sort_values("champion_probability", ascending=True).copy()
    chart_df["champion_label"] = chart_df["champion_probability"].map(format_percent)
    chart = px.bar(
        chart_df,
        x="champion_probability",
        y="team_display",
        orientation="h",
        text="champion_label",
        hover_data=["elo", "history_score", "recent_form"],
        labels={"champion_probability": "奪冠率", "team_display": "國家隊"},
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
        margin=dict(l=10, r=50, t=10, b=10),
    )
    st.plotly_chart(chart, use_container_width=True)

    ranking = top20.reset_index(drop=True).copy()
    ranking["排名"] = ranking.index + 1
    ranking["國旗"] = ranking["flag_emoji"]
    ranking["奪冠率"] = ranking["champion_probability"].map(format_percent)
    st.dataframe(
        ranking[["排名", "國旗", "team_zh", "elo", "奪冠率"]].rename(
            columns={"team_zh": "國家隊", "elo": "Elo 分數"}
        ),
        use_container_width=True,
        hide_index=True,
    )


def group_qualification_page() -> None:
    page_header("小組出線機率分析", "V7 小組出線率、分組排名表與互動式 Plotly 長條圖")
    simulation_df = v7_simulation().copy()
    groups = group_lookup()
    simulation_df["group"] = simulation_df["team"].map(groups).fillna("未分組")

    selected_group = st.selectbox("選擇小組", ["全部"] + sorted(simulation_df["group"].unique().tolist()))
    filtered = simulation_df if selected_group == "全部" else simulation_df[simulation_df["group"] == selected_group]
    filtered = filtered.sort_values("group_qualified_probability", ascending=False).copy()

    chart_df = filtered.sort_values("group_qualified_probability", ascending=True).copy()
    chart_df["probability_label"] = chart_df["group_qualified_probability"].map(format_percent)
    chart = px.bar(
        chart_df,
        x="group_qualified_probability",
        y="team_display",
        orientation="h",
        text="probability_label",
        color="group_qualified_probability",
        color_continuous_scale=["#415a77", GOLD_LIGHT],
        hover_data=["group", "elo", "champion_probability"],
        labels={"group_qualified_probability": "小組出線率", "team_display": "國家隊"},
    )
    chart.update_traces(textposition="outside")
    chart.update_xaxes(tickformat=".0%", range=[0, 1])
    chart.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color=INK,
        coloraxis_showscale=False,
        margin=dict(l=10, r=50, t=10, b=10),
    )
    st.plotly_chart(chart, use_container_width=True)

    table = filtered[["group", "team_display", "elo", "group_qualified_probability", "champion_probability"]].copy()
    table["小組出線率"] = table["group_qualified_probability"].map(format_percent)
    table["奪冠率"] = table["champion_probability"].map(format_percent)
    table["排名"] = table.groupby("group")["group_qualified_probability"].rank(method="first", ascending=False).astype(int)
    st.dataframe(
        table[["group", "排名", "team_display", "elo", "小組出線率", "奪冠率"]].rename(
            columns={"group": "小組", "team_display": "國家隊", "elo": "Elo 分數"}
        ),
        use_container_width=True,
        hide_index=True,
    )


def advancement_probability_page() -> None:
    page_header("淘汰賽晉級機率分析", "V7 以 10000 次 Monte Carlo 模擬估算各階段晉級機率")
    simulation_df = v7_simulation().copy()
    stage_columns = v7_stage_columns()
    stage_labels = v7_stage_labels()

    team_options = simulation_df.sort_values("team_zh")["team_display"].tolist()
    selected_display = st.selectbox("選擇國家隊", team_options)
    selected_row = simulation_df[simulation_df["team_display"] == selected_display].iloc[0]

    probability_df = pd.DataFrame(
        [(stage_labels[column], selected_row[column]) for column in stage_columns],
        columns=["階段", "機率"],
    )
    probability_df["百分比"] = probability_df["機率"].map(format_percent)

    cols = st.columns(4)
    with cols[0]:
        display_card("國家隊", selected_row["team_display"], f"Elo {int(selected_row['elo'])}")
    with cols[1]:
        display_card("小組出線率", format_percent(float(selected_row["group_qualified_probability"])), "Monte Carlo")
    with cols[2]:
        display_card("決賽機率", format_percent(float(selected_row["final_probability"])), "淘汰賽模擬")
    with cols[3]:
        display_card("奪冠機率", format_percent(float(selected_row["champion_probability"])), f"{V7_SIMULATIONS:,} 次模擬")

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
    )
    st.plotly_chart(chart, use_container_width=True)

    table = simulation_df[["team_display", "elo", *stage_columns]].copy()
    for column in stage_columns:
        table[column] = table[column].map(format_percent)
    st.dataframe(
        table.rename(columns={"team_display": "國家隊", "elo": "Elo 分數", **stage_labels}),
        use_container_width=True,
        hide_index=True,
    )


def player_database_page() -> None:
    page_header("球員資料庫", "V8 真實名單資料：關鍵字搜尋、國家/位置篩選、身價與年齡排序")
    database = player_database(players_df, team_meta_df)
    source_text = database["data_source"].mode().iloc[0] if not database.empty else "無資料"
    st.caption(f"資料來源：{source_text}｜目前載入 {len(database):,} 名球員、{database['team'].nunique()} 支國家隊")

    teams = ["全部"] + database["national_team"].dropna().sort_values().unique().tolist()
    positions = ["全部"] + database["position_zh"].dropna().sort_values().unique().tolist()

    cols = st.columns([1.0, 0.9, 1.2, 1.0])
    selected_team = cols[0].selectbox("國家隊", teams)
    selected_position = cols[1].selectbox("位置", positions)
    keyword = cols[2].text_input("輸入球員姓名", "")
    sort_mode = cols[3].selectbox("排序", ["身價高到低", "身價低到高", "年齡高到低", "年齡低到高", "出賽多到少"])

    filtered = database.copy()
    if selected_team != "全部":
        filtered = filtered[filtered["national_team"] == selected_team]
    if selected_position != "全部":
        filtered = filtered[filtered["position_zh"] == selected_position]
    if keyword.strip():
        filtered = filtered[filtered["player_name"].str.contains(keyword.strip(), case=False, na=False)]
    sort_map = {
        "身價高到低": ("market_value_eur_m", False),
        "身價低到高": ("market_value_eur_m", True),
        "年齡高到低": ("age", False),
        "年齡低到高": ("age", True),
        "出賽多到少": ("national_caps", False),
    }
    sort_column, ascending = sort_map[sort_mode]
    filtered = filtered.sort_values(sort_column, ascending=ascending)

    cols = st.columns(4)
    with cols[0]:
        display_card("球員筆數", str(len(filtered)), "目前篩選")
    with cols[1]:
        display_card("國家隊數", str(filtered["team"].nunique()), "目前篩選")
    with cols[2]:
        display_card("平均年齡", f"{filtered['age'].mean():.1f}" if not filtered.empty else "0.0", "歲")
    with cols[3]:
        display_card("總身價", f"€{filtered['market_value_eur_m'].sum():.1f}M", "無資料以 0 計")

    table = filtered[
        [
            "national_team",
            "player_name",
            "jersey_number",
            "position_zh",
            "age",
            "height_cm",
            "preferred_foot",
            "market_value_eur_m",
            "club",
            "national_caps",
            "national_goals",
            "goal_rate",
            "data_source",
        ]
    ].copy()
    table["goal_rate"] = table["goal_rate"].map(format_percent)
    st.dataframe(
        table.rename(
            columns={
                "national_team": "國家隊",
                "player_name": "球員姓名",
                "jersey_number": "背號",
                "position_zh": "位置",
                "age": "年齡",
                "height_cm": "身高(cm)",
                "preferred_foot": "慣用腳",
                "market_value_eur_m": "身價(百萬歐元)",
                "club": "所屬俱樂部",
                "national_caps": "國家隊出賽",
                "national_goals": "國家隊進球",
                "goal_rate": "進球率",
                "data_source": "資料來源",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )


def national_team_center_page() -> None:
    page_header("國家隊資料中心", "完整名單、平均年齡、總身價、世界排名與小組出線率")
    database = player_database(players_df, team_meta_df)
    summary = squad_summary(database)
    simulation_df = v7_simulation()[["team", "group_qualified_probability"]].copy()
    summary = summary.merge(simulation_df, on="team", how="left")

    teams = summary["national_team"].sort_values().tolist()
    selected_team = st.selectbox("選擇國家隊", teams)
    selected_summary = summary[summary["national_team"] == selected_team].iloc[0]
    roster = database[database["team"] == selected_summary["team"]].copy()

    cols = st.columns(5)
    with cols[0]:
        display_card("球員數", str(int(selected_summary["players"])), "名單資料")
    with cols[1]:
        display_card("平均年齡", f"{selected_summary['average_age']:.1f}", "歲")
    with cols[2]:
        display_card("總身價", f"€{selected_summary['total_market_value_eur_m']:.1f}M", "fallback/API")
    with cols[3]:
        display_card("世界排名", str(int(selected_summary["fifa_ranking"])), "FIFA ranking")
    with cols[4]:
        display_card("小組出線率", format_percent(float(selected_summary["group_qualified_probability"])), f"{V7_SIMULATIONS:,} 次模擬")

    position_chart = roster.groupby("position_zh", as_index=False)["player_name"].count()
    chart = px.bar(
        position_chart,
        x="position_zh",
        y="player_name",
        text="player_name",
        color="player_name",
        color_continuous_scale=["#415a77", GOLD_LIGHT],
        labels={"position_zh": "位置", "player_name": "人數"},
    )
    chart.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color=INK,
        coloraxis_showscale=False,
    )
    st.plotly_chart(chart, use_container_width=True)

    roster_table = roster[
        [
            "player_name",
            "jersey_number",
            "position_zh",
            "age",
            "height_cm",
            "market_value_eur_m",
            "national_caps",
            "national_goals",
            "data_source",
        ]
    ].copy()
    st.dataframe(
        roster_table.rename(
            columns={
                "player_name": "球員姓名",
                "jersey_number": "背號",
                "position_zh": "位置",
                "age": "年齡",
                "height_cm": "身高(cm)",
                "market_value_eur_m": "身價(百萬歐元)",
                "national_caps": "國家隊出賽",
                "national_goals": "國家隊進球",
                "data_source": "資料來源",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )


def live_time_text(value) -> str:
    if value is None or pd.isna(value):
        return "時間待定"
    timestamp = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(timestamp):
        return str(value)
    return timestamp.tz_convert("Asia/Taipei").strftime("%Y/%m/%d %H:%M")


def live_matches_page() -> None:
    page_header("即時賽況", "API 優先載入真實賽況；無 API key、無資料或額度不足時自動使用 fallback 展示資料")
    refresh_seconds = st.selectbox("刷新頻率", [30, 45, 60], index=1)
    st.markdown(
        f"<meta http-equiv='refresh' content='{refresh_seconds}'>",
        unsafe_allow_html=True,
    )

    target_date = pd.Timestamp.now(tz="Asia/Taipei").date()
    live_matches, live_events, data_source = load_live_data(
        st.secrets,
        live_matches_df,
        live_events_df,
        target_date=target_date,
    )

    st.caption(f"時區：台灣時間（UTC+8）｜刷新頻率：{refresh_seconds} 秒｜資料來源：{data_source}")
    if "fallback" in data_source.lower():
        st.warning("展示資料／模擬即時賽況：目前未取得真實 API 資料，畫面使用本地 fallback CSV，避免部署時空白或壞掉。")
    else:
        st.success("目前即時賽況由 API 載入；若 API 未提供事件或技術統計，該區塊會顯示待補資料。")

    if live_matches.empty:
        st.info("今日沒有可顯示的比賽資料。")
        return

    def option_label(row: pd.Series) -> str:
        return (
            f"{live_time_text(row.get('scheduled_time'))} | "
            f"{team_name(row.get('home_team', 'TBD'))} "
            f"{row.get('home_score', 0)}-{row.get('away_score', 0)} "
            f"{team_name(row.get('away_team', 'TBD'))}"
        )

    labels = {
        option_label(row): row["live_match_id"]
        for _, row in live_matches.iterrows()
    }
    selected = st.selectbox("選擇即時比賽", list(labels.keys()))
    live_match_id = labels[selected]
    row = live_matches[live_matches["live_match_id"] == live_match_id].iloc[0]
    scheduled_time = live_time_text(row.get("scheduled_time"))
    venue = row.get("venue", "待官方公布")

    st.markdown(
        f"""
        <div class="display-card score-card">
          <div class="score-teams">{html.escape(team_name(row.get('home_team', 'TBD')))} vs {html.escape(team_name(row.get('away_team', 'TBD')))}</div>
          <div class="score-value">{row.get('home_score', 0)} : {row.get('away_score', 0)}</div>
          <div class="score-note">比賽時間：{html.escape(scheduled_time)} ｜ 狀態：{html.escape(str(row.get('status', '待官方公布')))} ｜ 分鐘：{int(row.get('minute', 0) or 0)}' ｜ 場地：{html.escape(str(venue))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cols = st.columns(3)
    with cols[0]:
        display_card("射門", f"{int(row.get('home_shots', 0))} : {int(row.get('away_shots', 0))}")
    with cols[1]:
        display_card("控球率", f"{int(row.get('home_possession', 50))}% : {int(row.get('away_possession', 50))}%")
    with cols[2]:
        display_card("角球", f"{int(row.get('home_corners', 0))} : {int(row.get('away_corners', 0))}")

    st.subheader("比賽事件")
    events = live_events[live_events["live_match_id"] == live_match_id].copy()
    if events.empty:
        st.info("目前 API 或 fallback 資料沒有提供此場事件。")
    else:
        events["事件"] = events["event_type"].map(live_event_label)
        events["球隊"] = events["team"].map(lambda team: f"{flag(team)} {team_name(team, with_flag=False)}")
        st.dataframe(
            events[["minute", "事件", "球隊", "player", "detail"]].rename(
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
            {"項目": "射門", row["home_team"]: row.get("home_shots", 0), row["away_team"]: row.get("away_shots", 0)},
            {"項目": "控球率", row["home_team"]: row.get("home_possession", 50), row["away_team"]: row.get("away_possession", 50)},
            {"項目": "角球", row["home_team"]: row.get("home_corners", 0), row["away_team"]: row.get("away_corners", 0)},
        ]
    )
    stats_long = stats.melt(id_vars="項目", var_name="球隊", value_name="數值")
    chart = px.bar(
        stats_long,
        x="項目",
        y="數值",
        color="球隊",
        barmode="group",
        color_discrete_sequence=[GOLD, "#8aa0c3"],
    )
    chart.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color=INK)
    st.plotly_chart(chart, use_container_width=True)


def player_database_page() -> None:
    page_header("球員資料庫", "國家隊球員篩選、姓名搜尋與位置查詢")
    st.info("2026 世界盃最終名單若尚未完整公布，本頁先使用本地 fallback 球員資料。後續可接 API-Football 或 football-data.org 名單資料，API key 由 Streamlit secrets 管理。")

    database = player_database(players_df, team_meta_df)
    if database.empty:
        st.warning("目前沒有球員資料可顯示。")
        return

    teams = ["全部"] + database["national_team"].dropna().sort_values().unique().tolist()
    positions = ["全部"] + database["position_zh"].dropna().sort_values().unique().tolist()

    cols = st.columns([1.1, 1.0, 1.4])
    selected_team = cols[0].selectbox("國家隊", teams)
    selected_position = cols[1].selectbox("位置", positions)
    keyword = cols[2].text_input("搜尋球員姓名", "")

    filtered = database.copy()
    if selected_team != "全部":
        filtered = filtered[filtered["national_team"] == selected_team]
    if selected_position != "全部":
        filtered = filtered[filtered["position_zh"] == selected_position]
    if keyword.strip():
        filtered = filtered[filtered["player_name"].str.contains(keyword.strip(), case=False, na=False)]

    kpi_cols = st.columns(3)
    with kpi_cols[0]:
        display_card("球員筆數", str(len(filtered)), "目前顯示資料")
    with kpi_cols[1]:
        display_card("國家隊數", str(filtered["team"].nunique()), "篩選後")
    with kpi_cols[2]:
        avg_form = filtered["recent_form_rating"].mean() if not filtered.empty else 0
        display_card("平均近況", f"{avg_form:.1f}", "fallback 評分")

    table = filtered[
        [
            "national_team",
            "player_name",
            "jersey_number",
            "position_zh",
            "age",
            "club",
            "national_caps",
            "national_goals",
            "goal_rate",
            "recent_form_rating",
        ]
    ].copy()
    table["goal_rate"] = table["goal_rate"].map(format_percent)
    st.dataframe(
        table.rename(
            columns={
                "national_team": "國家隊",
                "player_name": "球員姓名",
                "jersey_number": "背號",
                "position_zh": "位置",
                "age": "年齡",
                "club": "所屬俱樂部",
                "national_caps": "國家隊出賽",
                "national_goals": "國家隊進球",
                "goal_rate": "進球率",
                "recent_form_rating": "近況評分",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )


def player_database_page() -> None:
    page_header("球員資料庫", "V7 球員搜尋、國家篩選、位置篩選與互動式資料表")
    st.info("若 API 尚未提供完整 2026 最終名單，本頁會使用本地 fallback 球員資料並保留資料來源欄位。")

    database = player_database(players_df, team_meta_df)
    teams = ["全部"] + database["national_team"].dropna().sort_values().unique().tolist()
    positions = ["全部"] + database["position_zh"].dropna().sort_values().unique().tolist()

    cols = st.columns([1.1, 1.0, 1.4])
    selected_team = cols[0].selectbox("國家隊", teams)
    selected_position = cols[1].selectbox("位置", positions)
    keyword = cols[2].text_input("輸入球員姓名", "")

    filtered = database.copy()
    if selected_team != "全部":
        filtered = filtered[filtered["national_team"] == selected_team]
    if selected_position != "全部":
        filtered = filtered[filtered["position_zh"] == selected_position]
    if keyword.strip():
        filtered = filtered[filtered["player_name"].str.contains(keyword.strip(), case=False, na=False)]

    cols = st.columns(4)
    with cols[0]:
        display_card("球員筆數", str(len(filtered)), "目前篩選")
    with cols[1]:
        display_card("國家隊數", str(filtered["team"].nunique()), "目前篩選")
    with cols[2]:
        display_card("平均年齡", f"{filtered['age'].mean():.1f}" if not filtered.empty else "0.0", "歲")
    with cols[3]:
        display_card("總身價", f"€{filtered['market_value_eur_m'].sum():.1f}M", "fallback/API")

    table = filtered[
        [
            "national_team",
            "player_name",
            "jersey_number",
            "position_zh",
            "age",
            "height_cm",
            "market_value_eur_m",
            "club",
            "national_caps",
            "national_goals",
            "goal_rate",
            "data_source",
        ]
    ].copy()
    table["goal_rate"] = table["goal_rate"].map(format_percent)
    st.dataframe(
        table.rename(
            columns={
                "national_team": "國家隊",
                "player_name": "球員姓名",
                "jersey_number": "背號",
                "position_zh": "位置",
                "age": "年齡",
                "height_cm": "身高(cm)",
                "market_value_eur_m": "身價(百萬歐元)",
                "club": "所屬俱樂部",
                "national_caps": "國家隊出賽",
                "national_goals": "國家隊進球",
                "goal_rate": "進球率",
                "data_source": "資料來源",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )


def national_team_center_page() -> None:
    page_header("國家隊資料中心", "V8：完整名單、世界排名、Elo、總身價與各階段晉級率")
    database = player_database(players_df, team_meta_df)
    summary = squad_summary(database)
    simulation_df = v7_simulation()[["team", *v7_stage_columns()]].copy()
    summary = summary.merge(simulation_df, on="team", how="left")

    teams = summary["national_team"].sort_values().tolist()
    selected_team = st.selectbox("選擇國家隊", teams)
    selected_summary = summary[summary["national_team"] == selected_team].iloc[0]
    roster = database[database["team"] == selected_summary["team"]].copy()

    cols = st.columns(6)
    with cols[0]:
        display_card("世界排名", str(int(selected_summary["fifa_ranking"])), "FIFA")
    with cols[1]:
        display_card("Elo Rating", str(int(selected_summary["elo"])), "模型評分")
    with cols[2]:
        display_card("平均年齡", f"{selected_summary['average_age']:.1f}", "歲")
    with cols[3]:
        display_card("總身價", f"€{selected_summary['total_market_value_eur_m']:.1f}M", "無資料以 0 計")
    with cols[4]:
        display_card("球員數", str(int(selected_summary["players"])), "名單")
    with cols[5]:
        display_card("奪冠率", format_percent(float(selected_summary["champion_probability"])), f"{V7_SIMULATIONS:,} 次")

    stage_labels = v7_stage_labels()
    stage_df = pd.DataFrame(
        [(stage_labels[col], selected_summary[col]) for col in v7_stage_columns()],
        columns=["階段", "機率"],
    )
    stage_df["百分比"] = stage_df["機率"].map(format_percent)
    col_a, col_b = st.columns([1.1, 0.9])
    with col_a:
        chart = px.bar(
            stage_df,
            x="階段",
            y="機率",
            text="百分比",
            color="機率",
            color_continuous_scale=["#415a77", GOLD_LIGHT],
            labels={"機率": "晉級機率"},
        )
        chart.update_traces(textposition="outside")
        chart.update_yaxes(tickformat=".0%", range=[0, 1])
        chart.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color=INK, coloraxis_showscale=False)
        st.plotly_chart(chart, use_container_width=True)
    with col_b:
        position_chart = roster.groupby("position_zh", as_index=False)["player_name"].count()
        pie = px.pie(position_chart, values="player_name", names="position_zh", hole=0.42, color_discrete_sequence=[GOLD, "#8aa0c3", "#28a745", "#dc3545"])
        pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color=INK, legend_title_text="位置")
        st.plotly_chart(pie, use_container_width=True)

    table = roster[
        [
            "player_name",
            "jersey_number",
            "position_zh",
            "age",
            "height_cm",
            "preferred_foot",
            "market_value_eur_m",
            "club",
            "national_caps",
            "national_goals",
        ]
    ].copy()
    st.dataframe(
        table.rename(
            columns={
                "player_name": "球員姓名",
                "jersey_number": "背號",
                "position_zh": "位置",
                "age": "年齡",
                "height_cm": "身高(cm)",
                "preferred_foot": "慣用腳",
                "market_value_eur_m": "身價(百萬歐元)",
                "club": "所屬俱樂部",
                "national_caps": "國家隊出賽",
                "national_goals": "國家隊進球",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )


def worldcup_simulator_page() -> None:
    page_header("世界盃模擬器", "獨立 Monte Carlo 模擬頁：選擇次數、開始模擬、查看各隊晉級與奪冠率")
    disclaimer_box()
    st.info(
        "本頁僅使用 Elo、近期狀態、世界盃歷史表現與 Poisson 進球分布進行展示型 Monte Carlo 模擬。"
        "結果屬於展示資料／模擬資料，不保證準確，也不包含即時比分、控球率或射門數。"
    )

    simulations = st.radio("模擬次數", [1000, 5000, 10000], index=0, horizontal=True)
    start = st.button("開始模擬", type="primary", use_container_width=True)

    if start:
        with st.spinner(f"正在執行 {int(simulations):,} 次 Monte Carlo 模擬..."):
            sim_df = run_worldcup_monte_carlo(
                fixtures_df,
                team_meta_df,
                wc_team_stats_df,
                matches_df,
                simulations=int(simulations),
            )
        data_label = f"即時計算模擬結果｜{int(simulations):,} 次"
    else:
        try:
            sim_df = pd.read_csv("data/monte_carlo_results.csv")
            simulations = int(sim_df.get("simulation_count", pd.Series([simulations])).iloc[0])
            data_label = f"展示資料／模擬資料｜本地 CSV｜{int(simulations):,} 次"
        except Exception:
            sim_df = cached_tournament_simulation(
                fixtures_df,
                team_meta_df,
                wc_team_stats_df,
                matches_df,
                int(simulations),
            )
            data_label = f"展示資料／模擬資料｜fallback 計算｜{int(simulations):,} 次"

    required = [
        "team_display",
        "elo",
        "group_qualified_probability",
        "round_16_probability",
        "round_8_probability",
        "semi_final_probability",
        "final_probability",
        "champion_probability",
    ]
    missing = [column for column in required if column not in sim_df.columns]
    if missing:
        st.warning(f"模擬結果缺少欄位：{', '.join(missing)}。已停止顯示，避免頁面報錯。")
        return

    sim_df = sim_df.sort_values("champion_probability", ascending=False).reset_index(drop=True)
    champion = sim_df.iloc[0]
    cols = st.columns(4)
    with cols[0]:
        display_card("模擬次數", f"{int(simulations):,}", "Monte Carlo")
    with cols[1]:
        display_card("參賽隊伍", str(len(sim_df)), "本地賽程資料")
    with cols[2]:
        display_card("最高冠軍率", format_percent(float(champion["champion_probability"])), champion["team_display"])
    with cols[3]:
        display_card("資料狀態", "展示資料／模擬資料", data_label)

    st.subheader("冠軍率排行榜")
    ranking = sim_df.head(20).copy()
    ranking["排名"] = ranking.index + 1
    ranking["冠軍率"] = ranking["champion_probability"].map(format_percent)
    ranking["決賽率"] = ranking["final_probability"].map(format_percent)
    st.dataframe(
        ranking[["排名", "team_display", "elo", "決賽率", "冠軍率"]].rename(
            columns={"team_display": "國家隊", "elo": "Elo"}
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Top 10 冠軍機率長條圖")
    top10 = sim_df.head(10).sort_values("champion_probability", ascending=True).copy()
    top10["label"] = top10["champion_probability"].map(format_percent)
    bar = px.bar(
        top10,
        x="champion_probability",
        y="team_display",
        orientation="h",
        text="label",
        color="champion_probability",
        color_continuous_scale=["#415a77", GOLD_LIGHT],
        labels={"champion_probability": "冠軍率", "team_display": "國家隊"},
    )
    bar.update_traces(textposition="outside")
    bar.update_xaxes(tickformat=".0%")
    bar.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color=INK,
        coloraxis_showscale=False,
        margin=dict(l=10, r=50, t=10, b=10),
    )
    st.plotly_chart(bar, use_container_width=True)

    st.subheader("晉級率總表")
    table = sim_df[
        [
            "team_display",
            "group_qualified_probability",
            "round_16_probability",
            "round_8_probability",
            "semi_final_probability",
            "final_probability",
            "champion_probability",
        ]
    ].copy()
    rename_map = {
        "team_display": "國家隊",
        "group_qualified_probability": "小組出線率",
        "round_16_probability": "16強率",
        "round_8_probability": "8強率",
        "semi_final_probability": "4強率",
        "final_probability": "決賽率",
        "champion_probability": "冠軍率",
    }
    for column in table.columns:
        if column != "team_display":
            table[column] = table[column].map(format_percent)
    st.dataframe(table.rename(columns=rename_map), use_container_width=True, hide_index=True)

    st.subheader("階段機率熱度圖")
    heat = sim_df.head(16)[
        [
            "team_display",
            "group_qualified_probability",
            "round_16_probability",
            "round_8_probability",
            "semi_final_probability",
            "final_probability",
            "champion_probability",
        ]
    ].copy()
    heat = heat.set_index("team_display").rename(columns={
        "group_qualified_probability": "小組出線率",
        "round_16_probability": "16強率",
        "round_8_probability": "8強率",
        "semi_final_probability": "4強率",
        "final_probability": "決賽率",
        "champion_probability": "冠軍率",
    })
    heatmap = px.imshow(heat, aspect="auto", color_continuous_scale=["#071426", GOLD_LIGHT], labels=dict(color="機率"))
    heatmap.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color=INK)
    st.plotly_chart(heatmap, use_container_width=True)

    st.subheader("模型說明")
    st.markdown(
        """
        - 以本地 2026 世界盃展示賽程作為模擬骨架。
        - 球隊強度由 Elo Rating、近期狀態與 2002～2022 世界盃歷史表現組成。
        - 單場比分使用 Poisson 進球模型抽樣。
        - 淘汰賽平手時以強度差轉換勝率決定晉級隊伍。
        - 結果僅供分析展示，不保證準確，不構成投注建議。
        """
    )


def presentation_mode_page() -> None:
    dashboard_page()
    return
    sim_df = v7_simulation()
    rankings = build_elo_rankings(team_meta_df).head(20)
    player_db = player_database(players_df, team_meta_df)
    live_matches, _, live_source = load_live_data(st.secrets, live_matches_df, live_events_df, target_date=pd.Timestamp.now(tz="Asia/Taipei").date())

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
    live_matches, _, live_source = load_live_data(st.secrets, live_matches_df, live_events_df, target_date=pd.Timestamp.now(tz="Asia/Taipei").date())

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
        ("操作流程", "首頁 → 單場分析 → 模擬器 → 資料庫", "適合課堂快速展示與口頭報告。"),
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


def _player_center_data() -> pd.DataFrame:
    data = player_database(players_df, team_meta_df).copy()
    defaults = {
        "player_name": "未提供",
        "team_zh": "未提供",
        "position": "未提供",
        "age": 0,
        "height_cm": 0,
        "club": "未提供",
        "national_caps": 0,
        "national_goals": 0,
        "assists": 0,
        "market_value_eur_m": 0.0,
        "fifa_ranking": 0,
        "flag_emoji": "",
        "goal_rate": 0.0,
        "recent_form_rating": 0.0,
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
        "recent_form_rating",
    ]
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce").fillna(0)

    def clean_display_text(value) -> str:
        text = str(value or "").strip()
        if not text or text.lower() == "nan":
            return "資料待補"
        mojibake_markers = ["�", "獺", "穩", "禳", "????"]
        if any(marker in text for marker in mojibake_markers):
            return "資料待補"
        return text

    for column in ["player_name", "team_zh", "position", "club"]:
        data[column] = data[column].map(clean_display_text)
    data["team_label"] = data["flag_emoji"].fillna("") + " " + data["team_zh"].fillna(data["team"])
    data["player_score"] = (
        data["recent_form_rating"] * 2
        + data["goal_rate"] * 10
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
                table = rows[["player", "position", "appearances", "goals", "assists", "impact_score", "is_available"]].copy()
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
    prediction_cards(row)

    probabilities = prediction_to_frame(prediction)
    probability_display = probabilities.copy()
    probability_display.iloc[:, 1] = probability_display.iloc[:, 1].map(format_percent)
    st.dataframe(
        probability_display,
        use_container_width=True,
        hide_index=True,
    )

    render_worldcup_history_block(row["home_team"], row["away_team"])
    render_h2h_summary_block(row["home_team"], row["away_team"])
    render_key_players_block(row["home_team"], row["away_team"])
    _render_player_impact(row["home_team"], row["away_team"])


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

    display_card("模擬次數", f"{int(simulations):,}", "Monte Carlo")
    top10 = sim_df.head(10).sort_values("champion_probability", ascending=True).copy()
    top10["label"] = top10["champion_probability"].map(format_percent)
    chart = px.bar(
        top10,
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


def betting_page() -> None:
    page_header("市場機率分析", "將模型機率與賠率隱含機率融合，提供風險參考")
    disclaimer_box()
    row = selected_fixture()
    prediction = predict_match(matches_df, row["home_team"], row["away_team"], wc_team_stats_df)
    table = market_probability_table(row, prediction)

    st.subheader(f"{fixture_time_text(row)} ｜ {matchup_text(row)}")
    display = table.copy()
    for column in ["model_probability", "market_probability", "fused_probability"]:
        display[column] = display[column].map(format_percent)
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
    st.warning("風險提醒：本頁僅做市場機率與模型機率比較，不提供下注功能，不保證賽果或獲利。")


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

    cols = st.columns(len(summary))
    for col, (_, row) in zip(cols, summary.iterrows()):
        with col:
            display_card(str(row["team"]), f"xG {row['xg']:.2f}", f"實際進球 {int(row['goals'])}")

    st.subheader("xG vs 實際進球")
    comparison = summary.melt(id_vars="team", value_vars=["xg", "goals"], var_name="指標", value_name="數值")
    bar = px.bar(comparison, x="team", y="數值", color="指標", barmode="group", color_discrete_sequence=[GOLD, "#8aa0c3"])
    bar.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color=INK)
    st.plotly_chart(bar, use_container_width=True)

    st.subheader("xG 累積走勢")
    match_shots["cumulative_xg"] = match_shots.groupby("team")["xg_value"].cumsum()
    line = px.line(match_shots, x="minute", y="cumulative_xg", color="team", markers=True, labels={"minute": "分鐘", "cumulative_xg": "累積 xG"})
    line.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color=INK)
    st.plotly_chart(line, use_container_width=True)

    st.subheader("球員射門 xG 排行")
    st.dataframe(
        match_shots.sort_values("xg_value", ascending=False)[["minute", "team", "player", "body_part", "situation", "is_goal", "xg_value"]].rename(
            columns={
                "minute": "分鐘",
                "team": "球隊",
                "player": "球員",
                "body_part": "身體部位",
                "situation": "射門情境",
                "is_goal": "是否進球",
                "xg_value": "xG",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("xG 分析")
    for line_text in xg_analysis_text(summary):
        st.markdown(f"- {line_text}")
    st.caption("預測權重參考：Elo 40%、近期狀態 20%、歷史成績 15%、xG 表現 25%。")


def dashboard_page() -> None:
    st.markdown(
        """
        <div class="hero">
          <div>
            <div class="hero-kicker">WORLD CUP INTELLIGENCE CENTER</div>
            <div class="hero-title">世界盃情報中心</div>
            <div class="hero-copy">
              整合 2026 世界盃賽程、Elo、球員資料、xG、市場機率與 Monte Carlo 模擬，
              提供可解釋的足球資料分析。
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

    cols = st.columns(4)
    with cols[0]:
        display_card("近期賽程", str(len(fixture_odds_df.head(6))), "台灣時間 UTC+8")
    with cols[1]:
        display_card("預測比分", f"{prediction.predicted_home_goals} : {prediction.predicted_away_goals}", matchup_text(row))
    with cols[2]:
        display_card("冠軍率最高", format_percent(float(champion["champion_probability"])), str(champion["team_display"]))
    with cols[3]:
        display_card("模型組合", "Elo + Poisson + xG", "可解釋分析")

    left, right = st.columns([1.1, 0.9])
    with left:
        st.subheader("近期賽程")
        st.dataframe(fixtures_with_flags(fixture_odds_df.head(5)), use_container_width=True, hide_index=True)
    with right:
        st.subheader("冠軍機率 Top 5")
        top5 = sim_df.head(5).sort_values("champion_probability", ascending=True).copy()
        top5["label"] = top5["champion_probability"].map(format_percent)
        chart = px.bar(top5, x="champion_probability", y="team_display", orientation="h", text="label", color="champion_probability", color_continuous_scale=["#415a77", GOLD_LIGHT])
        chart.update_xaxes(tickformat=".0%")
        chart.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color=INK, coloraxis_showscale=False)
        st.plotly_chart(chart, use_container_width=True)


if page == "首頁儀表板":
    dashboard_page()
elif page == "世界盃賽程表":
    fixtures_page()
elif page == "賽程頁":
    fixtures_page()
elif page == "單場分析頁":
    match_analysis_page()
elif page == "投注分析頁":
    betting_page()
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
elif page == "歷史世界盃數據分析":
    worldcup_history_page()
elif page == "國家隊世界盃戰績":
    team_record_page()
elif page == "歷史交手分析":
    head_to_head_page()
else:
    disclaimer_page()

footer()
