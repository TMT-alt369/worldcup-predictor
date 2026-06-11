from __future__ import annotations


def mobile_css() -> str:
    return """
    <style>
    div[data-testid="stDataFrame"], div[data-testid="stTable"] {
        max-width: 100%;
        overflow-x: auto;
    }
    div[data-testid="stDataFrame"] div[role="grid"] {
        min-width: 680px;
    }
    .element-container:has(.js-plotly-plot) {
        max-width: 100%;
        overflow-x: auto;
    }
    .stButton > button {
        min-height: 44px;
        border-radius: 8px;
        font-weight: 800;
    }
    @media (max-width: 768px) {
        [data-testid="stSidebar"] {
            max-width: min(88vw, 330px);
        }
        [data-testid="stSidebar"] [role="radiogroup"] label {
            padding: 9px 10px;
            font-size: 0.96rem;
        }
        .block-container {
            padding: 1rem 0.82rem 7.5rem !important;
            max-width: 100% !important;
        }
        [data-testid="stSidebar"] {
            transform: translateX(0);
        }
        .hero {
            grid-template-columns: 1fr !important;
            min-height: auto !important;
            padding: 22px !important;
            gap: 16px !important;
        }
        .hero-title {
            font-size: 2.14rem !important;
            line-height: 1.15 !important;
        }
        .hero-copy {
            font-size: 1rem !important;
        }
        .hero-visual {
            min-height: 190px !important;
        }
        .trophy {
            font-size: 5.2rem !important;
        }
        .page-title h1 {
            font-size: 1.62rem !important;
        }
        .page-title p {
            font-size: 0.98rem !important;
        }
        .display-card, .confidence-wrap {
            padding: 16px !important;
            margin-bottom: 12px !important;
            width: 100% !important;
            max-width: 100% !important;
        }
        .card-value {
            font-size: 1.7rem !important;
            overflow-wrap: anywhere;
        }
        .card-note, .card-label {
            font-size: 0.92rem !important;
            line-height: 1.55 !important;
        }
        .score-card {
            padding: 28px 14px !important;
            text-align: center !important;
        }
        .score-value {
            font-size: 4.6rem !important;
        }
        .score-teams {
            font-size: 1rem !important;
            overflow-wrap: anywhere;
        }
        div[data-testid="column"] {
            width: 100% !important;
            flex: 1 1 100% !important;
            min-width: 100% !important;
        }
        div[data-testid="stDataFrame"] {
            border: 1px solid rgba(214, 178, 94, 0.18);
            border-radius: 8px;
        }
        div[data-testid="stDataFrame"] div[role="grid"] {
            min-width: 760px;
            font-size: 0.94rem;
        }
        .js-plotly-plot, .plot-container {
            width: 100% !important;
        }
        .js-plotly-plot .main-svg {
            max-height: 360px;
        }
        .stSelectbox, .stRadio, .stNumberInput, .stSlider, .stTextInput {
            margin-bottom: 0.72rem;
        }
        .stButton > button {
            width: 100%;
            margin: 0.25rem 0;
            min-height: 52px !important;
            font-size: 1rem !important;
        }
        .stTabs [role="tablist"] {
            overflow-x: auto;
            flex-wrap: nowrap;
        }
        .stTabs [role="tab"] {
            min-width: max-content;
            padding: 10px 14px;
        }
        .risk-badge {
            margin-left: 0;
            margin-top: 6px;
        }
        footer, .footer {
            padding-bottom: 7rem !important;
        }
        [data-testid="stToolbar"], [data-testid="stDecoration"] {
            max-width: 100vw !important;
        }
    }
    </style>
    """
