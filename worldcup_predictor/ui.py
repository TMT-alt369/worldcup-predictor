import pandas as pd
import streamlit as st


def format_percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def disclaimer_box() -> None:
    st.warning(
        "本網站內容僅供資料分析與學習參考，不構成保證獲利或下注指示。"
        "投注具有風險，請自行判斷並承擔結果。"
    )


def signal_dataframe(signals) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "市場": signal.market,
                "賠率": signal.odds,
                "模型機率": format_percent(signal.model_probability),
                "隱含機率": format_percent(signal.implied_probability),
                "優勢值": format_percent(signal.edge),
                "建議": signal.recommendation,
                "風險": signal.risk_level,
            }
            for signal in signals
        ]
    )
