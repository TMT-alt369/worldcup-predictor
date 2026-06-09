import pandas as pd
import streamlit as st


def format_percent(value: float) -> str:
    return f"{float(value) * 100:.1f}%"


def disclaimer_box() -> None:
    st.warning(
        "本網站內容僅供足球資料分析參考，不構成下注指示，也不承諾任何獲利。"
        "請自行判斷風險。"
    )


def signal_dataframe(signals) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "市場": signal.market,
                "賠率": signal.odds,
                "模型機率": format_percent(signal.model_probability),
                "隱含機率": format_percent(signal.implied_probability),
                "差距": format_percent(signal.edge),
                "分析訊號": signal.recommendation,
                "風險等級": signal.risk_level,
            }
            for signal in signals
        ]
    )
