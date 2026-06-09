# World Cup Data Platform

Streamlit 世界盃資料平台，整合 2026 世界盃賽程、單場比分預測、勝平負機率、投注風險輔助、歷史世界盃資料、球隊資料、球員資料、Elo 世界排名、冠軍機率與晉級機率模擬。

本網站僅供資料分析參考，不提供下注、金流、會員錢包或獲利承諾。

## 功能

- 首頁儀表板：近期賽程、預測摘要、冠軍機率、晉級機率與即時足球資料摘要
- 世界盃賽程表：2026 世界盃賽程、階段、球場、城市與台灣時間
- 單場分析：預測比分、勝平負機率、信心分數、歷史戰績、歷史交手與關鍵球員
- 投注分析：勝平負機率、賠率隱含機率、風險等級與責任提醒
- 即時實況：支援世界盃資料模式與真實足球即時資料模式
- 球員資料中心：球員表格、搜尋、國家篩選、位置篩選與核心球員 Top5
- 球隊資料庫：國旗、洲別、FIFA 排名、Elo 分數與球隊資訊
- Elo 世界排名：完整 Elo 排名與 Top 10 視覺化
- 冠軍機率預測：Elo、Poisson 與 Monte Carlo 模擬
- 晉級機率分析：小組出線、16 強、8 強、4 強、決賽與冠軍機率
- 模型回測：回測場次、命中率、ROI、平均信心分數與信心分層
- 免責聲明：資料限制、模型限制與投注風險說明

## 即時資料來源

即時實況頁提供兩種模式：

- 世界盃展示資料：使用本地 `data/mock_live_matches.csv` 與 `data/mock_live_events.csv`
- 真實足球即時資料：優先查詢 API-Football；若無可用資料，自動改查 ESPN Scoreboard API；若 ESPN 也無資料，回到本地資料

前端只顯示簡潔狀態：

- `🟢 真實 API 即時資料`
- `🟡 展示資料`

## Streamlit Secrets

API key 不應寫入 `.py`、CSV、README 或 GitHub Repository。

如需啟用 API-Football，請到 Streamlit Cloud：

```text
Manage app -> Settings -> Secrets
```

新增 Secret 欄位：

```text
FOOTBALL_API_KEY
```

欄位值請填入你的 API-Football 私密 key。若未設定，系統會使用 ESPN Scoreboard API；若 ESPN 也無資料，才會使用本地資料。

## 技術架構

- Python
- Streamlit
- Pandas
- Plotly
- Requests
- 本地 CSV 資料
- Streamlit Secrets

## 本機執行

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

網站預設開啟：

```text
http://localhost:8501
```

## 部署

專案可部署到 Streamlit Community Cloud。

部署條件：

- `app.py` 位於專案根目錄
- `requirements.txt` 位於專案根目錄
- `data/` 包含必要 CSV
- API key 只放在 Streamlit Secrets

## 免責聲明

本網站所有預測、模型分數、賠率比較與風險提示僅供資料分析參考，不代表實際賽果，也不保證命中率或投資報酬。請勿將本網站視為投注保證工具。
