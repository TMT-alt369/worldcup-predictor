# 世界盃智慧預測平台

Streamlit 製作的 2026 世界盃資料與預測平台，整合賽程、即時資料 fallback、Elo、Poisson、xG、市場機率、球員影響模型與 Monte Carlo 模擬。

本專案僅供足球資料分析與機率學習，不提供下注功能，也不保證任何預測結果。

## 主要功能

- 世界盃賽程表與台灣時間顯示
- 單場比分預測、勝平負機率與智慧模型信心指數
- 即時世界盃中心：賽果輸入、小組積分、動態 Elo 與機率重新計算
- 球員影響分析：球員評分、缺陣模擬、勝率/xG/冠軍率變化
- AI 世界盃模擬器：1000、5000、10000 次 Monte Carlo 模擬
- 冠軍率 Top20、晉級率總表、常見決賽組合、黑馬排行榜與洲別冠軍機率
- Elo 世界排名、xG 模型分析、球隊與球員資料庫
- 市場機率分析與賠率試算中心
- 手機版優化：側邊欄收合、卡片單欄、表格橫向滑動與圖表自適應

## V24

新增 AI 世界盃蒙地卡羅模擬器：

- `utils/ai_monte_carlo_engine.py`
- 頁面：AI 世界盃模擬器
- 支援 1000 / 5000 / 10000 次模擬
- 輸出冠軍率 Top20、各階段晉級率、常見決賽組合、黑馬榜與洲別冠軍率

## V25

終極版首頁與手機版優化：

- 首頁新增今日焦點賽事、即時世界盃總覽、奪冠熱門 Top10、晉級機率摘要、AI 今日分析摘要、xG 摘要與市場機率摘要
- 側邊欄維持分類導航，並預設收合
- 手機版加強卡片、表格、圖表、按鈕與底部留白

## API Key

若要啟用 API-Football，請在 Streamlit Cloud 設定 Secrets：

```text
FOOTBALL_API_KEY="your_api_football_key"
```

若沒有 API key，系統會自動使用 ESPN 或本地 mock/fallback 資料，避免頁面中斷。

## 本機執行

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

開啟：

```text
http://localhost:8501
```

## 部署

本專案可直接部署到 Streamlit Community Cloud：

- 入口檔：`app.py`
- 依賴：`requirements.txt`
- 資料：`data/`
- Secrets：只在 Streamlit Cloud 後台設定，不提交到 GitHub

## 免責聲明

所有模型、機率、xG、市場機率與模擬結果僅供資料分析參考。足球比賽受臨場狀態、陣容、戰術、天氣與隨機因素影響，請理性看待所有預測結果。
