# 世足比分預測網站 MVP

這是一個使用 Streamlit 製作的世界盃比分預測與資料分析展示網站。網站提供 2026 世界盃賽程、單場比分預測、勝平負機率、投注輔助訊號、歷史世界盃資料、球隊/球員分析、模型回測與模擬即時賽況。

> 本專案僅供資料分析、學習與作品展示使用，不提供下注、金流或保證獲利。

## 功能

- 首頁儀表板：賽程摘要、目前回測命中率、歷史世界盃視覺化
- 賽程頁：2026 世界盃真實賽程，時間統一顯示台灣時間 UTC+8
- 單場分析頁：預測比分、勝平負機率、信心分數、歷史戰績、關鍵球員、歷史交手
- 投注分析頁：1X2 賠率、隱含機率、價值投注訊號與風險等級
- 模型回測頁：目前回測命中率、高信心場次命中率、ROI 與信心分層
- 冠軍機率預測：Elo、Poisson、近期狀態、歷史表現與 Monte Carlo 模擬的奪冠/晉級機率
- 晉級機率分析：選擇單一國家隊，查看小組出線、16 強、8 強、4 強、決賽與奪冠機率
- Elo 世界排名：依 Elo Rating 顯示完整排名表與前 10 名排行榜
- 即時賽況頁：本地 mock 資料展示比分、事件、控球率、射門數
- 免責聲明頁：資料與投注風險說明

## 技術架構

- Python
- Streamlit
- Pandas
- NumPy
- Plotly

## 專案結構

```text
.
├── app.py
├── requirements.txt
├── README.md
├── data/
│   ├── fixtures_real_2026.csv
│   ├── team_meta.csv
│   ├── players.csv
│   ├── historical_matches.csv
│   ├── odds.csv
│   ├── mock_live_matches.csv
│   ├── mock_live_events.csv
│   └── worldcup/
│       ├── worldcup_matches_2002_2022.csv
│       ├── worldcup_team_stats_2002_2022.csv
│       ├── worldcup_champions_2002_2022.csv
│       ├── worldcup_top4_2002_2022.csv
│       └── worldcup_head_to_head_2002_2022.csv
└── worldcup_predictor/
    ├── backtest.py
    ├── betting.py
    ├── data_loader.py
    ├── history.py
    ├── model.py
    └── ui.py
```

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

## Streamlit Community Cloud 部署

部署入口檔案：

```text
app.py
```

Python 依賴：

```text
requirements.txt
```

本專案不使用 `.env`，也不會把 API key 寫入公開檔案。即時賽況若需要真實資料，請透過 Streamlit Secrets 設定 `FOOTBALL_API_KEY`；未設定時會自動使用 repo 內的本地 CSV 展示資料。

建議在 Streamlit Community Cloud 的 Advanced settings 選擇 Python 3.12。

## 資料來源

歷史世界盃資料主要參考 Fjelstul World Cup Database：

- Repository: https://github.com/jfjelstul/worldcup
- Author: Joshua C. Fjelstul, Ph.D.
- License: CC-BY-SA 4.0

2026 賽程、球員、賠率與即時賽況資料為 MVP 展示用本地 CSV。即時賽況頁明確標示為「展示資料／模擬即時賽況」。

## 模型說明

MVP 保留簡單可解釋模型：

- Poisson 進球模型
- Elo 評分
- 近期狀態
- 世界盃歷史表現輔助權重
- 關鍵球員進球率作為回測分層與展示輔助
- 冠軍機率頁使用 1000 次 Monte Carlo 模擬，估算小組出線、16 強、8 強、4 強、決賽與冠軍機率
- 晉級機率分析頁重用同一套 1000 次 Monte Carlo 模擬結果，提供單隊階段機率視角
- Elo 世界排名頁使用 `team_meta.csv` 的 Elo 欄位，提供完整排序、級距與 Top 10 視覺化

模型輸出僅供資料分析參考，不代表實際比賽結果。

## 免責聲明

本網站不提供下注、金流、帳戶、會員或任何投注交易功能。所有預測、賠率分析與投注建議僅供資料分析與作品展示參考，不保證命中率或獲利。請自行承擔風險。

## V5 真實即時資訊與球員資料庫

V5 新增兩個展示重點：

- 即時賽況頁：優先讀取真實足球 API，顯示今日比賽、狀態、比分、時間、主客隊、場地與事件資料；若 API 無資料、額度不足或未設定 key，會自動使用 `data/mock_live_matches.csv` 與 `data/mock_live_events.csv` fallback。
- 球員資料庫頁：支援國家隊篩選、位置篩選與球員姓名搜尋，顯示球員姓名、背號、位置、年齡、俱樂部、國家隊、出賽數、進球數、進球率與近況評分。

目前 2026 世界盃最終球員名單若尚未完整公布，球員資料庫會先使用 `data/players.csv` 作為本地展示資料，並在頁面上清楚標示 fallback 來源。後續可在 `worldcup_predictor/players.py` 中擴充 API 匯入流程。

### Streamlit Secrets 設定

不要把 API key 寫進程式碼或提交到 GitHub。請在 Streamlit Cloud：

```text
Manage app -> Settings -> Secrets
```

加入以下任一組 key：

請新增一個 Secret 欄位：`FOOTBALL_API_KEY`。欄位值填入你在 API-Football 取得的私密 key，請不要提交到 GitHub。

V7 會優先嘗試 `API-Football` 以取得事件與技術統計，失敗、無資料或未設定 Secrets 時會使用本地 fallback CSV。

## V7 世界盃智慧預測中心

V7 延伸既有 V1~V5，不重做模型主流程，新增：

- 真實即時資訊：Streamlit Secrets 中存在 `FOOTBALL_API_KEY` 時優先使用 API-Football 讀取今日比賽、比分、比賽狀態、事件與技術統計；無 key、API 無資料或額度不足時使用本地 fallback。
- 球員資料庫：新增 `data/players_2026.csv`，欄位包含姓名、國籍、位置、年齡、身高、身價、國家隊出賽與進球、資料來源。若未匯入官方完整名單，頁面會標示 fallback 資料來源。
- 冠軍機率 TOP20：使用 Elo Rating、Poisson 進球模型與 10000 次 Monte Carlo Simulation，顯示排名、國旗、Elo 與奪冠率。
- 小組出線機率分析：以 Plotly 長條圖與排名表顯示各隊小組出線率。
- 淘汰賽晉級機率分析：顯示 16 強、8 強、4 強、決賽與奪冠機率。
- 國家隊資料中心：顯示單隊完整名單、平均年齡、總身價、世界排名與小組出線率。

所有新增圖表皆使用 Plotly，可滑鼠查看數值。V7 仍不提供下注、金流、會員或保證命中率功能。

## V8 真實資料版

V8 將球員資料庫升級為真實 squad-list 資料版本：

- `data/players_2026.csv` 目前由公開 `2026 FIFA World Cup squads` 名單表建立，載入 48 隊、1,244 名球員。
- 已包含：姓名、國家、位置、年齡、所屬俱樂部、國家隊出賽、國家隊進球。
- 身高、慣用腳、身價欄位已保留；若目前來源未提供，頁面會顯示待資料源補齊或以 0 計入總身價。
- 球員資料庫支援關鍵字搜尋、國家篩選、位置篩選、身價排序與年齡排序。
- 國家隊資料中心新增：世界排名、Elo Rating、平均年齡、總身價、球員名單、小組出線率、16 強率、8 強率、4 強率、決賽率、奪冠率。
- 新增「世界盃模擬器」頁，可選 1000、5000、10000 次 Monte Carlo 模擬，輸出單隊奪冠率、決賽率、四強率、八強率。
- 新增「專題展示模式」頁，整合世界排名 TOP20、奪冠機率 TOP20、即時賽況、球員資料統計與國家隊比較。

API-Football secrets 可額外設定：

請在 Streamlit Cloud Secrets 新增 `FOOTBALL_API_KEY` 欄位；欄位值只放在 Streamlit Cloud，不要寫入 repo。

若沒有 API key，即時賽況會使用 Fallback Dataset，頁面會清楚標示資料來源。

## V8 專題展示強化版

本次 V8 以「穩定展示」為優先，保留既有部署版本並強化專題展示流程：

- 修正「世界盃模擬器」：改為獨立 Monte Carlo 模擬頁，提供 1000 / 5000 / 10000 次選擇、「開始模擬」按鈕、冠軍率排行榜、晉級率總表、Top 10 冠軍機率圖與模型說明。
- 首頁儀表板：優先展示近期賽程、預測比分、冠軍機率 Top 5、晉級摘要、即時實況來源、Monte Carlo 摘要與模型說明。
- 側邊欄分類：改為「功能分類」加「頁面」選擇，降低手機版選單長度。
- 球員資料庫：新增 `data/player_database.csv`，支援國家隊篩選、位置篩選、球員搜尋與關鍵球員 Top 10。
- 球隊資料庫：新增 `data/team_database.csv`，支援洲別篩選、國家搜尋、Elo Top 10 與各洲參賽隊伍統計。
- 專題展示模式：改為卡片式排版，包含專題名稱、動機、功能架構、資料來源、模型說明、Monte Carlo、限制、未來發展與操作流程。
- 所有 fallback 與推估欄位皆標示為展示資料／模擬資料。
