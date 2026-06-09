# 世界盃情報中心

以 Streamlit 建立的世界盃資料平台，整合賽程、比分預測、Elo 世界排名、球員資料、市場機率、xG 模型與 Monte Carlo 世界盃模擬。

本網站僅供資料分析參考，不提供下注、金流、會員錢包或獲利承諾。

## 功能

- 世界盃賽程表與近期賽程
- 單場比分預測、勝平負機率與信心分數
- 市場機率分析：模型機率 70% + 市場隱含機率 30%
- Elo 世界排名與近期賽果更新
- 球隊資料庫、球員資料庫與關鍵球員影響模型
- 冠軍機率、晉級機率與世界盃 Monte Carlo 模擬
- xG 預期進球模型分析
- 即時比分中心：API-Football 優先，ESPN fallback，本地資料備援

## Streamlit Secrets

API key 不要寫入 `.py`、CSV、README 或 GitHub Repository。

在 Streamlit Cloud 設定：

```text
Manage app -> Settings -> Secrets
```

新增：

```text
FOOTBALL_API_KEY="你的 API-Football Key"
```

若未設定 API key，系統會自動使用 ESPN 或本地備援資料，避免頁面空白。

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

- `app.py` 位於專案根目錄
- `requirements.txt` 位於專案根目錄
- `data/` 內含必要 CSV 備援資料
- API key 只透過 Streamlit Secrets 管理

## 免責聲明

所有預測與模擬結果都可能失準。請將本網站視為足球資料分析與可解釋模型研究工具，不代表實際賽果或任何獲利承諾。
