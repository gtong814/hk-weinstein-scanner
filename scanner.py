import io
import os
import time
import warnings
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import requests
import yfinance as yf

warnings.filterwarnings("ignore")

# ==========================================
# 1. 參數設定 (對應 Pine Script Inputs)
# ==========================================
CONFIG = {
    "ma_len": 30,          # 均線週期 (週線基準)
    "slope_lb": 4,         # 均線斜率回看週數
    "slope_thresh": 0.3,   # 斜率閾值 (%)
    "vol_len": 20,         # 成交量均線週期
    "vol_mult": 1.5,       # 突破爆量倍數
    "bench_sym": "^HSI",   # 基準指數 (Yahoo Finance 恒指代號)
    "mrs_len": 52,         # Mansfield RS 基準週數
    "use_mrs_flt": True,   # 啟用 Mansfield RS > 0 過濾突破
    "req_mrs_up": True,    # 要求 Mansfield RS 斜率向上
    "ao_fast": 5,          # AO 快線週期
    "ao_slow": 34,         # AO 慢線週期
    "ext_pct": 25.0,       # Wave 5 均線正乖離警示 (%)
    "atr_len": 14,         # 週線 ATR 波動率週期
    "atr_mult": 1.0,       # 一週預期波幅倍數
    "min_weekly_turnover": 5_000_000,  # 最低週成交額過濾 (HKD)，設 0 則不過濾仙股/冷門股
    "batch_size": 100,     # 每次批量下載股票數量
}

HKEX_URL = "https://www.hkex.com.hk/eng/services/trading/securities/securitieslists/ListOfSecurities.xlsx"


# ==========================================
# 2. 獲取全港上市股票名單 (HKEX 官方資料)
# ==========================================
def fetch_hk_stock_list() -> dict:
    """從 HKEX 下載最新全港股名單，回傳 {ticker: name} 字典"""
    print("正在從 HKEX 下載全港上市股票名單...")
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(HKEX_URL, headers=headers, timeout=30)
    resp.raise_for_status()

    # HKEX Excel 前兩行通常是標題，動態尋找 "Stock Code" 所在列
    raw_df = pd.read_excel(io.BytesIO(resp.content), header=None)
    header_row = None
    for i in range(10):
        row_vals = raw_df.iloc[i].astype(str).tolist()
        if any("Stock Code" in v for v in row_vals):
            header_row = i
            break

    df = pd.read_excel(io.BytesIO(resp.content), skiprows=header_row)
    df.columns = [str(c).strip() for c in df.columns]

    # 只保留股票 (Equity)，排除債券、窩輪、牛熊證與 ETF
    equities = df[df["Category"].astype(str).str.strip().str.lower() == "equity"].copy()

    stock_map = {}
    for _, row in equities.iterrows():
        code_raw = str(row["Stock Code"]).strip()
        if not code_raw.isdigit():
            continue
        code_int = int(code_raw)
        # Yahoo Finance 港股格式：4位數補零 (如 0700.HK)，5位數維持原樣
        yf_ticker = f"{code_int:04d}.HK" if code_int < 10000 else f"{code_int}.HK"
        name = str(row["Name of Securities"]).strip()
        stock_map[yf_ticker] = name

    print(f"成功獲取 {len(stock_map)} 隻香港上市股票。")
    return stock_map


# ==========================================
# 3. 技術指標輔助函數 (對齊 TradingView)
# ==========================================
def calc_rma(series: pd.Series, length: int) -> pd.Series:
    """TradingView ta.rma (Wilder's Smoothing)"""
    return series.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()


def calc_atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int) -> pd.Series:
    """TradingView ta.atr"""
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return calc_rma(tr, length)


# ==========================================
# 4. 單股核心運算 (完整移植 Pine Script 邏輯)
# ==========================================
def analyze_stock(ticker: str, name: str, df: pd.DataFrame, bench_close: pd.Series) -> dict | None:
    if df is None or df.empty:
        return None

    df = df.dropna(subset=["Close", "High", "Low", "Volume"]).copy()
    # 需要至少 60 週數據以計算 52 週 Mansfield RS 與 30 週均線
    if len(df) < 60:
        return None

    # 確保索引無時區差異並與基準指數對齊
    df.index = pd.to_datetime(df.index).tz_localize(None)
    aligned_bench = bench_close.reindex(df.index, method="ffill")

    w_close = df["Close"]
    w_high = df["High"]
    w_low = df["Low"]
    w_vol = df["Volume"]

    # 流動性過濾 (最後一週成交額)
    last_turnover = float(w_close.iloc[-1] * w_vol.iloc[-1])
    if last_turnover < CONFIG["min_weekly_turnover"]:
        return None

    # 均線與斜率
    w_ma30 = w_close.rolling(CONFIG["ma_len"]).mean()
    w_vol_ma = w_vol.rolling(CONFIG["vol_len"]).mean()
    w_ma_slope = (w_ma30 - w_ma30.shift(CONFIG["slope_lb"])) / w_ma30.shift(CONFIG["slope_lb"]) * 100
    w_atr = calc_atr(w_high, w_low, w_close, CONFIG["atr_len"])

    slope_thresh = CONFIG["slope_thresh"]
    is_ma_up = w_ma_slope > slope_thresh
    is_ma_down = w_ma_slope < -slope_thresh
    is_ma_flat = (~is_ma_up) & (~is_ma_down)

    # Mansfield Relative Strength (MRS)
    rp = (w_close / aligned_bench) * 100
    rp_sma52 = rp.rolling(CONFIG["mrs_len"]).mean()
    mansfield_rs = ((rp / rp_sma52) - 1) * 100

    if CONFIG["req_mrs_up"]:
        is_mrs_bull = (mansfield_rs > 0) & (mansfield_rs > mansfield_rs.shift(1))
    else:
        is_mrs_bull = mansfield_rs > 0

    mrs_pass = pd.Series(True, index=df.index) if not CONFIG["use_mrs_flt"] else is_mrs_bull

    # Stan Weinstein 四階段狀態機 (逐週遞推)
    stages = np.ones(len(df), dtype=int)
    c_arr = w_close.values
    ma_arr = w_ma30.values
    up_arr = is_ma_up.values
    down_arr = is_ma_down.values
    flat_arr = is_ma_flat.values
    slope_arr = w_ma_slope.values

    for i in range(1, len(df)):
        if np.isnan(ma_arr[i]) or np.isnan(slope_arr[i]):
            stages[i] = stages[i - 1]
            continue

        prev_s = stages[i - 1]
        if c_arr[i] > ma_arr[i] and up_arr[i]:
            stages[i] = 2
        elif c_arr[i] < ma_arr[i] and down_arr[i]:
            stages[i] = 4
        elif c_arr[i] <= ma_arr[i] and (flat_arr[i] or slope_arr[i] > -slope_thresh) and prev_s == 4:
            stages[i] = 1
        elif c_arr[i] >= ma_arr[i] and flat_arr[i] and prev_s == 2:
            stages[i] = 3
        else:
            if c_arr[i] > ma_arr[i]:
                stages[i] = 3 if down_arr[i] else 2
            else:
                stages[i] = 2 if up_arr[i] else 4

    stage_s = pd.Series(stages, index=df.index)

    # 突破訊號
    raw_s2_breakout = (stage_s == 2) & (stage_s.shift(1) != 2) & (w_vol > w_vol_ma * CONFIG["vol_mult"])
    valid_s2_breakout = raw_s2_breakout & mrs_pass

    # 波浪理論主升浪 (Wave 3) 與 Wave 5 動能判定
    hl2 = (w_high + w_low) / 2
    ao = hl2.rolling(CONFIG["ao_fast"]).mean() - hl2.rolling(CONFIG["ao_slow"]).mean()
    ao_highest_20 = ao.rolling(20).max()
    price_highest_20 = w_high.rolling(20).max()

    ao_rising = (ao > ao.shift(1)) & (ao > 0)
    ao_is_highest = ao == ao_highest_20
    price_is_highest = w_high == price_highest_20
    ma_extension = (w_close - w_ma30) / w_ma30 * 100

    is_wave3_momentum = (stage_s == 2) & is_mrs_bull & ao_rising & ao_is_highest & (w_vol > w_vol_ma)
    is_wave5_warn = (stage_s == 2) & price_is_highest & (ao < ao_highest_20) & (ma_extension > CONFIG["ext_pct"])

    w3_trigger = is_wave3_momentum & (~is_wave3_momentum.shift(1).fillna(False))
    w5_trigger = is_wave5_warn & (~is_wave5_warn.shift(1).fillna(False))

    # 取最新一週數據
    idx = -1
    cur_close = float(w_close.iloc[idx])
    cur_ma30 = float(w_ma30.iloc[idx])
    cur_slope = float(w_ma_slope.iloc[idx])
    cur_mrs = float(mansfield_rs.iloc[idx])
    cur_atr = float(w_atr.iloc[idx])
    cur_ext = float(ma_extension.iloc[idx])
    cur_stage = int(stage_s.iloc[idx])
    cur_mrs_bull = bool(is_mrs_bull.iloc[idx])
    cur_w3_mom = bool(is_wave3_momentum.iloc[idx])
    cur_w5_warn = bool(is_wave5_warn.iloc[idx])

    if any(np.isnan([cur_close, cur_ma30, cur_slope, cur_mrs, cur_atr])):
        return None

    # 未來一星期走勢預測計算 (1-Week Quantitative Projection)
    weekly_drift_pct = cur_slope / CONFIG["slope_lb"]
    if cur_stage == 2:
        drift_weight = 1.5 if cur_w3_mom else (-0.3 if cur_w5_warn else 1.0)
    elif cur_stage == 4:
        drift_weight = 1.2
    else:
        drift_weight = 0.2

    fcst_base_price = cur_close * (1 + (weekly_drift_pct * drift_weight / 100))
    if cur_stage == 2 and cur_mrs_bull and not cur_w5_warn:
        fcst_base_adj = fcst_base_price + (cur_atr * 0.25)
    elif cur_stage == 4:
        fcst_base_adj = fcst_base_price - (cur_atr * 0.25)
    else:
        fcst_base_adj = fcst_base_price

    fcst_bull_high = fcst_base_adj + (cur_atr * CONFIG["atr_mult"] * (0.85 if cur_stage == 2 else 0.6))
    fcst_bear_low = fcst_base_adj - (cur_atr * CONFIG["atr_mult"] * (0.85 if cur_stage == 4 else 0.6))
    pct_base = (fcst_base_adj - cur_close) / cur_close * 100

    # 訊號標籤
    signals = []
    if bool(valid_s2_breakout.iloc[idx]):
        signals.append("S2強勢突破")
    if bool(w3_trigger.iloc[idx]):
        signals.append("W3主升")
    if bool(w5_trigger.iloc[idx]):
        signals.append("W5背離")

    stage_map = {1: "Stage 1 (築底)", 2: "Stage 2 (主升)", 3: "Stage 3 (頭部)", 4: "Stage 4 (下跌)"}

    return {
        "Ticker": ticker,
        "Name": name,
        "Close": round(cur_close, 3),
        "Stage": cur_stage,
        "Stage_Label": stage_map[cur_stage],
        "Signal": " | ".join(signals) if signals else "",
        "Mansfield_RS": round(cur_mrs, 2),
        "MA30": round(cur_ma30, 3),
        "MA30_Slope_%": round(cur_slope, 2),
        "MA30_Ext_%": round(cur_ext, 2),
        "1W_Base": round(fcst_base_adj, 3),
        "1W_Base_%": round(pct_base, 2),
        "1W_Low": round(fcst_bear_low, 3),
        "1W_High": round(fcst_bull_high, 3),
        "Weekly_Turnover_HKD": int(last_turnover),
    }


# ==========================================
# 5. Telegram 通知 (選用)
# ==========================================
def send_telegram_message(text: str):
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    # Telegram 單則訊息上限 4096 字元
    requests.post(url, json={"chat_id": chat_id, "text": text[:4000], "parse_mode": "Markdown