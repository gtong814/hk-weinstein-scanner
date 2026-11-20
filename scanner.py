import io
import json
import os
import re
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
    "ma_len": 30,                      # 均線週期 (週線基準)
    "slope_lb": 4,                     # 均線斜率回看週數
    "slope_thresh": 0.3,               # 斜率閾值 (%)
    "vol_len": 20,                     # 成交量均線週期
    "vol_mult": 1.5,                   # 突破爆量倍數
    "bench_sym": "^HSI",               # 基準指數 (恒生指數)
    "mrs_len": 52,                     # Mansfield RS 基準週數
    "use_mrs_flt": True,               # 啟用 Mansfield RS > 0 過濾突破
    "req_mrs_up": True,                # 要求 Mansfield RS 斜率向上
    "ao_fast": 5,                      # AO 快線週期
    "ao_slow": 34,                     # AO 慢線週期
    "ext_pct": 25.0,                   # Wave 5 均線正乖離警示 (%)
    "atr_len": 14,                     # 週線 ATR 波動率週期
    "atr_mult": 1.0,                   # 一週預期波幅倍數
    "min_weekly_turnover": 5_000_000,  # 最低週成交額過濾 (HKD)，設 0 則不過濾
    "batch_size": 100,                 # 每次批量下載股票數量
    "keep_history_days": 30,           # results 資料夾保留歷史 CSV 天數
}

HKEX_URL = "https://www.hkex.com.hk/eng/services/trading/securities/securitieslists/ListOfSecurities.xlsx"


# ==========================================
# 2. 獲取全港上市股票名單 (HKEX 官方資料)
# ==========================================
def fetch_hk_stock_list() -> dict:
    print("正在從 HKEX 下載全港上市股票名單...")
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(HKEX_URL, headers=headers, timeout=30)
    resp.raise_for_status()

    raw_df = pd.read_excel(io.BytesIO(resp.content), header=None)
    header_row = None
    for i in range(10):
        row_vals = raw_df.iloc[i].astype(str).tolist()
        if any("Stock Code" in v for v in row_vals):
            header_row = i
            break

    df = pd.read_excel(io.BytesIO(resp.content), skiprows=header_row)
    df.columns = [str(c).strip() for c in df.columns]

    equities = df[df["Category"].astype(str).str.strip().str.lower() == "equity"].copy()

    stock_map = {}
    for _, row in equities.iterrows():
        code_raw = str(row["Stock Code"]).strip()
        if not code_raw.isdigit():
            continue
        code_int = int(code_raw)
        yf_ticker = f"{code_int:04d}.HK" if code_int < 10000 else f"{code_int}.HK"
        name = str(row["Name of Securities"]).strip()
        stock_map[yf_ticker] = name

    print(f"成功獲取 {len(stock_map)} 隻香港上市股票。")
    return stock_map


# ==========================================
# 3. 技術指標輔助函數 (對齊 TradingView)
# ==========================================
def calc_rma(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()


def calc_atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int) -> pd.Series:
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


def cleanup_old_csvs(folder: str, keep_days: int, today_dt: datetime):
    """刪除超過 keep_days 天的歷史日期 CSV（保留 latest_*.csv 與報告）"""
    cutoff_date = (today_dt - timedelta(days=keep_days)).date()
    pattern = re.compile(r"^hk_(?:all_stages|signals)_(\d{4}-\d{2}-\d{2})\.csv$")
    removed = 0
    for fname in os.listdir(folder):
        m = pattern.match(fname)
        if not m:
            continue
        file_date = datetime.strptime(m.group(1), "%Y-%m-%d").date()
        if file_date < cutoff_date:
            os.remove(os.path.join(folder, fname))
            removed += 1
    if removed:
        print(f"已自動清理 {removed} 個超過 {keep_days} 天的舊 CSV 檔案。")


# ==========================================
# 4. 單股核心運算 (完整移植 Pine Script 邏輯)
# ==========================================
def analyze_stock(ticker: str, name: str, df: pd.DataFrame, bench_close: pd.Series) -> dict | None:
    if df is None or df.empty:
        return None

    df = df.dropna(subset=["Close", "High", "Low", "Volume"]).copy()
    if len(df) < 60:
        return None

    df.index = pd.to_datetime(df.index).tz_localize(None)
    aligned_bench = bench_close.reindex(df.index, method="ffill")

    w_close = df["Close"]
    w_high = df["High"]
    w_low = df["Low"]
    w_vol = df["Volume"]

    last_turnover = float(w_close.iloc[-1] * w_vol.iloc[-1])
    if last_turnover < CONFIG["min_weekly_turnover"]:
        return None

    w_ma30 = w_close.rolling(CONFIG["ma_len"]).mean()
    w_vol_ma = w_vol.rolling(CONFIG["vol_len"]).mean()
    w_ma_slope = (w_ma30 - w_ma30.shift(CONFIG["slope_lb"])) / w_ma30.shift(CONFIG["slope_lb"]) * 100
    w_atr = calc_atr(w_high, w_low, w_close, CONFIG["atr_len"])

    slope_thresh = CONFIG["slope_thresh"]
    is_ma_up = w_ma_slope > slope_thresh
    is_ma_down = w_ma_slope < -slope_thresh
    is_ma_flat = (~is_ma_up) & (~is_ma_down)

    rp = (w_close / aligned_bench) * 100
    rp_sma52 = rp.rolling(CONFIG["mrs_len"]).mean()
    mansfield_rs = ((rp / rp_sma52) - 1) * 100

    if CONFIG["req_mrs_up"]:
        is_mrs_bull = (mansfield_rs > 0) & (mansfield_rs > mansfield_rs.shift(1))
    else:
        is_mrs_bull = mansfield_rs > 0

    mrs_pass = pd.Series(True, index=df.index) if not CONFIG["use_mrs_flt"] else is_mrs_bull

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

    raw_s2_breakout = (stage_s == 2) & (stage_s.shift(1) != 2) & (w_vol > w_vol_ma * CONFIG["vol_mult"])
    valid_s2_breakout = raw_s2_breakout & mrs_pass

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
# 5. 主執行流程
# ==========================================
def main():
    hk_tz = timezone(timedelta(hours=8))
    now_hk = datetime.now(hk_tz)
    run_date = now_hk.strftime("%Y-%m-%d")
    os.makedirs("results", exist_ok=True)
    os.makedirs("docs", exist_ok=True)

    print(f"正在下載基準指數 {CONFIG['bench_sym']} 週線數據...")
    bench_df = yf.download(CONFIG["bench_sym"], period="3y", interval="1wk", progress=False, auto_adjust=True)
    if isinstance(bench_df.columns, pd.MultiIndex):
        bench_df.columns = bench_df.columns.get_level_values(0)
    bench_close = bench_df["Close"].dropna()
    bench_close.index = pd.to_datetime(bench_close.index).tz_localize(None)

    stock_map = fetch_hk_stock_list()
    tickers = list(stock_map.keys())
    batch_size = CONFIG["batch_size"]

    results = []
    total_batches = (len(tickers) + batch_size - 1) // batch_size

    for b in range(total_batches):
        batch = tickers[b * batch_size : (b + 1) * batch_size]
        print(f"掃描進度：第 {b + 1}/{total_batches} 批 ({len(batch)} 隻)...")
        try:
            data = yf.download(
                batch,
                period="3y",
                interval="1wk",
                group_by="ticker",
                threads=True,
                progress=False,
                auto_adjust=True,
            )
        except Exception as e:
            print(f"第 {b + 1} 批下載失敗：{e}")
            continue

        for t in batch:
            try:
                stock_df = data[t] if len(batch) > 1 else data
                res = analyze_stock(t, stock_map[t], stock_df, bench_close)
                if res:
                    results.append(res)
            except Exception:
                continue

        time.sleep(1)

    if not results:
        print("未獲取到任何有效股票數據。")
        return

    res_df = pd.DataFrame(results).sort_values(by=["Mansfield_RS"], ascending=False)

    # 輸出 CSV 檔案
    res_df.to_csv(f"results/hk_all_stages_{run_date}.csv", index=False, encoding="utf-8-sig")
    res_df.to_csv("results/latest_all.csv", index=False, encoding="utf-8-sig")

    sig_df = res_df[res_df["Signal"] != ""].copy()
    sig_df.to_csv(f"results/hk_signals_{run_date}.csv", index=False, encoding="utf-8-sig")
    sig_df.to_csv("results/latest_signals.csv", index=False, encoding="utf-8-sig")

    # 自動清理超過 30 天的舊歷史 CSV
    cleanup_old_csvs("results", CONFIG["keep_history_days"], now_hk)

    # 輸出 Markdown 報告
    s2_df = sig_df[sig_df["Signal"].str.contains("S2強勢突破")]
    w3_df = sig_df[sig_df["Signal"].str.contains("W3主升")]
    w5_df = sig_df[sig_df["Signal"].str.contains("W5背離")]

    md_lines = [
        f"# 全港股 Weinstein + Mansfield RS 掃描報告 ({run_date})",
        f"- 有效掃描股票數（週成交額 ≥ HK${CONFIG['min_weekly_turnover']:,}）：**{len(res_df)}** 隻",
        f"- **S2 強勢突破**：{len(s2_df)} 隻 | **W3 主升爆發**：{len(w3_df)} 隻 | **W5 過熱背離**：{len(w5_df)} 隻\n",
    ]

    def append_table(title: str, sub_df: pd.DataFrame):
        md_lines.append(f"## {title} ({len(sub_df)} 隻)")
        if sub_df.empty:
            md_lines.append("_本週無符合條件之股票_\n")
            return
        cols = ["Ticker", "Name", "Close", "Mansfield_RS", "MA30_Slope_%", "MA30_Ext_%", "1W_Base", "1W_Base_%", "1W_Low", "1W_High"]
        md_lines.append(sub_df[cols].head(30).to_markdown(index=False))
        md_lines.append("")

    append_table("S2 強勢突破 (RS+ 爆量)", s2_df)
    append_table("W3 主升浪爆發", w3_df)
    append_table("W5 末升段背離警示", w5_df)

    with open("results/LATEST_REPORT.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    # 輸出給 GitHub Pages 儀表板讀取的 JSON
    dashboard_data = {
        "updated_at": now_hk.strftime("%Y-%m-%d %H:%M HKT"),
        "min_turnover": CONFIG["min_weekly_turnover"],
        "total_scanned": len(res_df),
        "counts": {
            "s2_breakout": int(len(s2_df)),
            "w3_impulse": int(len(w3_df)),
            "w5_warn": int(len(w5_df)),
            "stage2_total": int((res_df["Stage"] == 2).sum()),
        },
        "stocks": res_df.to_dict(orient="records"),
    }
    with open("docs/data.json", "w", encoding="utf-8") as f:
        json.dump(dashboard_data, f, ensure_ascii=False)

    print("掃描完成！報告與網頁數據已更新。")


if __name__ == "__main__":
    main()
