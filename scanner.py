import os
import io
import time
import requests
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime

# ==========================================
# 1. 參數設定 (對應 Pine Script Inputs)
# ==========================================
CONFIG = {
    "ma_len": 30,            # 均線週期 (週線基準)
    "slope_lb": 4,           # 均線斜率回看週數
    "vol_len": 20,           # 成交量均線週期
    "vol_mult": 1.5,         # 突破爆量倍數
    "slope_thresh": 0.3,     # 均線斜率門檻 (%)
    "bench_sym": "^HSI",     # 基準指數 (恒生指數)
    "mrs_len": 52,           # Mansfield RS 基準週數
    "use_mrs_flt": True,     # 啟用 Mansfield RS > 0 過濾
    "req_mrs_up": True,      # 要求 Mansfield RS 斜率向上
    "ao_fast": 5,            # AO 快線週期
    "ao_slow": 34,           # AO 慢線週期
    "ext_pct": 25.0,         # Wave 5 均線正乖離警示 (%)
    "atr_len": 14,           # 週線 ATR 週期
    "atr_mult": 1.0,         # 一週預期波幅倍數
    "batch_size": 100,       # 每次向 Yahoo Finance 請求的股票數量
    "min_weekly_vol": 500000 # 過濾過低流動性仙股 (可設為 0 關閉)
}


# ==========================================
# 2. 自動獲取全港上市股票名單 (HKEX 官方 Excel)
# ==========================================
def get_all_hk_tickers() -> list[str]:
    print("正在從港交所 (HKEX) 下載全港上市股票名單...")
    url = "https://www.hkex.com.hk/eng/services/trading/securities/securitieslists/ListOfSecurities.xlsx"
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()

    df = pd.read_excel(io.BytesIO(resp.content), header=2)
    equities = df[df["Category"].astype(str).str.strip() == "Equity"].copy()

    tickers = []
    for code in equities["Stock Code"]:
        try:
            code_int = int(code)
            ticker = f"{code_int:04d}.HK" if code_int < 10000 else f"{code_int}.HK"
            tickers.append(ticker)
        except (ValueError, TypeError):
            continue

    tickers = sorted(list(set(tickers)))
    print(f"成功取得 {len(tickers)} 隻香港上市股票。")
    return tickers


# ==========================================
# 3. 指標計算引擎 (完全還原 Pine Script v6)
# ==========================================
def calc_rma(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(alpha=1 / length, adjust=False).mean()


def analyze_stock(ticker: str, df: pd.DataFrame, bench_close: pd.Series, cfg: dict) -> dict | None:
    df = df.dropna(subset=["Close", "High", "Low", "Volume"]).copy()
    if len(df) < max(cfg["mrs_len"], cfg["ma_len"], cfg["ao_slow"]) + 10:
        return None

    df["Bench_Close"] = bench_close.reindex(df.index).ffill()
    df = df.dropna(subset=["Bench_Close"])
    if len(df) < 60:
        return None

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    vol = df["Volume"]

    if vol.iloc[-1] < cfg["min_weekly_vol"]:
        return None

    w_ma30 = close.rolling(cfg["ma_len"]).mean()
    w_vol_ma = vol.rolling(cfg["vol_len"]).mean()
    w_ma_slope = (w_ma30 - w_ma30.shift(cfg["slope_lb"])) / w_ma30.shift(cfg["slope_lb"]) * 100

    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    w_atr = calc_rma(tr, cfg["atr_len"])

    is_ma_up = w_ma_slope > cfg["slope_thresh"]
    is_ma_down = w_ma_slope < -cfg["slope_thresh"]
    is_ma_flat = (~is_ma_up) & (~is_ma_down)

    rp = (close / df["Bench_Close"]) * 100
    rp_sma52 = rp.rolling(cfg["mrs_len"]).mean()
    mansfield_rs = ((rp / rp_sma52) - 1) * 100

    is_mrs_bull = (mansfield_rs > 0) & (
        (not cfg["req_mrs_up"]) | (mansfield_rs > mansfield_rs.shift(1))
    )
    mrs_pass = (not cfg["use_mrs_flt"]) | is_mrs_bull

    stages = np.ones(len(df), dtype=int)
    c_arr = close.values
    ma_arr = w_ma30.values
    up_arr = is_ma_up.values
    dn_arr = is_ma_down.values
    fl_arr = is_ma_flat.values
    sl_arr = w_ma_slope.values

    for i in range(1, len(df)):
        if np.isnan(ma_arr[i]) or np.isnan(sl_arr[i]):
            stages[i] = stages[i - 1]
            continue

        prev_s = stages[i - 1]
        if c_arr[i] > ma_arr[i] and up_arr[i]:
            stages[i] = 2
        elif c_arr[i] < ma_arr[i] and dn_arr[i]:
            stages[i] = 4
        elif c_arr[i] <= ma_arr[i] and (fl_arr[i] or sl_arr[i] > -cfg["slope_thresh"]) and prev_s == 4:
            stages[i] = 1
        elif c_arr[i] >= ma_arr[i] and fl_arr[i] and prev_s == 2:
            stages[i] = 3
        else:
            if c_arr[i] > ma_arr[i]:
                stages[i] = 3 if dn_arr[i] else 2
            else:
                stages[i] = 2 if up_arr[i] else 4

    stage_s = pd.Series(stages, index=df.index)

    raw_s2_breakout = (stage_s == 2) & (stage_s.shift(1) != 2) & (vol > w_vol_ma * cfg["vol_mult"])
    valid_s2_breakout = raw_s2_breakout & mrs_pass
    weak_s2_breakout = raw_s2_breakout & (~mrs_pass)

    hl2 = (high + low) / 2
    ao = hl2.rolling(cfg["ao_fast"]).mean() - hl2.rolling(cfg["ao_slow"]).mean()
    ao_highest_20 = ao.rolling(20).max()
    price_highest_20 = high.rolling(20).max()

    ao_rising = (ao > ao.shift(1)) & (ao > 0)
    ao_is_highest = ao == ao_highest_20
    price_is_highest = high == price_highest_20
    ma_extension = (close - w_ma30) / w_ma30 * 100

    is_wave3 = (stage_s == 2) & is_mrs_bull & ao_rising & ao_is_highest & (vol > w_vol_ma)
    is_wave5 = (stage_s == 2) & price_is_highest & (ao < ao_highest_20) & (ma_extension > cfg["ext_pct"])

    w3_trigger = is_wave3 & (~is_wave3.shift(1).fillna(False))
    w5_trigger = is_wave5 & (~is_wave5.shift(1).fillna(False))

    idx = -1
    cur_close = float(close.iloc[idx])
    cur_ma30 = float(w_ma30.iloc[idx])
    cur_slope = float(w_ma_slope.iloc[idx])
    cur_mrs = float(mansfield_rs.iloc[idx])
    cur_ext = float(ma_extension.iloc[idx])
    cur_atr = float(w_atr.iloc[idx])
    cur_stage = int(stage_s.iloc[idx])
    cur_w3 = bool(is_wave3.iloc[idx])
    cur_w5 = bool(is_wave5.iloc[idx])
    cur_mrs_bull = bool(is_mrs_bull.iloc[idx])

    if np.isnan(cur_ma30) or np.isnan(cur_mrs) or np.isnan(cur_atr):
        return None

    weekly_drift_pct = cur_slope / cfg["slope_lb"]
    if cur_stage == 2:
        drift_weight = 1.5 if cur_w3 else (-0.3 if cur_w5 else 1.0)
    elif cur_stage == 4:
        drift_weight = 1.2
    else:
        drift_weight = 0.2

    fcst_base_price = cur_close * (1 + (weekly_drift_pct * drift_weight / 100))
    if cur_stage == 2 and cur_mrs_bull and not cur_w5:
        fcst_base_adj = fcst_base_price + (cur_atr * 0.25)
    elif cur_stage == 4:
        fcst_base_adj = fcst_base_price - (cur_atr * 0.25)
    else:
        fcst_base_adj = fcst_base_price

    fcst_bull_high = fcst_base_adj + (cur_atr * cfg["atr_mult"] * (0.85 if cur_stage == 2 else 0.6))
    fcst_bear_low = fcst_base_adj - (cur_atr * cfg["atr_mult"] * (0.85 if cur_stage == 4 else 0.6))

    pct_base = (fcst_base_adj - cur_close) / cur_close * 100

    signals = []
    if bool(valid_s2_breakout.iloc[idx]):
        signals.append("S2強勢突破")
    elif bool(weak_s2_breakout.iloc[idx]):
        signals.append("RS弱勢突破")
    if bool(w3_trigger.iloc[idx]):
        signals.append("W3主升爆發")
    if bool(w5_trigger.iloc[idx]):
        signals.append("W5背離警示")

    stage_map = {1: "Stage 1 (築底)", 2: "Stage 2 (主升)", 3: "Stage 3 (頭部)", 4: "Stage 4 (下跌)"}

    return {
        "Ticker": ticker,
        "Close": round(cur_close, 3),
        "Stage": stage_map[cur_stage],
        "Signals": " | ".join(signals) if signals else "-",
        "Mansfield_RS": round(cur_mrs, 2),
        "30W_MA": round(cur_ma30, 3),
        "30W_Slope_%": round(cur_slope, 2),
        "MA_Ext_%": round(cur_ext, 2),
        "1W_Base": round(fcst_base_adj, 3),
        "1W_Base_%": round(pct_base, 2),
        "1W_Low": round(fcst_bear_low, 3),
        "1W_High": round(fcst_bull_high, 3),
        "Weekly_Vol": int(vol.iloc[idx])
    }


# ==========================================
# 4. 主程式：批量掃描全港股並輸出結果
# ==========================================
def main():
    tickers = get_all_hk_tickers()

    print(f"下載基準指數 ({CONFIG['bench_sym']}) 週線數據...")
    bench_df = yf.download(CONFIG["bench_sym"], period="3y", interval="1wk", progress=False)
    if isinstance(bench_df.columns, pd.MultiIndex):
        bench_close = bench_df["Close"].iloc[:, 0]
    else:
        bench_close = bench_df["Close"]

    results = []
    batch_size = CONFIG["batch_size"]

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i + batch_size]
        print(f"正在掃描進度: {i + 1} ~ {min(i + batch_size, len(tickers))} / {len(tickers)}...")
        try:
            data = yf.download(
                batch,
                period="3y",
                interval="1wk",
                group_by="ticker",
                threads=True,
                progress=False
            )
            for t in batch:
                try:
                    df_t = data[t] if len(batch) > 1 else data
                    if df_t.empty:
                        continue
                    res = analyze_stock(t, df_t, bench_close, CONFIG)
                    if res:
                        results.append(res)
                except Exception:
                    continue
        except Exception as e:
            print(f"批次下載異常: {e}")
        time.sleep(1)

    if not results:
        print("未找到符合條件的數據。")
        return

    res_df = pd.DataFrame(results)
    os.makedirs("results", exist_ok=True)
    today_str = datetime.now().strftime("%Y-%m-%d")

    full_path = f"results/hk_all_stocks_{today_str}.csv"
    res_df.to_csv(full_path, index=False, encoding="utf-8-sig")

    focus_df = res_df[res_df["Signals"] != "-"].sort_values(by="Mansfield_RS", ascending=False)
    focus_path = f"results/hk_signals_{today_str}.csv"
    focus_df.to_csv(focus_path, index=False, encoding="utf-8-sig")

    print(f"\n掃描完成！共分析 {len(res_df)} 隻有效港股，其中 {len(focus_df)} 隻觸發關鍵訊號。")
    print(f"完整結果: {full_path}")
    print(f"焦點訊號: {focus_path}")


if __name__ == "__main__":
    main()
