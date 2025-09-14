# scalper/backtest_runner.py

import sys, os
import pandas as pd
import json
import time
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scalper_strategy import calculate_ut_signals, _passes_min_body_filter, _calculate_sl_tp
from core.logger import global_logger as logger

# Silence logger
logger.logger.setLevel(logging.ERROR)
logger.log_debug = lambda *a, **k: None
logger.log_info = lambda *a, **k: None
logger.log_warning = lambda *a, **k: None

# === Load config ===
CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "config.json")
with open(CONFIG_PATH, "r") as f:
    CONFIG = json.load(f)

settings = CONFIG["scalper_settings"]

filters = settings.get("filters", {})
use_trend_filter = filters.get("use_trend_filter", False)
use_time_filter = filters.get("use_time_filter", False)
use_min_body = filters.get("use_min_body", True)

# === Symbol list & CSV paths ===
# ⚠️ Adjust paths to where your CSVs are stored
SYMBOLS = {
    "BTCUSDT": r"C:\Users\rahul\Downloads\csv data\btcusdt_5m_july_august_2025.csv",
    "ETHUSDT": r"C:\Users\rahul\Downloads\csv data\ethusdt_5m_july_august_2025.csv",
    "SOLUSDT": r"C:\Users\rahul\Downloads\csv data\solusdt_5m_july_august_2025.csv",
    "BNBUSDT": r"C:\Users\rahul\Downloads\csv data\bnbusdt_5m_july_august_2025.csv"
}

# === Function to run backtest for one symbol ===
def run_backtest(symbol: str, csv_path: str, settings: dict):
    try:
        df = pd.read_csv(csv_path)

        df.columns = [c.lower() for c in df.columns]
        df = df.rename(columns={"open_time": "time"})
        df = df[["time", "open", "high", "low", "close", "volume"]]

        df["time"] = pd.to_numeric(df["time"], errors="coerce")
        df["time"] = pd.to_datetime(df["time"], unit="ms", errors="coerce")
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna().reset_index(drop=True)

        # === Precompute signals ===
        df = calculate_ut_signals(df, settings)
        if use_trend_filter and settings.get("ema_filter_period", 0) > 0:
            df["ema"] = df["close"].ewm(span=settings["ema_filter_period"]).mean()

        warmup = settings.get("min_candles", 1000)
        trades = []
        open_trade = None

        start_time = time.time()

        for i in range(warmup, len(df)):
            row = df.iloc[i]
            price = row["close"]

            # === Filters ===
            if use_time_filter:
                start_hour, end_hour = settings.get("allowed_trading_hours", [0, 24])
                if not (start_hour <= row["time"].hour < end_hour):
                    continue

            if use_min_body:
                if not _passes_min_body_filter(df.iloc[: i + 1], settings):
                    continue

            if use_trend_filter:
                if row["close"] < row["ema"] and row["ut_buy_signal"] == 1.0:
                    continue
                if row["close"] > row["ema"] and row["ut_sell_signal"] == 1.0:
                    continue

            # === Signal check ===
            side, sltp = None, None
            if row["ut_buy_signal"] == 1.0:
                side, sltp = "LONG", _calculate_sl_tp(df.iloc[: i + 1], settings, "LONG", price)
            elif row["ut_sell_signal"] == 1.0:
                side, sltp = "SHORT", _calculate_sl_tp(df.iloc[: i + 1], settings, "SHORT", price)

            # === Entry ===
            if side and not open_trade:
                open_trade = {
                    "side": side,
                    "entry_time": row["time"],
                    "entry_price": price,
                    "sltp": sltp,
                    "partial_taken": False
                }

            # === Manage Open Trade ===
            elif open_trade:
                sl, tp = open_trade["sltp"].sl, open_trade["sltp"].tp

                # --- Partial TP (50% at 1:1 RR) ---
                if not open_trade["partial_taken"] and sl != open_trade["entry_price"]:
                    if open_trade["side"] == "LONG":
                        target_partial = open_trade["entry_price"] + (open_trade["entry_price"] - sl)
                        if price >= target_partial:
                            trades.append({
                                **open_trade,
                                "exit_time": row["time"],
                                "exit_price": price,
                                "result": "PARTIAL_50",
                                "pnl_pct": (price - open_trade["entry_price"]) / open_trade["entry_price"] * 100 * 0.5
                            })
                            open_trade["sltp"].sl = open_trade["entry_price"]  # Move SL to BE
                            open_trade["partial_taken"] = True
                    else:
                        target_partial = open_trade["entry_price"] - (sl - open_trade["entry_price"])
                        if price <= target_partial:
                            trades.append({
                                **open_trade,
                                "exit_time": row["time"],
                                "exit_price": price,
                                "result": "PARTIAL_50",
                                "pnl_pct": (open_trade["entry_price"] - price) / open_trade["entry_price"] * 100 * 0.5
                            })
                            open_trade["sltp"].sl = open_trade["entry_price"]  # Move SL to BE
                            open_trade["partial_taken"] = True

                # --- Final Exit ---
                sl, tp = open_trade["sltp"].sl, open_trade["sltp"].tp
                if open_trade["side"] == "LONG":
                    if price <= sl:
                        trades.append({**open_trade, "exit_time": row["time"], "exit_price": sl, "result": "LOSS"})
                        open_trade = None
                    elif price >= tp:
                        trades.append({**open_trade, "exit_time": row["time"], "exit_price": tp, "result": "WIN"})
                        open_trade = None
                else:
                    if price >= sl:
                        trades.append({**open_trade, "exit_time": row["time"], "exit_price": sl, "result": "LOSS"})
                        open_trade = None
                    elif price <= tp:
                        trades.append({**open_trade, "exit_time": row["time"], "exit_price": tp, "result": "WIN"})
                        open_trade = None

        elapsed = time.time() - start_time

        # === Results ===
        results = pd.DataFrame(trades)
        if not results.empty:
            if "pnl_pct" not in results.columns:
                results["pnl_pct"] = (
                    (results["exit_price"] - results["entry_price"]) / results["entry_price"]
                    * results["side"].map({"LONG": 1, "SHORT": -1}) * 100
                )
            results["equity_curve"] = results["pnl_pct"].cumsum()

            print(f"\n===== Backtest Summary ({symbol} 5m | Jul–Aug 2025) =====")
            print(f"Trades       : {len(results)}")
            print(f"Wins / Losses: {sum(results['result'].str.contains('WIN'))} / {sum(results['result'].str.contains('LOSS'))}")
            print(f"Win rate     : {sum(results['result'].str.contains('WIN'))/len(results)*100:.2f}%")
            print(f"Avg PnL      : {results['pnl_pct'].mean():.3f}%")
            print(f"Median PnL   : {results['pnl_pct'].median():.3f}%")
            print(f"Cumulative   : {results['pnl_pct'].sum():.2f}%")
            print(f"Max Drawdown : {results['equity_curve'].min():.2f}%")
            print(f"⏱️ Runtime   : {elapsed:.2f} sec for {len(df)} candles")

            save_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), f"{symbol.lower()}_scalper_backtest.csv")
            results.to_csv(save_path, index=False)
            print(f"Trades saved -> {save_path}")
        else:
            print(f"\n===== Backtest Summary ({symbol}) =====")
            print("No trades generated.")

    except Exception as e:
        print(f"❌ Error backtesting {symbol}: {e}")

# === Run for all symbols ===
for sym, path in SYMBOLS.items():
    if os.path.exists(path):
        run_backtest(sym, path, settings)
    else:
        print(f"⚠️ Skipping {sym}, CSV not found at {path}")
