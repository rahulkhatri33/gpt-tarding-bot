# ml_engine/trainer/relabel_triple_barrier.py

import os
import pandas as pd
from ml_engine.data_labeler.triple_barrier_labeler import (
    label_with_triple_barrier,
    label_with_triple_barrier_short
)
from core.config import get_triple_barrier_config

# === CONFIG ===
ENRICHED_DIR = "data/enriched_1h/"
LABELED_DIR = "data/labeled_1h/"
os.makedirs(LABELED_DIR, exist_ok=True)

# === Read thresholds from config ===
TB_CONFIG = get_triple_barrier_config()
TP_PCT = TB_CONFIG.get("tp_pct", 0.0325)
SL_PCT = TB_CONFIG.get("sl_pct", 0.015)
MAX_HOLD_BARS = TB_CONFIG.get("max_hold_bars", 4)

def label_all():
    files = sorted([f for f in os.listdir(ENRICHED_DIR) if f.endswith(".csv")])
    print(f"🔍 Found {len(files)} enriched files")

    total_long, total_short = 0, 0
    skipped = 0

    for fname in files:
        symbol = fname.replace("_1h_enriched.csv", "")
        path = os.path.join(ENRICHED_DIR, fname)

        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"❌ Failed to read {fname}: {e}")
            skipped += 1
            continue

        print(f"\n📈 Labeling {symbol}...")

        # --- LONG ---
        try:
            labeled_long = label_with_triple_barrier(
                df.copy(),
                tp_threshold=TP_PCT,
                sl_threshold=SL_PCT,
                max_hold_minutes=MAX_HOLD_BARS
            )
            if 'label' in labeled_long.columns:
                long_out = os.path.join(LABELED_DIR, f"{symbol}_labeled_long.csv")
                labeled_long.dropna(subset=["label"]).to_csv(long_out, index=False)
                count = labeled_long['label'].notna().sum()
                print(f"✅ Saved LONG → {long_out} | Rows: {count}")
                total_long += 1
            else:
                print(f"⚠️ Skipped LONG for {symbol} — no 'label' column.")
        except Exception as e:
            print(f"❌ Error labeling LONG for {symbol}: {e}")

        # --- SHORT ---
        try:
            labeled_short = label_with_triple_barrier_short(
                df.copy(),
                tp_threshold=TP_PCT,
                sl_threshold=SL_PCT,
                max_hold_minutes=MAX_HOLD_BARS
            )
            if 'label' in labeled_short.columns:
                short_out = os.path.join(LABELED_DIR, f"{symbol}_labeled_short.csv")
                labeled_short.dropna(subset=["label"]).to_csv(short_out, index=False)
                count = labeled_short['label'].notna().sum()
                print(f"✅ Saved SHORT → {short_out} | Rows: {count}")
                total_short += 1
            else:
                print(f"⚠️ Skipped SHORT for {symbol} — no 'label' column.")
        except Exception as e:
            print(f"❌ Error labeling SHORT for {symbol}: {e}")

    print("\n🏁 Labeling complete.")
    print(f"✅ Labeled: {total_long} long + {total_short} short")
    print(f"⚠️ Skipped due to errors: {skipped}")

if __name__ == "__main__":
    label_all()
