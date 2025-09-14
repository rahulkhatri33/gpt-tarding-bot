# ml_engine/trainer/prune_binary.py

import os
import pandas as pd
import json

# === CONFIG ===
LABELED_DIR = "data/labeled_1h_binary/"
PRUNED_DIR = "data/pruned_1h_binary/"
TOP_FEATURES_PATH = os.path.abspath("config/top_features.json")

os.makedirs(PRUNED_DIR, exist_ok=True)

# === LOAD CONFIG ===
with open(TOP_FEATURES_PATH, "r") as f:
    top_features = json.load(f)

for fname in os.listdir(LABELED_DIR):
    if not fname.endswith(".csv"):
        continue

    path = os.path.join(LABELED_DIR, fname)
    df = pd.read_csv(path)

    symbol = fname.split("_")[0]
    direction = fname.split("_")[-1].replace(".csv", "")
    key = f"{symbol}_{direction}"

    if key not in top_features:
        print(f"⚠️ Skipping {fname} — no features found in top_features.json")
        continue

    selected_features = top_features[key]
    required_columns = selected_features + ["timestamp", "binary_label"]

    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        print(f"❌ Missing columns in {fname}: {missing}")
        continue

    pruned_df = df[required_columns].copy()
    output_path = os.path.join(PRUNED_DIR, fname)
    pruned_df.to_csv(output_path, index=False)
    print(f"✅ Pruned binary {fname} → {output_path}")

print("\n🏁 Binary pruning complete.")
