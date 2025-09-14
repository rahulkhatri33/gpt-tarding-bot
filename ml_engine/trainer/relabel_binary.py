# ml_engine/trainer/relabel_binary.py

import os
import pandas as pd

LABELED_DIR = "data/labeled_1h/"
BINARY_LABELED_DIR = "data/labeled_1h_binary/"

os.makedirs(BINARY_LABELED_DIR, exist_ok=True)

files = [f for f in os.listdir(LABELED_DIR) if f.endswith(".csv")]
print(f"🔍 Found {len(files)} labeled files")

for fname in files:
    path = os.path.join(LABELED_DIR, fname)
    df = pd.read_csv(path)

    if "label" not in df.columns:
        print(f"⚠️ Skipping {fname} — missing 'label' column")
        continue

    df["binary_label"] = df["label"].apply(lambda x: 1 if x == 2 else 0)
    out_path = os.path.join(BINARY_LABELED_DIR, fname)
    df.to_csv(out_path, index=False)
    print(f"✅ Saved binary-labeled file → {out_path}")

print("\n🏁 Binary relabeling complete.")
