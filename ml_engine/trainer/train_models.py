# ml_engine/trainer/train_models.py

import os
import pandas as pd
import xgboost as xgb
import joblib
from sklearn.metrics import classification_report

# === CONFIG ===
PRUNED_DIR = "data/pruned_1h_binary/"
MODEL_DIR = "ml_models_binary/"
CONF_LOG_DIR = "confidence_logs_binary/"
os.makedirs(CONF_LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

TP_WEIGHT = 8.0

CONFIG = {
    "base_pairs": [
  "ADAUSDT", "APEUSDT", "ARBUSDT", "BNBUSDT", "BTCUSDT", "DOGEUSDT",
  "ETHUSDT", "FILUSDT", "HBARUSDT", "OPUSDT", "QNTUSDT", "SOLUSDT",
  "THETAUSDT", "UNIUSDT", "VETUSDT", "XRPUSDT"
]
}

def train_binary_xgb(symbol, direction):
    fname = f"{symbol}_labeled_{direction}.csv"
    path = os.path.join(PRUNED_DIR, fname)
    if not os.path.exists(path):
        return

    df = pd.read_csv(path)

    if "binary_label" not in df.columns:
        print(f"⚠️ Missing binary_label in {fname}")
        return

    df = df.dropna()
    if df.empty:
        print(f"⚠️ Empty data after dropping NA: {fname}")
        return

    X = df.drop(columns=["binary_label", "timestamp"], errors="ignore")
    y = df["binary_label"]

    if y.nunique() < 2:
        print(f"⚠️ Only one class in {fname}, skipping.")
        return

    sample_weights = [TP_WEIGHT if val == 1 else 1.0 for val in y]

    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.01,
        subsample=0.7,
        colsample_bytree=0.7,
        eval_metric="logloss",
        use_label_encoder=False
    )

    model.fit(X, y, sample_weight=sample_weights)
    preds = model.predict(X)
    probas = model.predict_proba(X)
    avg_conf = probas[:, 1].mean()

    print(f"✅ Binary XGB for {symbol} | {direction.upper()} | Samples={len(X)}")
    print(classification_report(y, preds, target_names=["Not TP", "TP"]))
    print(f"📈 Average TP Confidence: {avg_conf:.4f}")
    print("-" * 60)

    out_dir = os.path.join(MODEL_DIR, symbol)
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(model, os.path.join(out_dir, f"{symbol}_xgb_{direction}.pkl"))

    conf_df = pd.DataFrame({
        "true_label": y.values,
        "confidence": probas[:, 1]
    })
    conf_df.to_csv(os.path.join(CONF_LOG_DIR, f"{symbol}_{direction}_conf.csv"), index=False)

if __name__ == "__main__":
    for direction in ["long", "short"]:
        for symbol in CONFIG["base_pairs"]:
            train_binary_xgb(symbol, direction)
