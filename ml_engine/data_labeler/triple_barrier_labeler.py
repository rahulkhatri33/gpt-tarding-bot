# ml_engine/data_labeler/triple_barrier_labeler.py

import pandas as pd
from core.config import get_triple_barrier_config

# === Pull shared thresholds directly from config ===
TB_CONFIG = get_triple_barrier_config()
TP_PCT = TB_CONFIG.get("tp_pct", 0.0325)
SL_PCT = TB_CONFIG.get("sl_pct", 0.015)
MAX_HOLD_BARS = TB_CONFIG.get("max_hold_bars", 4)

def label_with_triple_barrier(df: pd.DataFrame) -> pd.DataFrame:
    """
    Long-side triple barrier labeling (strict order):
    - If SL hits first → label 0
    - If TP hits first → label 2
    - If neither hit → label 1 (TIME)

    Returns 'label': 0 = SL, 1 = TIME, 2 = TP
    """
    df = df.copy().reset_index(drop=True)
    labels = []

    for i in range(len(df) - MAX_HOLD_BARS):
        entry_price = df.loc[i, "close"]
        tp_price = entry_price * (1 + TP_PCT)
        sl_price = entry_price * (1 - SL_PCT)
        window = df.loc[i + 1:i + MAX_HOLD_BARS]

        label = 1  # default TIME

        for _, row in window.iterrows():
            if row["low"] <= sl_price:
                label = 0  # SL hit first
                break
            if row["high"] >= tp_price:
                label = 2  # TP hit first
                break

        labels.append(label)

    labels += [None] * MAX_HOLD_BARS
    df["label"] = labels
    return df


def label_with_triple_barrier_short(df: pd.DataFrame) -> pd.DataFrame:
    """
    Short-side triple barrier labeling (strict order):
    - If SL hits first → label 0
    - If TP hits first → label 2
    - If neither hit → label 1 (TIME)

    Returns 'label': 0 = SL, 1 = TIME, 2 = TP
    """
    df = df.copy().reset_index(drop=True)
    labels = []

    for i in range(len(df) - MAX_HOLD_BARS):
        entry_price = df.loc[i, "close"]
        tp_price = entry_price * (1 - TP_PCT)
        sl_price = entry_price * (1 + SL_PCT)
        window = df.loc[i + 1:i + MAX_HOLD_BARS]

        label = 1  # default TIME

        for _, row in window.iterrows():
            if row["high"] >= sl_price:
                label = 0  # SL hit first
                break
            if row["low"] <= tp_price:
                label = 2  # TP hit first
                break

        labels.append(label)

    labels += [None] * MAX_HOLD_BARS
    df["label"] = labels
    return df
