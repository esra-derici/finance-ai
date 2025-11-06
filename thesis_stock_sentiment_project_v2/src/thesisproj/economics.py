
from __future__ import annotations
import numpy as np, pandas as pd

def simple_long_strategy(df: pd.DataFrame, cost_bps: float = 5.0, thresh: float = 0.0) -> dict:
    # df needs columns: date, y_true, y_pred, y_prev
    # Signal = 1 if predicted next-day change > thresh, else 0 (flat)
    pred_ret = (df["y_pred"] - df["y_prev"]) / np.maximum(1e-9, df["y_prev"])
    true_ret = (df["y_true"] - df["y_prev"]) / np.maximum(1e-9, df["y_prev"])

    signal = (pred_ret > thresh).astype(int)
    # trades when signal changes from 0->1 or 1->0 (assume close-to-close)
    trades = signal.diff().fillna(0).abs()
    cost = trades * (cost_bps/10000.0)

    strat_ret = signal * true_ret - cost
    cum = (1 + strat_ret).cumprod()
    ann_ret = (cum.iloc[-1] ** (252/len(cum))) - 1 if len(cum) > 0 else 0.0
    sharpe = np.sqrt(252) * strat_ret.mean() / (strat_ret.std() + 1e-12)

    return {"AnnReturn": float(ann_ret), "Sharpe": float(sharpe),
            "AvgDailyRet": float(strat_ret.mean()), "Vol": float(strat_ret.std()),
            "Turnover": float(trades.mean())}
