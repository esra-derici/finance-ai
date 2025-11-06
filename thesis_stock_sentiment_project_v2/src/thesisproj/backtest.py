
from __future__ import annotations
import numpy as np, pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Callable, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

try:
    import xgboost as xgb
    HAVE_XGB = True
except Exception:
    HAVE_XGB = False

@dataclass
class ModelSpec:
    name: str
    build: Callable[[], object]  # returns an estimator with fit/predict

def get_model_registry(random_state: int = 42) -> Dict[str, ModelSpec]:
    reg: Dict[str, ModelSpec] = {
        "RF": ModelSpec("RF", lambda: RandomForestRegressor(n_estimators=400, random_state=random_state, n_jobs=-1)),
        "GBM": ModelSpec("GBM", lambda: GradientBoostingRegressor(random_state=random_state)),
    }
    if HAVE_XGB:
        reg["XGB"] = ModelSpec("XGB", lambda: xgb.XGBRegressor(random_state=random_state, n_estimators=400, max_depth=4, subsample=0.9, colsample_bytree=0.9, tree_method="hist"))
    return reg

def chrono_splits(n_rows: int, n_splits: int, test_size: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    # expanding window walk-forward
    indices = np.arange(n_rows)
    splits = []
    train_end_min = n_rows - n_splits*test_size
    if train_end_min < 20:
        train_end_min = 20
    for i in range(n_splits):
        test_start = n_rows - (n_splits - i)*test_size
        test_end = test_start + test_size
        train_idx = indices[:test_start]
        test_idx = indices[test_start:test_end]
        splits.append((train_idx, test_idx))
    return splits

def fit_predict(estimator, Xtr, ytr, Xte):
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xte_s = scaler.transform(Xte)
    estimator.fit(Xtr_s, ytr)
    pred = estimator.predict(Xte_s)
    return pred

def directional_accuracy(y_prev, y, yhat):
    dy_true = y - y_prev; dy_pred = yhat - y_prev
    return float(np.mean(np.sign(dy_true) == np.sign(dy_pred))) * 100.0

def metrics(y, yhat, yprev) -> Dict[str, float]:
    mae  = float(np.mean(np.abs(y-yhat)))
    rmse = float(np.sqrt(np.mean((y-yhat)**2)))
    mape = float(np.mean(np.abs((y-yhat)/np.maximum(1e-9, np.abs(y))))) * 100.0
    ss_res = float(np.sum((y-yhat)**2)); ss_tot = float(np.sum((y - np.mean(y))**2) + 1e-12)
    r2 = 1 - ss_res/ss_tot
    da = directional_accuracy(yprev, y, yhat)
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2, "DA(%)": da}

def walk_forward_eval(df_feat: pd.DataFrame, feature_cols: List[str], y_col: str, prev_col: str,
                      models: List[str], n_splits: int = 5, test_size: int = 20, random_state: int = 42) -> pd.DataFrame:
    X = df_feat[feature_cols].values
    y = df_feat[y_col].values
    yprev = df_feat[prev_col].values
    splits = chrono_splits(len(df_feat), n_splits=n_splits, test_size=test_size)
    registry = get_model_registry(random_state=random_state)

    records = []
    for mname in models:
        est = registry[mname].build()
        all_fold = []
        for fold, (tr, te) in enumerate(splits, 1):
            yhat = fit_predict(est, X[tr], y[tr], X[te])
            rec = metrics(y[te], yhat, yprev[te])
            rec.update({"model": mname, "fold": fold})
            all_fold.append(rec)
        records.extend(all_fold)

    out = pd.DataFrame(records)
    # add group stats (mean±std)
    agg = out.groupby("model").agg(["mean","std"]).reset_index()
    # flatten columns
    agg.columns = ["model"] + [f"{m}_{stat}" for m,stat in agg.columns.tolist()[1:]]
    return out, agg
