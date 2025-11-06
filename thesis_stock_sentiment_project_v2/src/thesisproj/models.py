
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
def build_feature_target(df: pd.DataFrame):
    df = df.sort_values("date").reset_index(drop=True)
    y = df["close"].shift(-1).values
    Xdf = df.drop(columns=["close"]).select_dtypes(include=["number"])
    X = Xdf.values; cols = list(Xdf.columns)
    X, y = X[:-1], y[:-1]
    return X, y, cols
def naive_yesterday(prev_close: np.ndarray):
    return prev_close
class RFModel:
    def __init__(self, n_estimators=400, random_state=42):
        self.model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
        self.scaler = StandardScaler()
    def fit(self, Xtr, ytr):
        Xs = self.scaler.fit_transform(Xtr)
        self.model.fit(Xs, ytr)
    def predict(self, X):
        Xs = self.scaler.transform(X)
        return self.model.predict(Xs)
