# src/thesisproj/features.py
import numpy as np
import pandas as pd

def add_market_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # güvenlik: close kesin numeric olsun
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    out["ret1"] = out["close"].pct_change()
    out["logret1"] = np.log(out["close"]).diff()
    out["ret5"] = out["close"].pct_change(5)
    out["sma5"] = out["close"].rolling(5).mean()
    out["sma10"] = out["close"].rolling(10).mean()
    out["vol10"] = out["close"].pct_change().rolling(10).std()
    out["dow"] = out["date"].dt.dayofweek
    return out.fillna(method="bfill").fillna(method="ffill")

def daily_news_agg(news: pd.DataFrame, ticker: str) -> pd.DataFrame:
    df = news[news["ticker"] == ticker].copy()
    if df.empty:
        # boşsa kolon tipleri korunmuş bir DataFrame döndür
        return pd.DataFrame({
            "date": pd.Series([], dtype="datetime64[ns]"),
            "sent_mean": pd.Series([], dtype="float"),
            "news_cnt": pd.Series([], dtype="float"),
        })
    g = df.groupby("date").agg(
        sent_mean=("sentiment", "mean"),
        news_cnt=("sentiment", "size"),
    ).reset_index()
    return g

def add_lagged_sentiment(market: pd.DataFrame, daily_sent: pd.DataFrame, max_lag: int = 5) -> pd.DataFrame:
    df = market.merge(daily_sent, on="date", how="left").sort_values("date")
    df[["sent_mean", "news_cnt"]] = df[["sent_mean", "news_cnt"]].fillna(0.0)
    for k in range(1, max_lag + 1):
        df[f"sent_mean_lag{k}"] = df["sent_mean"].shift(k).fillna(0.0)
        df[f"news_cnt_lag{k}"] = df["news_cnt"].shift(k).fillna(0.0)
    return df
