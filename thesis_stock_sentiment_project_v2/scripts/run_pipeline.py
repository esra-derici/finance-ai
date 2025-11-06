
import argparse
import pandas as pd
from pathlib import Path
from thesisproj.config import Config
from thesisproj.data import load_prices, load_news
from thesisproj.features import add_market_features, daily_news_agg, add_lagged_sentiment
from thesisproj.split import chrono_split
from thesisproj.models import build_feature_target, naive_yesterday, RFModel
from thesisproj.evaluate import summarize
from thesisproj.plot import plot_price_vs_pred

def run_for_ticker(cfg: Config, ticker: str) -> pd.DataFrame:
    prices = load_prices(cfg.data_dir, ticker)
    news = load_news(cfg.data_dir)

    market = add_market_features(prices)
    sent_daily = daily_news_agg(news, ticker)
    feat = add_lagged_sentiment(market, sent_daily, max_lag=5)

    X, y, cols = build_feature_target(feat)
    dates = feat["date"].iloc[1:].reset_index(drop=True)
    prev_close = feat["close"].iloc[:-1].reset_index(drop=True)

    df_xy = pd.DataFrame(X, columns=cols)
    df_xy["date"] = dates
    df_xy["close_prev"] = prev_close
    df_xy["y"] = y

    train, test = chrono_split(df_xy, cfg.test_days)
    Xtr, ytr = train[cols].values, train["y"].values
    Xte, yte = test[cols].values, test["y"].values
    prev_test = test["close_prev"].values

    naive_pred = naive_yesterday(prev_test)

    model = RFModel(n_estimators=cfg.n_estimators, random_state=cfg.random_state)
    model.fit(Xtr, ytr)
    yhat = model.predict(Xte)

    base_m = summarize(yte, naive_pred, prev_test)
    rf_m = summarize(yte, yhat, prev_test)

    Path(cfg.results_dir).mkdir(parents=True, exist_ok=True)
    pred_df = pd.DataFrame({"date": test["date"].values, "y_true": yte, "y_pred": yhat, "y_naive": naive_pred})
    pred_df.to_csv(Path(cfg.results_dir)/f"predictions_{ticker}.csv", index=False)
    plot_price_vs_pred(pred_df[["date","y_true","y_pred"]], str(Path(cfg.results_dir)/"figures"/f"price_vs_pred_{ticker}.png"),
                       f"{ticker} True vs Pred")

    return pd.DataFrame([{"ticker": ticker, "model":"NaiveYesterday", **base_m},
                         {"ticker": ticker, "model":"RandomForest", **rf_m}])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="THYAO")
    args = parser.parse_args()
    cfg = Config()
    tickers = cfg.tickers if args.ticker.upper()=="ALL" else [args.ticker]
    metrics = pd.concat([run_for_ticker(cfg, t) for t in tickers], ignore_index=True)
    metrics.to_csv(Path(cfg.results_dir)/"metrics.csv", index=False)
    print("\n=== Metrics ===")
    print(metrics.to_string(index=False))

if __name__ == "__main__":
    main()
