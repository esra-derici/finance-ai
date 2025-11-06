
import argparse
import pandas as pd
from pathlib import Path
# EN ÜSTTE (diğer importlardan önce) güvenli yol:
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Sonra hepsi tutarlı olsun:
from thesisproj.config import Config
from thesisproj.data import load_prices, load_news
from thesisproj.features import add_market_features, daily_news_agg, add_lagged_sentiment
from thesisproj.models import build_feature_target, RFModel
from thesisproj.backtest import walk_forward_eval
from thesisproj.evaluate import summarize
from thesisproj.plot import plot_price_vs_pred
from thesisproj.economics import simple_long_strategy
from thesisproj.export import save_table, ensure_dir
from thesisproj.split import chrono_split  # << önceki satırdaki "src.thesisproj" karışıklığını kaldırdık


def prepare_features(cfg: Config, ticker: str, use_sentiment: bool, max_lag: int) -> pd.DataFrame:
    prices = load_prices(cfg.data_dir, ticker)
    news = load_news(cfg.data_dir)
    mkt = add_market_features(prices)
    if use_sentiment:
        sent = daily_news_agg(news, ticker)
        feat = add_lagged_sentiment(mkt, sent, max_lag=max_lag)
    else:
        feat = mkt.copy()
        feat["sent_mean"] = 0.0
        feat["news_cnt"] = 0.0
    X, y, cols = build_feature_target(feat)
    df = pd.DataFrame(X, columns=cols)
    df["date"] = feat["date"].iloc[1:].reset_index(drop=True)
    df["y"] = y
    df["y_prev"] = feat["close"].iloc[:-1].reset_index(drop=True)
    return df, cols

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="THYAO")
    ap.add_argument("--models", default="RF,GBM", help="comma list among RF,GBM,XGB")
    ap.add_argument("--walk_splits", type=int, default=5)
    ap.add_argument("--walk_test", type=int, default=20)
    ap.add_argument("--use_sentiment", default="yes")
    ap.add_argument("--lags", default="5")
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--cost_bps", type=float, default=5.0)
    args = ap.parse_args()

    cfg = Config()
    results = Path(cfg.results_dir)
    paper_dir = ensure_dir(results/"paper_tables")

    tickers = cfg.tickers if args.ticker.upper()=="ALL" else [args.ticker]
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    lags_list = [int(x) for x in args.lags.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    use_sent = args.use_sentiment.lower() in ("yes","true","1")

    all_back = []; all_eco = []; all_lastpred = []

    for t in tickers:
        for lag in lags_list:
            for seed in seeds:
                # features + walk-forward
                df, cols = prepare_features(cfg, t, use_sent, lag)
                back_folds, back_agg = walk_forward_eval(df, cols, y_col="y", prev_col="y_prev",
                                                          models=models, n_splits=args.walk_splits,
                                                          test_size=args.walk_test, random_state=seed)
                back_folds["ticker"] = t; back_folds["lags"] = lag; back_folds["seed"] = seed
                back_agg["ticker"] = t; back_agg["lags"] = lag; back_agg["seed"] = seed
                all_back.append(back_agg)

                # train once on earliest (train) and evaluate last window for economics
                # emulate last fold test set for economics
                n = len(df)
                start = n - args.walk_test
                test = df.iloc[start:]
                # simple RF prediction on last window for economics chart
                m = RFModel()
                # train on prior data
                from thesisproj.split import chrono_split
                train_df = df.iloc[:start]
                Xtr, ytr = train_df[cols].values, train_df["y"].values
                Xte, yte = test[cols].values, test["y"].values
                m.fit(Xtr, ytr); yhat = m.predict(Xte)
                econ_in = pd.DataFrame({"date": test["date"].values, "y_true": yte, "y_pred": yhat, "y_prev": test["y_prev"].values})
                eco = simple_long_strategy(econ_in, cost_bps=args.cost_bps, thresh=0.0)
                eco["ticker"]=t; eco["lags"]=lag; eco["seed"]=seed; eco["model"]="RF"
                all_eco.append(eco)

                # save last predictions figure
                from thesisproj.plot import plot_price_vs_pred
                figpath = results/"figures"/f"adv_price_vs_pred_{t}_lags{lag}_seed{seed}.png"
                plot_price_vs_pred(econ_in.rename(columns={"y_true":"y_true","y_pred":"y_pred"}), str(figpath), f"{t} Last Window")

    back_df = pd.concat(all_back, ignore_index=True)
    eco_df = pd.DataFrame(all_eco)
    save_table(back_df, results/"paper_tables"/"walk_forward_summary.csv",
               results/"paper_tables"/"walk_forward_summary.tex",
               caption="Walk-forward mean±std metrics per model.", label="tab:walkforward")
    save_table(eco_df, results/"paper_tables"/"economics_summary.csv",
               results/"paper_tables"/"economics_summary.tex",
               caption="Economic backtest summary (simple long, cost in bps).", label="tab:economics")

    print("\nSaved paper tables to", results/"paper_tables")
    print("\nExample usage:")
    print("python scripts/run_advanced.py --ticker THYAO --models RF,GBM --walk_splits 5 --walk_test 20 --use_sentiment yes --lags 0,3,5 --seeds 42,1337 --cost_bps 10.0")

if __name__ == "__main__":
    main()
