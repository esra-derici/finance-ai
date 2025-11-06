
import argparse
from pathlib import Path
import yfinance as yf

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", default="THYAO.IS,BIMAS.IS,TUPRS.IS",
                    help="Comma-separated Yahoo tickers (BIST uses .IS suffix).")
    ap.add_argument("--start", default="2020-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--outdir", default="data/raw")
    args = ap.parse_args()

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    for t in [x.strip() for x in args.tickers.split(",")]:
        df = yf.download(t, start=args.start, end=args.end, progress=False)
        if df.empty:
            print("No data for", t); continue
        df = df.rename(columns={"Close":"close","Volume":"volume"})
        df = df[["close","volume"]].reset_index().rename(columns={"Date":"date"})
        out = Path(args.outdir)/f"price_{t.replace('.IS','')}.csv"
        df.to_csv(out, index=False)
        print("Wrote", out)

if __name__ == "__main__":
    main()
