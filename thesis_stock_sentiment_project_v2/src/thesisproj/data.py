from pathlib import Path
import pandas as pd
import numpy as np

def _to_float_series(s: pd.Series) -> pd.Series:
    # "1.234,56" → "1234.56", boşluk ve NBSP temizleme
    s = s.astype(str).str.replace("\u00a0", "", regex=False).str.replace(" ", "", regex=False)
    # Eğer virgül ondalık gibi kullanılmışsa nokta ile değiştir
    s = s.str.replace(".", "", regex=False)            # binlik ayırıcı nokta kaldır
    s = s.str.replace(",", ".", regex=False)           # ondalık virgülü noktaya çevir
    return pd.to_numeric(s, errors="coerce")

def load_prices(data_dir: str, ticker: str) -> pd.DataFrame:
    p = Path(data_dir)/"raw"/f"price_{ticker}.csv"
    df = pd.read_csv(p, parse_dates=["date"], dayfirst=False, encoding="utf-8")
    # Zorunlu numeric dönüşümleri
    if "close" in df.columns:
        df["close"] = _to_float_series(df["close"])
    else:
        raise ValueError(f"'close' kolonu yok: {p}")
    if "volume" in df.columns:
        # volume tipik olarak tam sayı; sadece binlik ayırıcıları sil
        vol = df["volume"].astype(str).str.replace("\u00a0","", regex=False).str.replace(" ","", regex=False).str.replace(",","", regex=False).str.replace(".","", regex=False)
        df["volume"] = pd.to_numeric(vol, errors="coerce")
    # Tarih ve NaN temizliği
    df = df.dropna(subset=["date","close"]).sort_values("date").reset_index(drop=True)
    return df

def load_news(data_dir: str) -> pd.DataFrame:
    p = Path(data_dir)/"raw"/"news.csv"
    df = pd.read_csv(p, parse_dates=["date"], dayfirst=False, encoding="utf-8")
    # sentiment’i garanti sayısal yap
    if "sentiment" in df.columns:
        df["sentiment"] = pd.to_numeric(df["sentiment"], errors="coerce").fillna(0.0)
    return df.sort_values(["date","ticker"]).reset_index(drop=True)
