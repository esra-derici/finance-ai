#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Google News RSS'ten başlık toplayıp (TR), sözlük tabanlı duygu skoru (-1..+1) hesaplar.
Çıktı: data/raw/news.csv (date,ticker,title,sentiment)

Kullanım örn:
python scripts/build_news_csv.py --tickers THYAO,BIMAS,TUPRS --start 2024-01-01 --end 2025-09-30 --out data/raw/news.csv
"""
import argparse, html, re
from urllib.parse import quote_plus
from datetime import datetime, timezone
import feedparser
import pandas as pd
from dateutil import parser as dparser

# --- basit TR normalizasyonu (eşleştirme kolaylığı) ---
TR_MAP = str.maketrans({"Ç":"c","ç":"c","Ğ":"g","ğ":"g","İ":"i","I":"i","ı":"i","Ö":"o","ö":"o","Ş":"s","ş":"s","Ü":"u","ü":"u"})
def norm_tr(s:str) -> str:
    return re.sub(r"[^a-z0-9\s]", " ", s.lower().translate(TR_MAP)).strip()

# --- ticker -> eşleşecek anahtar kelimeler (istediğin gibi genişlet) ---
TICKER_KEYWORDS = {
    "THYAO": ["thyao", "türk hava", "turkish airlines", "thy"],
    "BIMAS": ["bimas", "bim", "birleşik mağazalar", "birlesik magazalar"],
    "TUPRS": ["tupras", "tüpraş", "tupras rafineri", "rafineri"],
}

# --- mini TR duygu sözlüğü (örnek; istersen büyüt) ---
POS = [
    "artış", "artis", "yükseliş", "yukselis", "yükseldi", "rekor", "güçlü", "guclu", "iyileşme", "iyimser",
    "kazanç", "kazan", "büyüme", "buyume", "pozitif", "olumlu", "talep güçlü", "sipariş aldı", "yatırım"
]
NEG = [
    "düşüş", "dus", "gerileme", "zayıf", "zayif", "kayıp", "kayip", "zarar", "negatif", "olumsuz",
    "maliyet artışı", "maliyet baskısı", "soruşturma", "dava", "ceza", "iflas", "risk", "uyarı", "kesinti"
]
INTENSIFIERS = ["çok", "cok", "rekor", "sert", "keskin", "güçlü", "guclu"]
DAMPENERS    = ["hafif", "az", "sınırlı", "sinirli"]

def lexicon_score(text:str) -> float:
    t = norm_tr(text)
    toks = t.split()
    pos = sum(1 for w in toks for k in POS if k in w)
    neg = sum(1 for w in toks for k in NEG if k in w)
    # yoğunlaştırıcı / yumuşatıcı
    if any(w in toks for w in INTENSIFIERS): pos *= 1.2; neg *= 1.2
    if any(w in toks for w in DAMPENERS):    pos *= 0.8; neg *= 0.8
    s = (pos - neg) / (pos + neg + 1e-6)
    # sınırla
    return max(-1.0, min(1.0, s))

def guess_tickers(title:str) -> list:
    t = norm_tr(title)
    hits = []
    for tk, keys in TICKER_KEYWORDS.items():
        for kw in keys:
            if kw in t:
                hits.append(tk); break
    return hits

def google_news_rss(query:str) -> str:
    # TR sonuçları
    base = "https://news.google.com/rss/search?q="
    tail = "&hl=tr&gl=TR&ceid=TR:tr"
    return base + quote_plus(query) + tail

def fetch_for_ticker(ticker:str, start:str=None, end:str=None) -> pd.DataFrame:
    """
    Sorgu stratejisi: her ticker için hem isim anahtarlarını hem de ticker kodunu arıyoruz.
    """
    queries = [ticker] + TICKER_KEYWORDS.get(ticker, [])
    rows = []
    for q in queries:
        url = google_news_rss(q)
        feed = feedparser.parse(url)
        for e in feed.entries:
            title = html.unescape(getattr(e, "title", "")).strip()
            if not title: continue
            pub = getattr(e, "published", "") or getattr(e, "pubDate", "") or ""
            try:
                dt = dparser.parse(pub).date()
            except Exception:
                dt = datetime.now(timezone.utc).date()
            if start and dt < dparser.parse(start).date(): continue
            if end   and dt > dparser.parse(end).date():   continue
            rows.append({"date": dt.isoformat(), "ticker": ticker, "title": title, "source": getattr(e, "source", {}).get("title", ""), "url": getattr(e, "link", "")})
    if not rows:
        return pd.DataFrame(columns=["date","ticker","title","source","url"])
    df = pd.DataFrame(rows).drop_duplicates(subset=["date","title","ticker"]).reset_index(drop=True)
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", default="THYAO,BIMAS,TUPRS")
    ap.add_argument("--start", default=None)
    ap.add_argument("--end",   default=None)
    ap.add_argument("--out",   default="data/raw/news.csv")
    args = ap.parse_args()

    tickers = [x.strip().upper() for x in args.tickers.split(",") if x.strip()]
    all_df = []
    for t in tickers:
        df = fetch_for_ticker(t, start=args.start, end=args.end)
        if not df.empty:
            # sözlük skoru
            df["sentiment"] = df["title"].apply(lexicon_score)
            all_df.append(df[["date","ticker","title","sentiment"]])

    if not all_df:
        print("Uyarı: hiçbir haber bulunamadı. Anahtar kelimeleri genişletin.")
        return

    out = pd.concat(all_df, ignore_index=True).sort_values(["date","ticker"])
    # (opsiyonel) aynı gün aynı başlık birden çok tickere eşleşmişse, bırakıyoruz; istersen groupby ile tekilleştir.
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False, encoding="utf-8")
    print(f"Yazıldı: {args.out}  (satır: {len(out)})")

if __name__ == "__main__":
    from pathlib import Path
    main()
