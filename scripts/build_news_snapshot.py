"""Rebuild the web snapshot from original raw news using notebook-equivalent scoring."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from web_engine import ROOT, score_headline


def parse_date(value):
    if pd.isna(value):
        return pd.NaT
    dt = pd.to_datetime(value, errors='coerce', utc=True)
    if pd.notna(dt):
        return dt.tz_convert(None)
    s = str(value).replace('T', '').replace('Z', '').replace('-', '').replace(':', '')[:14]
    return pd.to_datetime(s, format='%Y%m%d%H%M%S', errors='coerce')


def build():
    raw = pd.read_csv(ROOT / 'data/raw/raw_news.csv')
    data = raw.dropna(subset=['title', 'url']).copy()
    data['title'] = data['title'].astype(str).str.strip()
    data = data[data['title'].str.len() > 5]
    data['article_date'] = data['seendate'].apply(parse_date)
    data = data.dropna(subset=['article_date']).drop_duplicates(subset=['nse_symbol', 'url'])
    data = data.sort_values(['nse_symbol', 'article_date']).reset_index(drop=True)
    scored = pd.DataFrame([score_headline(title) for title in data['title']])
    data = pd.concat([data, scored], axis=1)
    data['article_date'] = data['article_date'].dt.strftime('%Y-%m-%d')
    data = data.sort_values(['article_date', 'nse_symbol'], ascending=[False, True])
    out = ROOT / 'web_data/news_snapshot.csv'
    data.to_csv(out, index=False)
    print(f'Saved {len(data)} scored headlines to {out}')


if __name__ == '__main__':
    build()
