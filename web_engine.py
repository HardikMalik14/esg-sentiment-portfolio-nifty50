"""Portfolio reporting and the original notebook's VADER/keyword scoring."""
from __future__ import annotations
import json
import re
from functools import lru_cache
from pathlib import Path
import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

ROOT = Path(__file__).resolve().parent
NAMES = {'RF': 'Random Forest', 'GB': 'Gradient Boosting', 'Linear': 'Linear Regression', 'LASSO': 'LASSO', 'Equal': 'Equal-weight benchmark'}
KEYWORDS = json.loads((ROOT / 'web_data/keywords.json').read_text())
PATTERNS = {name: re.compile('|'.join(re.escape(str(k).lower().strip()) for k in words)) for name, words in KEYWORDS.items()}

@lru_cache(maxsize=1)
def analyzer():
    return SentimentIntensityAnalyzer()


def score_headline(title: str):
    tone = analyzer().polarity_scores(title)
    low = title.lower()
    counts = {name: len(pat.findall(low)) for name, pat in PATTERNS.items()}
    return {'sentiment_score': round(tone['compound'] * 100, 2),
            'positive_tone': round(tone['pos'] * 100, 2),
            'negative_tone': round(tone['neg'] * 100, 2),
            'neutral_tone': round(tone['neu'] * 100, 2),
            'env_score': counts['KW_ENVIRONMENTAL'], 'social_score': counts['KW_SOCIAL'],
            'gov_score': counts['KW_GOVERNANCE'], 'sdg_score': counts['KW_SDG'],
            'esg_score': counts['KW_ENVIRONMENTAL'] + counts['KW_SOCIAL'] + counts['KW_GOVERNANCE'],
            'controversy': int(counts['KW_CONTROVERSY'] > 0)}


def records(frame):
    return json.loads(frame.to_json(orient='records', date_format='iso'))


@lru_cache(maxsize=1)
def archived_returns():
    return pd.read_csv(ROOT / 'outputs/tables/portfolio_returns.csv')


@lru_cache(maxsize=1)
def archived_news():
    return pd.read_csv(ROOT / 'web_data/news_snapshot.csv').fillna('')


def validate_returns(frame):
    required = {'model', 'rebalance_date', 'monthly_return'}
    if not required.issubset(frame.columns):
        raise ValueError('CSV needs model, rebalance_date, and monthly_return columns. Returns must be decimals, e.g. 0.02 for 2%.')
    if not 2 <= len(frame) <= 600:
        raise ValueError('Use 2–600 monthly return rows.')
    data = frame[['model', 'rebalance_date', 'monthly_return']].copy()
    if data.isna().any().any():
        raise ValueError('Model, month, and return cannot be blank.')
    data['model'] = data['model'].astype(str).str.strip()
    if (data['model'].str.len() == 0).any() or data['model'].str.len().max() > 60:
        raise ValueError('Model names must contain 1–60 characters.')
    if data['model'].nunique() > 12:
        raise ValueError('Use at most 12 models.')
    if not data['rebalance_date'].astype(str).str.fullmatch(r'\d{4}-\d{2}').all():
        raise ValueError('Use YYYY-MM for rebalance_date, e.g. 2026-01.')
    parsed = pd.to_datetime(data['rebalance_date'], format='%Y-%m', errors='coerce')
    if parsed.isna().any():
        raise ValueError('Some month values are invalid.')
    data['monthly_return'] = pd.to_numeric(data['monthly_return'], errors='coerce')
    vals = data['monthly_return'].to_numpy(dtype=float)
    if not np.isfinite(vals).all() or (vals <= -1).any() or (vals > 10).any():
        raise ValueError('Monthly returns must be finite decimals greater than -1 and at most 10.')
    if data.duplicated(['model', 'rebalance_date']).any():
        raise ValueError('Each model must have exactly one return per month.')
    months = sorted(data['rebalance_date'].unique())
    expected = pd.period_range(months[0], months[-1], freq='M').astype(str).tolist()
    if months != expected:
        raise ValueError('The data must contain consecutive calendar months with no gaps.')
    if any(sorted(g['rebalance_date'].tolist()) != months for _, g in data.groupby('model')):
        raise ValueError('All models must cover the same months; missing returns are not filled with zero.')
    return data.sort_values(['model', 'rebalance_date']).reset_index(drop=True)


def portfolio_report(frame, capital=100000.0, risk_free=0.065, start=None, end=None):
    for bound in [start, end]:
        if bound and pd.isna(pd.to_datetime(bound, format='%Y-%m', errors='coerce')):
            raise ValueError('Date bounds must be valid YYYY-MM calendar months.')
    data = validate_returns(frame)
    all_months = sorted(data['rebalance_date'].unique())
    if start:
        data = data[data['rebalance_date'] >= start]
    if end:
        data = data[data['rebalance_date'] <= end]
    if data.empty:
        raise ValueError('No returns fall inside this date range.')
    rows, curves = [], []
    for code, group in data.groupby('model', sort=False):
        rets = group['monthly_return'].to_numpy(dtype=float)
        with np.errstate(over='ignore', invalid='ignore'):
            wealth = capital * np.cumprod(1 + rets)
        if not np.isfinite(wealth).all():
            raise ValueError('The compounded capital path exceeds numeric limits. Use a smaller range or realistic return values.')
        wealth_with_start = np.r_[capital, wealth]
        peaks = np.maximum.accumulate(wealth_with_start)
        drawdowns = wealth_with_start / peaks - 1
        # Match the notebook's annualization convention for archived comparison.
        ann_return = float((1 + rets.mean()) ** 12 - 1)
        vol = float(rets.std(ddof=1) * np.sqrt(12)) if len(rets) > 1 else None
        sharpe = float((ann_return - risk_free) / vol) if vol and vol > 1e-12 else None
        rows.append({'model': code, 'name': NAMES.get(code, code), 'months': len(rets),
                     'total_return': float(wealth[-1] / capital - 1), 'ending_value': float(wealth[-1]),
                     'pnl': float(wealth[-1] - capital), 'ann_return': ann_return, 'ann_vol': vol,
                     'sharpe': sharpe, 'max_drawdown': float(drawdowns.min())})
        for i, (_, row) in enumerate(group.iterrows()):
            curves.append({'model': code, 'month': row['rebalance_date'], 'monthly_return': float(rets[i]),
                           'value': float(wealth[i]), 'drawdown': float(drawdowns[i + 1])})
    rows.sort(key=lambda r: r['sharpe'] if r['sharpe'] is not None else -float('inf'), reverse=True)
    return {'performance': rows, 'curves': curves, 'available_months': all_months,
            'selected_months': sorted(data['rebalance_date'].unique()), 'capital': capital,
            'risk_free': risk_free, 'annualization': '(1 + mean monthly return)^12 - 1; sample volatility × sqrt(12)',
            'monthly_returns': records(data)}
