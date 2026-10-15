"""Vercel application for the existing ESG/ML research repository."""
from __future__ import annotations
import base64
import binascii
import io
import re
from pathlib import Path
from typing import Literal
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field
from web_engine import ROOT, NAMES, archived_news, archived_returns, portfolio_report, records, score_headline

app = FastAPI(title='Verdant — ESG & ML Research', version='1.0.0')


class ReportOptions(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    capital: float = Field(100000, ge=1, le=1e9)
    risk_free: float = Field(.065, ge=0, le=1)
    start: str | None = Field(None, pattern=r'^\d{4}-\d{2}$')
    end: str | None = Field(None, pattern=r'^\d{4}-\d{2}$')
    source: Literal['archive', 'upload'] = 'archive'
    file_data: str = Field('', max_length=700000)


class Headline(BaseModel):
    model_config = ConfigDict(extra='forbid')
    text: str = Field(min_length=5, max_length=2000)


@app.middleware('http')
async def limit_body(request: Request, call_next):
    if request.method == 'POST':
        body = bytearray()
        async for chunk in request.stream():
            body.extend(chunk)
            if len(body) > 800000:
                return JSONResponse({'detail': 'Request too large. Use a CSV under 500 KB.'}, status_code=413)
        request._body = bytes(body)
    return await call_next(request)


@app.get('/', include_in_schema=False)
def home():
    return FileResponse(ROOT / 'public/index.html')


@app.get('/api/health')
def health():
    return {'status': 'ok', 'project': 'esg-sentiment-portfolio-nifty50'}


@app.get('/api/metadata')
def metadata():
    news = archived_news()
    companies = pd.read_csv(ROOT / 'data/raw/nifty50_companies.csv')
    models = pd.read_csv(ROOT / 'outputs/models/model_performance.csv')
    monthly = archived_returns()
    return {'model_names': NAMES, 'companies': records(companies[['company_name', 'nse_symbol']]),
            'model_errors': records(models), 'news_count': len(news),
            'covered_companies': int(news['nse_symbol'].nunique()), 'universe_count': len(companies),
            'news_start': news['article_date'].min(), 'news_end': news['article_date'].max(),
            'months': sorted(monthly['rebalance_date'].unique()),
            'research_source': 'Original committed notebook outputs; monthly results are saved research, not live trading.',
            'limitations': ['The original stock selection averages predictions from inside each holding month (look-ahead bias).',
                            'Only five monthly observations are saved, including a partial final month.',
                            'Original results omit transaction costs. Annualization from this short sample is unstable.',
                            'ESG keyword counts and VADER headline sentiment are proxies, not verified company ESG ratings.',
                            'Daily return prediction R² is near zero for all four supervised models. No RL model is implemented.']}


@app.post('/api/report')
def report(o: ReportOptions):
    try:
        if o.start and o.end and o.start > o.end:
            raise ValueError('Start month must not be after end month.')
        if o.source == 'upload':
            try:
                raw = base64.b64decode(o.file_data, validate=True)
            except (binascii.Error, ValueError) as exc:
                raise ValueError('Invalid file encoding.') from exc
            if len(raw) > 500000:
                raise ValueError('Use a CSV under 500 KB.')
            data = pd.read_csv(io.BytesIO(raw), nrows=601)
        else:
            data = archived_returns()
        response = portfolio_report(data, o.capital, o.risk_free, o.start, o.end)
        response['source'] = 'Archived notebook results' if o.source == 'archive' else 'Uploaded monthly returns'
        response['is_archive'] = o.source == 'archive'
        return response
    except (ValueError, TypeError, KeyError) as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get('/api/news')
def news(symbol: str = Query('', max_length=30), search: str = Query('', max_length=120),
         category: Literal['all', 'environment', 'social', 'governance', 'sdg', 'controversy'] = 'all',
         tone: Literal['all', 'positive', 'negative', 'neutral'] = 'all',
         page: int = Query(1, ge=1, le=1000)):
    df = archived_news()
    if symbol:
        df = df[df['nse_symbol'] == symbol]
    if search.strip():
        df = df[df['title'].str.contains(search.strip(), case=False, regex=False)]
    field = {'environment': 'env_score', 'social': 'social_score', 'governance': 'gov_score', 'sdg': 'sdg_score', 'controversy': 'controversy'}.get(category)
    if field:
        df = df[df[field] > 0]
    if tone == 'positive':
        df = df[df['sentiment_score'] >= 5]
    elif tone == 'negative':
        df = df[df['sentiment_score'] <= -5]
    elif tone == 'neutral':
        df = df[df['sentiment_score'].between(-5, 5, inclusive='neither')]
    size = 12
    counts = df.groupby('company_name').size().sort_values(ascending=False).head(8)
    return {'rows': records(df.iloc[(page - 1) * size:page * size]), 'total': len(df), 'page': page,
            'pages': (len(df) + size - 1) // size,
            'average_sentiment': float(df['sentiment_score'].mean()) if len(df) else None,
            'controversy_share': float(df['controversy'].mean()) if len(df) else None,
            'coverage': [{'name': name, 'count': int(count)} for name, count in counts.items()]}


@app.post('/api/score')
def score(o: Headline):
    if len(o.text.strip()) < 5:
        raise HTTPException(400, 'Enter a headline with at least five visible characters.')
    return {'headline': o.text.strip(), **score_headline(o.text.strip())}


@app.get('/api/sample.csv')
def sample():
    text = archived_returns()[['model', 'rebalance_date', 'monthly_return']].to_csv(index=False)
    return Response(text, media_type='text/csv', headers={'Content-Disposition': 'attachment; filename="esg_monthly_returns.csv"'})
