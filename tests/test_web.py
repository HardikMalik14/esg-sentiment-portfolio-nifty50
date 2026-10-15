import ast
import base64
import json
from pathlib import Path
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from index import app
from web_engine import KEYWORDS, archived_returns, portfolio_report, score_headline

client = TestClient(app)


def test_home_metadata_and_archived_provenance():
    assert 'verdant' in client.get('/').text
    assert client.get('/api/health').status_code == 200
    d = client.get('/api/metadata').json()
    assert d['universe_count'] == 50
    assert d['news_count'] > 8000
    assert len(d['model_errors']) == 4
    assert any('look-ahead' in s for s in d['limitations'])


def test_archive_statistics_match_notebook_outputs():
    r = client.post('/api/report', json={})
    assert r.status_code == 200, r.text
    d = r.json()
    saved = pd.read_csv('outputs/tables/portfolio_performance.csv').set_index('model')
    assert len(d['curves']) == 25
    for row in d['performance']:
        s = saved.loc[row['model']]
        assert round(row['ann_return'], 4) == s['Ann_Return']
        assert round(row['ann_vol'], 4) == s['Ann_Vol']
        assert round(row['sharpe'], 4) == s['Sharpe']


def test_changing_capital_scales_values_but_not_percentage_returns():
    a = portfolio_report(archived_returns(), capital=100000)
    b = portfolio_report(archived_returns(), capital=200000)
    for x, y in zip(a['performance'], b['performance']):
        np.testing.assert_allclose(y['ending_value'], x['ending_value'] * 2)
        np.testing.assert_allclose(y['total_return'], x['total_return'])
        np.testing.assert_allclose(y['sharpe'], x['sharpe'])


def test_risk_free_changes_sharpe_and_date_filter_changes_returns():
    a = portfolio_report(archived_returns(), risk_free=.065)
    b = portfolio_report(archived_returns(), risk_free=.02)
    by_model = {r['model']: r for r in b['performance']}
    assert all(row['sharpe'] < by_model[row['model']]['sharpe'] for row in a['performance'])
    one = portfolio_report(archived_returns(), start='2026-03', end='2026-03')
    assert one['selected_months'] == ['2026-03']
    assert all(r['sharpe'] is None and r['ann_vol'] is None for r in one['performance'])


def test_upload_recalculates_from_values_and_does_not_mutate_archive():
    body = b'model,rebalance_date,monthly_return\nCustom,2025-01,0.1\nCustom,2025-02,-0.05\n'
    r = client.post('/api/report', json={'source': 'upload', 'file_data': base64.b64encode(body).decode()})
    assert r.status_code == 200, r.text
    row = r.json()['performance'][0]
    np.testing.assert_allclose(row['ending_value'], 104500)
    np.testing.assert_allclose(row['max_drawdown'], -.05)
    assert r.json()['is_archive'] is False
    assert len(client.post('/api/report', json={}).json()['performance']) == 5


def test_upload_rejects_missing_months_duplicates_and_infinite_returns():
    for body in [
        'model,rebalance_date,monthly_return\nX,2025-01,.1\nX,2025-03,.2',
        'model,rebalance_date,monthly_return\nX,2025-01,.1\nX,2025-01,.2',
        'model,rebalance_date,monthly_return\nX,2025-01,inf\nX,2025-02,.2',
        'model,rebalance_date,monthly_return\nX,2025-01,.1\nY,2025-02,.2',
    ]:
        r = client.post('/api/report', json={'source': 'upload', 'file_data': base64.b64encode(body.encode()).decode()})
        assert r.status_code == 400, r.text
    assert client.post('/api/report', content=b'x' * 800001).status_code == 413


def test_news_filters_search_and_pagination():
    all_news = client.get('/api/news').json()
    assert len(all_news['rows']) == 12
    second = client.get('/api/news?page=2').json()
    assert second['rows'][0]['url'] != all_news['rows'][0]['url']
    q = client.get('/api/news?symbol=RELIANCE&category=environment&tone=positive').json()
    assert q['total'] > 0
    assert all(row['nse_symbol'] == 'RELIANCE' and row['env_score'] > 0 and row['sentiment_score'] >= 5 for row in q['rows'])
    assert client.get('/api/news?search=UNLIKELY_UNIQUE_STRING_54321').json()['total'] == 0


def test_keyword_definitions_are_exactly_the_original_notebook_lists():
    notebook = json.loads(Path('esg_sentiment_portfolio.ipynb').read_text())
    original = {}
    for node in ast.parse(''.join(notebook['cells'][4]['source'])).body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id.startswith('KW_'):
            original[node.targets[0].id] = ast.literal_eval(node.value)
    assert KEYWORDS == original


def test_live_headline_scoring_reacts_to_input():
    a = client.post('/api/score', json={'text': 'Company celebrates excellent growth in renewable energy and worker safety'}).json()
    b = client.post('/api/score', json={'text': 'Company faces fraud scandal, corruption lawsuit and toxic pollution'}).json()
    assert a['sentiment_score'] > b['sentiment_score']
    assert a['env_score'] > 0 and a['social_score'] > 0
    assert b['controversy'] == 1
    assert client.post('/api/score', json={'text': '     '}).status_code == 400


def test_sample_csv_is_valid_input():
    r = client.get('/api/sample.csv')
    response = client.post('/api/report', json={'source': 'upload', 'file_data': base64.b64encode(r.content).decode()})
    assert response.status_code == 200
    assert len(response.json()['curves']) == 25
