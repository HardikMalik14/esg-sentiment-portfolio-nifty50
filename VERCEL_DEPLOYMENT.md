# Deploy Verdant from the original ESG repository

Original repository: https://github.com/HardikMalik14/esg-sentiment-portfolio-nifty50

## Apply the update using Windows PowerShell

Save `ESG_Original_Repo_Vercel_Update.zip` in Downloads. The following creates a local clone of your existing GitHub repository, extracts only the added/updated files over it, and pushes the commit back to that same repository. It does not create a new GitHub repository.

```powershell
cd "$HOME\Downloads"
git clone https://github.com/HardikMalik14/esg-sentiment-portfolio-nifty50.git ESG-original-deploy
cd .\ESG-original-deploy
Expand-Archive -LiteralPath "$HOME\Downloads\ESG_Original_Repo_Vercel_Update.zip" -DestinationPath . -Force
git add .
git commit -m "Add redesigned ESG research dashboard and Vercel deployment"
git push origin main
```

Run each command after the previous one succeeds. If the local `ESG-original-deploy` directory already exists, use that checkout and pull the current `main` before extracting; do not run `git init` or force-push. The ZIP has no enclosing folder and contains no `.git` history.

## Vercel

1. Open https://vercel.com/new and import **HardikMalik14/esg-sentiment-portfolio-nifty50**.
2. Select **FastAPI** as Framework Preset. Set Root Directory to `./`.
3. Leave Build Command, Output Directory and Install Command overrides disabled. No environment variables are required.
4. Click Deploy. Once ready, click Visit.
5. The first view loads original notebook monthly returns automatically. Select different models, change starting capital/risk-free assumption/date range, then click Recalculate.
6. Open Signals & news, filter headlines, and try your own headline in the scorer.
7. Share the production `.vercel.app` URL in your portfolio and repository About → Website field.

Reference: https://vercel.com/docs/frameworks/backend/fastapi

`pyproject.toml` explicitly sets the Python entrypoint to `index:app`. Python 3.12 is selected in `.python-version`. The service uses FastAPI, pandas, NumPy and VADER; research/notebook dependencies remain in `requirements-notebook.txt`. The original notebook and committed research outputs are preserved.

## What is live and what is archived

- **Archived**: the notebook's model training, predictions, monthly portfolio return series and original prediction-error metrics. These saved artifacts are not presented as a live market feed.
- **Calculated per request**: capital growth, total return, annualized return/volatility, Sharpe ratio, maximum drawdown, and an analysis of uploaded monthly return CSVs.
- **Calculated from your text**: VADER sentiment and original ESG/SDG/controversy keyword counts for custom headlines.
- **Prepared snapshot**: the original 8,755 news items scored with the notebook-equivalent scoring logic, then filtered by the website. Regenerate using `python scripts/build_news_snapshot.py` after updating `data/raw/raw_news.csv`. The scorer's keyword lists are copied exactly from the original notebook configuration.

There is no RL agent, new training pipeline, live market-data download, stock recommendation engine or new portfolio-optimization algorithm. This is a deployment and presentation of the existing supervised ML project, with interactive reporting and headline scoring. Holdings/weights are not invented: the committed outputs do not include selected stock identities for every monthly portfolio, so the site does not show fabricated holdings.

## Using your own CSV

Use three columns:

```csv
model,rebalance_date,monthly_return
MyModel,2025-01,0.02
MyModel,2025-02,-0.01
```

Returns are decimals: `0.02` means +2%. Month format is YYYY-MM. All models must cover identical consecutive months. Duplicate model-months, missing observations, non-finite returns, and returns ≤ −100% are rejected. Uploads are limited to 500 KB, 600 rows and 12 models. The downloadable example is the original portfolio return series in this three-column format. Uploads are processed in memory; no upload database or history is created.

Uploaded return data recalculates reporting statistics. It does not fit a new model. Archived prediction-error metrics remain in Research notes and are labeled separately.

## Calculation definitions

- Capital path: starting capital multiplied by cumulative product of (1 + monthly return), with a starting point before the first selected month.
- Total period return: ending capital / starting capital − 1.
- Annualized return: (1 + arithmetic mean monthly return)^12 − 1, following the original notebook convention.
- Annualized volatility: sample standard deviation of monthly returns × sqrt(12).
- Sharpe: (annualized return − selected risk-free assumption) / annualized volatility.
- Drawdown: wealth / running peak − 1, including the initial capital.
- One-month selections have no sample volatility or Sharpe; these display as unavailable.

Changing starting capital scales rupee values without changing percentage returns or Sharpe. Changing risk-free assumptions changes Sharpe, not the portfolio's return path. Changing the focus model changes the selected metrics and highlighting; it does not retrain a model.

## Material research limitations

The original backtest selects stocks using predictions from inside the same holding month (look-ahead bias). Only five monthly observations are saved and the final month is partial. Original returns omit transaction costs. The site preserves these limitations prominently and does not correct or conceal the original results. ESG keyword scores are content proxies rather than audited ESG ratings. The original daily prediction R² values are near zero.

## Local development

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn index:app --reload
```

Visit http://127.0.0.1:8000. To run tests, install `requirements-dev.txt`, then `python -m pytest -q`.

## Troubleshooting

- **ModuleNotFoundError**: ensure `web_engine.py`, `web_data/keywords.json`, and `web_data/news_snapshot.csv` were committed with `index.py`.
- **404**: root must be `./` and Framework Preset FastAPI.
- **Missing saved CSV**: keep `outputs/tables`, `outputs/models/model_performance.csv`, and `data/raw/nifty50_companies.csv` in the repository. These are already tracked upstream.
- **Invalid upload**: use the downloadable example format and decimal returns.
- **Failed deployment**: expand Build Logs and share the first error with its nearby lines.

Cloud deployment is performed in your Vercel account. Local validation cannot establish that the public cloud deployment has succeeded.

## Validation performed

10 backend tests passed, including exact comparisons to the saved notebook statistics, capital scaling, risk-free and date-range changes, custom CSV arithmetic, validation, news filtering and live input-sensitive scoring. Headless-browser checks passed for archive loading, model/chart controls, CSV upload, one-month edge cases, export, restoring the archive, news filters, headline scoring and mobile layout, with no JavaScript errors.

The raw news snapshot includes article dates outside the notebook's market-data/backtest window. The explorer preserves those supplied dates and labels this distinction; it is not a verified contemporaneous news feed.
