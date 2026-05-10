[README.md](https://github.com/user-attachments/files/27565162/README.md)
# fin-forecast# Stock Analyzer & Forecasting Engine

Web app for equity analysis with fundamentals, valuation, technical indicators, forecasts, risk metrics, and an optional generated investment pitch.

This is a decision-support tool, not investment advice.

## Features

- Browser dashboard backed by a FastAPI JSON API
- Streamlit interface retained as an alternate local UI
- Ticker lookup and S&P 500 stock finder in the Streamlit UI
- yfinance-backed price, company, financial statement, and analyst estimate data
- FRED-backed macro/risk-free-rate support, with a safe fallback when no key is configured
- Fundamental, valuation, technical, forecasting, and risk analysis tabs
- Mathematical-finance-aware Monte Carlo using exact lognormal discretization,
  antithetic Brownian increments, dividend-adjusted risk-neutral drift, and a
  numeraire check
- Optional Anthropic or Groq investment pitch generation, with a local template fallback
- DuckDB and Parquet runtime cache

## Local Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
uvicorn web_app.main:app --reload
```

Open `http://localhost:8000`.

The legacy Streamlit interface is still available:

```bash
streamlit run app.py
```

## Configuration

All environment variables are optional:

- `FRED_API_KEY`: fetches the current 10-year Treasury rate. If missing, the app uses a 4.5% default.
- `ANTHROPIC_API_KEY`: enables Anthropic-backed pitch generation.
- `GROQ_API_KEY`: optional fallback for pitch generation when Anthropic is unavailable.
- `ANTHROPIC_MODEL`: overrides the Anthropic model name. Defaults to `claude-sonnet-4-6`.

Do not commit `.env`; use your deployment provider's secrets UI.

## Deployment

### Streamlit Community Cloud

1. Push this project to a Git repository.
2. Create a Streamlit app pointing at `app.py`.
3. Set Python to `3.12` if the provider asks.
4. Add any secrets from `.env.example`.
5. Deploy.

The app creates `data/` cache files at runtime. Those files are intentionally ignored by git.

### Docker

```bash
docker build -t stock-forecast .
docker run --env-file .env -p 8000:8000 stock-forecast
```

### Render, Railway, or Heroku-style Hosts

Use the included `Procfile`. Set the start command to:

```bash
uvicorn web_app.main:app --host 0.0.0.0 --port=$PORT
```

## Verification

```bash
python -m compileall -q app.py analysis core data_layer ui transcriber
python -m unittest discover -s tests
```

## Optional Transcriber Utility

`transcriber/` is a separate Streamlit app for local video transcription. It requires `ffmpeg` and `transcriber/requirements.txt`:

```bash
pip install -r transcriber/requirements.txt
streamlit run transcriber/app.py
```
