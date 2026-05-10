"""Central configuration — all constants, thresholds, weights, TTLs."""
from __future__ import annotations
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
PARQUET_DIR = DATA_DIR / "parquet"
DUCKDB_PATH = DATA_DIR / "cache.duckdb"

DATA_DIR.mkdir(exist_ok=True)
PARQUET_DIR.mkdir(exist_ok=True)

# ── API Keys ───────────────────────────────────────────────────────────────────
FRED_API_KEY: str = os.getenv("FRED_API_KEY", "")

# ── Cache TTLs (seconds) ───────────────────────────────────────────────────────
PRICE_TTL_SECONDS = 86_400        # 1 trading day
FUNDAMENTALS_TTL_SECONDS = 604_800  # 7 days
ESTIMATES_TTL_SECONDS = 604_800
MACRO_TTL_SECONDS = 86_400
INFO_TTL_SECONDS = 86_400

# ── Data Fetch Params ──────────────────────────────────────────────────────────
PRICE_HISTORY_YEARS = 10
FINANCIALS_YEARS = 5
RETRY_MAX = 3
RETRY_INITIAL_DELAY = 1.0
RETRY_MAX_DELAY = 30.0

# ── Monte Carlo ────────────────────────────────────────────────────────────────
MC_NUM_PATHS = 10_000
MC_HORIZONS_MONTHS = [1, 3, 6, 12]

# ── Composite Score Weights (must sum to 1.0) ──────────────────────────────────
DEFAULT_WEIGHTS: dict[str, float] = {
    "valuation": 0.30,
    "fundamental": 0.30,
    "forecast": 0.15,
    "technical": 0.15,
    "risk": 0.10,
}

# ── Rating Bands (0-100 composite score) ──────────────────────────────────────
RATING_BANDS = [
    (80, 100, "Strong Buy"),
    (65,  79, "Buy"),
    (45,  64, "Hold"),
    (30,  44, "Sell"),
    (0,   29, "Strong Sell"),
]
CONFIDENCE_BAND_HALF_WIDTH = 15.0  # +/- this * (1 - confidence)

# ── Scoring Thresholds ─────────────────────────────────────────────────────────
# Maps raw metric values to 0-100 sub-scores via threshold bands.
# Each entry: list of (upper_bound, score) pairs, lowest score if below all.

ROIC_THRESHOLDS = [(0.05, 20), (0.10, 40), (0.15, 60), (0.20, 75), (0.25, 90), (1.0, 100)]
ROE_THRESHOLDS  = [(0.05, 20), (0.10, 40), (0.15, 60), (0.20, 75), (0.25, 90), (1.0, 100)]
NET_MARGIN_THRESHOLDS = [(-0.05, 10), (0.0, 25), (0.05, 45), (0.10, 60), (0.20, 80), (1.0, 100)]
REVENUE_CAGR_THRESHOLDS = [(-0.05, 10), (0.0, 30), (0.03, 50), (0.07, 65), (0.12, 80), (0.20, 95), (1.0, 100)]
FCF_CONVERSION_THRESHOLDS = [(0.0, 10), (0.3, 30), (0.6, 55), (0.8, 70), (1.0, 85), (1.5, 100)]
NET_DEBT_EBITDA_THRESHOLDS = [(0.0, 100), (1.0, 85), (2.0, 70), (3.0, 50), (5.0, 25), (100.0, 10)]  # lower is better
INTEREST_COVERAGE_THRESHOLDS = [(1.5, 10), (3.0, 30), (5.0, 55), (8.0, 75), (15.0, 90), (100.0, 100)]

# ── FRED Series IDs ────────────────────────────────────────────────────────────
FRED_SERIES = {
    "treasury_10y": "DGS10",
    "fed_funds": "FEDFUNDS",
}

# ── Sector ETF Map ─────────────────────────────────────────────────────────────
SECTOR_ETF_MAP: dict[str, str] = {
    "Technology": "XLK",
    "Health Care": "XLV",
    "Financials": "XLF",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Industrials": "XLI",
    "Energy": "XLE",
    "Materials": "XLB",
    "Real Estate": "XLRE",
    "Utilities": "XLU",
    "Communication Services": "XLC",
}

# ── Macro Index Tickers ────────────────────────────────────────────────────────
MACRO_INDICES = {
    "sp500": "^GSPC",
    "russell2000": "^RUT",
}

# ── Technical Analysis ─────────────────────────────────────────────────────────
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
BOLLINGER_WINDOW = 20
BOLLINGER_STD = 2
