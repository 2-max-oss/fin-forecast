"""FRED macro data provider."""
from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from config import FRED_API_KEY, FRED_SERIES, MACRO_INDICES
from core.exceptions import ConfigurationError, DataFetchError
from data_layer.cache import CacheManager
from data_layer.retry import with_retry

logger = logging.getLogger(__name__)


class FredProvider:
    def __init__(self, cache: CacheManager | None = None):
        self._cache = cache or CacheManager.get()
        self._fred = None

    def _get_fred(self):
        if self._fred is None:
            if not FRED_API_KEY or FRED_API_KEY == "your_fred_api_key_here":
                raise ConfigurationError(
                    "FRED_API_KEY not set. Get a free key at https://fred.stlouisfed.org/docs/api/api_key.html "
                    "and add it to your .env file."
                )
            try:
                from fredapi import Fred
                self._fred = Fred(api_key=FRED_API_KEY)
            except ImportError:
                raise ConfigurationError("fredapi package not installed. Run: pip install fredapi")
        return self._fred

    def get_series(self, series_id: str) -> pd.Series:
        cached = self._cache.get_macro(series_id)
        if cached is not None:
            return cached
        return self._fetch_series(series_id)

    @with_retry()
    def _fetch_series(self, series_id: str) -> pd.Series:
        fred = self._get_fred()
        series = fred.get_series(series_id)
        if series is None or series.empty:
            raise DataFetchError(f"Empty data for FRED series {series_id}")
        series = series.dropna()
        if hasattr(series.index, "tz") and series.index.tz is not None:
            series.index = series.index.tz_localize(None)
        self._cache.set_macro(series_id, series)
        return series

    def get_treasury_10y(self) -> pd.Series:
        return self.get_series(FRED_SERIES["treasury_10y"])

    def get_fed_funds_rate(self) -> pd.Series:
        return self.get_series(FRED_SERIES["fed_funds"])

    def get_risk_free_rate(self) -> float:
        """Latest 10-year Treasury yield as a decimal."""
        if not FRED_API_KEY or FRED_API_KEY == "your_fred_api_key_here":
            logger.info("FRED_API_KEY not configured — using 4.5%% default risk-free rate")
            return 0.045
        try:
            series = self.get_treasury_10y()
            latest = float(series.dropna().iloc[-1])
            return latest / 100.0
        except Exception as e:
            logger.warning("Could not fetch risk-free rate: %s — using 4.5%% default", e)
            return 0.045


class MacroContext:
    """Aggregates macro data needed by analysis modules."""

    def __init__(self, fred: FredProvider | None = None):
        self._fred = fred or FredProvider()

    def get_risk_free_rate(self) -> float:
        return self._fred.get_risk_free_rate()

    def get_sp500_history(self) -> Optional[pd.Series]:
        """Fetch SPY price history for beta calculation (via yfinance)."""
        try:
            from data_layer.yfinance_provider import YFinanceProvider
            provider = YFinanceProvider()
            history = provider.get_price_history("SPY", years=5)
            col = "adj_close" if "adj_close" in history.df.columns else "close"
            return history.df[col]
        except Exception as e:
            logger.warning("Could not fetch SPY history: %s", e)
            return None

    def get_sector_etf_history(self, etf_ticker: str) -> Optional[pd.Series]:
        try:
            from data_layer.yfinance_provider import YFinanceProvider
            provider = YFinanceProvider()
            history = provider.get_price_history(etf_ticker, years=3)
            col = "adj_close" if "adj_close" in history.df.columns else "close"
            return history.df[col]
        except Exception as e:
            logger.warning("Could not fetch %s history: %s", etf_ticker, e)
            return None
