"""yfinance data provider implementation."""
from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

from core.exceptions import DataFetchError, DataValidationError
from core.types import (
    AnalystEstimates,
    CompanyInfo,
    FinancialStatement,
    Financials,
    PeriodType,
    PriceHistory,
)
from data_layer.cache import CacheManager
from data_layer.retry import with_retry

logger = logging.getLogger(__name__)


def _normalize_df_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names to lower-snake-case."""
    df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]
    return df


def _normalize_price_df(raw: pd.DataFrame) -> pd.DataFrame:
    """Standardize yfinance OHLCV DataFrame."""
    df = raw.copy()
    # Flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

    # Rename 'adj_close' variants
    rename = {}
    for col in df.columns:
        if "adj" in col and "close" in col:
            rename[col] = "adj_close"
        elif col == "close":
            rename[col] = "close"
    df = df.rename(columns=rename)

    if "adj_close" not in df.columns and "close" in df.columns:
        df["adj_close"] = df["close"]

    # Normalize timezone
    if hasattr(df.index, "tz") and df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    df = df.sort_index()
    return df


def _validate_price_df(df: pd.DataFrame, ticker: str) -> None:
    """Basic quality checks on price data."""
    if df is None or df.empty:
        raise DataValidationError(f"Empty price data for {ticker}")
    col = "adj_close" if "adj_close" in df.columns else "close"
    if (df[col] <= 0).any():
        logger.warning("Non-positive prices detected for %s — may indicate stale/bad data", ticker)
    if not df.index.is_monotonic_increasing:
        raise DataValidationError(f"Price index is not monotonically increasing for {ticker}")


def _normalize_financial_stmt(df: pd.DataFrame, period_type: PeriodType, ticker: str) -> FinancialStatement:
    """Transpose yfinance financial statement into (date_index x item_columns) form."""
    if df is None or df.empty:
        return FinancialStatement(ticker=ticker, period_type=period_type, df=pd.DataFrame())

    # yfinance returns items as rows, dates as columns — transpose
    result = df.T.copy()
    # Ensure date index
    result.index = pd.to_datetime(result.index, errors="coerce")
    result = result[result.index.notna()]
    result = result.sort_index(ascending=False)
    # Normalize column names
    result.columns = [str(c).lower().strip().replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "") for c in result.columns]
    # Cast to float
    for col in result.columns:
        result[col] = pd.to_numeric(result[col], errors="coerce")
    return FinancialStatement(ticker=ticker, period_type=period_type, df=result)


class YFinanceProvider:
    """Fetches and caches equity data via yfinance."""

    def __init__(self, cache: CacheManager | None = None):
        self._cache = cache or CacheManager.get()

    # ── Price ──────────────────────────────────────────────────────────────────
    def get_price_history(self, ticker: str, years: int = 10) -> PriceHistory:
        cached = self._cache.get_price(ticker)
        if cached is not None:
            logger.debug("Price cache hit: %s", ticker)
            return PriceHistory(ticker=ticker, df=cached)

        return self._fetch_price(ticker, years)

    @with_retry()
    def _fetch_price(self, ticker: str, years: int) -> PriceHistory:
        start = (datetime.utcnow() - timedelta(days=years * 365 + 30)).strftime("%Y-%m-%d")
        t = yf.Ticker(ticker)
        raw = t.history(start=start, auto_adjust=True, actions=True)

        if raw is None or raw.empty:
            raise DataFetchError(f"No price data returned for {ticker}")

        df = _normalize_price_df(raw)
        _validate_price_df(df, ticker)

        # Sanity check: compare latest close to market price from info
        try:
            info = t.fast_info
            market_price = getattr(info, "last_price", None)
            if market_price and abs(df["adj_close"].iloc[-1] / market_price - 1) > 0.5:
                logger.warning(
                    "%s: cached close %.2f vs market price %.2f — forcing re-download",
                    ticker, df["adj_close"].iloc[-1], market_price,
                )
                raw = t.history(start=start, auto_adjust=True, actions=True)
                df = _normalize_price_df(raw)
        except Exception:
            pass

        self._cache.set_price(ticker, df)
        time.sleep(0.3)  # Gentle rate limiting
        return PriceHistory(ticker=ticker, df=df)

    # ── Financials ─────────────────────────────────────────────────────────────
    def get_financials(self, ticker: str) -> Financials:
        cached = self._cache.get_financials(ticker)
        if cached is not None:
            logger.debug("Fundamentals cache hit: %s", ticker)
            return self._dict_to_financials(ticker, cached)

        return self._fetch_financials(ticker)

    @with_retry()
    def _fetch_financials(self, ticker: str) -> Financials:
        t = yf.Ticker(ticker)
        time.sleep(0.5)

        stmts = {
            "income_annual":       t.income_stmt,
            "income_quarterly":    t.quarterly_income_stmt,
            "balance_annual":      t.balance_sheet,
            "balance_quarterly":   t.quarterly_balance_sheet,
            "cashflow_annual":     t.cashflow,
            "cashflow_quarterly":  t.quarterly_cashflow,
        }

        # Cache raw DataFrames as financials dict
        cache_payload: dict[str, pd.DataFrame] = {}
        for name, df in stmts.items():
            if df is not None and not df.empty:
                transposed = df.T.copy()
                transposed.index = pd.to_datetime(transposed.index, errors="coerce")
                transposed = transposed[transposed.index.notna()].sort_index(ascending=False)
                transposed.columns = [
                    str(c).lower().strip().replace(" ", "_").replace("/", "_")
                    .replace("(", "").replace(")", "")
                    for c in transposed.columns
                ]
                for col in transposed.columns:
                    transposed[col] = pd.to_numeric(transposed[col], errors="coerce")
                cache_payload[name] = transposed

        self._cache.set_financials(ticker, cache_payload)

        result = Financials(
            ticker=ticker,
            income_stmt=_normalize_financial_stmt(stmts.get("income_annual"), PeriodType.ANNUAL, ticker),
            income_stmt_quarterly=_normalize_financial_stmt(stmts.get("income_quarterly"), PeriodType.QUARTERLY, ticker),
            balance_sheet=_normalize_financial_stmt(stmts.get("balance_annual"), PeriodType.ANNUAL, ticker),
            balance_sheet_quarterly=_normalize_financial_stmt(stmts.get("balance_quarterly"), PeriodType.QUARTERLY, ticker),
            cashflow=_normalize_financial_stmt(stmts.get("cashflow_annual"), PeriodType.ANNUAL, ticker),
            cashflow_quarterly=_normalize_financial_stmt(stmts.get("cashflow_quarterly"), PeriodType.QUARTERLY, ticker),
        )
        time.sleep(0.3)
        return result

    def _dict_to_financials(self, ticker: str, cached: dict[str, pd.DataFrame]) -> Financials:
        def _get(key: str, period_type: PeriodType) -> FinancialStatement:
            df = cached.get(key, pd.DataFrame())
            return FinancialStatement(ticker=ticker, period_type=period_type, df=df)

        return Financials(
            ticker=ticker,
            income_stmt=_get("income_annual", PeriodType.ANNUAL),
            income_stmt_quarterly=_get("income_quarterly", PeriodType.QUARTERLY),
            balance_sheet=_get("balance_annual", PeriodType.ANNUAL),
            balance_sheet_quarterly=_get("balance_quarterly", PeriodType.QUARTERLY),
            cashflow=_get("cashflow_annual", PeriodType.ANNUAL),
            cashflow_quarterly=_get("cashflow_quarterly", PeriodType.QUARTERLY),
        )

    # ── Company Info ───────────────────────────────────────────────────────────
    def get_info(self, ticker: str) -> CompanyInfo:
        cached = self._cache.get_info(ticker)
        if cached:
            return CompanyInfo(
                ticker=ticker,
                name=cached.get("name") or ticker,
                sector=cached.get("sector") or "",
                industry=cached.get("industry") or "",
                market_cap=cached.get("market_cap") or 0.0,
                shares_outstanding=cached.get("shares_outstanding") or 0.0,
                dividend_yield=cached.get("dividend_yield"),
                enterprise_value=cached.get("enterprise_value"),
                currency=cached.get("currency") or "USD",
                exchange=cached.get("exchange") or "",
                description=cached.get("description") or "",
                website=cached.get("website") or "",
                country=cached.get("country") or "",
                employees=cached.get("employees"),
            )
        return self._fetch_info(ticker)

    @with_retry()
    def _fetch_info(self, ticker: str) -> CompanyInfo:
        t = yf.Ticker(ticker)
        info: dict = {}
        try:
            info = t.info or {}
        except Exception as e:
            logger.warning("Could not fetch info for %s: %s", ticker, e)

        def g(key, default=None):
            val = info.get(key, default)
            if val in (None, "N/A", ""):
                return default
            return val

        payload = {
            "name": g("longName") or g("shortName") or ticker,
            "sector": g("sector", ""),
            "industry": g("industry", ""),
            "market_cap": g("marketCap", 0.0),
            "shares_outstanding": g("sharesOutstanding", 0.0),
            "enterprise_value": g("enterpriseValue"),
            "dividend_yield": g("dividendYield"),
            "currency": g("currency", "USD"),
            "exchange": g("exchange", ""),
            "description": g("longBusinessSummary", ""),
            "website": g("website", ""),
            "country": g("country", ""),
            "employees": g("fullTimeEmployees"),
        }
        self._cache.set_info(ticker, payload)
        time.sleep(0.3)

        return CompanyInfo(ticker=ticker, **payload)

    # ── Analyst Estimates ──────────────────────────────────────────────────────
    def get_estimates(self, ticker: str) -> Optional[AnalystEstimates]:
        cached = self._cache.get_estimates(ticker)
        if cached:
            return AnalystEstimates(
                ticker=ticker,
                forward_revenue=cached.get("forward_revenue"),
                forward_ebitda=cached.get("forward_ebitda"),
                forward_eps=cached.get("forward_eps"),
                forward_pe=cached.get("forward_pe"),
                target_mean_price=cached.get("target_mean_price"),
                target_median_price=cached.get("target_median_price"),
                recommendation=cached.get("recommendation"),
                num_analysts=cached.get("num_analysts"),
            )
        return self._fetch_estimates(ticker)

    @with_retry()
    def _fetch_estimates(self, ticker: str) -> Optional[AnalystEstimates]:
        t = yf.Ticker(ticker)
        info: dict = {}
        try:
            info = t.info or {}
        except Exception:
            pass

        def g(key):
            val = info.get(key)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                return None
            return val

        payload = {
            "forward_revenue":     g("totalRevenue"),
            "forward_ebitda":      g("ebitda"),
            "forward_eps":         g("forwardEps"),
            "forward_pe":          g("forwardPE"),
            "target_mean_price":   g("targetMeanPrice"),
            "target_median_price": g("targetMedianPrice"),
            "recommendation":      g("recommendationKey"),
            "num_analysts":        g("numberOfAnalystOpinions"),
        }
        self._cache.set_estimates(ticker, payload)
        time.sleep(0.3)

        if all(v is None for v in payload.values()):
            return None
        return AnalystEstimates(ticker=ticker, **payload)
