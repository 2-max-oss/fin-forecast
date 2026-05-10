"""Service layer for the browser web app.

This module keeps the HTTP layer thin and reuses the same model code as the
Streamlit app.
"""
from __future__ import annotations

import math
import sys
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.forecasting import analyze_forecast
from analysis.fundamental import analyze_fundamentals
from analysis.risk import analyze_risk
from analysis.technical import analyze_technical
from analysis.valuation import analyze_valuation
from config import SECTOR_ETF_MAP
from core.scoring import CompositeScorer
from core.types import CompanyInfo, CompositeResult, ModuleScore
from data_layer.fred_provider import MacroContext
from data_layer.yfinance_provider import YFinanceProvider


class AnalysisError(RuntimeError):
    """Raised when a ticker cannot produce any useful analysis."""


def _clean(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (datetime, date, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return _clean(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if is_dataclass(value):
        return _clean(asdict(value))
    return value


def _score_dict(score: ModuleScore | None) -> dict[str, Any] | None:
    if score is None:
        return None
    return _clean(
        {
            "name": score.name,
            "score": score.score,
            "confidence": score.confidence,
            "components": score.components,
            "warnings": score.warnings,
        }
    )


def _company_dict(info: CompanyInfo) -> dict[str, Any]:
    return _clean(
        {
            "ticker": info.ticker,
            "name": info.name,
            "sector": info.sector,
            "industry": info.industry,
            "marketCap": info.market_cap,
            "sharesOutstanding": info.shares_outstanding,
            "dividendYield": info.dividend_yield,
            "enterpriseValue": info.enterprise_value,
            "currency": info.currency,
            "exchange": info.exchange,
            "country": info.country,
            "employees": info.employees,
            "website": info.website,
            "description": info.description,
        }
    )


def _composite_dict(composite: CompositeResult) -> dict[str, Any]:
    return _clean(
        {
            "overallScore": composite.overall_score,
            "confidence": composite.confidence,
            "rating": composite.rating.value,
            "confidenceBand": composite.confidence_band,
            "possibleRatings": [r.value for r in composite.possible_ratings],
            "weightsUsed": composite.weights_used,
            "componentScores": composite.component_scores,
            "modules": [_score_dict(ms) for ms in composite.module_scores],
            "computedAt": composite.computed_at,
        }
    )


def _fundamental_dict(result: Any) -> dict[str, Any]:
    return _clean(
        {
            "score": _score_dict(result.score),
            "metrics": {
                "revenueCagr5y": result.revenue_cagr_5y,
                "epsCagr5y": result.eps_cagr_5y,
                "fcfGrowthYoy": result.fcf_growth_yoy,
                "grossMargin": result.gross_margin,
                "operatingMargin": result.operating_margin,
                "netMargin": result.net_margin,
                "fcfMargin": result.fcf_margin,
                "roic": result.roic,
                "roe": result.roe,
                "roa": result.roa,
                "netDebtToEbitda": result.net_debt_to_ebitda,
                "interestCoverage": result.interest_coverage,
                "currentRatio": result.current_ratio,
                "quickRatio": result.quick_ratio,
                "fcfConversion": result.fcf_conversion,
            },
        }
    )


def _multiple_dict(snapshot: Any) -> dict[str, Any] | None:
    if snapshot is None:
        return None
    return _clean(
        {
            "name": snapshot.name,
            "current": snapshot.current,
            "historical5yLow": snapshot.historical_5y_low,
            "historical5yMedian": snapshot.historical_5y_median,
            "historical5yHigh": snapshot.historical_5y_high,
            "percentile": snapshot.percentile,
        }
    )


def _valuation_dict(result: Any) -> dict[str, Any]:
    return _clean(
        {
            "score": _score_dict(result.score),
            "currentPrice": result.current_price,
            "fairValueLow": result.fair_value_low,
            "fairValueMid": result.fair_value_mid,
            "fairValueHigh": result.fair_value_high,
            "upsidePct": result.upside_pct,
            "multiples": [
                m
                for m in [
                    _multiple_dict(result.pe_trailing),
                    _multiple_dict(result.pe_forward),
                    _multiple_dict(result.ev_ebitda),
                    _multiple_dict(result.ev_sales),
                    _multiple_dict(result.pb),
                    _multiple_dict(result.fcf_yield),
                ]
                if m is not None
            ],
            "dcf": {
                "fairValue": result.dcf.fair_value,
                "assumptions": result.dcf.assumptions,
            }
            if result.dcf
            else None,
            "reverseDcf": {
                "impliedGrowthRate": result.reverse_dcf.implied_growth_rate,
                "currentPrice": result.reverse_dcf.current_price,
                "consensusGrowthRate": result.reverse_dcf.consensus_growth_rate,
                "historicalGrowthRate": result.reverse_dcf.historical_growth_rate,
            }
            if result.reverse_dcf
            else None,
        }
    )


def _technical_dict(result: Any) -> dict[str, Any]:
    return _clean(
        {
            "score": _score_dict(result.score),
            "sma20": result.sma_20,
            "sma50": result.sma_50,
            "sma200": result.sma_200,
            "ema20": result.ema_20,
            "ema50": result.ema_50,
            "ema200": result.ema_200,
            "maSignals": result.ma_signals,
            "rsi14": result.rsi_14,
            "macdLine": result.macd_line,
            "macdSignal": result.macd_signal,
            "macdHistogram": result.macd_histogram,
            "bollingerUpper": result.bollinger_upper,
            "bollingerLower": result.bollinger_lower,
            "bollingerPctB": result.bollinger_pct_b,
            "supportLevels": result.support_levels,
            "resistanceLevels": result.resistance_levels,
            "vwap": result.vwap,
            "volumeVsAvg20d": result.volume_vs_avg_20d,
            "abnormalVolume": result.abnormal_volume,
        }
    )


def _dist_dict(dist: Any) -> dict[str, Any]:
    return _clean(
        {
            "horizonMonths": dist.horizon_months,
            "mean": dist.mean,
            "median": dist.median,
            "p5": dist.p5,
            "p10": dist.p10,
            "p25": dist.p25,
            "p75": dist.p75,
            "p90": dist.p90,
            "p95": dist.p95,
            "probPositive": dist.prob_positive,
            "currentPrice": dist.current_price,
        }
    )


def _forecast_dict(result: Any) -> dict[str, Any]:
    return _clean(
        {
            "score": _score_dict(result.score),
            "ensemble": [_dist_dict(v) for _, v in sorted(result.ensemble.items())],
            "monteCarlo": [_dist_dict(v) for _, v in sorted(result.monte_carlo.items())],
            "riskNeutral": [_dist_dict(v) for _, v in sorted(result.risk_neutral.items())],
            "mlDirectional": result.ml_directional,
            "mathFinance": result.math_finance,
            "backtests": [
                {
                    "method": bt.method,
                    "mae": bt.mae,
                    "rmse": bt.rmse,
                    "directionalAccuracy": bt.directional_accuracy,
                    "naiveBaselineMae": bt.naive_baseline_mae,
                    "beatsNaive": bt.beats_naive,
                }
                for bt in result.backtests
            ],
        }
    )


def _risk_dict(result: Any) -> dict[str, Any]:
    return _clean(
        {
            "score": _score_dict(result.score),
            "realizedVol30d": result.realized_vol_30d,
            "realizedVol90d": result.realized_vol_90d,
            "realizedVol365d": result.realized_vol_365d,
            "betaSpy": result.beta_spy,
            "betaSector": result.beta_sector,
            "sectorEtf": result.sector_etf,
            "maxDrawdown1y": result.max_drawdown_1y,
            "maxDrawdown3y": result.max_drawdown_3y,
            "currentDrawdown": result.current_drawdown,
            "sharpe1y": result.sharpe_1y,
            "sortino1y": result.sortino_1y,
            "downsideDeviation1y": result.downside_deviation_1y,
            "kellyFraction": result.kelly_fraction,
            "quarterKelly": result.quarter_kelly,
            "kellyEdge": result.kelly_edge,
            "impliedVolAtm": result.implied_vol_atm,
        }
    )


def _run_module(name: str, errors: dict[str, str], fn: Any) -> Any | None:
    try:
        return fn()
    except Exception as exc:
        errors[name] = str(exc)
        return None


def analyze_ticker(ticker: str) -> dict[str, Any]:
    ticker = ticker.strip().upper()
    if not ticker or len(ticker) > 12:
        raise AnalysisError("Enter a valid ticker symbol.")

    provider = YFinanceProvider()
    macro = MacroContext()
    errors: dict[str, str] = {}

    try:
        info = provider.get_info(ticker)
        prices = provider.get_price_history(ticker, years=10)
        financials = provider.get_financials(ticker)
        estimates = provider.get_estimates(ticker)
    except Exception as exc:
        raise AnalysisError(f"Data fetch failed for {ticker}: {exc}") from exc

    risk_free_rate = macro.get_risk_free_rate()
    spy_prices = macro.get_sp500_history()
    sector_ticker = SECTOR_ETF_MAP.get(info.sector)
    sector_prices = macro.get_sector_etf_history(sector_ticker) if sector_ticker else None

    fundamental = _run_module(
        "fundamental",
        errors,
        lambda: analyze_fundamentals(financials, info),
    )
    valuation = _run_module(
        "valuation",
        errors,
        lambda: analyze_valuation(prices, financials, info, estimates, risk_free_rate),
    )
    technical = _run_module("technical", errors, lambda: analyze_technical(prices))
    forecast = _run_module(
        "forecast",
        errors,
        lambda: analyze_forecast(
            prices,
            risk_free_rate=risk_free_rate,
            dividend_yield=info.dividend_yield or 0.0,
        ),
    )
    risk = _run_module(
        "risk",
        errors,
        lambda: analyze_risk(
            prices,
            info,
            forecast,
            risk_free_rate,
            spy_prices,
            sector_prices,
        ),
    )

    module_results = [fundamental, valuation, technical, forecast, risk]
    module_scores = [r.score for r in module_results if r is not None]
    if not module_scores:
        raise AnalysisError("All analysis modules failed.")

    composite = CompositeScorer().compute(module_scores)
    price_col = "adj_close" if "adj_close" in prices.df.columns else "close"

    return _clean(
        {
            "ticker": ticker,
            "company": _company_dict(info),
            "currentPrice": prices.latest_close,
            "priceDate": prices.df.index[-1],
            "macro": {
                "riskFreeRate": risk_free_rate,
                "sectorEtf": sector_ticker,
            },
            "composite": _composite_dict(composite),
            "modules": {
                "fundamental": _fundamental_dict(fundamental) if fundamental else None,
                "valuation": _valuation_dict(valuation) if valuation else None,
                "technical": _technical_dict(technical) if technical else None,
                "forecast": _forecast_dict(forecast) if forecast else None,
                "risk": _risk_dict(risk) if risk else None,
            },
            "errors": errors,
            "history": {
                "latest": [
                    {
                        "date": d.isoformat(),
                        "price": float(v) if math.isfinite(float(v)) else None,
                    }
                    for d, v in prices.df[price_col].dropna().tail(260).items()
                ]
            },
        }
    )

