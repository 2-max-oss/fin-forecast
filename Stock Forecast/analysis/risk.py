"""Risk analysis and position sizing module."""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from config import SECTOR_ETF_MAP
from core.types import (
    CompanyInfo, ForecastResult, ModuleScore, PriceHistory, RiskResult, safe_divide
)

logger = logging.getLogger(__name__)

TRADING_DAYS = 252


def _log_returns(price_series: pd.Series) -> pd.Series:
    return np.log(price_series / price_series.shift(1)).dropna()


def _realized_vol(returns: pd.Series, window: int) -> Optional[float]:
    """Annualized realized volatility."""
    if len(returns) < window:
        return None
    recent = returns.iloc[-window:]
    return float(recent.std() * np.sqrt(TRADING_DAYS))


def _compute_beta(stock_returns: pd.Series, market_returns: pd.Series) -> Optional[float]:
    """OLS beta of stock vs market."""
    aligned = pd.concat([stock_returns, market_returns], axis=1).dropna()
    if len(aligned) < 60:
        return None
    aligned.columns = ["stock", "market"]
    cov = aligned.cov().loc["stock", "market"]
    var = aligned["market"].var()
    return safe_divide(cov, var)


def _max_drawdown(price_series: pd.Series) -> Optional[float]:
    if len(price_series) < 2:
        return None
    rolling_max = price_series.cummax()
    drawdown = (price_series - rolling_max) / rolling_max
    return float(drawdown.min())


def _current_drawdown(price_series: pd.Series) -> Optional[float]:
    if price_series.empty:
        return None
    peak = price_series.cummax().iloc[-1]
    current = price_series.iloc[-1]
    return safe_divide(current - peak, peak)


def _sharpe(returns: pd.Series, risk_free_daily: float = 0.045 / 252) -> Optional[float]:
    if len(returns) < 30:
        return None
    excess = returns - risk_free_daily
    std = excess.std()
    if std == 0:
        return None
    return float(excess.mean() / std * np.sqrt(TRADING_DAYS))


def _sortino(returns: pd.Series, risk_free_daily: float = 0.045 / 252) -> Optional[float]:
    if len(returns) < 30:
        return None
    excess = returns - risk_free_daily
    downside = excess[excess < 0]
    if len(downside) < 5:
        return None
    downside_std = downside.std()
    if downside_std == 0:
        return None
    return float(excess.mean() / downside_std * np.sqrt(TRADING_DAYS))


def _downside_deviation(returns: pd.Series, target: float = 0.0) -> Optional[float]:
    if len(returns) < 30:
        return None
    below_target = returns[returns < target]
    if len(below_target) == 0:
        return 0.0
    return float(np.sqrt(np.mean(below_target ** 2)) * np.sqrt(TRADING_DAYS))


def _kelly_criterion(
    forecast: Optional[ForecastResult],
    risk_free_rate: float = 0.045,
) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """Return (full_kelly, quarter_kelly, edge) from 12m forecast distribution."""
    if forecast is None or 12 not in forecast.ensemble:
        return None, None, None

    dist = forecast.ensemble[12]
    expected_return = (dist.mean / dist.current_price - 1.0) if dist.current_price > 0 else None
    if expected_return is None:
        return None, None, None

    edge = expected_return - risk_free_rate
    vol_annual = (dist.p95 - dist.p5) / (2 * 1.645 * dist.current_price) if dist.current_price > 0 else 0.20
    vol_annual = max(vol_annual, 0.05)  # floor

    kelly = safe_divide(edge, vol_annual ** 2)
    if kelly is None:
        return None, None, edge
    kelly = max(-1.0, min(1.0, kelly))  # cap at +/-100%
    quarter_kelly = kelly * 0.25
    return kelly, quarter_kelly, edge


def _get_implied_vol(ticker: str) -> Optional[float]:
    """Attempt to get ATM implied vol from options chain."""
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        expirations = t.options
        if not expirations:
            return None
        # Use nearest expiration ~30 days out
        from datetime import datetime, timedelta
        target = datetime.utcnow() + timedelta(days=30)
        nearest = min(
            expirations,
            key=lambda d: abs((datetime.strptime(d, "%Y-%m-%d") - target).days)
        )
        chain = t.option_chain(nearest)
        if chain is None:
            return None
        current_price = t.fast_info.last_price
        calls = chain.calls
        if calls is None or calls.empty or current_price is None:
            return None
        # Find ATM strike
        calls["distance"] = abs(calls["strike"] - current_price)
        atm = calls.nsmallest(1, "distance")
        if atm.empty:
            return None
        iv = atm["impliedVolatility"].iloc[0]
        if iv and not np.isnan(iv) and 0.01 < iv < 5.0:
            return float(iv)
    except Exception as e:
        logger.debug("Options IV fetch failed: %s", e)
    return None


def _score_risk(
    vol_365d: Optional[float],
    beta: Optional[float],
    sharpe: Optional[float],
    max_dd_1y: Optional[float],
) -> tuple[float, dict[str, float]]:
    """Lower risk → higher score. Attractive risk-adjusted returns → higher score."""
    components: dict[str, float] = {}

    # Volatility: <20% = good(70), 20-30% = ok(50), 30-40% = elevated(35), >40% = high(20)
    if vol_365d is not None:
        if vol_365d < 0.20:
            components["volatility"] = 70.0
        elif vol_365d < 0.30:
            components["volatility"] = 55.0
        elif vol_365d < 0.40:
            components["volatility"] = 35.0
        else:
            components["volatility"] = 20.0

    # Beta: <0.8=defensive(65), 0.8-1.2=market(55), 1.2-1.8=aggressive(45), >1.8=high(30)
    if beta is not None:
        if beta < 0.8:
            components["beta"] = 65.0
        elif beta < 1.2:
            components["beta"] = 55.0
        elif beta < 1.8:
            components["beta"] = 45.0
        else:
            components["beta"] = 30.0

    # Sharpe: <0=bad(20), 0-0.5(40), 0.5-1.0(60), 1.0-2.0(80), >2.0(95)
    if sharpe is not None:
        if sharpe < 0:
            components["sharpe"] = 20.0
        elif sharpe < 0.5:
            components["sharpe"] = 40.0
        elif sharpe < 1.0:
            components["sharpe"] = 60.0
        elif sharpe < 2.0:
            components["sharpe"] = 80.0
        else:
            components["sharpe"] = 95.0

    # Max drawdown: >-50% = bad(15), -30 to -50(30), -20 to -30(50), -10 to -20(65), <-10(80)
    if max_dd_1y is not None:
        dd = max_dd_1y  # negative value
        if dd < -0.5:
            components["drawdown"] = 15.0
        elif dd < -0.3:
            components["drawdown"] = 30.0
        elif dd < -0.2:
            components["drawdown"] = 50.0
        elif dd < -0.1:
            components["drawdown"] = 65.0
        else:
            components["drawdown"] = 80.0

    if not components:
        return 50.0, components
    return float(np.mean(list(components.values()))), components


def analyze_risk(
    price_history: PriceHistory,
    info: CompanyInfo,
    forecast: Optional[ForecastResult] = None,
    risk_free_rate: float = 0.045,
    spy_prices: Optional[pd.Series] = None,
    sector_prices: Optional[pd.Series] = None,
) -> RiskResult:
    warnings: list[str] = []
    df = price_history.df
    col = "adj_close" if "adj_close" in df.columns else "close"
    price = df[col].dropna()

    if price.empty:
        warnings.append("No price data available for risk analysis")
        return RiskResult(
            score=ModuleScore(name="risk", score=50.0, confidence=0.0, warnings=warnings)
        )

    returns = _log_returns(price)
    returns_1y = returns.last("252B") if len(returns) >= 252 else returns
    price_1y   = price.last("252B")   if len(price) >= 252   else price
    price_3y   = price.last("756B")   if len(price) >= 756   else price

    # ── Volatility ─────────────────────────────────────────────────────────────
    vol_30d  = _realized_vol(returns, 30)
    vol_90d  = _realized_vol(returns, 90)
    vol_365d = _realized_vol(returns, 252)

    # ── Beta ──────────────────────────────────────────────────────────────────
    beta_spy: Optional[float] = None
    if spy_prices is not None:
        spy_ret = _log_returns(spy_prices)
        beta_spy = _compute_beta(returns, spy_ret)
        if beta_spy is None:
            warnings.append("Beta vs SPY: insufficient aligned history")
    else:
        warnings.append("SPY price data unavailable — beta not computed")

    # Sector beta
    beta_sector: Optional[float] = None
    sector_etf = SECTOR_ETF_MAP.get(info.sector)
    if sector_prices is not None and sector_etf:
        sect_ret = _log_returns(sector_prices)
        beta_sector = _compute_beta(returns, sect_ret)

    # ── Drawdown ──────────────────────────────────────────────────────────────
    max_dd_1y  = _max_drawdown(price_1y)
    max_dd_3y  = _max_drawdown(price_3y)
    curr_dd    = _current_drawdown(price)

    # ── Ratios ────────────────────────────────────────────────────────────────
    rfd = risk_free_rate / TRADING_DAYS
    sharpe_1y  = _sharpe(returns_1y, rfd)
    sortino_1y = _sortino(returns_1y, rfd)
    downside_dev = _downside_deviation(returns_1y)

    # ── Kelly ─────────────────────────────────────────────────────────────────
    kelly, quarter_kelly, edge = _kelly_criterion(forecast, risk_free_rate)

    # ── Implied vol ───────────────────────────────────────────────────────────
    impl_vol = _get_implied_vol(price_history.ticker)

    # ── Score ─────────────────────────────────────────────────────────────────
    available_count = sum(1 for v in [vol_365d, beta_spy, sharpe_1y, max_dd_1y] if v is not None)
    confidence = min(1.0, available_count / 4)

    overall_score, components = _score_risk(vol_365d, beta_spy, sharpe_1y, max_dd_1y)

    module_score = ModuleScore(
        name="risk",
        score=round(overall_score, 1),
        confidence=round(confidence, 3),
        components=components,
        warnings=warnings,
    )

    return RiskResult(
        score=module_score,
        realized_vol_30d=vol_30d,
        realized_vol_90d=vol_90d,
        realized_vol_365d=vol_365d,
        beta_spy=beta_spy,
        beta_sector=beta_sector,
        sector_etf=sector_etf,
        max_drawdown_1y=max_dd_1y,
        max_drawdown_3y=max_dd_3y,
        current_drawdown=curr_dd,
        sharpe_1y=sharpe_1y,
        sortino_1y=sortino_1y,
        downside_deviation_1y=downside_dev,
        kelly_fraction=kelly,
        quarter_kelly=quarter_kelly,
        kelly_edge=edge,
        implied_vol_atm=impl_vol,
    )
