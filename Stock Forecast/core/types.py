"""All shared dataclasses, enums, and type aliases."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd


# ── Enums ──────────────────────────────────────────────────────────────────────

class Rating(Enum):
    STRONG_BUY  = "Strong Buy"
    BUY         = "Buy"
    HOLD        = "Hold"
    SELL        = "Sell"
    STRONG_SELL = "Strong Sell"


class PeriodType(Enum):
    QUARTERLY = "quarterly"
    ANNUAL    = "annual"


# ── Utility helpers ────────────────────────────────────────────────────────────

def safe_divide(
    numerator: Optional[float], denominator: Optional[float]
) -> Optional[float]:
    if numerator is None or denominator is None or denominator == 0:
        return None
    try:
        result = numerator / denominator
        return None if (np.isnan(result) or np.isinf(result)) else result
    except Exception:
        return None


def safe_pct_change(
    new: Optional[float], old: Optional[float]
) -> Optional[float]:
    if new is None or old is None:
        return None
    return safe_divide(new - old, old)


def threshold_score(value: Optional[float], thresholds: list[tuple[float, int]]) -> Optional[float]:
    """Map a raw metric value to a 0-100 score via threshold bands."""
    if value is None or np.isnan(value):
        return None
    for upper, score in thresholds:
        if value <= upper:
            return float(score)
    return float(thresholds[-1][1])


# ── Data Layer Types ───────────────────────────────────────────────────────────

@dataclass
class PriceHistory:
    """Daily OHLCV data (adjusted)."""
    ticker: str
    df: pd.DataFrame          # Index: DatetimeIndex; cols: open,high,low,close,adj_close,volume
    currency: str = "USD"
    fetched_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def latest_close(self) -> float:
        col = "adj_close" if "adj_close" in self.df.columns else "close"
        return float(self.df[col].iloc[-1])

    @property
    def years_available(self) -> float:
        if len(self.df) < 2:
            return 0.0
        span = self.df.index[-1] - self.df.index[0]
        return span.days / 365.25


@dataclass
class FinancialStatement:
    ticker: str
    period_type: PeriodType
    df: pd.DataFrame          # Index: period_end (Timestamp); columns: line items


@dataclass
class Financials:
    ticker: str
    income_stmt: FinancialStatement
    income_stmt_quarterly: FinancialStatement
    balance_sheet: FinancialStatement
    balance_sheet_quarterly: FinancialStatement
    cashflow: FinancialStatement
    cashflow_quarterly: FinancialStatement


@dataclass
class AnalystEstimates:
    ticker: str
    forward_revenue: Optional[float] = None
    forward_ebitda: Optional[float] = None
    forward_eps: Optional[float] = None
    forward_pe: Optional[float] = None
    target_mean_price: Optional[float] = None
    target_median_price: Optional[float] = None
    recommendation: Optional[str] = None
    num_analysts: Optional[int] = None


@dataclass
class CompanyInfo:
    ticker: str
    name: str
    sector: str
    industry: str
    market_cap: float
    shares_outstanding: float
    dividend_yield: Optional[float] = None
    enterprise_value: Optional[float] = None
    currency: str = "USD"
    exchange: str = ""
    description: str = ""
    website: str = ""
    country: str = ""
    employees: Optional[int] = None


# ── Module Score ───────────────────────────────────────────────────────────────

@dataclass
class ModuleScore:
    name: str
    score: float                           # 0-100
    confidence: float                      # 0.0-1.0
    components: dict[str, float] = field(default_factory=dict)   # sub-scores 0-100
    warnings: list[str] = field(default_factory=list)
    computed_at: datetime = field(default_factory=datetime.utcnow)


# ── Analysis Module Result Types ───────────────────────────────────────────────

@dataclass
class FundamentalResult:
    score: ModuleScore
    # Growth
    revenue_cagr_5y: Optional[float] = None
    eps_cagr_5y: Optional[float] = None
    fcf_growth_yoy: Optional[float] = None
    # Profitability
    gross_margin: Optional[float] = None
    operating_margin: Optional[float] = None
    net_margin: Optional[float] = None
    fcf_margin: Optional[float] = None
    # Efficiency
    roic: Optional[float] = None
    roe: Optional[float] = None
    roa: Optional[float] = None
    asset_turnover: Optional[float] = None
    # Balance sheet
    net_debt_to_ebitda: Optional[float] = None
    interest_coverage: Optional[float] = None
    current_ratio: Optional[float] = None
    quick_ratio: Optional[float] = None
    # Quality
    fcf_conversion: Optional[float] = None
    accruals_ratio: Optional[float] = None
    earnings_stability: Optional[float] = None
    # Trend data for charts
    margin_history: Optional[pd.DataFrame] = None   # index=year, cols=[gross,operating,net,fcf]
    growth_history: Optional[pd.DataFrame] = None   # index=year, cols=[revenue,eps,fcf]
    roic_history: Optional[pd.Series] = None


@dataclass
class MultiplesSnapshot:
    name: str
    current: float
    historical_5y_low: Optional[float] = None
    historical_5y_high: Optional[float] = None
    historical_5y_median: Optional[float] = None
    percentile: Optional[float] = None     # 0-100 where current falls in 5y range


@dataclass
class DCFResult:
    fair_value: float
    sensitivity_table: pd.DataFrame        # index=WACC values, cols=terminal_growth values
    assumptions: dict[str, float] = field(default_factory=dict)


@dataclass
class ReverseDCFResult:
    implied_growth_rate: float
    current_price: float
    consensus_growth_rate: Optional[float] = None
    historical_growth_rate: Optional[float] = None
    assumptions: dict[str, float] = field(default_factory=dict)


@dataclass
class ValuationResult:
    score: ModuleScore
    pe_trailing: Optional[MultiplesSnapshot] = None
    pe_forward: Optional[MultiplesSnapshot] = None
    ev_ebitda: Optional[MultiplesSnapshot] = None
    ev_sales: Optional[MultiplesSnapshot] = None
    pb: Optional[MultiplesSnapshot] = None
    fcf_yield: Optional[MultiplesSnapshot] = None
    dcf: Optional[DCFResult] = None
    reverse_dcf: Optional[ReverseDCFResult] = None
    fair_value_low: Optional[float] = None
    fair_value_mid: Optional[float] = None
    fair_value_high: Optional[float] = None
    current_price: float = 0.0
    upside_pct: Optional[float] = None


@dataclass
class TechnicalResult:
    score: ModuleScore
    sma_20: Optional[float] = None
    sma_50: Optional[float] = None
    sma_200: Optional[float] = None
    ema_20: Optional[float] = None
    ema_50: Optional[float] = None
    ema_200: Optional[float] = None
    ma_signals: dict[str, str] = field(default_factory=dict)
    rsi_14: Optional[float] = None
    macd_line: Optional[float] = None
    macd_signal: Optional[float] = None
    macd_histogram: Optional[float] = None
    bollinger_upper: Optional[float] = None
    bollinger_lower: Optional[float] = None
    bollinger_pct_b: Optional[float] = None
    support_levels: list[float] = field(default_factory=list)
    resistance_levels: list[float] = field(default_factory=list)
    vwap: Optional[float] = None
    volume_vs_avg_20d: Optional[float] = None
    abnormal_volume: bool = False
    indicator_df: Optional[pd.DataFrame] = None


@dataclass
class ForecastDistribution:
    horizon_months: int
    mean: float
    median: float
    p5: float
    p10: float
    p25: float
    p75: float
    p90: float
    p95: float
    prob_positive: float                   # P(return > 0)
    current_price: float = 0.0


@dataclass
class MathematicalFinanceDiagnostics:
    scheme: str = "exact_lognormal"
    physical_drift: Optional[float] = None
    risk_neutral_drift: Optional[float] = None
    annualized_volatility: Optional[float] = None
    adjusted_volatility: Optional[float] = None
    volatility_regime_factor: Optional[float] = None
    risk_free_rate: Optional[float] = None
    dividend_yield: Optional[float] = None
    martingale_error_12m: Optional[float] = None
    antithetic_variates: bool = False
    notes: list[str] = field(default_factory=list)


@dataclass
class BacktestMetrics:
    method: str
    mae: float
    rmse: float
    directional_accuracy: float
    naive_baseline_mae: float
    beats_naive: bool = False


@dataclass
class ForecastResult:
    score: ModuleScore
    monte_carlo: dict[int, ForecastDistribution] = field(default_factory=dict)
    risk_neutral: dict[int, ForecastDistribution] = field(default_factory=dict)
    time_series: dict[int, ForecastDistribution] = field(default_factory=dict)
    ml_directional: dict[int, float] = field(default_factory=dict)
    ensemble: dict[int, ForecastDistribution] = field(default_factory=dict)
    backtests: list[BacktestMetrics] = field(default_factory=list)
    mc_sample_paths: Optional[np.ndarray] = None   # shape: (50, num_days)
    math_finance: Optional[MathematicalFinanceDiagnostics] = None


@dataclass
class RiskResult:
    score: ModuleScore
    realized_vol_30d: Optional[float] = None
    realized_vol_90d: Optional[float] = None
    realized_vol_365d: Optional[float] = None
    beta_spy: Optional[float] = None
    beta_sector: Optional[float] = None
    sector_etf: Optional[str] = None
    max_drawdown_1y: Optional[float] = None
    max_drawdown_3y: Optional[float] = None
    current_drawdown: Optional[float] = None
    sharpe_1y: Optional[float] = None
    sortino_1y: Optional[float] = None
    downside_deviation_1y: Optional[float] = None
    kelly_fraction: Optional[float] = None
    quarter_kelly: Optional[float] = None
    kelly_edge: Optional[float] = None
    implied_vol_atm: Optional[float] = None


@dataclass
class CompositeResult:
    overall_score: float
    confidence: float
    rating: Rating
    confidence_band: tuple[float, float]
    possible_ratings: list[Rating]
    weights_used: dict[str, float]
    component_scores: dict[str, float]     # module_name -> weighted contribution (0-100)
    module_scores: list[ModuleScore]
    computed_at: datetime = field(default_factory=datetime.utcnow)
