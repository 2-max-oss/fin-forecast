"""Fundamental analysis module."""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from config import (
    ROIC_THRESHOLDS, ROE_THRESHOLDS, NET_MARGIN_THRESHOLDS,
    REVENUE_CAGR_THRESHOLDS, FCF_CONVERSION_THRESHOLDS,
    NET_DEBT_EBITDA_THRESHOLDS, INTEREST_COVERAGE_THRESHOLDS,
)
from core.types import (
    CompanyInfo, Financials, FundamentalResult, ModuleScore,
    safe_divide, threshold_score,
)

logger = logging.getLogger(__name__)


# ── Column lookup helpers ─────────────────────────────────────────────────────

def _find_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    """Return the first matching column name in df (case-insensitive partial match)."""
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        cand_l = cand.lower()
        if cand_l in cols_lower:
            return cols_lower[cand_l]
        for col_l, col_orig in cols_lower.items():
            if cand_l in col_l:
                return col_orig
    return None


def _get_series(df: pd.DataFrame, candidates: list[str]) -> Optional[pd.Series]:
    col = _find_col(df, candidates)
    if col is None or df.empty:
        return None
    return pd.to_numeric(df[col], errors="coerce")


def _latest(series: Optional[pd.Series]) -> Optional[float]:
    if series is None or series.empty:
        return None
    val = series.dropna().iloc[0] if not series.dropna().empty else None
    if val is None:
        return None
    try:
        f = float(val)
        return None if (np.isnan(f) or np.isinf(f)) else f
    except Exception:
        return None


def _cagr(series: Optional[pd.Series], years: int = 5) -> Optional[float]:
    """Compute CAGR over last *years* observations."""
    if series is None:
        return None
    s = series.dropna()
    if len(s) < 2:
        return None
    s = s.sort_index(ascending=False)   # most recent first
    n = min(years, len(s) - 1)
    end_val = _latest(s)
    start_val = float(s.iloc[n]) if n < len(s) else None
    if end_val is None or start_val is None or start_val <= 0 or end_val <= 0:
        return None
    try:
        result = (end_val / start_val) ** (1.0 / n) - 1.0
        return None if (np.isnan(result) or np.isinf(result)) else result
    except Exception:
        return None


# ── Main analysis function ────────────────────────────────────────────────────

def analyze_fundamentals(financials: Financials, info: CompanyInfo) -> FundamentalResult:
    warnings: list[str] = []
    available = 0
    expected = 17  # total metrics attempted

    inc = financials.income_stmt.df
    bal = financials.balance_sheet.df
    cf  = financials.cashflow.df

    # ── Revenue ───────────────────────────────────────────────────────────────
    revenue_series = _get_series(inc, ["total_revenue", "totalrevenue", "revenue", "total revenue"])
    revenue_latest = _latest(revenue_series)

    # ── Net income ────────────────────────────────────────────────────────────
    net_income_series = _get_series(inc, ["net_income", "netincome", "net income", "net income common stockholders"])
    net_income_latest = _latest(net_income_series)

    # ── EPS (basic) ───────────────────────────────────────────────────────────
    eps_series = _get_series(inc, ["basic_eps", "diluted_eps", "basiceps", "eps", "earningspershare"])

    # ── Gross profit ─────────────────────────────────────────────────────────
    gross_profit_series = _get_series(inc, ["gross_profit", "grossprofit", "gross profit"])
    gross_profit_latest = _latest(gross_profit_series)

    # ── Operating income ──────────────────────────────────────────────────────
    operating_income_series = _get_series(inc, [
        "operating_income", "operatingincome", "ebit", "operating income"
    ])
    operating_income_latest = _latest(operating_income_series)

    # ── Free cash flow ────────────────────────────────────────────────────────
    cfo_series = _get_series(cf, [
        "operating_cash_flow", "cash_from_operations", "total_cash_from_operating_activities",
        "cash_flow_from_continuing_operating_activities", "operatingcashflow"
    ])
    capex_series = _get_series(cf, [
        "capital_expenditure", "capex", "purchase_of_ppe",
        "capital_expenditures", "purchase_of_property_plant_and_equipment"
    ])
    cfo_latest = _latest(cfo_series)
    capex_latest = _latest(capex_series)

    # FCF = CFO - |capex|  (capex is usually negative in yfinance)
    fcf_latest: Optional[float] = None
    fcf_series: Optional[pd.Series] = None
    if cfo_series is not None and capex_series is not None:
        fcf_series = cfo_series + capex_series  # capex already negative
        fcf_latest = _latest(fcf_series)
        if fcf_latest is None and cfo_latest is not None:
            fcf_series = cfo_series
            fcf_latest = cfo_latest

    # ── EBITDA ────────────────────────────────────────────────────────────────
    ebitda_series = _get_series(inc, ["ebitda", "normalized_ebitda"])
    ebitda_latest = _latest(ebitda_series)
    if ebitda_latest is None:
        da_series = _get_series(cf, [
            "depreciation_and_amortization", "depreciation",
            "depreciation_amortization_depletion", "reconciled_depreciation"
        ])
        da_latest = _latest(da_series)
        if operating_income_latest is not None and da_latest is not None:
            ebitda_latest = operating_income_latest + abs(da_latest)

    # ── Balance sheet items ───────────────────────────────────────────────────
    total_assets_series     = _get_series(bal, ["total_assets", "totalassets"])
    total_equity_series     = _get_series(bal, [
        "stockholders_equity", "total_equity_gross_minority_interest",
        "common_stock_equity", "stockholdersequity", "total_stockholders_equity"
    ])
    total_debt_series       = _get_series(bal, [
        "total_debt", "long_term_debt", "longterm_debt",
        "long_term_debt_and_capital_lease_obligation"
    ])
    cash_series             = _get_series(bal, [
        "cash_and_cash_equivalents", "cash", "cash_and_short_term_investments",
        "cash_cash_equivalents_and_short_term_investments"
    ])
    current_assets_series   = _get_series(bal, ["current_assets", "totalcurrentassets", "total_current_assets"])
    current_liab_series     = _get_series(bal, ["current_liabilities", "totalcurrentliabilities", "total_current_liabilities"])
    inventory_series        = _get_series(bal, ["inventory", "inventories", "finished_goods"])
    invested_capital_series = _get_series(bal, ["invested_capital", "investedcapital"])

    total_assets_latest  = _latest(total_assets_series)
    total_equity_latest  = _latest(total_equity_series)
    total_debt_latest    = _latest(total_debt_series)
    cash_latest          = _latest(cash_series)
    curr_assets_latest   = _latest(current_assets_series)
    curr_liab_latest     = _latest(current_liab_series)
    inventory_latest     = _latest(inventory_series)
    invested_cap_latest  = _latest(invested_capital_series)

    # Net debt
    net_debt: Optional[float] = None
    if total_debt_latest is not None and cash_latest is not None:
        net_debt = total_debt_latest - cash_latest

    # Interest expense
    interest_expense_series = _get_series(inc, [
        "interest_expense", "interest expense", "interest_expense_non_operating"
    ])
    interest_expense_latest = _latest(interest_expense_series)

    # ── COMPUTE METRICS ───────────────────────────────────────────────────────

    # Growth
    revenue_cagr = _cagr(revenue_series)
    if revenue_cagr is not None: available += 1
    else: warnings.append("Revenue CAGR: insufficient data")

    eps_cagr = _cagr(eps_series)
    if eps_cagr is not None: available += 1
    else: warnings.append("EPS CAGR: insufficient data")

    fcf_growth_yoy: Optional[float] = None
    if fcf_series is not None and len(fcf_series.dropna()) >= 2:
        s = fcf_series.dropna().sort_index(ascending=False)
        fcf_growth_yoy = safe_divide(s.iloc[0] - s.iloc[1], abs(s.iloc[1]))
        if fcf_growth_yoy is not None: available += 1
    else:
        warnings.append("FCF growth: insufficient data")

    # Profitability
    gross_margin    = safe_divide(gross_profit_latest, revenue_latest)
    operating_margin = safe_divide(operating_income_latest, revenue_latest)
    net_margin      = safe_divide(net_income_latest, revenue_latest)
    fcf_margin      = safe_divide(fcf_latest, revenue_latest)
    for m, name in [(gross_margin, "Gross margin"), (operating_margin, "Op margin"),
                    (net_margin, "Net margin"), (fcf_margin, "FCF margin")]:
        if m is not None: available += 1
        else: warnings.append(f"{name}: insufficient data")

    # Efficiency
    nopat: Optional[float] = None
    if operating_income_latest is not None:
        nopat = operating_income_latest * 0.79  # approximate post-tax
    if invested_cap_latest is None and total_assets_latest is not None and total_equity_latest is not None:
        invested_cap_latest = total_equity_latest + (total_debt_latest or 0) - (cash_latest or 0)
    roic = safe_divide(nopat, invested_cap_latest)
    roe  = safe_divide(net_income_latest, total_equity_latest)
    roa  = safe_divide(net_income_latest, total_assets_latest)
    asset_turnover = safe_divide(revenue_latest, total_assets_latest)
    for m, name in [(roic, "ROIC"), (roe, "ROE"), (roa, "ROA"), (asset_turnover, "Asset turnover")]:
        if m is not None: available += 1
        else: warnings.append(f"{name}: insufficient data")

    # Balance sheet
    net_debt_to_ebitda = safe_divide(net_debt, ebitda_latest) if ebitda_latest and ebitda_latest > 0 else None
    interest_coverage: Optional[float] = None
    if interest_expense_latest is not None and interest_expense_latest != 0:
        interest_coverage = safe_divide(ebitda_latest, abs(interest_expense_latest))
    current_ratio = safe_divide(curr_assets_latest, curr_liab_latest)
    quick_ratio: Optional[float] = None
    if curr_assets_latest is not None and inventory_latest is not None and curr_liab_latest is not None:
        quick_ratio = safe_divide(curr_assets_latest - abs(inventory_latest), curr_liab_latest)
    for m, name in [(net_debt_to_ebitda, "Net debt/EBITDA"), (interest_coverage, "Interest coverage"),
                    (current_ratio, "Current ratio")]:
        if m is not None: available += 1
        else: warnings.append(f"{name}: insufficient data")

    # Quality
    fcf_conversion = safe_divide(fcf_latest, net_income_latest)

    # Accruals ratio = (Net income - FCF) / avg total assets
    accruals_ratio: Optional[float] = None
    if net_income_latest is not None and fcf_latest is not None and total_assets_latest is not None and total_assets_latest > 0:
        accruals_ratio = (net_income_latest - fcf_latest) / total_assets_latest

    # Earnings stability = 1 - CV of trailing EPS
    earnings_stability: Optional[float] = None
    if eps_series is not None:
        s = eps_series.dropna()
        if len(s) >= 3 and s.mean() != 0:
            cv = s.std() / abs(s.mean())
            earnings_stability = max(0.0, 1.0 - cv)

    for m, name in [(fcf_conversion, "FCF conversion"), (earnings_stability, "Earnings stability")]:
        if m is not None: available += 1
        else: warnings.append(f"{name}: insufficient data")

    # ── Trend data ────────────────────────────────────────────────────────────
    margin_history: Optional[pd.DataFrame] = None
    if revenue_series is not None and not revenue_series.dropna().empty:
        rev = revenue_series.dropna().sort_index()
        frames: dict[str, pd.Series] = {}
        if gross_profit_series is not None:
            frames["gross"] = (gross_profit_series.dropna() / rev).dropna()
        if operating_income_series is not None:
            frames["operating"] = (operating_income_series.dropna() / rev).dropna()
        if net_income_series is not None:
            frames["net"] = (net_income_series.dropna() / rev).dropna()
        if fcf_series is not None:
            frames["fcf"] = (fcf_series.dropna() / rev).dropna()
        if frames:
            margin_history = pd.DataFrame(frames).sort_index()

    growth_history: Optional[pd.DataFrame] = None
    frames_g: dict[str, pd.Series] = {}
    if revenue_series is not None:
        frames_g["revenue"] = revenue_series.dropna().sort_index()
    if eps_series is not None:
        frames_g["eps"] = eps_series.dropna().sort_index()
    if fcf_series is not None:
        frames_g["fcf"] = fcf_series.dropna().sort_index()
    if frames_g:
        growth_history = pd.DataFrame(frames_g).sort_index()

    roic_history: Optional[pd.Series] = None
    if total_equity_series is not None and net_income_series is not None:
        try:
            nopat_series = net_income_series.dropna() * 0.79
            denom_series = total_equity_series.dropna()
            if total_debt_series is not None:
                denom_series = denom_series.add(total_debt_series.dropna(), fill_value=0)
            denom_series = denom_series.replace(0, float("nan"))
            roic_series = (nopat_series / denom_series).dropna().sort_index()
            if not roic_series.empty:
                roic_history = roic_series
        except Exception:
            pass

    # ── Score computation ─────────────────────────────────────────────────────
    confidence = min(1.0, available / expected)

    sub_scores: dict[str, float] = {}

    def _add_score(key: str, val, thresholds):
        s = threshold_score(val, thresholds)
        if s is not None:
            sub_scores[key] = s

    _add_score("revenue_cagr",    revenue_cagr,       REVENUE_CAGR_THRESHOLDS)
    _add_score("net_margin",      net_margin,         NET_MARGIN_THRESHOLDS)
    _add_score("roic",            roic,               ROIC_THRESHOLDS)
    _add_score("roe",             roe,                ROE_THRESHOLDS)
    _add_score("fcf_conversion",  fcf_conversion,     FCF_CONVERSION_THRESHOLDS)
    _add_score("net_debt_ebitda", net_debt_to_ebitda, NET_DEBT_EBITDA_THRESHOLDS)
    _add_score("interest_coverage", interest_coverage, INTEREST_COVERAGE_THRESHOLDS)

    # EPS CAGR bonus/penalty
    if eps_cagr is not None:
        sub_scores["eps_cagr"] = threshold_score(eps_cagr, REVENUE_CAGR_THRESHOLDS) or 50.0

    # Earnings stability
    if earnings_stability is not None:
        sub_scores["earnings_stability"] = earnings_stability * 100.0

    # FCF margin
    if fcf_margin is not None:
        sub_scores["fcf_margin"] = threshold_score(fcf_margin, NET_MARGIN_THRESHOLDS) or 50.0

    overall = float(np.mean(list(sub_scores.values()))) if sub_scores else 50.0
    overall = max(0.0, min(100.0, overall))

    module_score = ModuleScore(
        name="fundamental",
        score=round(overall, 1),
        confidence=round(confidence, 3),
        components=sub_scores,
        warnings=warnings,
    )

    return FundamentalResult(
        score=module_score,
        revenue_cagr_5y=revenue_cagr,
        eps_cagr_5y=eps_cagr,
        fcf_growth_yoy=fcf_growth_yoy,
        gross_margin=gross_margin,
        operating_margin=operating_margin,
        net_margin=net_margin,
        fcf_margin=fcf_margin,
        roic=roic,
        roe=roe,
        roa=roa,
        asset_turnover=asset_turnover,
        net_debt_to_ebitda=net_debt_to_ebitda,
        interest_coverage=interest_coverage,
        current_ratio=current_ratio,
        quick_ratio=quick_ratio,
        fcf_conversion=fcf_conversion,
        accruals_ratio=accruals_ratio,
        earnings_stability=earnings_stability,
        margin_history=margin_history,
        growth_history=growth_history,
        roic_history=roic_history,
    )
