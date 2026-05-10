"""Valuation module: multiples, DCF, reverse DCF, blended fair value."""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from core.types import (
    AnalystEstimates, CompanyInfo, DCFResult, Financials,
    ModuleScore, MultiplesSnapshot, PriceHistory, ReverseDCFResult,
    ValuationResult, safe_divide,
)

logger = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _find_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        cand_l = cand.lower()
        if cand_l in cols_lower:
            return cols_lower[cand_l]
        for cl, co in cols_lower.items():
            if cand_l in cl:
                return co
    return None


def _get_val(df: pd.DataFrame, candidates: list[str], idx: int = 0) -> Optional[float]:
    col = _find_col(df, candidates)
    if col is None or df.empty:
        return None
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    if idx >= len(s):
        return None
    v = float(s.iloc[idx])
    return None if (np.isnan(v) or np.isinf(v)) else v


def _multiples_snapshot(name: str, current: Optional[float], hist: pd.Series) -> Optional[MultiplesSnapshot]:
    if current is None or np.isnan(current):
        return None
    h = hist.dropna()
    if h.empty:
        return MultiplesSnapshot(name=name, current=current)
    low = float(h.quantile(0.05))
    high = float(h.quantile(0.95))
    median = float(h.median())
    pct = float((h < current).mean() * 100)
    return MultiplesSnapshot(
        name=name,
        current=current,
        historical_5y_low=low,
        historical_5y_high=high,
        historical_5y_median=median,
        percentile=round(pct, 1),
    )


# ── Historical multiple series from price + financial history ─────────────────

def _build_pe_series(price_df: pd.DataFrame, income_df: pd.DataFrame) -> pd.Series:
    """Trailing twelve months P/E for each available annual period."""
    price_col = "adj_close" if "adj_close" in price_df.columns else "close"
    results = {}
    eps_col = _find_col(income_df, ["diluted_eps", "basic_eps", "eps"])
    if eps_col is None:
        return pd.Series(dtype=float)
    for date, row in income_df.iterrows():
        eps = row.get(eps_col)
        if eps is None or (isinstance(eps, float) and (np.isnan(eps) or eps <= 0)):
            continue
        # Price at end of that fiscal year (or nearest trading day)
        nearby = price_df[price_col].asof(date)
        if nearby is None or np.isnan(nearby):
            continue
        results[date] = nearby / eps
    return pd.Series(results, dtype=float).sort_index()


def _build_ev_ebitda_series(
    price_df: pd.DataFrame,
    income_df: pd.DataFrame,
    balance_df: pd.DataFrame,
    shares_outstanding: float,
) -> pd.Series:
    price_col = "adj_close" if "adj_close" in price_df.columns else "close"
    results = {}
    ebitda_col = _find_col(income_df, ["ebitda"])
    debt_col   = _find_col(balance_df, ["total_debt", "long_term_debt"])
    cash_col   = _find_col(balance_df, ["cash_and_cash_equivalents", "cash"])

    if ebitda_col is None:
        return pd.Series(dtype=float)

    for date, row in income_df.iterrows():
        ebitda = row.get(ebitda_col)
        if ebitda is None or (isinstance(ebitda, float) and (np.isnan(ebitda) or ebitda <= 0)):
            continue
        p = price_df[price_col].asof(date)
        if p is None or np.isnan(p):
            continue
        mktcap = p * shares_outstanding

        # Get balance sheet items for that date (or nearest prior)
        debt = balance_df[debt_col].asof(date) if debt_col and not balance_df.empty else 0.0
        cash = balance_df[cash_col].asof(date) if cash_col and not balance_df.empty else 0.0
        debt = float(debt) if not np.isnan(debt) else 0.0
        cash = float(cash) if not np.isnan(cash) else 0.0
        ev = mktcap + debt - cash
        results[date] = safe_divide(ev, ebitda)
    return pd.Series({k: v for k, v in results.items() if v is not None}, dtype=float).sort_index()


# ── DCF ───────────────────────────────────────────────────────────────────────

def _dcf_value(
    fcf_base: float,
    wacc: float,
    terminal_growth: float,
    revenue_growth_rates: list[float],
    fcf_margin_trajectory: list[float],
    revenue_base: float,
    projection_years: int = 7,
) -> float:
    """Two-stage DCF. Returns per-share equivalent (divide by shares externally)."""
    fcf = fcf_base
    pv_sum = 0.0

    for i, (g, m) in enumerate(zip(revenue_growth_rates, fcf_margin_trajectory)):
        revenue = revenue_base * (1 + g) ** (i + 1)
        fcf = revenue * m
        pv_sum += fcf / (1 + wacc) ** (i + 1)

    # Terminal value (Gordon Growth)
    terminal_fcf = fcf * (1 + terminal_growth)
    if wacc <= terminal_growth:
        terminal_growth = wacc - 0.01  # Prevent division by zero
    tv = terminal_fcf / (wacc - terminal_growth)
    pv_terminal = tv / (1 + wacc) ** projection_years

    return pv_sum + pv_terminal


def compute_dcf(
    financials: Financials,
    info: CompanyInfo,
    wacc: float = 0.10,
    terminal_growth: float = 0.03,
    projection_years: int = 7,
    risk_free_rate: float = 0.045,
) -> Optional[DCFResult]:
    inc = financials.income_stmt.df
    cf  = financials.cashflow.df

    revenue_latest = _get_val(inc, ["total_revenue", "totalrevenue", "revenue"])
    cfo_latest     = _get_val(cf,  ["operating_cash_flow", "cash_from_operations",
                                     "total_cash_from_operating_activities",
                                     "cash_flow_from_continuing_operating_activities"])
    capex_latest   = _get_val(cf,  ["capital_expenditure", "capex", "purchase_of_ppe",
                                     "capital_expenditures",
                                     "purchase_of_property_plant_and_equipment"])
    net_income = _get_val(inc, ["net_income", "netincome"])
    op_income  = _get_val(inc, ["operating_income", "ebit"])
    op_margin  = safe_divide(op_income, revenue_latest)

    if revenue_latest is None or revenue_latest <= 0:
        return None

    fcf_base = (cfo_latest or 0) + (capex_latest or 0)
    if abs(fcf_base) < 1:
        fcf_base = net_income * 0.85 if net_income else revenue_latest * 0.05

    fcf_margin = safe_divide(fcf_base, revenue_latest) or 0.08

    # Revenue growth trajectory: linear decay from recent CAGR to terminal
    revenue_series = inc.get(_find_col(inc, ["total_revenue", "revenue"]) or "", pd.Series())
    if isinstance(revenue_series, pd.Series) and len(revenue_series.dropna()) >= 3:
        s = revenue_series.dropna().sort_index(ascending=False)
        n = min(5, len(s) - 1)
        hist_cagr = ((s.iloc[0] / s.iloc[n]) ** (1.0 / n) - 1) if s.iloc[n] > 0 else 0.05
        hist_cagr = max(-0.05, min(0.30, float(hist_cagr)))
    else:
        hist_cagr = 0.07

    growth_start = hist_cagr
    growth_end   = terminal_growth + 0.01

    growth_rates  = np.linspace(growth_start, growth_end, projection_years).tolist()
    margin_traj   = np.linspace(fcf_margin, fcf_margin * 1.1, projection_years).tolist()

    # Sensitivity table: WACC vs terminal growth
    wacc_range = np.linspace(max(0.06, wacc - 0.03), wacc + 0.03, 7)
    tgr_range  = np.linspace(0.01, 0.05, 5)

    table_data: dict[float, dict[float, float]] = {}
    shares = info.shares_outstanding or 1

    for w in wacc_range:
        table_data[round(w, 4)] = {}
        for tg in tgr_range:
            total_value = _dcf_value(
                fcf_base, w, tg, growth_rates, margin_traj, revenue_latest, projection_years
            )
            table_data[round(w, 4)][round(tg, 4)] = round(total_value / shares, 2)

    sensitivity_df = pd.DataFrame(table_data).T
    sensitivity_df.index.name = "WACC"
    sensitivity_df.columns.name = "Terminal Growth"

    base_value = _dcf_value(fcf_base, wacc, terminal_growth, growth_rates, margin_traj,
                             revenue_latest, projection_years)
    fair_value = round(base_value / shares, 2)

    return DCFResult(
        fair_value=fair_value,
        sensitivity_table=sensitivity_df,
        assumptions={
            "wacc": wacc,
            "terminal_growth": terminal_growth,
            "projection_years": projection_years,
            "fcf_base": fcf_base,
            "revenue_base": revenue_latest,
            "fcf_margin": fcf_margin,
            "hist_cagr": hist_cagr,
        },
    )


def compute_reverse_dcf(
    current_price: float,
    financials: Financials,
    info: CompanyInfo,
    estimates: Optional[AnalystEstimates],
    wacc: float = 0.10,
    projection_years: int = 7,
) -> Optional[ReverseDCFResult]:
    inc = financials.income_stmt.df
    cf  = financials.cashflow.df

    revenue_latest = _get_val(inc, ["total_revenue", "totalrevenue", "revenue"])
    cfo_latest     = _get_val(cf,  ["operating_cash_flow", "cash_from_operations",
                                     "total_cash_from_operating_activities",
                                     "cash_flow_from_continuing_operating_activities"])
    capex_latest   = _get_val(cf,  ["capital_expenditure", "capex",
                                     "capital_expenditures"])
    net_income = _get_val(inc, ["net_income", "netincome"])

    if revenue_latest is None or revenue_latest <= 0 or current_price <= 0:
        return None

    fcf_base = (cfo_latest or 0) + (capex_latest or 0)
    if abs(fcf_base) < 1:
        fcf_base = (net_income or revenue_latest * 0.05) * 0.85

    fcf_margin = safe_divide(fcf_base, revenue_latest) or 0.08
    shares = info.shares_outstanding or 1
    target_total_value = current_price * shares

    # Binary search for the constant revenue growth rate that equates DCF to market price
    def _pv_at_growth(g: float) -> float:
        rates = [g] * projection_years
        margins = [fcf_margin] * projection_years
        return _dcf_value(fcf_base, wacc, 0.025, rates, margins, revenue_latest, projection_years)

    lo, hi = -0.10, 0.50
    for _ in range(50):
        mid = (lo + hi) / 2.0
        val = _pv_at_growth(mid)
        if val < target_total_value:
            lo = mid
        else:
            hi = mid
    implied_growth = round((lo + hi) / 2.0, 4)

    # Consensus growth (from estimates if available)
    consensus_growth: Optional[float] = None
    if estimates and estimates.forward_revenue and revenue_latest:
        consensus_growth = safe_divide(
            estimates.forward_revenue - revenue_latest, revenue_latest
        )

    # Historical growth
    historical_growth: Optional[float] = None
    rev_col = _find_col(inc, ["total_revenue", "revenue"])
    if rev_col and not inc.empty:
        rev_s = pd.to_numeric(inc[rev_col], errors="coerce").dropna().sort_index(ascending=False)
        n = min(5, len(rev_s) - 1)
        if n >= 1 and rev_s.iloc[n] > 0:
            historical_growth = float((rev_s.iloc[0] / rev_s.iloc[n]) ** (1.0 / n) - 1)

    return ReverseDCFResult(
        implied_growth_rate=implied_growth,
        current_price=current_price,
        consensus_growth_rate=consensus_growth,
        historical_growth_rate=historical_growth,
        assumptions={"wacc": wacc, "projection_years": projection_years, "fcf_margin": fcf_margin},
    )


# ── Scoring ───────────────────────────────────────────────────────────────────

def _score_valuation(
    upside_pct: Optional[float],
    pe_percentile: Optional[float],
    ev_ebitda_percentile: Optional[float],
    dcf_upside: Optional[float],
) -> tuple[float, dict[str, float]]:
    components: dict[str, float] = {}

    # Upside to fair value
    if upside_pct is not None:
        if upside_pct > 0.50:
            components["upside"] = 95.0
        elif upside_pct > 0.25:
            components["upside"] = 80.0
        elif upside_pct > 0.10:
            components["upside"] = 65.0
        elif upside_pct > -0.05:
            components["upside"] = 50.0
        elif upside_pct > -0.20:
            components["upside"] = 35.0
        else:
            components["upside"] = 15.0

    # P/E percentile: lower percentile = cheaper = better score
    if pe_percentile is not None:
        components["pe_relative"] = 100.0 - pe_percentile

    # EV/EBITDA percentile
    if ev_ebitda_percentile is not None:
        components["ev_ebitda_relative"] = 100.0 - ev_ebitda_percentile

    # DCF upside
    if dcf_upside is not None:
        if dcf_upside > 0.30:
            components["dcf"] = 85.0
        elif dcf_upside > 0.10:
            components["dcf"] = 65.0
        elif dcf_upside > -0.10:
            components["dcf"] = 50.0
        elif dcf_upside > -0.30:
            components["dcf"] = 30.0
        else:
            components["dcf"] = 15.0

    if not components:
        return 50.0, components
    return float(np.mean(list(components.values()))), components


# ── Main function ─────────────────────────────────────────────────────────────

def analyze_valuation(
    price_history: PriceHistory,
    financials: Financials,
    info: CompanyInfo,
    estimates: Optional[AnalystEstimates] = None,
    risk_free_rate: float = 0.045,
) -> ValuationResult:
    warnings: list[str] = []
    current_price = price_history.latest_close
    price_df = price_history.df
    inc = financials.income_stmt.df
    bal = financials.balance_sheet.df

    shares = info.shares_outstanding or 0
    mktcap = info.market_cap or (current_price * shares)
    ev = info.enterprise_value or mktcap

    # Align balance sheet to price df index for historical multiple series
    bal_sorted = bal.sort_index() if not bal.empty else bal
    inc_sorted = inc.sort_index() if not inc.empty else inc

    # ── Current multiples ─────────────────────────────────────────────────────
    # TTM EPS
    eps_ttm = _get_val(inc, ["diluted_eps", "basic_eps", "eps"])
    # Trailing P/E
    pe_trailing_current: Optional[float] = None
    if eps_ttm and eps_ttm > 0:
        pe_trailing_current = safe_divide(current_price, eps_ttm)

    # Forward P/E
    pe_forward_current: Optional[float] = estimates.forward_pe if estimates else None
    if pe_forward_current is None and estimates and estimates.forward_eps and estimates.forward_eps > 0:
        pe_forward_current = safe_divide(current_price, estimates.forward_eps)

    # EV/EBITDA current
    ebitda_latest = _get_val(inc, ["ebitda", "normalized_ebitda"])
    if ebitda_latest is None:
        op_income = _get_val(inc, ["operating_income", "ebit"])
        da_col    = _find_col(financials.cashflow.df, ["depreciation_and_amortization", "depreciation"])
        da_val    = _get_val(financials.cashflow.df, ["depreciation_and_amortization", "depreciation"])
        if op_income is not None and da_val is not None:
            ebitda_latest = op_income + abs(da_val)
    ev_ebitda_current = safe_divide(ev, ebitda_latest)

    # EV/Sales
    rev_latest = _get_val(inc, ["total_revenue", "totalrevenue", "revenue"])
    ev_sales_current = safe_divide(ev, rev_latest)

    # P/B
    eq_latest = _get_val(bal, ["stockholders_equity", "total_equity_gross_minority_interest",
                                "common_stock_equity"])
    bvps = safe_divide(eq_latest, shares)
    pb_current = safe_divide(current_price, bvps) if bvps and bvps > 0 else None

    # FCF yield
    cfo = _get_val(financials.cashflow.df, ["operating_cash_flow", "cash_from_operations",
                                             "total_cash_from_operating_activities",
                                             "cash_flow_from_continuing_operating_activities"])
    capex = _get_val(financials.cashflow.df, ["capital_expenditure", "capex",
                                               "capital_expenditures"])
    fcf = (cfo or 0) + (capex or 0) if (cfo is not None) else None
    fcf_yield_current = safe_divide(fcf, mktcap) if fcf is not None else None

    # ── Historical multiple series ─────────────────────────────────────────────
    pe_hist = _build_pe_series(price_df, inc_sorted) if not inc_sorted.empty else pd.Series()
    ev_ebitda_hist = (
        _build_ev_ebitda_series(price_df, inc_sorted, bal_sorted, shares)
        if (not inc_sorted.empty and shares > 0) else pd.Series()
    )

    # Build simple EV/Sales and P/B history from available data
    ev_sales_hist = pd.Series(dtype=float)
    if not inc_sorted.empty and shares > 0:
        rev_col = _find_col(inc_sorted, ["total_revenue", "totalrevenue", "revenue"])
        price_col = "adj_close" if "adj_close" in price_df.columns else "close"
        if rev_col:
            for date, row in inc_sorted.iterrows():
                r = row.get(rev_col)
                if r and r > 0:
                    p = price_df[price_col].asof(date)
                    if p and not np.isnan(p):
                        ev_s = (p * shares) / r
                        ev_sales_hist[date] = ev_s

    # ── Snapshots ─────────────────────────────────────────────────────────────
    pe_trailing_snap  = _multiples_snapshot("P/E (TTM)", pe_trailing_current, pe_hist)
    pe_forward_snap   = _multiples_snapshot("P/E (Fwd)", pe_forward_current, pe_hist) if pe_forward_current else None
    ev_ebitda_snap    = _multiples_snapshot("EV/EBITDA", ev_ebitda_current, ev_ebitda_hist)
    ev_sales_snap     = _multiples_snapshot("EV/Sales", ev_sales_current, ev_sales_hist)
    pb_snap           = _multiples_snapshot("P/B", pb_current, pd.Series())
    fcf_yield_snap    = _multiples_snapshot("FCF Yield", fcf_yield_current, pd.Series())

    # Warn on extreme percentiles
    for snap in [pe_trailing_snap, ev_ebitda_snap, ev_sales_snap]:
        if snap and snap.percentile is not None:
            if snap.percentile >= 80:
                warnings.append(f"{snap.name} at {snap.percentile:.0f}th historical percentile — elevated")
            elif snap.percentile <= 20:
                warnings.append(f"{snap.name} at {snap.percentile:.0f}th historical percentile — compressed")

    # ── DCF ───────────────────────────────────────────────────────────────────
    # Estimate WACC: risk_free + equity_risk_premium * beta (use 1.0 default beta)
    equity_risk_premium = 0.055  # typical ERP
    beta = 1.0
    wacc = risk_free_rate + equity_risk_premium * beta
    wacc = max(0.07, min(0.18, wacc))

    dcf_result = compute_dcf(financials, info, wacc=wacc, terminal_growth=0.03,
                              risk_free_rate=risk_free_rate)
    if dcf_result is None:
        warnings.append("DCF: insufficient financial data")

    # ── Reverse DCF ───────────────────────────────────────────────────────────
    reverse_dcf_result = compute_reverse_dcf(current_price, financials, info, estimates,
                                              wacc=wacc)
    if reverse_dcf_result is None:
        warnings.append("Reverse DCF: insufficient data")

    # ── Blended fair value ────────────────────────────────────────────────────
    fair_estimates: list[float] = []

    # DCF estimate
    if dcf_result and dcf_result.fair_value > 0:
        fair_estimates.append(dcf_result.fair_value)

    # Multiples-based: fair price at median P/E / EV multiples
    if pe_trailing_snap and pe_trailing_snap.historical_5y_median and eps_ttm:
        fair_pe = pe_trailing_snap.historical_5y_median * eps_ttm
        if fair_pe > 0:
            fair_estimates.append(fair_pe)

    if ev_ebitda_snap and ev_ebitda_snap.historical_5y_median and ebitda_latest and shares > 0:
        debt_c = _get_val(bal, ["total_debt", "long_term_debt"]) or 0
        cash_c = _get_val(bal, ["cash_and_cash_equivalents", "cash",
                                 "cash_and_short_term_investments"]) or 0
        fair_ev = ev_ebitda_snap.historical_5y_median * ebitda_latest
        fair_mktcap = fair_ev - debt_c + cash_c
        fair_price = safe_divide(fair_mktcap, shares)
        if fair_price and fair_price > 0:
            fair_estimates.append(fair_price)

    # Analyst target
    if estimates and estimates.target_median_price and estimates.target_median_price > 0:
        fair_estimates.append(estimates.target_median_price)

    fair_value_mid: Optional[float] = None
    fair_value_low: Optional[float] = None
    fair_value_high: Optional[float] = None

    if fair_estimates:
        fair_value_mid  = round(float(np.median(fair_estimates)), 2)
        fair_value_low  = round(float(np.percentile(fair_estimates, 25)), 2)
        fair_value_high = round(float(np.percentile(fair_estimates, 75)), 2)

    upside_pct = safe_divide(
        (fair_value_mid - current_price) if fair_value_mid else None,
        current_price
    )

    # DCF upside for scoring
    dcf_upside = safe_divide(
        (dcf_result.fair_value - current_price) if dcf_result else None,
        current_price
    )

    # ── Score ─────────────────────────────────────────────────────────────────
    available_count = sum(1 for v in [pe_trailing_snap, ev_ebitda_snap, dcf_result, upside_pct] if v is not None)
    confidence = min(1.0, available_count / 4)

    overall_score, components = _score_valuation(
        upside_pct,
        pe_trailing_snap.percentile if pe_trailing_snap else None,
        ev_ebitda_snap.percentile if ev_ebitda_snap else None,
        dcf_upside,
    )

    module_score = ModuleScore(
        name="valuation",
        score=round(overall_score, 1),
        confidence=round(confidence, 3),
        components=components,
        warnings=warnings,
    )

    return ValuationResult(
        score=module_score,
        pe_trailing=pe_trailing_snap,
        pe_forward=pe_forward_snap,
        ev_ebitda=ev_ebitda_snap,
        ev_sales=ev_sales_snap,
        pb=pb_snap,
        fcf_yield=fcf_yield_snap,
        dcf=dcf_result,
        reverse_dcf=reverse_dcf_result,
        fair_value_low=fair_value_low,
        fair_value_mid=fair_value_mid,
        fair_value_high=fair_value_high,
        current_price=current_price,
        upside_pct=upside_pct,
    )
