"""Stock Analyzer & Forecasting Engine — Streamlit entry point."""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Optional

import streamlit as st

# ── Ensure project root is on sys.path ────────────────────────────────────────
ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stock Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Imports (after path setup) ────────────────────────────────────────────────
from config import DEFAULT_WEIGHTS, SECTOR_ETF_MAP
from core.scoring import CompositeScorer
from core.types import CompanyInfo, CompositeResult
from data_layer.cache import CacheManager
from data_layer.fred_provider import FredProvider, MacroContext
from data_layer.yfinance_provider import YFinanceProvider
from analysis.fundamental import analyze_fundamentals
from analysis.technical import analyze_technical
from analysis.risk import analyze_risk
from analysis.valuation import analyze_valuation
from analysis.forecasting import analyze_forecast
from analysis.pitch import generate_pitch
from data_layer.edgar import fetch_news_and_filings
from data_layer.stock_search import search_stocks, build_sp500_metadata, _cache_is_fresh
from ui.components import render_banner, render_score_decomposition
from ui.summary_tab import render_summary
from ui.fundamental_tab import render_fundamental
from ui.valuation_tab import render_valuation
from ui.technical_tab import render_technical
from ui.forecasting_tab import render_forecasting
from ui.risk_tab import render_risk
from ui.pitch_tab import render_pitch


# ── Cached data fetch ─────────────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def fetch_all_data(ticker: str):
    provider = YFinanceProvider()
    info      = provider.get_info(ticker)
    prices    = provider.get_price_history(ticker, years=10)
    financials = provider.get_financials(ticker)
    estimates = provider.get_estimates(ticker)
    return prices, financials, info, estimates


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_macro(sector: str):
    macro = MacroContext()
    spy = macro.get_sp500_history()
    sector_etf = SECTOR_ETF_MAP.get(sector)
    sector_prices = macro.get_sector_etf_history(sector_etf) if sector_etf else None
    risk_free_rate = macro.get_risk_free_rate()
    return spy, sector_prices, risk_free_rate


# ── Cached analysis ───────────────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def run_fundamental_cached(ticker: str):
    prices, financials, info, _ = fetch_all_data(ticker)
    return analyze_fundamentals(financials, info)


@st.cache_data(ttl=300, show_spinner=False)
def run_technical_cached(ticker: str):
    prices, _, _, _ = fetch_all_data(ticker)
    return analyze_technical(prices)


@st.cache_data(ttl=300, show_spinner=False)
def run_valuation_cached(ticker: str, risk_free_rate: float):
    prices, financials, info, estimates = fetch_all_data(ticker)
    return analyze_valuation(prices, financials, info, estimates, risk_free_rate)


@st.cache_data(ttl=300, show_spinner=False)
def run_forecast_cached(
    ticker: str,
    risk_free_rate: float | None = None,
    dividend_yield: float = 0.0,
):
    prices, _, _, _ = fetch_all_data(ticker)
    return analyze_forecast(
        prices,
        risk_free_rate=risk_free_rate,
        dividend_yield=dividend_yield,
    )


@st.cache_data(ttl=300, show_spinner=False)
def run_risk_cached(ticker: str, risk_free_rate: float):
    prices, _, info, _ = fetch_all_data(ticker)
    spy, sector_prices, _ = fetch_macro(info.sector)
    forecast = run_forecast_cached(ticker, risk_free_rate, info.dividend_yield or 0.0)
    return analyze_risk(prices, info, forecast, risk_free_rate, spy, sector_prices)


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_news_and_filings_cached(ticker: str):
    """Cache news & filings for 30 min — EDGAR rate limits are lenient but finite."""
    return fetch_news_and_filings(ticker)


# ── Settings sidebar ──────────────────────────────────────────────────────────

def render_settings() -> dict:
    with st.sidebar:
        st.header("⚙ Settings")

        st.subheader("Composite Weights")
        st.caption("Adjust weights. They will be normalized to sum to 100%.")
        w_val  = st.slider("Valuation",    0, 50, int(DEFAULT_WEIGHTS["valuation"]  * 100), 5)
        w_fund = st.slider("Fundamentals", 0, 50, int(DEFAULT_WEIGHTS["fundamental"] * 100), 5)
        w_fore = st.slider("Forecast",     0, 30, int(DEFAULT_WEIGHTS["forecast"]   * 100), 5)
        w_tech = st.slider("Technical",    0, 30, int(DEFAULT_WEIGHTS["technical"]  * 100), 5)
        w_risk = st.slider("Risk",         0, 30, int(DEFAULT_WEIGHTS["risk"]       * 100), 5)

        total = w_val + w_fund + w_fore + w_tech + w_risk
        if total == 0:
            total = 1
        weights = {
            "valuation":   w_val  / total,
            "fundamental": w_fund / total,
            "forecast":    w_fore / total,
            "technical":   w_tech / total,
            "risk":        w_risk / total,
        }

        st.divider()
        st.subheader("Cache")
        if st.button("🔄 Clear Cache for Ticker"):
            st.session_state["clear_cache"] = True

    return weights


# ── Main app ──────────────────────────────────────────────────────────────────

def main():
    st.title("📈 Stock Analyzer & Forecasting Engine")
    st.caption("Decision-support tool for equity analysis. Not investment advice.")

    weights = render_settings()

    # ── Input row 1: direct ticker ────────────────────────────────────────────
    col_input, col_btn = st.columns([4, 1])
    with col_input:
        ticker_raw = st.text_input(
            "Ticker Symbol",
            value=st.session_state.get("last_ticker", "AAPL"),
            placeholder="e.g. AAPL, MSFT, NVDA",
            label_visibility="collapsed",
        )
    with col_btn:
        analyze_clicked = st.button("Analyze", type="primary", use_container_width=True)

    # ── Input row 2: natural language stock finder ────────────────────────────
    st.caption("— or describe what you're looking for —")
    col_search, col_search_btn = st.columns([4, 1])
    with col_search:
        search_query = st.text_input(
            "Stock finder",
            value="",
            placeholder='e.g. "defensive dividend consumer staples" or "high growth semiconductor"',
            label_visibility="collapsed",
            key="stock_search_query",
        )
    with col_search_btn:
        search_clicked = st.button("Find Stocks", use_container_width=True)

    # Run search and show clickable suggestions
    if search_clicked and search_query.strip():
        # Build / refresh the S&P 500 metadata cache if needed
        sp500_meta = None
        if not _cache_is_fresh():
            st.info("Building S&P 500 index for the first time — this takes ~15 seconds and won't repeat.")
            progress_bar = st.progress(0)
            status_text  = st.empty()

            def _on_progress(done: int, total: int) -> None:
                pct = done / total
                progress_bar.progress(pct)
                status_text.caption(f"Fetching metadata: {done}/{total} stocks")

            sp500_meta = build_sp500_metadata(progress_callback=_on_progress)
            progress_bar.empty()
            status_text.empty()

        with st.spinner(f"Scanning S&P 500 for '{search_query.strip()}'..."):
            suggestions = search_stocks(search_query.strip(), max_results=6, metadata=sp500_meta)
        if suggestions:
            st.write("**Best matches from the S&P 500** — click to open analysis:")
            btn_cols = st.columns(min(len(suggestions), 6))
            for col, s in zip(btn_cols, suggestions):
                help_text = (
                    f"{s.name}\n"
                    f"{s.sector} · {s.industry}\n"
                    f"Matched: {s.match_reason}"
                )
                if col.button(
                    s.ticker,
                    help=help_text,
                    key=f"suggest_{s.ticker}",
                    type="primary" if s == suggestions[0] else "secondary",
                ):
                    st.session_state["last_ticker"] = s.ticker
                    st.session_state["auto_analyze"] = True
                    st.rerun()
            # Show full names beneath the buttons
            name_cols = st.columns(min(len(suggestions), 6))
            for col, s in zip(name_cols, suggestions):
                col.caption(f"{s.name[:22]}")
        else:
            st.info("No S&P 500 matches found. Try different keywords or enter a ticker directly.")

    # ── Auto-analyze when a suggestion was clicked ────────────────────────────
    auto_analyze = st.session_state.pop("auto_analyze", False)

    ticker = ticker_raw.strip().upper()

    # Clear cache if requested
    if st.session_state.get("clear_cache"):
        CacheManager.get().invalidate(ticker)
        st.cache_data.clear()
        st.session_state["clear_cache"] = False
        st.success(f"Cache cleared for {ticker}")

    if not ticker:
        st.info("Enter a ticker symbol above and click Analyze.")
        return

    if not (analyze_clicked or auto_analyze or st.session_state.get("last_ticker") == ticker):
        st.info(f"Press **Analyze** to run analysis for **{ticker}**.")
        return

    st.session_state["last_ticker"] = ticker

    # ── Banner placeholder (fills after all modules complete) ─────────────────
    banner_placeholder = st.empty()

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tabs = st.tabs(["Summary", "Fundamental", "Valuation", "Technical", "Forecasting", "Risk", "Pitch"])

    results = {}
    errors  = {}

    # ── Fetch data first ──────────────────────────────────────────────────────
    with st.spinner(f"Fetching data for {ticker}..."):
        try:
            prices, financials, info, estimates = fetch_all_data(ticker)
            spy, sector_prices, risk_free_rate = fetch_macro(info.sector)
        except Exception as e:
            st.error(f"Data fetch failed for **{ticker}**: {e}")
            st.caption("Check the ticker symbol. Common causes: invalid ticker, network issue, or rate limit.")
            return

    current_price = prices.latest_close

    # ── Run each module and render its tab ────────────────────────────────────

    # Fundamental
    with tabs[1]:
        with st.status("Analyzing fundamentals...", expanded=False) as s:
            try:
                r = analyze_fundamentals(financials, info)
                results["fundamental"] = r
                render_fundamental(r)
                s.update(label=f"Fundamentals ✓ — Score: {r.score.score:.0f}/100", state="complete")
            except Exception as e:
                errors["fundamental"] = str(e)
                st.error(f"Fundamental analysis failed: {e}")
                s.update(label="Fundamentals ✗", state="error")

    # Technical
    with tabs[3]:
        with st.status("Running technical analysis...", expanded=False) as s:
            try:
                r = analyze_technical(prices)
                results["technical"] = r
                render_technical(r, ticker)
                s.update(label=f"Technical ✓ — Score: {r.score.score:.0f}/100", state="complete")
            except Exception as e:
                errors["technical"] = str(e)
                st.error(f"Technical analysis failed: {e}")
                s.update(label="Technical ✗", state="error")

    # Valuation
    with tabs[2]:
        with st.status("Running valuation analysis...", expanded=False) as s:
            try:
                r = analyze_valuation(prices, financials, info, estimates, risk_free_rate)
                results["valuation"] = r
                render_valuation(r)
                s.update(label=f"Valuation ✓ — Score: {r.score.score:.0f}/100", state="complete")
            except Exception as e:
                errors["valuation"] = str(e)
                st.error(f"Valuation analysis failed: {e}")
                s.update(label="Valuation ✗", state="error")

    # Forecasting
    with tabs[4]:
        with st.status("Running forecasting models (Monte Carlo, Prophet, ML)...", expanded=False) as s:
            try:
                r = analyze_forecast(
                    prices,
                    risk_free_rate=risk_free_rate,
                    dividend_yield=info.dividend_yield or 0.0,
                )
                results["forecast"] = r
                render_forecasting(r, current_price)
                s.update(label=f"Forecasting ✓ — Score: {r.score.score:.0f}/100", state="complete")
            except Exception as e:
                errors["forecast"] = str(e)
                st.error(f"Forecasting failed: {e}")
                s.update(label="Forecasting ✗", state="error")

    # Risk
    with tabs[5]:
        with st.status("Computing risk metrics...", expanded=False) as s:
            try:
                forecast = results.get("forecast")
                r = analyze_risk(prices, info, forecast, risk_free_rate, spy, sector_prices)
                results["risk"] = r
                render_risk(r, prices)
                s.update(label=f"Risk ✓ — Score: {r.score.score:.0f}/100", state="complete")
            except Exception as e:
                errors["risk"] = str(e)
                st.error(f"Risk analysis failed: {e}")
                s.update(label="Risk ✗", state="error")

    # ── News & filings (non-blocking, best-effort) ────────────────────────────
    news_filings = None
    try:
        news_filings = fetch_news_and_filings_cached(ticker)
    except Exception as _nf_exc:
        logger.debug("News/filings fetch failed: %s", _nf_exc)

    # ── Composite scoring ─────────────────────────────────────────────────────
    module_scores = [r.score for r in results.values() if r is not None]

    if not module_scores:
        st.error("All analysis modules failed. Cannot compute composite score.")
        return

    scorer = CompositeScorer(weights=weights)
    composite = scorer.compute(module_scores)

    # ── Summary tab ───────────────────────────────────────────────────────────
    with tabs[0]:
        render_summary(composite)

    # ── Pitch tab ─────────────────────────────────────────────────────────────
    with tabs[6]:
        st.subheader("AI Investment Pitch")
        st.caption(
            "Generates a structured one-page investment pitch grounded in the "
            "quantitative analysis above. Uses configured AI keys when available, "
            "otherwise falls back to a local template."
        )

        gen_key = f"pitch_{ticker}"
        pitch_key = f"pitch_result_{ticker}"

        if st.button("Generate Pitch", type="primary", key=f"gen_btn_{ticker}"):
            st.session_state[gen_key] = True
            st.session_state.pop(pitch_key, None)

        if st.session_state.get(gen_key) and pitch_key not in st.session_state:
            with st.spinner("Generating investment pitch with Claude AI..."):
                try:
                    pitch_result = generate_pitch(
                        ticker=ticker,
                        info=info,
                        composite=composite,
                        current_price=current_price,
                        fundamental=results.get("fundamental"),
                        valuation=results.get("valuation"),
                        technical=results.get("technical"),
                        forecast=results.get("forecast"),
                        risk=results.get("risk"),
                        news_filings=news_filings,
                    )
                    st.session_state[pitch_key] = pitch_result
                    st.session_state[gen_key] = False
                    render_pitch(pitch_result, ticker, current_price)
                except Exception as _pe:
                    st.session_state[gen_key] = False
                    render_pitch(None, ticker, current_price,
                                 error_msg=str(_pe))
        elif pitch_key in st.session_state:
            render_pitch(st.session_state[pitch_key], ticker, current_price)
        else:
            render_pitch(None, ticker, current_price)

    # ── Banner ────────────────────────────────────────────────────────────────
    with banner_placeholder.container():
        render_banner(composite, info.name, ticker, current_price)

    # ── Footer ────────────────────────────────────────────────────────────────
    st.divider()
    col_a, col_b, col_c = st.columns(3)
    col_a.caption(f"Price data: {prices.df.index[-1].strftime('%Y-%m-%d')}")
    col_b.caption(f"Company: {info.name} · {info.sector}")
    col_c.caption("Data via yfinance & FRED. Not investment advice.")


if __name__ == "__main__":
    main()
