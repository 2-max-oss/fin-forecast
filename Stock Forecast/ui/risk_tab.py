"""Risk and position sizing UI tab."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.types import PriceHistory, RiskResult
from ui.components import fmt_pct, fmt_num, render_warnings


def render_risk(result: RiskResult, price_history: PriceHistory | None = None) -> None:
    render_warnings(result.score.warnings)

    # ── Volatility ────────────────────────────────────────────────────────────
    st.subheader("Volatility")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("30-Day Realized Vol",  fmt_pct(result.realized_vol_30d))
    c2.metric("90-Day Realized Vol",  fmt_pct(result.realized_vol_90d))
    c3.metric("365-Day Realized Vol", fmt_pct(result.realized_vol_365d))
    c4.metric("Implied Vol (ATM)",    fmt_pct(result.implied_vol_atm) if result.implied_vol_atm else "N/A")

    # Vol term structure bar chart
    vol_data = {}
    if result.realized_vol_30d  is not None: vol_data["30d Realized"]  = result.realized_vol_30d
    if result.realized_vol_90d  is not None: vol_data["90d Realized"]  = result.realized_vol_90d
    if result.realized_vol_365d is not None: vol_data["365d Realized"] = result.realized_vol_365d
    if result.implied_vol_atm   is not None: vol_data["Implied (~30d)"] = result.implied_vol_atm

    if vol_data:
        fig = go.Figure(go.Bar(
            x=list(vol_data.keys()),
            y=[v * 100 for v in vol_data.values()],
            marker_color=["#1976D2", "#42A5F5", "#90CAF9", "#FFA726"][:len(vol_data)],
            text=[f"{v*100:.1f}%" for v in vol_data.values()],
            textposition="outside",
        ))
        fig.update_layout(
            height=220,
            yaxis_ticksuffix="%",
            yaxis_title="Annualized Volatility",
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Beta ──────────────────────────────────────────────────────────────────
    st.subheader("Beta")
    c1, c2 = st.columns(2)
    c1.metric("Beta vs SPY",
              fmt_num(result.beta_spy, 2) if result.beta_spy is not None else "N/A")
    c2.metric(f"Beta vs {result.sector_etf or 'Sector ETF'}",
              fmt_num(result.beta_sector, 2) if result.beta_sector is not None else "N/A")

    # ── Risk Ratios ───────────────────────────────────────────────────────────
    st.subheader("Risk-Adjusted Ratios (Trailing 1Y)")
    c1, c2, c3 = st.columns(3)
    c1.metric("Sharpe Ratio",       fmt_num(result.sharpe_1y, 2)  if result.sharpe_1y  is not None else "N/A")
    c2.metric("Sortino Ratio",      fmt_num(result.sortino_1y, 2) if result.sortino_1y is not None else "N/A")
    c3.metric("Downside Deviation", fmt_pct(result.downside_deviation_1y) if result.downside_deviation_1y is not None else "N/A")

    # ── Drawdown ──────────────────────────────────────────────────────────────
    st.subheader("Drawdown")
    c1, c2, c3 = st.columns(3)
    c1.metric("Max Drawdown (1Y)",  fmt_pct(result.max_drawdown_1y)  if result.max_drawdown_1y  is not None else "N/A")
    c2.metric("Max Drawdown (3Y)",  fmt_pct(result.max_drawdown_3y)  if result.max_drawdown_3y  is not None else "N/A")
    c3.metric("Current Drawdown",   fmt_pct(result.current_drawdown) if result.current_drawdown is not None else "N/A")

    # Drawdown chart
    if price_history is not None:
        df = price_history.df
        col = "adj_close" if "adj_close" in df.columns else "close"
        prices = df[col].dropna()
        if len(prices) >= 100:
            prices_3y = prices.last("756B") if len(prices) >= 756 else prices
            rolling_max = prices_3y.cummax()
            drawdown = (prices_3y - rolling_max) / rolling_max * 100

            fig = go.Figure(go.Scatter(
                x=drawdown.index, y=drawdown.values,
                mode="lines",
                fill="tozeroy",
                fillcolor="rgba(213,0,0,0.12)",
                line=dict(color="#D50000", width=1),
                name="Drawdown",
            ))
            fig.update_layout(
                height=200,
                yaxis_ticksuffix="%",
                yaxis_title="Drawdown",
                margin=dict(l=10, r=10, t=10, b=10),
            )
            st.plotly_chart(fig, use_container_width=True)

    # ── Position Sizing (Kelly) ───────────────────────────────────────────────
    st.subheader("Position Sizing — Fractional Kelly")
    if result.quarter_kelly is not None:
        pct = result.quarter_kelly * 100
        edge = result.kelly_edge
        st.info(
            f"**Quarter-Kelly Suggested Allocation: {pct:.1f}% of portfolio**\n\n"
            f"Edge (expected return above risk-free): {fmt_pct(edge)}\n\n"
            f"Full Kelly: {fmt_pct(result.kelly_fraction)} — capped to prevent pathological sizing. "
            f"Quarter-Kelly ({pct:.1f}%) is the conservative default."
        )
        st.caption(
            "⚠ Kelly sizing is derived from model forecasts and historical volatility. "
            "It is a mathematical guideline, not a recommendation. "
            "Negative Kelly implies the model sees negative expected excess returns."
        )
    else:
        st.info("Position sizing unavailable — requires 12-month forecast distribution.")
