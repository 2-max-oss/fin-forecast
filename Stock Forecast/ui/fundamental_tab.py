"""Fundamental analysis tab."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core.types import FundamentalResult
from ui.components import fmt_pct, fmt_num, metric_row, render_warnings


def render_fundamental(result: FundamentalResult) -> None:
    render_warnings(result.score.warnings)

    # ── Growth ────────────────────────────────────────────────────────────────
    st.subheader("Growth")
    metric_row([
        ("Revenue CAGR (5Y)",  fmt_pct(result.revenue_cagr_5y), None),
        ("EPS CAGR (5Y)",      fmt_pct(result.eps_cagr_5y),    None),
        ("FCF Growth (YoY)",   fmt_pct(result.fcf_growth_yoy), None),
    ])

    if result.growth_history is not None and not result.growth_history.empty:
        fig = go.Figure()
        gh = result.growth_history
        if "revenue" in gh.columns:
            fig.add_trace(go.Bar(
                x=gh.index.astype(str), y=gh["revenue"],
                name="Revenue", marker_color="#1976D2"
            ))
        fig.update_layout(
            title="Revenue History",
            height=220,
            margin=dict(l=10, r=10, t=40, b=10),
            yaxis_tickprefix="$",
            bargap=0.2,
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Profitability ─────────────────────────────────────────────────────────
    st.subheader("Profitability")
    metric_row([
        ("Gross Margin",     fmt_pct(result.gross_margin),     None),
        ("Operating Margin", fmt_pct(result.operating_margin), None),
        ("Net Margin",       fmt_pct(result.net_margin),       None),
        ("FCF Margin",       fmt_pct(result.fcf_margin),       None),
    ])

    if result.margin_history is not None and not result.margin_history.empty:
        mh = result.margin_history * 100
        fig = go.Figure()
        colors = {"gross": "#42A5F5", "operating": "#66BB6A", "net": "#FFA726", "fcf": "#AB47BC"}
        for col in mh.columns:
            fig.add_trace(go.Scatter(
                x=mh.index.astype(str),
                y=mh[col],
                mode="lines+markers",
                name=col.title(),
                line=dict(color=colors.get(col, "#888"), width=2),
            ))
        fig.update_layout(
            title="Margin Trends (%)",
            height=240,
            margin=dict(l=10, r=10, t=40, b=10),
            yaxis_ticksuffix="%",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Capital Efficiency ────────────────────────────────────────────────────
    st.subheader("Capital Efficiency")
    metric_row([
        ("ROIC",           fmt_pct(result.roic),          None),
        ("ROE",            fmt_pct(result.roe),           None),
        ("ROA",            fmt_pct(result.roa),           None),
        ("Asset Turnover", fmt_num(result.asset_turnover, 2) + "x" if result.asset_turnover else "N/A", None),
    ])

    if result.roic_history is not None and not result.roic_history.empty:
        fig = go.Figure(go.Scatter(
            x=result.roic_history.index.astype(str),
            y=result.roic_history * 100,
            mode="lines+markers",
            line=dict(color="#00BCD4", width=2),
            fill="tozeroy",
            fillcolor="rgba(0,188,212,0.1)",
        ))
        fig.update_layout(
            title="ROIC History (%)",
            height=200,
            margin=dict(l=10, r=10, t=40, b=10),
            yaxis_ticksuffix="%",
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Balance Sheet ─────────────────────────────────────────────────────────
    st.subheader("Balance Sheet")
    metric_row([
        ("Net Debt / EBITDA",  fmt_num(result.net_debt_to_ebitda, 1) + "x" if result.net_debt_to_ebitda is not None else "N/A", None),
        ("Interest Coverage",  fmt_num(result.interest_coverage,  1) + "x" if result.interest_coverage is not None else "N/A", None),
        ("Current Ratio",      fmt_num(result.current_ratio,      2)                if result.current_ratio is not None else "N/A", None),
        ("Quick Ratio",        fmt_num(result.quick_ratio,        2)                if result.quick_ratio is not None else "N/A", None),
    ])

    # ── Quality ───────────────────────────────────────────────────────────────
    st.subheader("Quality")
    metric_row([
        ("FCF Conversion",     fmt_num(result.fcf_conversion,    2) + "x" if result.fcf_conversion is not None else "N/A", None),
        ("Accruals Ratio",     fmt_pct(result.accruals_ratio)                if result.accruals_ratio is not None else "N/A", None),
        ("Earnings Stability", fmt_num(result.earnings_stability, 2)         if result.earnings_stability is not None else "N/A", None),
    ])
