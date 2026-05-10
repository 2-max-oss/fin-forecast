"""Forecasting module UI tab."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.types import ForecastResult
from ui.components import fmt_pct, fmt_num, render_warnings


def _horizon_label(months: int) -> str:
    return f"{months}M"


def render_forecasting(result: ForecastResult, current_price: float = 0) -> None:
    render_warnings(result.score.warnings)

    if not result.ensemble:
        st.info("No forecast data available.")
        return

    # ── Horizon summary table ─────────────────────────────────────────────────
    st.subheader("Forecast Summary")
    rows = []
    for h, dist in sorted(result.ensemble.items()):
        S0 = dist.current_price or current_price
        exp_ret = (dist.median / S0 - 1.0) if S0 > 0 else None
        ml_prob = result.ml_directional.get(h)
        rows.append({
            "Horizon": f"{h} months",
            "Median Price": f"${dist.median:,.2f}",
            "Expected Return": fmt_pct(exp_ret),
            "P(Positive)": f"{dist.prob_positive:.1%}",
            "ML P(Up)": f"{ml_prob:.1%}" if ml_prob is not None else "N/A",
            "5th %ile": f"${dist.p5:,.2f}",
            "95th %ile": f"${dist.p95:,.2f}",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    _render_math_finance_assumptions(result)

    # ── Fan chart ─────────────────────────────────────────────────────────────
    st.subheader("12-Month Distribution Fan Chart")
    _render_fan_chart(result, current_price)

    # ── Monte Carlo paths ─────────────────────────────────────────────────────
    if result.mc_sample_paths is not None and len(result.mc_sample_paths) > 0:
        st.subheader("Monte Carlo Sample Paths (50 of 10,000)")
        _render_mc_paths(result.mc_sample_paths, current_price)

    # ── Backtest results ──────────────────────────────────────────────────────
    if result.backtests:
        st.subheader("Backtest Performance vs. Naive Baseline")
        bt_rows = []
        for bt in result.backtests:
            bt_rows.append({
                "Method": bt.method.replace("_", " ").title(),
                "MAE": f"{bt.mae:.4f}" if bt.mae else "N/A",
                "RMSE": f"{bt.rmse:.4f}" if bt.rmse else "N/A",
                "Directional Acc.": f"{bt.directional_accuracy:.1%}",
                "Naive Baseline MAE": f"{bt.naive_baseline_mae:.4f}" if bt.naive_baseline_mae else "N/A",
                "Beats Naive": "✅" if bt.beats_naive else "⚠️ No",
            })
        st.dataframe(pd.DataFrame(bt_rows), use_container_width=True, hide_index=True)
        st.caption(
            "Note: Individual equity returns are dominated by noise. "
            "A directional accuracy of 55%+ on out-of-sample data should be treated with caution."
        )


def _render_math_finance_assumptions(result: ForecastResult) -> None:
    diag = result.math_finance
    if diag is None:
        return

    st.subheader("Mathematical Finance Assumptions")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Physical Drift", fmt_pct(diag.physical_drift))
    col2.metric("Risk-Neutral Drift", fmt_pct(diag.risk_neutral_drift))
    col3.metric("Adjusted Volatility", fmt_pct(diag.adjusted_volatility))
    col4.metric("12M Martingale Error", fmt_pct(diag.martingale_error_12m))

    assumption_rows = [
        {"Item": "Scheme", "Value": diag.scheme},
        {
            "Item": "Volatility Regime Factor",
            "Value": fmt_num(diag.volatility_regime_factor, 2),
        },
        {"Item": "Risk-Free Rate", "Value": fmt_pct(diag.risk_free_rate)},
        {"Item": "Dividend Yield", "Value": fmt_pct(diag.dividend_yield)},
        {
            "Item": "Variance Reduction",
            "Value": "Antithetic Brownian increments" if diag.antithetic_variates else "None",
        },
    ]
    st.dataframe(pd.DataFrame(assumption_rows), use_container_width=True, hide_index=True)

    if result.risk_neutral:
        rn_rows = []
        for h, dist in sorted(result.risk_neutral.items()):
            years = h / 12
            risk_free = diag.risk_free_rate or 0.0
            discounted_mean = dist.mean * np.exp(-risk_free * years)
            martingale_error = (
                discounted_mean / dist.current_price - 1.0
                if dist.current_price
                else None
            )
            rn_rows.append({
                "Horizon": f"{h} months",
                "Risk-Neutral Mean": f"${dist.mean:,.2f}",
                "Discounted Mean": f"${discounted_mean:,.2f}",
                "Martingale Error": fmt_pct(martingale_error, 2),
                "P(Positive)": f"{dist.prob_positive:.1%}",
            })
        st.dataframe(pd.DataFrame(rn_rows), use_container_width=True, hide_index=True)


def _render_fan_chart(result: ForecastResult, current_price: float) -> None:
    if 12 not in result.ensemble:
        st.info("12-month ensemble forecast unavailable.")
        return

    dist = result.ensemble[12]
    S0 = dist.current_price or current_price
    h_days = 12 * 21

    # Build a simple fan from today → 12m horizon
    x = [0, h_days]
    fig = go.Figure()

    # 5-95 band
    fig.add_trace(go.Scatter(
        x=x + x[::-1], y=[S0, dist.p5, dist.p95, S0],
        fill="toself", fillcolor="rgba(25,118,210,0.06)",
        line=dict(width=0), name="5-95th %ile", showlegend=True,
    ))
    # 10-90 band
    fig.add_trace(go.Scatter(
        x=x + x[::-1], y=[S0, dist.p10, dist.p90, S0],
        fill="toself", fillcolor="rgba(25,118,210,0.10)",
        line=dict(width=0), name="10-90th %ile", showlegend=True,
    ))
    # 25-75 band
    fig.add_trace(go.Scatter(
        x=x + x[::-1], y=[S0, dist.p25, dist.p75, S0],
        fill="toself", fillcolor="rgba(25,118,210,0.18)",
        line=dict(width=0), name="25-75th %ile", showlegend=True,
    ))
    # Median line
    fig.add_trace(go.Scatter(
        x=x, y=[S0, dist.median],
        mode="lines", line=dict(color="#1976D2", width=2),
        name=f"Median ${dist.median:,.2f}",
    ))
    # Current price line
    fig.add_hline(y=S0, line_dash="dash", line_color="gray",
                  annotation_text=f"Current ${S0:,.2f}")

    fig.update_layout(
        height=350,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(title="Trading Days"),
        yaxis=dict(title="Price", tickprefix="$"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_mc_paths(paths: np.ndarray, current_price: float) -> None:
    fig = go.Figure()
    n_show = min(50, paths.shape[0])
    x = list(range(paths.shape[1]))

    for i in range(n_show):
        fig.add_trace(go.Scatter(
            x=x, y=paths[i],
            mode="lines",
            line=dict(width=0.5, color="rgba(25,118,210,0.20)"),
            showlegend=False,
        ))

    if current_price > 0:
        fig.add_hline(y=current_price, line_dash="dash", line_color="red",
                      annotation_text=f"Current ${current_price:,.2f}")

    fig.update_layout(
        height=300,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(title="Trading Days"),
        yaxis=dict(title="Price", tickprefix="$"),
    )
    st.plotly_chart(fig, use_container_width=True)
