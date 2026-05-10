"""Valuation module UI tab."""
from __future__ import annotations

from typing import Optional

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from core.types import ValuationResult
from ui.components import fmt_pct, fmt_num, render_warnings, render_multiples_table


def render_valuation(result: ValuationResult) -> None:
    render_warnings(result.score.warnings)

    current = result.current_price

    # ── Fair Value Range Bar ──────────────────────────────────────────────────
    if result.fair_value_mid is not None:
        st.subheader("Fair Value Estimate")
        col1, col2, col3, col4 = st.columns(4)
        upside_str = fmt_pct(result.upside_pct) if result.upside_pct is not None else "N/A"
        upside_color = "normal" if (result.upside_pct or 0) >= 0 else "inverse"
        col1.metric("Current Price", f"${current:,.2f}")
        col2.metric("Fair Value (Low)", f"${result.fair_value_low:,.2f}" if result.fair_value_low else "N/A")
        col3.metric("Fair Value (Mid)", f"${result.fair_value_mid:,.2f}")
        col4.metric("Upside / Downside", upside_str, delta_color=upside_color)

        # Visual bar
        fvl = result.fair_value_low  or current * 0.9
        fvh = result.fair_value_high or current * 1.1
        fvm = result.fair_value_mid
        fig = go.Figure()
        fig.add_shape(type="rect", x0=fvl, x1=fvh, y0=0, y1=1,
                      fillcolor="rgba(25,118,210,0.15)", line_width=0)
        fig.add_vline(x=current, line_color="red", line_dash="dash",
                      annotation_text=f"Current ${current:.2f}", annotation_position="top right")
        fig.add_vline(x=fvm, line_color="#1976D2", line_width=2,
                      annotation_text=f"Fair Value ${fvm:.2f}", annotation_position="top left")
        x_min = min(fvl * 0.85, current * 0.85)
        x_max = max(fvh * 1.15, current * 1.15)
        fig.update_layout(
            height=100,
            margin=dict(l=20, r=20, t=30, b=10),
            xaxis=dict(range=[x_min, x_max], tickprefix="$"),
            yaxis=dict(visible=False),
            plot_bgcolor="white",
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Multiples Table ───────────────────────────────────────────────────────
    st.subheader("Valuation Multiples vs. History")
    snaps = [result.pe_trailing, result.pe_forward, result.ev_ebitda,
             result.ev_sales, result.pb, result.fcf_yield]
    render_multiples_table([s for s in snaps if s is not None])

    # ── DCF ───────────────────────────────────────────────────────────────────
    if result.dcf is not None:
        st.subheader("Discounted Cash Flow")
        dcf = result.dcf
        st.metric("DCF Fair Value / Share", f"${dcf.fair_value:,.2f}")
        with st.expander("DCF Assumptions"):
            a = dcf.assumptions
            cols = st.columns(4)
            cols[0].metric("WACC", fmt_pct(a.get("wacc")))
            cols[1].metric("Terminal Growth", fmt_pct(a.get("terminal_growth")))
            cols[2].metric("Projection Years", str(int(a.get("projection_years", 7))))
            cols[3].metric("FCF Margin", fmt_pct(a.get("fcf_margin")))

        st.write("**Sensitivity Table** (DCF Fair Value per Share)")
        df = dcf.sensitivity_table
        if not df.empty:
            # Format as dollars
            styled = df.style.format("${:.2f}").background_gradient(
                cmap="RdYlGn", axis=None
            )
            try:
                st.dataframe(styled, use_container_width=True)
            except Exception:
                st.dataframe(df.applymap(lambda x: f"${x:.2f}"), use_container_width=True)

    # ── Reverse DCF ───────────────────────────────────────────────────────────
    if result.reverse_dcf is not None:
        st.subheader("Reverse DCF — What Is the Market Pricing In?")
        rdcf = result.reverse_dcf
        c1, c2, c3 = st.columns(3)
        c1.metric("Implied Growth Rate", fmt_pct(rdcf.implied_growth_rate))
        c2.metric("Consensus Growth",
                  fmt_pct(rdcf.consensus_growth_rate) if rdcf.consensus_growth_rate is not None else "N/A")
        c3.metric("Historical Growth (5Y)",
                  fmt_pct(rdcf.historical_growth_rate) if rdcf.historical_growth_rate is not None else "N/A")

        impl = rdcf.implied_growth_rate
        hist = rdcf.historical_growth_rate
        if impl is not None and hist is not None:
            if impl > hist * 1.5 and impl > 0.10:
                st.warning(
                    f"Market is pricing in {impl:.1%} annual growth — "
                    f"significantly above the {hist:.1%} historical rate. "
                    "This implies substantial expectations for acceleration."
                )
            elif impl < hist * 0.5 and hist > 0.03:
                st.success(
                    f"Market is pricing in only {impl:.1%} growth vs. "
                    f"{hist:.1%} historical — potential value opportunity."
                )
