"""Reusable Streamlit UI components."""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.types import CompositeResult, Rating


# ── Color helpers ─────────────────────────────────────────────────────────────

RATING_COLORS = {
    Rating.STRONG_BUY:  "#00C853",
    Rating.BUY:         "#69F0AE",
    Rating.HOLD:        "#FFD600",
    Rating.SELL:        "#FF6D00",
    Rating.STRONG_SELL: "#D50000",
}

RATING_BG = {
    Rating.STRONG_BUY:  "#E8F5E9",
    Rating.BUY:         "#F1F8E9",
    Rating.HOLD:        "#FFFDE7",
    Rating.SELL:        "#FFF3E0",
    Rating.STRONG_SELL: "#FFEBEE",
}

def score_color(score: float) -> str:
    if score >= 65:
        return "#00C853"
    elif score >= 45:
        return "#FFD600"
    else:
        return "#D50000"


def pct_color(pct: Optional[float]) -> str:
    if pct is None:
        return "gray"
    return "#00C853" if pct >= 0 else "#D50000"


def fmt_pct(v: Optional[float], decimals: int = 1) -> str:
    if v is None:
        return "N/A"
    return f"{v * 100:+.{decimals}f}%"


def fmt_num(v: Optional[float], decimals: int = 2) -> str:
    if v is None:
        return "N/A"
    return f"{v:.{decimals}f}"


def fmt_large(v: Optional[float]) -> str:
    """Format large numbers with B/M/K suffix."""
    if v is None:
        return "N/A"
    abs_v = abs(v)
    if abs_v >= 1e12:
        return f"${v/1e12:.2f}T"
    if abs_v >= 1e9:
        return f"${v/1e9:.2f}B"
    if abs_v >= 1e6:
        return f"${v/1e6:.1f}M"
    return f"${v:,.0f}"


# ── Banner ────────────────────────────────────────────────────────────────────

def render_banner(composite: CompositeResult, name: str, ticker: str, price: float) -> None:
    color = RATING_COLORS[composite.rating]
    bg    = RATING_BG[composite.rating]

    bands_str = ""
    if len(composite.possible_ratings) > 1:
        alt = " / ".join(r.value for r in composite.possible_ratings)
        bands_str = f" ({alt})"

    st.markdown(f"""
    <div style="background:{bg};border-left:6px solid {color};padding:16px 20px;border-radius:8px;margin-bottom:16px">
        <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap">
            <div>
                <span style="font-size:1.8rem;font-weight:700;color:#1a1a1a">{ticker.upper()}</span>
                <span style="font-size:1rem;color:#555;margin-left:12px">{name}</span>
            </div>
            <div style="font-size:1.6rem;font-weight:600;color:#1a1a1a">${price:,.2f}</div>
        </div>
        <div style="margin-top:10px;display:flex;align-items:center;gap:16px;flex-wrap:wrap">
            <span style="background:{color};color:white;padding:4px 14px;border-radius:20px;font-weight:700;font-size:1.1rem">
                {composite.rating.value}{bands_str}
            </span>
            <span style="color:#555">Score: <strong>{composite.overall_score:.1f}/100</strong></span>
            <span style="color:#555">Confidence: <strong>{composite.confidence:.0%}</strong></span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── Composite decomposition chart ─────────────────────────────────────────────

def render_score_decomposition(composite: CompositeResult) -> None:
    names = []
    scores = []
    weighted = []
    weights = []

    for ms in composite.module_scores:
        w = composite.weights_used.get(ms.name, 0.0)
        names.append(ms.name.title())
        scores.append(ms.score)
        weighted.append(composite.component_scores.get(ms.name, 0.0))
        weights.append(w)

    col1, col2 = st.columns([2, 1])

    with col1:
        fig = go.Figure()
        bar_colors = [score_color(s) for s in scores]
        fig.add_trace(go.Bar(
            x=scores,
            y=names,
            orientation="h",
            marker_color=bar_colors,
            text=[f"{s:.1f}" for s in scores],
            textposition="outside",
            name="Raw Score",
        ))
        fig.update_layout(
            title="Module Scores (0–100)",
            xaxis=dict(range=[0, 110], title="Score"),
            height=280,
            margin=dict(l=10, r=40, t=40, b=10),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        df_decomp = pd.DataFrame({
            "Module": names,
            "Score": [f"{s:.1f}" for s in scores],
            "Weight": [f"{w:.0%}" for w in weights],
            "Contribution": [f"{c:.1f}" for c in weighted],
        })
        st.dataframe(df_decomp, use_container_width=True, hide_index=True)


# ── Metric card row ───────────────────────────────────────────────────────────

def metric_row(metrics: list[tuple[str, str, str | None]]) -> None:
    """Display a row of metric cards. Each tuple: (label, value, delta)."""
    cols = st.columns(len(metrics))
    for col, (label, value, delta) in zip(cols, metrics):
        with col:
            if delta:
                st.metric(label=label, value=value, delta=delta)
            else:
                st.metric(label=label, value=value)


# ── Warnings display ──────────────────────────────────────────────────────────

def render_warnings(warnings: list[str], title: str = "Data Warnings") -> None:
    if not warnings:
        return
    with st.expander(f"⚠ {title} ({len(warnings)})", expanded=False):
        for w in warnings:
            st.caption(f"• {w}")


# ── Multiples table ───────────────────────────────────────────────────────────

def render_multiples_table(snaps: list) -> None:
    """Render a table of MultiplesSnapshot objects."""
    from core.types import MultiplesSnapshot
    rows = []
    for s in snaps:
        if s is None:
            continue
        percentile_str = f"{s.percentile:.0f}th" if s.percentile is not None else "N/A"
        # Color the percentile cell
        if s.percentile is not None and s.percentile >= 80:
            pct_display = f"🔴 {percentile_str}"
        elif s.percentile is not None and s.percentile <= 20:
            pct_display = f"🟢 {percentile_str}"
        else:
            pct_display = f"🟡 {percentile_str}"

        rows.append({
            "Metric": s.name,
            "Current": f"{s.current:.1f}x" if s.current else "N/A",
            "5Y Low": f"{s.historical_5y_low:.1f}x" if s.historical_5y_low else "N/A",
            "5Y Median": f"{s.historical_5y_median:.1f}x" if s.historical_5y_median else "N/A",
            "5Y High": f"{s.historical_5y_high:.1f}x" if s.historical_5y_high else "N/A",
            "Percentile": pct_display,
        })
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ── Radar chart for summary tab ───────────────────────────────────────────────

def render_radar(composite: CompositeResult) -> None:
    names = [ms.name.title() for ms in composite.module_scores]
    scores = [ms.score for ms in composite.module_scores]
    if not names:
        return

    fig = go.Figure(go.Scatterpolar(
        r=scores + [scores[0]],
        theta=names + [names[0]],
        fill="toself",
        line_color="#1976D2",
        fillcolor="rgba(25,118,210,0.15)",
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=False,
        height=300,
        margin=dict(l=40, r=40, t=20, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)


# ── Mini sparkline ────────────────────────────────────────────────────────────

def mini_line(series: pd.Series, color: str = "#1976D2", height: int = 120) -> go.Figure:
    fig = go.Figure(go.Scatter(
        x=series.index, y=series.values,
        mode="lines",
        line=dict(color=color, width=2),
        fill="tozeroy",
        fillcolor=f"rgba(25,118,210,0.08)",
    ))
    fig.update_layout(
        height=height,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig
