"""Technical analysis UI tab."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from core.types import TechnicalResult
from ui.components import fmt_num, render_warnings


def render_technical(result: TechnicalResult, ticker: str = "") -> None:
    render_warnings(result.score.warnings)

    ind = result.indicator_df

    if ind is None or ind.empty:
        st.info("No indicator data available.")
        return

    # ── Signal chips ──────────────────────────────────────────────────────────
    st.subheader("Signals")
    sig_cols = st.columns(4)

    def _signal_chip(label: str, value: str, idx: int) -> None:
        color = "#00C853" if "bullish" in value.lower() or "above" in value.lower() or "golden" in value.lower() else \
                "#D50000" if "bearish" in value.lower() or "below" in value.lower() or "death" in value.lower() else \
                "#FFD600"
        sig_cols[idx % 4].markdown(
            f'<div style="background:{color}22;border-left:3px solid {color};'
            f'padding:6px 10px;border-radius:4px;margin:4px 0;font-size:0.85rem">'
            f'<strong>{label}</strong>: {value}</div>',
            unsafe_allow_html=True,
        )

    i = 0
    for k, v in result.ma_signals.items():
        _signal_chip(k.replace("_", " ").title(), v, i)
        i += 1

    if result.rsi_14 is not None:
        rsi_label = "Overbought" if result.rsi_14 > 70 else "Oversold" if result.rsi_14 < 30 else "Neutral"
        _signal_chip("RSI", f"{result.rsi_14:.1f} ({rsi_label})", i)
        i += 1

    if result.macd_histogram is not None:
        macd_sig = "Bullish" if result.macd_histogram > 0 else "Bearish"
        _signal_chip("MACD Histogram", f"{result.macd_histogram:.3f} ({macd_sig})", i)
        i += 1

    if result.abnormal_volume:
        _signal_chip("Volume", f"{result.volume_vs_avg_20d:.1f}x avg — abnormal", i)
        i += 1

    # ── Main price chart ──────────────────────────────────────────────────────
    st.subheader("Price Chart")
    recent = ind.last("504B") if len(ind) >= 100 else ind  # ~2 years

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.6, 0.2, 0.2],
        vertical_spacing=0.02,
        subplot_titles=("Price & MAs", "RSI (14)", "MACD"),
    )

    p_col = "price"

    # Price & MAs
    fig.add_trace(go.Scatter(x=recent.index, y=recent[p_col], name="Price",
                              line=dict(color="#1976D2", width=1.5)), row=1, col=1)

    for col, color, name in [
        ("sma_20", "#FFA726", "SMA 20"),
        ("sma_50", "#66BB6A", "SMA 50"),
        ("sma_200", "#EF5350", "SMA 200"),
    ]:
        if col in recent.columns:
            fig.add_trace(go.Scatter(x=recent.index, y=recent[col], name=name,
                                      line=dict(color=color, width=1, dash="dot")), row=1, col=1)

    # Bollinger Bands
    if "bb_upper" in recent.columns and "bb_lower" in recent.columns:
        fig.add_trace(go.Scatter(x=recent.index, y=recent["bb_upper"],
                                  name="BB Upper", line=dict(color="gray", width=1, dash="dash"),
                                  showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=recent.index, y=recent["bb_lower"],
                                  name="BB Lower", line=dict(color="gray", width=1, dash="dash"),
                                  fill="tonexty", fillcolor="rgba(150,150,150,0.05)",
                                  showlegend=False), row=1, col=1)

    # Support & resistance
    for level in result.support_levels[:3]:
        fig.add_hline(y=level, line_dash="dot", line_color="#00C853",
                      annotation_text=f"S {level:.2f}", row=1, col=1)
    for level in result.resistance_levels[:3]:
        fig.add_hline(y=level, line_dash="dot", line_color="#D50000",
                      annotation_text=f"R {level:.2f}", row=1, col=1)

    # RSI
    if "rsi_14" in recent.columns:
        fig.add_trace(go.Scatter(x=recent.index, y=recent["rsi_14"], name="RSI",
                                  line=dict(color="#AB47BC", width=1.5)), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red",   row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

    # MACD
    if "macd_histogram" in recent.columns:
        colors = ["#00C853" if v > 0 else "#D50000" for v in recent["macd_histogram"].fillna(0)]
        fig.add_trace(go.Bar(x=recent.index, y=recent["macd_histogram"],
                              name="MACD Histogram", marker_color=colors), row=3, col=1)
    if "macd_line" in recent.columns:
        fig.add_trace(go.Scatter(x=recent.index, y=recent["macd_line"], name="MACD",
                                  line=dict(color="#1976D2", width=1)), row=3, col=1)
    if "macd_signal" in recent.columns:
        fig.add_trace(go.Scatter(x=recent.index, y=recent["macd_signal"], name="Signal",
                                  line=dict(color="#FF9800", width=1)), row=3, col=1)

    fig.update_layout(
        height=600,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    fig.update_xaxes(rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

    # ── Latest values ─────────────────────────────────────────────────────────
    st.subheader("Current Indicator Values")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.write("**Moving Averages**")
        st.write(f"SMA 20: {fmt_num(result.sma_20, 2)}")
        st.write(f"SMA 50: {fmt_num(result.sma_50, 2)}")
        st.write(f"SMA 200: {fmt_num(result.sma_200, 2)}")
        st.write(f"EMA 20: {fmt_num(result.ema_20, 2)}")
    with c2:
        st.write("**Oscillators**")
        st.write(f"RSI (14): {fmt_num(result.rsi_14, 1)}")
        st.write(f"MACD Line: {fmt_num(result.macd_line, 3)}")
        st.write(f"MACD Signal: {fmt_num(result.macd_signal, 3)}")
        st.write(f"Bollinger %B: {fmt_num(result.bollinger_pct_b, 3)}")
    with c3:
        st.write("**Volume & Levels**")
        st.write(f"VWAP: {fmt_num(result.vwap, 2)}")
        st.write(f"Vol vs 20d Avg: {fmt_num(result.volume_vs_avg_20d, 2)}x")
        st.write(f"Supports: {', '.join(f'{s:.2f}' for s in result.support_levels[:3])}")
        st.write(f"Resistances: {', '.join(f'{r:.2f}' for r in result.resistance_levels[:3])}")
