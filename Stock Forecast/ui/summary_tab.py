"""Summary / composite score UI tab."""
from __future__ import annotations

import streamlit as st

from core.types import CompositeResult
from ui.components import render_radar, render_score_decomposition, render_warnings


def render_summary(composite: CompositeResult) -> None:
    col1, col2 = st.columns([1, 2])

    with col1:
        render_radar(composite)

    with col2:
        render_score_decomposition(composite)

    # ── All warnings aggregated ───────────────────────────────────────────────
    all_warnings: list[str] = []
    for ms in composite.module_scores:
        for w in ms.warnings:
            all_warnings.append(f"[{ms.name.title()}] {w}")

    if all_warnings:
        render_warnings(all_warnings, "All Module Warnings")

    # ── Confidence explanation ────────────────────────────────────────────────
    band_low, band_high = composite.confidence_band
    st.caption(
        f"Confidence band: {band_low:.1f}–{band_high:.1f} "
        f"(overall confidence: {composite.confidence:.0%}). "
        "A wide band reflects incomplete data. Component scores are unweighted for direct comparison."
    )

    # ── Disclaimer ────────────────────────────────────────────────────────────
    st.markdown(
        """---
        <small>
        <strong>Disclaimer:</strong> This tool is for informational and educational purposes only.
        It does not constitute investment advice, a recommendation to buy or sell any security,
        or a solicitation of any transaction. All analytical outputs are model-derived and subject
        to model error, data error, survivorship bias, look-ahead bias, and regime changes.
        Users are solely responsible for their investment decisions.
        </small>""",
        unsafe_allow_html=True,
    )
