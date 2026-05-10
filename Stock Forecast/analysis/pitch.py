"""Investment pitch narrative generator.

Uses Anthropic when configured, Groq as a fallback when configured, and a
deterministic local template when no API key is available.

Output structure:
  - Investment thesis paragraph
  - 3–5 catalysts
  - 3–5 key risks
  - Valuation summary sentence
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import anthropic
import groq as groq_sdk

from core.types import (
    CompanyInfo, CompositeResult, FundamentalResult,
    ForecastResult, RiskResult, ValuationResult, TechnicalResult,
)
from data_layer.edgar import NewsAndFilings

logger = logging.getLogger(__name__)

_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")

_SYSTEM_PROMPT = """\
Act as a top-tier equity research analyst at a leading investment firm. \
Write a clear, concise, and compelling investment pitch for the company described by the data below. \
The pitch should be easy to understand for a non-expert audience while still demonstrating strong financial reasoning. \
Spell out all acronyms and avoid jargon — write as if explaining to a smart friend, not a finance professor.

Format your response using EXACTLY these section headers (no deviations):

## Company Overview
(2–3 sentences) Briefly explain what the company does, the market it operates in, and its competitive position.

## Investment Thesis
(3–5 bullet points) The core reasons this is a strong investment. Focus on durable advantages, growth drivers, and competitive positioning. Write in plain English — no acronyms.

## Key Catalysts
(3–4 bullet points) Upcoming events or trends that could drive the stock price higher. Be specific and forward-looking.

## Risks
(3–4 bullet points) Realistic downside risks — market, operational, or macroeconomic. Be honest, not dismissive.

## Valuation Perspective
(1 short paragraph) Explain whether the stock appears undervalued, fairly valued, or overvalued. \
Reference simple metrics like price-to-earnings ratio, free cash flow yield, or how it compares to peers. Avoid abbreviations.

## Conclusion
(2–3 sentences) Provide a clear buy / hold / sell recommendation with a brief justification tied to the data.

Rules:
- Total response must be 400–500 words.
- Tone: sharp, confident, professional — like a pitch to sophisticated but non-specialist investors.
- No disclaimers, no caveats about model limitations, no meta-commentary.
- Every claim must be grounded in the data provided."""


@dataclass
class PitchResult:
    ticker: str
    company_name: str
    pitch_text: str                          # Full markdown from the model
    company_overview: str = ""
    investment_thesis: list[str] = field(default_factory=list)
    key_catalysts: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    valuation_perspective: str = ""
    conclusion: str = ""
    model_used: str = _MODEL
    generated_at: datetime = field(default_factory=datetime.utcnow)
    token_usage: dict = field(default_factory=dict)


def _fmt_pct(v: Optional[float]) -> str:
    return f"{v*100:+.1f}%" if v is not None else "N/A"


def _fmt_x(v: Optional[float]) -> str:
    return f"{v:.1f}x" if v is not None else "N/A"


def _build_data_block(
    ticker: str,
    info: CompanyInfo,
    composite: CompositeResult,
    fundamental: Optional[FundamentalResult],
    valuation: Optional[ValuationResult],
    technical: Optional[TechnicalResult],
    forecast: Optional[ForecastResult],
    risk: Optional[RiskResult],
    news_filings: Optional[NewsAndFilings],
    current_price: float,
) -> str:
    """Serialize all relevant numbers into a compact text block for the prompt."""
    lines: list[str] = [
        f"TICKER: {ticker}",
        f"COMPANY: {info.name}",
        f"SECTOR/INDUSTRY: {info.sector} / {info.industry}",
        f"MARKET CAP: ${info.market_cap/1e9:.1f}B" if info.market_cap else "MARKET CAP: N/A",
        f"CURRENT PRICE: ${current_price:.2f}",
        f"COMPOSITE SCORE: {composite.overall_score:.1f}/100",
        f"RATING: {composite.rating.value}",
        f"CONFIDENCE: {composite.confidence:.0%}",
        "",
        "--- MODULE SCORES ---",
    ]
    for ms in composite.module_scores:
        lines.append(f"  {ms.name.upper()}: {ms.score:.1f}/100")

    if fundamental:
        lines += [
            "",
            "--- FUNDAMENTALS ---",
            f"  Revenue CAGR 5Y: {_fmt_pct(fundamental.revenue_cagr_5y)}",
            f"  EPS CAGR 5Y:     {_fmt_pct(fundamental.eps_cagr_5y)}",
            f"  Gross Margin:    {_fmt_pct(fundamental.gross_margin)}",
            f"  Operating Margin:{_fmt_pct(fundamental.operating_margin)}",
            f"  Net Margin:      {_fmt_pct(fundamental.net_margin)}",
            f"  FCF Margin:      {_fmt_pct(fundamental.fcf_margin)}",
            f"  ROIC:            {_fmt_pct(fundamental.roic)}",
            f"  ROE:             {_fmt_pct(fundamental.roe)}",
            f"  Net Debt/EBITDA: {_fmt_x(fundamental.net_debt_to_ebitda)}",
            f"  Interest Coverage:{_fmt_x(fundamental.interest_coverage)}",
        ]

    if valuation:
        lines += [
            "",
            "--- VALUATION ---",
        ]
        if valuation.pe_trailing:
            lines.append(f"  P/E (TTM): {_fmt_x(valuation.pe_trailing.current)} "
                         f"(5Y median {_fmt_x(valuation.pe_trailing.historical_5y_median)})")
        if valuation.pe_forward:
            lines.append(f"  P/E (Fwd): {_fmt_x(valuation.pe_forward.current)}")
        if valuation.ev_ebitda:
            lines.append(f"  EV/EBITDA: {_fmt_x(valuation.ev_ebitda.current)}")
        if valuation.fcf_yield:
            lines.append(f"  FCF Yield: {_fmt_x(valuation.fcf_yield.current)}")
        if valuation.fair_value_mid and valuation.current_price:
            lines.append(f"  DCF Fair Value (mid): ${valuation.fair_value_mid:.2f} "
                         f"(upside: {_fmt_pct(valuation.upside_pct)})")
        if valuation.reverse_dcf:
            lines.append(f"  Implied Growth (Reverse DCF): "
                         f"{_fmt_pct(valuation.reverse_dcf.implied_growth_rate)}")

    if forecast:
        ens_12 = forecast.ensemble.get(12) or forecast.monte_carlo.get(12)
        if ens_12:
            lines += [
                "",
                "--- 12-MONTH FORECAST (ENSEMBLE) ---",
                f"  Mean Return:      {_fmt_pct(ens_12.mean / current_price - 1 if current_price else None)}",
                f"  P(Positive):      {ens_12.prob_positive:.0%}",
                f"  P10/P90 range:    ${ens_12.p10:.2f} – ${ens_12.p90:.2f}",
            ]

    if risk:
        lines += [
            "",
            "--- RISK ---",
            f"  30d Realized Vol:  {_fmt_pct(risk.realized_vol_30d)}",
            f"  Beta (vs SPY):     {f'{risk.beta_spy:.2f}' if risk.beta_spy else 'N/A'}",
            f"  Sharpe 1Y:         {f'{risk.sharpe_1y:.2f}' if risk.sharpe_1y else 'N/A'}",
            f"  Max Drawdown 1Y:   {_fmt_pct(risk.max_drawdown_1y)}",
            f"  Quarter-Kelly:     {_fmt_pct(risk.quarter_kelly)}",
        ]

    if technical:
        lines += [
            "",
            "--- TECHNICAL ---",
            f"  RSI (14d): {f'{technical.rsi_14:.1f}' if technical.rsi_14 else 'N/A'}",
            f"  Price vs 200d MA: {'above' if technical.sma_200 and current_price > technical.sma_200 else 'below'}",
        ]
        if technical.ma_signals:
            sigs = ", ".join(f"{k}:{v}" for k, v in list(technical.ma_signals.items())[:4])
            lines.append(f"  MA Signals: {sigs}")

    if news_filings:
        recent_headlines = [n.title for n in news_filings.news[:8] if n.title]
        if recent_headlines:
            lines += ["", "--- RECENT NEWS HEADLINES ---"]
            for h in recent_headlines:
                lines.append(f"  • {h}")

        catalyst_signals = news_filings.all_catalysts
        if catalyst_signals:
            lines += ["", f"--- DETECTED CATALYST SIGNALS: {', '.join(catalyst_signals)} ---"]

        recent_filings = news_filings.filings[:5]
        if recent_filings:
            lines += ["", "--- RECENT SEC FILINGS ---"]
            for f in recent_filings:
                date_str = f.filed_at.strftime("%Y-%m-%d") if f.filed_at else "?"
                lines.append(f"  {f.form_type} ({date_str}): {f.description}")

        if news_filings.risk_factors:
            lines += ["", "--- KEY 10-K RISK FACTORS (excerpts) ---"]
            for rf in news_filings.risk_factors[:6]:
                lines.append(f"  • {rf[:250]}")

    return "\n".join(lines)


def _parse_sections(text: str) -> dict[str, str]:
    """Extract named sections from the structured pitch text."""
    sections: dict[str, str] = {}
    current_key: str | None = None
    current_lines: list[str] = []

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("## "):
            if current_key:
                sections[current_key] = "\n".join(current_lines).strip()
            current_key = stripped[3:].strip()
            current_lines = []
        else:
            current_lines.append(line)

    if current_key:
        sections[current_key] = "\n".join(current_lines).strip()

    return sections


def _extract_bullets(text: str) -> list[str]:
    """Extract bullet-point lines from a section body."""
    bullets: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("- ") or stripped.startswith("• ") or stripped.startswith("* "):
            bullets.append(stripped[2:].strip())
    return bullets


def _generate_template_pitch(
    ticker: str,
    info: CompanyInfo,
    composite: CompositeResult,
    current_price: float,
    fundamental: Optional[FundamentalResult],
    valuation: Optional[ValuationResult],
    technical: Optional[TechnicalResult],
    forecast: Optional[ForecastResult],
    risk: Optional[RiskResult],
    news_filings: Optional[NewsAndFilings],
) -> PitchResult:
    """Build a structured pitch from templates — no API key required."""

    rating   = composite.rating.value
    score    = composite.overall_score
    conf     = composite.confidence

    # ── Investment Thesis ─────────────────────────────────────────────────────
    thesis_parts: list[str] = [
        f"{info.name} ({ticker}) scores {score:.0f}/100 on our composite model, "
        f"yielding a **{rating}** rating at {conf:.0%} confidence."
    ]

    if fundamental:
        if fundamental.revenue_cagr_5y is not None:
            direction = "growing" if fundamental.revenue_cagr_5y > 0 else "contracting"
            thesis_parts.append(
                f"Revenue has been {direction} at a {abs(fundamental.revenue_cagr_5y)*100:.1f}% "
                f"5-year CAGR"
                + (f" with {fundamental.net_margin*100:.1f}% net margins." if fundamental.net_margin else ".")
            )
        if fundamental.roic is not None:
            quality = "strong" if fundamental.roic > 0.15 else ("adequate" if fundamental.roic > 0.08 else "weak")
            thesis_parts.append(f"Capital returns are {quality} with ROIC of {fundamental.roic*100:.1f}%.")

    if valuation and valuation.upside_pct is not None:
        direction = "upside" if valuation.upside_pct >= 0 else "downside"
        thesis_parts.append(
            f"Our DCF mid-point implies {abs(valuation.upside_pct)*100:.1f}% {direction} "
            f"from the current price of ${current_price:.2f}."
        )

    # ── Company Overview ──────────────────────────────────────────────────────
    company_overview = (
        f"{info.name} is a {info.sector.lower()} company operating in the {info.industry} space"
        + (f", headquartered in {info.country}" if info.country else "")
        + "."
    )
    if info.description:
        company_overview += " " + info.description[:300].rstrip(".") + "."

    # ── Investment Thesis ─────────────────────────────────────────────────────
    investment_thesis: list[str] = []

    if fundamental:
        if fundamental.revenue_cagr_5y is not None and fundamental.revenue_cagr_5y > 0.05:
            investment_thesis.append(
                f"Strong revenue growth — the company has grown its top line at "
                f"{fundamental.revenue_cagr_5y*100:.1f}% per year over the past five years, "
                f"well above average for its sector."
            )
        if fundamental.roic is not None and fundamental.roic > 0.12:
            investment_thesis.append(
                f"High returns on invested capital of {fundamental.roic*100:.1f}% indicate "
                f"the business earns significantly more than its cost of capital — a hallmark of quality companies."
            )
        if fundamental.fcf_margin is not None and fundamental.fcf_margin > 0.10:
            investment_thesis.append(
                f"Healthy free cash flow margin of {fundamental.fcf_margin*100:.1f}% means the business "
                f"generates real cash after reinvestment — funding dividends, buybacks, or further growth."
            )
        if fundamental.net_debt_to_ebitda is not None and fundamental.net_debt_to_ebitda < 1.5:
            investment_thesis.append(
                f"Conservative balance sheet with net debt less than {fundamental.net_debt_to_ebitda:.1f} times "
                f"annual operating earnings, giving the company flexibility in a downturn."
            )
        if fundamental.operating_margin is not None and fundamental.operating_margin > 0.20:
            investment_thesis.append(
                f"Wide operating margins of {fundamental.operating_margin*100:.1f}% suggest durable pricing "
                f"power and a strong competitive position."
            )

    if not investment_thesis:
        investment_thesis = [
            f"The company scores {score:.0f} out of 100 on our composite model, placing it in the {rating} category.",
            "Management has demonstrated the ability to generate returns above the cost of capital over time.",
            "The business model benefits from recurring revenue or structural demand tailwinds in its sector.",
        ]

    investment_thesis = investment_thesis[:5]

    # ── Key Catalysts ─────────────────────────────────────────────────────────
    key_catalysts: list[str] = []

    if news_filings:
        signal_labels = {
            "earnings_beat":  "The company recently beat earnings expectations, suggesting the business is performing better than analysts anticipated.",
            "guidance_raise": "Management raised its forward guidance, signaling confidence in near-term business momentum.",
            "buyback":        "An active share repurchase program is underway, which reduces the number of shares outstanding and boosts value per share.",
            "dividend":       "A recent dividend increase reflects strong cash generation and a shareholder-friendly capital allocation strategy.",
            "ma":             "Merger and acquisition activity could create strategic value or expand the company's addressable market.",
            "product_launch": "A new product launch or service expansion has the potential to open up additional revenue streams.",
            "regulatory":     "A recent regulatory approval unlocks a significant new market opportunity.",
        }
        for signal in news_filings.all_catalysts:
            label = signal_labels.get(signal)
            if label and label not in key_catalysts:
                key_catalysts.append(label)

        for n in news_filings.news[:4]:
            if n.title and len(n.title) > 20 and len(key_catalysts) < 4:
                key_catalysts.append(f"Recent news: {n.title[:130]}")

    if forecast:
        ens_12 = forecast.ensemble.get(12) or forecast.monte_carlo.get(12)
        if ens_12 and ens_12.prob_positive > 0.60:
            key_catalysts.append(
                f"Our forecast models assign a {ens_12.prob_positive:.0%} probability of a positive "
                f"12-month return, with a projected price range of ${ens_12.p10:.2f} to ${ens_12.p90:.2f}."
            )

    if technical and technical.rsi_14 is not None and technical.rsi_14 < 40:
        key_catalysts.append(
            f"The stock appears technically oversold with a Relative Strength Index reading of "
            f"{technical.rsi_14:.0f} — historically a setup for a rebound."
        )

    if not key_catalysts:
        key_catalysts = [
            "The upcoming earnings release could serve as a catalyst if results beat current expectations.",
            "A broader sector recovery or easing interest rates could re-rate the stock higher.",
            "Continued share buybacks or a dividend initiation could attract income-focused investors.",
        ]

    key_catalysts = key_catalysts[:4]

    # ── Risks ─────────────────────────────────────────────────────────────────
    risks: list[str] = []

    if news_filings:
        miss_signals = [s for s in news_filings.all_catalysts if s in ("earnings_miss", "guidance_cut")]
        if miss_signals:
            risks.append("A recent earnings miss or guidance reduction suggests near-term business conditions may be deteriorating.")
        if news_filings.risk_factors:
            for rf in news_filings.risk_factors[:2]:
                cleaned = rf[:220].rstrip(".")
                if len(cleaned) > 60:
                    risks.append(cleaned + ".")

    if risk:
        if risk.realized_vol_30d is not None and risk.realized_vol_30d > 0.35:
            risks.append(
                f"The stock has been highly volatile — moving at an annualised rate of "
                f"{risk.realized_vol_30d*100:.0f}% — which could mean sharp swings in either direction."
            )
        if risk.max_drawdown_1y is not None and risk.max_drawdown_1y < -0.25:
            risks.append(
                f"The stock fell as much as {abs(risk.max_drawdown_1y)*100:.0f}% from its peak in the past year, "
                f"indicating meaningful downside when sentiment turns."
            )
        if risk.beta_spy is not None and risk.beta_spy > 1.4:
            risks.append(
                f"With a market sensitivity (beta) of {risk.beta_spy:.1f}, this stock tends to fall "
                f"harder than the broader market during sell-offs."
            )

    if fundamental:
        if fundamental.net_debt_to_ebitda is not None and fundamental.net_debt_to_ebitda > 3.0:
            risks.append(
                f"The company carries significant debt — approximately {fundamental.net_debt_to_ebitda:.1f} times "
                f"its annual operating earnings — which could become a burden if interest rates rise or earnings fall."
            )

    if valuation and valuation.pe_trailing and valuation.pe_trailing.percentile is not None:
        if valuation.pe_trailing.percentile > 80:
            risks.append(
                f"The stock's price-to-earnings ratio is near the top of its five-year historical range "
                f"(at the {valuation.pe_trailing.percentile:.0f}th percentile), leaving little room for the valuation to expand further."
            )

    if not risks:
        risks = [
            "Macroeconomic slowdown or rising interest rates could weigh on both earnings and the stock's valuation.",
            "Competitive pressure from new entrants or larger players could erode market share over time.",
            "Execution risk remains if the company fails to deliver on its growth strategy.",
        ]

    risks = risks[:4]

    # ── Valuation Perspective ─────────────────────────────────────────────────
    val_parts: list[str] = []
    if valuation:
        if valuation.pe_trailing:
            val_parts.append(
                f"a price-to-earnings ratio of {valuation.pe_trailing.current:.1f} times trailing earnings"
                + (f" (the five-year median is {valuation.pe_trailing.historical_5y_median:.1f} times)"
                   if valuation.pe_trailing.historical_5y_median else "")
            )
        if valuation.ev_ebitda:
            val_parts.append(f"an enterprise value to operating earnings ratio of {valuation.ev_ebitda.current:.1f} times")
        if valuation.fair_value_mid and valuation.upside_pct is not None:
            direction = "above" if valuation.upside_pct < 0 else "below"
            val_parts.append(
                f"our discounted cash flow model suggests a fair value of ${valuation.fair_value_mid:.2f}, "
                f"which is {abs(valuation.upside_pct)*100:.1f}% {direction} the current price"
            )

    if val_parts:
        verdict = "undervalued" if (valuation and valuation.upside_pct and valuation.upside_pct > 0.10) else \
                  "overvalued"  if (valuation and valuation.upside_pct and valuation.upside_pct < -0.10) else \
                  "fairly valued"
        valuation_perspective = (
            f"The stock appears {verdict}, trading at {'; '.join(val_parts)}. "
            f"Compared to its own history and peers, this valuation "
            + ("offers an attractive entry point." if verdict == "undervalued" else
               "leaves limited margin of safety." if verdict == "overvalued" else
               "looks broadly in line with fundamentals.")
        )
    else:
        valuation_perspective = (
            "Valuation data is limited, but the composite model score suggests the "
            f"market's current pricing is broadly consistent with the {rating} rating."
        )

    # ── Conclusion ────────────────────────────────────────────────────────────
    action_map = {
        "Strong Buy": "a Strong Buy",
        "Buy":        "a Buy",
        "Hold":       "a Hold",
        "Sell":       "a Sell",
        "Strong Sell":"a Strong Sell",
    }
    action = action_map.get(rating, "a Hold")
    conclusion = (
        f"Based on the analysis above, we rate {ticker} {action}. "
        f"The composite model assigns a score of {score:.0f} out of 100 with {conf:.0%} confidence. "
        + ("The combination of strong fundamentals and reasonable valuation makes this a compelling opportunity."
           if score >= 65 else
           "We would wait for a better entry point or improvement in fundamentals before adding exposure."
           if score < 45 else
           "We recommend holding current positions while monitoring the next earnings release for directional confirmation.")
    )

    # ── Assemble full text ────────────────────────────────────────────────────
    bullet_fmt = lambda items: "\n".join(f"- {i}" for i in items)
    full_text = (
        f"## Company Overview\n{company_overview}\n\n"
        f"## Investment Thesis\n{bullet_fmt(investment_thesis)}\n\n"
        f"## Key Catalysts\n{bullet_fmt(key_catalysts)}\n\n"
        f"## Risks\n{bullet_fmt(risks)}\n\n"
        f"## Valuation Perspective\n{valuation_perspective}\n\n"
        f"## Conclusion\n{conclusion}"
    )

    return PitchResult(
        ticker=ticker,
        company_name=info.name,
        pitch_text=full_text,
        company_overview=company_overview,
        investment_thesis=investment_thesis,
        key_catalysts=key_catalysts,
        risks=risks,
        valuation_perspective=valuation_perspective,
        conclusion=conclusion,
        model_used="template (no API key)",
        token_usage={},
    )


def _generate_groq_pitch(
    ticker: str,
    info: CompanyInfo,
    composite: CompositeResult,
    current_price: float,
    fundamental: Optional[FundamentalResult],
    valuation: Optional[ValuationResult],
    technical: Optional[TechnicalResult],
    forecast: Optional[ForecastResult],
    risk: Optional[RiskResult],
    news_filings: Optional[NewsAndFilings],
    groq_key: str,
) -> PitchResult:
    """Generate pitch via Groq (free tier). Uses llama-3.3-70b-versatile."""
    client = groq_sdk.Groq(api_key=groq_key)

    data_block = _build_data_block(
        ticker, info, composite, fundamental, valuation,
        technical, forecast, risk, news_filings, current_price,
    )
    user_message = (
        f"Generate a concise investment pitch for {ticker} ({info.name}) "
        f"based on the following quantitative analysis data:\n\n{data_block}"
    )

    full_text = ""
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": user_message},
            ],
            max_tokens=1500,
            temperature=0.4,
            stream=True,
        )
        for chunk in response:
            delta = chunk.choices[0].delta.content or ""
            full_text += delta
    except Exception as exc:
        logger.warning("Groq API failed — falling back to template: %s", exc)
        return _generate_template_pitch(
            ticker, info, composite, current_price,
            fundamental, valuation, technical, forecast, risk, news_filings,
        )

    sections = _parse_sections(full_text)
    return PitchResult(
        ticker=ticker,
        company_name=info.name,
        pitch_text=full_text,
        company_overview=sections.get("Company Overview", ""),
        investment_thesis=_extract_bullets(sections.get("Investment Thesis", "")),
        key_catalysts=_extract_bullets(sections.get("Key Catalysts", "")),
        risks=_extract_bullets(sections.get("Risks", "")),
        valuation_perspective=sections.get("Valuation Perspective", ""),
        conclusion=sections.get("Conclusion", ""),
        model_used="llama-3.3-70b (Groq)",
        token_usage={},
    )


def generate_pitch(
    ticker: str,
    info: CompanyInfo,
    composite: CompositeResult,
    current_price: float,
    fundamental: Optional[FundamentalResult] = None,
    valuation: Optional[ValuationResult] = None,
    technical: Optional[TechnicalResult] = None,
    forecast: Optional[ForecastResult] = None,
    risk: Optional[RiskResult] = None,
    news_filings: Optional[NewsAndFilings] = None,
) -> PitchResult:
    """Generate an investment pitch.

    Priority: Anthropic API → Groq (free) → template (no key needed).
    """
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    groq_key = os.getenv("GROQ_API_KEY", "")

    if not api_key:
        if groq_key:
            logger.info("No Anthropic key — using Groq (free) pitch generator")
            return _generate_groq_pitch(
                ticker, info, composite, current_price,
                fundamental, valuation, technical, forecast, risk, news_filings, groq_key,
            )
        logger.info("No API keys found — using template pitch generator")
        return _generate_template_pitch(
            ticker, info, composite, current_price,
            fundamental, valuation, technical, forecast, risk, news_filings,
        )

    client = anthropic.Anthropic(api_key=api_key)

    data_block = _build_data_block(
        ticker, info, composite, fundamental, valuation,
        technical, forecast, risk, news_filings, current_price,
    )

    user_message = (
        f"Generate a concise investment pitch for {ticker} ({info.name}) "
        f"based on the following quantitative analysis data:\n\n{data_block}"
    )

    full_text = ""
    usage: dict = {}

    try:
        with client.messages.stream(
            model=_MODEL,
            max_tokens=1500,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
            thinking={"type": "adaptive"},
        ) as stream:
            for text_chunk in stream.text_stream:
                full_text += text_chunk

            final = stream.get_final_message()
            usage = {
                "input_tokens":  final.usage.input_tokens,
                "output_tokens": final.usage.output_tokens,
            }
    except anthropic.APIStatusError as exc:
        logger.warning("Claude API error (%s) — trying Groq fallback: %s", exc.status_code, exc.message)
        if groq_key:
            return _generate_groq_pitch(
                ticker, info, composite, current_price,
                fundamental, valuation, technical, forecast, risk, news_filings, groq_key,
            )
        return _generate_template_pitch(
            ticker, info, composite, current_price,
            fundamental, valuation, technical, forecast, risk, news_filings,
        )
    except Exception as exc:
        logger.warning("Claude API call failed — trying Groq fallback: %s", exc)
        if groq_key:
            return _generate_groq_pitch(
                ticker, info, composite, current_price,
                fundamental, valuation, technical, forecast, risk, news_filings, groq_key,
            )
        return _generate_template_pitch(
            ticker, info, composite, current_price,
            fundamental, valuation, technical, forecast, risk, news_filings,
        )

    sections = _parse_sections(full_text)

    return PitchResult(
        ticker=ticker,
        company_name=info.name,
        pitch_text=full_text,
        company_overview=sections.get("Company Overview", ""),
        investment_thesis=_extract_bullets(sections.get("Investment Thesis", "")),
        key_catalysts=_extract_bullets(sections.get("Key Catalysts", "")),
        risks=_extract_bullets(sections.get("Risks", "")),
        valuation_perspective=sections.get("Valuation Perspective", ""),
        conclusion=sections.get("Conclusion", ""),
        token_usage=usage,
    )
