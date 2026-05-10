"""Technical analysis module."""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from config import RSI_OVERBOUGHT, RSI_OVERSOLD, BOLLINGER_WINDOW, BOLLINGER_STD
from core.types import ModuleScore, PriceHistory, TechnicalResult

logger = logging.getLogger(__name__)


def _price_col(df: pd.DataFrame) -> str:
    return "adj_close" if "adj_close" in df.columns else "close"


# ── Indicator calculations ────────────────────────────────────────────────────

def _sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()


def _ema(series: pd.Series, window: int) -> pd.Series:
    return series.ewm(span=window, adjust=False).mean()


def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=window - 1, min_periods=window).mean()
    avg_loss = loss.ewm(com=window - 1, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = _ema(series, fast)
    ema_slow = _ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = _ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def _bollinger(series: pd.Series, window: int = 20, std_dev: float = 2.0):
    mid = _sma(series, window)
    std = series.rolling(window=window, min_periods=window).std()
    upper = mid + std_dev * std
    lower = mid - std_dev * std
    pct_b = (series - lower) / (upper - lower).replace(0, np.nan)
    return upper, mid, lower, pct_b


def _vwap(df: pd.DataFrame) -> pd.Series:
    """Session-style VWAP (rolling 20-day for display)."""
    typ = (df["high"] + df["low"] + df[_price_col(df)]) / 3
    vol = df["volume"].replace(0, np.nan)
    cum_tp_vol = (typ * vol).rolling(20, min_periods=1).sum()
    cum_vol    = vol.rolling(20, min_periods=1).sum()
    return cum_tp_vol / cum_vol


def _support_resistance(series: pd.Series, order: int = 20, n: int = 5) -> tuple[list[float], list[float]]:
    """Find local extrema as support/resistance levels."""
    arr = series.values
    supports:    list[float] = []
    resistances: list[float] = []

    for i in range(order, len(arr) - order):
        window = arr[i - order: i + order + 1]
        if arr[i] == window.min():
            supports.append(float(arr[i]))
        if arr[i] == window.max():
            resistances.append(float(arr[i]))

    # Cluster and take strongest n levels
    def _cluster(levels: list[float], tol_pct: float = 0.02) -> list[float]:
        if not levels:
            return []
        levels = sorted(levels)
        clusters: list[list[float]] = [[levels[0]]]
        for lv in levels[1:]:
            if abs(lv - clusters[-1][-1]) / clusters[-1][-1] < tol_pct:
                clusters[-1].append(lv)
            else:
                clusters.append([lv])
        return [float(np.mean(c)) for c in clusters]

    supports    = sorted(_cluster(supports))[-n:]
    resistances = sorted(_cluster(resistances))[:n]
    return supports, resistances


# ── Crossover detection ───────────────────────────────────────────────────────

def _detect_crossovers(price: float, sma_20: float, sma_50: float, sma_200: float) -> dict[str, str]:
    signals = {}

    # Golden cross / death cross (50 vs 200)
    if not any(np.isnan(x) for x in [sma_50, sma_200]):
        if sma_50 > sma_200:
            signals["50_200_cross"] = "golden (bullish)"
        else:
            signals["50_200_cross"] = "death (bearish)"

    # Price vs MAs
    if not np.isnan(sma_200):
        signals["price_vs_200"] = "above" if price > sma_200 else "below"
    if not np.isnan(sma_50):
        signals["price_vs_50"] = "above" if price > sma_50 else "below"
    if not np.isnan(sma_20):
        signals["price_vs_20"] = "above" if price > sma_20 else "below"

    return signals


# ── Scoring ───────────────────────────────────────────────────────────────────

def _score_technicals(
    rsi: Optional[float],
    pct_b: Optional[float],
    macd_hist: Optional[float],
    price: float,
    sma_50: Optional[float],
    sma_200: Optional[float],
    volume_ratio: Optional[float],
) -> tuple[float, dict[str, float]]:
    components: dict[str, float] = {}

    # RSI: 30-70 = neutral(50), oversold = bullish potential(70), overbought = bearish(30)
    if rsi is not None and not np.isnan(rsi):
        if rsi < RSI_OVERSOLD:
            components["rsi"] = 72.0   # Oversold → potential reversal upside
        elif rsi > RSI_OVERBOUGHT:
            components["rsi"] = 28.0   # Overbought
        else:
            # Scale 30-70 → 50-65 (mild bullish in mid-range)
            components["rsi"] = 50.0 + (70.0 - rsi) / 40.0 * 15.0

    # Bollinger %B
    if pct_b is not None and not np.isnan(pct_b):
        if pct_b < 0.2:
            components["bollinger"] = 70.0   # Near lower band → oversold
        elif pct_b > 0.8:
            components["bollinger"] = 30.0   # Near upper band
        else:
            components["bollinger"] = 55.0

    # MACD histogram
    if macd_hist is not None and not np.isnan(macd_hist):
        components["macd"] = 65.0 if macd_hist > 0 else 35.0

    # Price vs key MAs
    ma_scores = []
    if sma_50 is not None and not np.isnan(sma_50):
        ma_scores.append(60.0 if price > sma_50 else 40.0)
    if sma_200 is not None and not np.isnan(sma_200):
        ma_scores.append(65.0 if price > sma_200 else 35.0)
    if ma_scores:
        components["trend"] = float(np.mean(ma_scores))

    if not components:
        return 50.0, components
    return float(np.mean(list(components.values()))), components


# ── Main function ─────────────────────────────────────────────────────────────

def analyze_technical(price_history: PriceHistory) -> TechnicalResult:
    warnings: list[str] = []
    df = price_history.df.copy()

    if df.empty or len(df) < 30:
        warnings.append("Insufficient price history for technical analysis")
        return TechnicalResult(
            score=ModuleScore(name="technical", score=50.0, confidence=0.0, warnings=warnings)
        )

    col = _price_col(df)
    price = df[col]
    current_price = float(price.iloc[-1])

    # Ensure required columns
    for required in ["high", "low", "volume"]:
        if required not in df.columns:
            df[required] = price  # fallback

    # ── Compute all indicators ────────────────────────────────────────────────
    ind = pd.DataFrame(index=df.index)
    ind["price"] = price

    ind["sma_20"]  = _sma(price, 20)
    ind["sma_50"]  = _sma(price, 50)
    ind["sma_200"] = _sma(price, 200)
    ind["ema_20"]  = _ema(price, 20)
    ind["ema_50"]  = _ema(price, 50)
    ind["ema_200"] = _ema(price, 200)

    ind["rsi_14"] = _rsi(price, 14)

    macd_l, macd_s, macd_h = _macd(price)
    ind["macd_line"]      = macd_l
    ind["macd_signal"]    = macd_s
    ind["macd_histogram"] = macd_h

    bb_upper, bb_mid, bb_lower, bb_pct_b = _bollinger(price, BOLLINGER_WINDOW, BOLLINGER_STD)
    ind["bb_upper"]  = bb_upper
    ind["bb_mid"]    = bb_mid
    ind["bb_lower"]  = bb_lower
    ind["bb_pct_b"]  = bb_pct_b

    ind["vwap"] = _vwap(df)

    # Volume
    ind["volume"] = df["volume"]
    vol_avg_20 = df["volume"].rolling(20, min_periods=5).mean()
    ind["vol_avg_20"] = vol_avg_20
    ind["vol_ratio"]  = df["volume"] / vol_avg_20

    # Latest values
    latest = ind.iloc[-1]

    def _safe(key: str) -> Optional[float]:
        val = latest.get(key, np.nan)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return None
        return float(val)

    sma_20  = _safe("sma_20")
    sma_50  = _safe("sma_50")
    sma_200 = _safe("sma_200")
    ema_20  = _safe("ema_20")
    ema_50  = _safe("ema_50")
    ema_200 = _safe("ema_200")
    rsi_14  = _safe("rsi_14")
    macd_line   = _safe("macd_line")
    macd_signal = _safe("macd_signal")
    macd_histogram = _safe("macd_histogram")
    bb_upper_val = _safe("bb_upper")
    bb_lower_val = _safe("bb_lower")
    bb_pct_b_val = _safe("bb_pct_b")
    vwap_val     = _safe("vwap")
    vol_ratio    = _safe("vol_ratio")

    abnormal_volume = bool(vol_ratio is not None and vol_ratio > 2.0)

    # ── Support / Resistance (trailing 12 months) ─────────────────────────────
    year_price = price.last("252B") if len(price) >= 100 else price
    try:
        supports, resistances = _support_resistance(year_price, order=10, n=5)
    except Exception:
        supports, resistances = [], []

    # ── MA signals ────────────────────────────────────────────────────────────
    ma_signals = {}
    if sma_50 is not None and sma_200 is not None:
        ma_signals = _detect_crossovers(current_price, sma_20 or current_price, sma_50, sma_200)

    # RSI signals
    if rsi_14 is not None:
        if rsi_14 > RSI_OVERBOUGHT:
            warnings.append(f"RSI {rsi_14:.1f} — overbought territory")
        elif rsi_14 < RSI_OVERSOLD:
            warnings.append(f"RSI {rsi_14:.1f} — oversold territory")

    # ── Score ─────────────────────────────────────────────────────────────────
    available_count = sum(1 for v in [rsi_14, macd_histogram, bb_pct_b_val, sma_50, sma_200] if v is not None)
    confidence = min(1.0, available_count / 5)

    overall_score, components = _score_technicals(
        rsi_14, bb_pct_b_val, macd_histogram, current_price, sma_50, sma_200, vol_ratio
    )

    module_score = ModuleScore(
        name="technical",
        score=round(overall_score, 1),
        confidence=round(confidence, 3),
        components=components,
        warnings=warnings,
    )

    return TechnicalResult(
        score=module_score,
        sma_20=sma_20,
        sma_50=sma_50,
        sma_200=sma_200,
        ema_20=ema_20,
        ema_50=ema_50,
        ema_200=ema_200,
        ma_signals=ma_signals,
        rsi_14=rsi_14,
        macd_line=macd_line,
        macd_signal=macd_signal,
        macd_histogram=macd_histogram,
        bollinger_upper=bb_upper_val,
        bollinger_lower=bb_lower_val,
        bollinger_pct_b=bb_pct_b_val,
        support_levels=supports,
        resistance_levels=resistances,
        vwap=vwap_val,
        volume_vs_avg_20d=vol_ratio,
        abnormal_volume=abnormal_volume,
        indicator_df=ind,
    )
