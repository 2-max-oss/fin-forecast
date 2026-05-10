"""Forecasting module: Monte Carlo, time-series (Prophet), ML directional, ensemble."""
from __future__ import annotations

import logging
import os
import tempfile
import warnings as pywarnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from config import MC_NUM_PATHS, MC_HORIZONS_MONTHS
from core.types import (
    BacktestMetrics,
    ForecastDistribution,
    ForecastResult,
    MathematicalFinanceDiagnostics,
    ModuleScore,
    PriceHistory,
    safe_divide,
)

logger = logging.getLogger(__name__)

TRADING_DAYS_PER_MONTH = 21

_MPLCONFIGDIR = Path(tempfile.gettempdir()) / "stock_forecast_matplotlib"
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))


def _log_returns(prices: pd.Series) -> pd.Series:
    return np.log(prices / prices.shift(1)).dropna()


def _annualized_stats(returns: pd.Series) -> tuple[float, float]:
    """(annualized drift, annualized volatility) from log returns."""
    mu = float(returns.mean()) * 252
    sigma = float(returns.std()) * np.sqrt(252)
    return mu, sigma


def _calibrate_lognormal_process(
    prices: pd.Series,
    lookback_years: int = 3,
) -> dict[str, float]:
    """Estimate GBM parameters from trailing log returns.

    The PDF's exact lognormal discretization evolves log(S), where the
    SDE drift is the observed annualized log drift plus half the variance.
    """
    ret = _log_returns(prices)
    days_back = lookback_years * 252
    if len(ret) > days_back:
        ret = ret.iloc[-days_back:]

    log_drift, sigma = _annualized_stats(ret)
    physical_drift = log_drift + 0.5 * sigma**2

    recent_sigma = float(ret.iloc[-30:].std()) * np.sqrt(252) if len(ret) >= 30 else sigma
    regime_factor = min(1.5, max(0.7, recent_sigma / sigma)) if sigma > 0 else 1.0
    sigma_adj = sigma * regime_factor

    return {
        "initial_price": float(prices.iloc[-1]),
        "log_drift": log_drift,
        "physical_drift": physical_drift,
        "annualized_volatility": sigma,
        "recent_volatility": recent_sigma,
        "volatility_regime_factor": regime_factor,
        "adjusted_volatility": sigma_adj,
    }


def _standard_normal_draws(
    num_paths: int,
    num_steps: int,
    rng: np.random.Generator,
    antithetic: bool = True,
) -> np.ndarray:
    if antithetic and num_paths > 1:
        half = int(np.ceil(num_paths / 2))
        base = rng.standard_normal((half, num_steps))
        return np.vstack([base, -base])[:num_paths]
    return rng.standard_normal((num_paths, num_steps))


def _simulate_lognormal_paths(
    initial_price: float,
    drift: float,
    volatility: float,
    num_paths: int,
    num_steps: int,
    rng: np.random.Generator,
    antithetic: bool = True,
) -> np.ndarray:
    """Exact Black-Scholes/GBM discretization of the log price process."""
    dt = 1.0 / 252
    z = _standard_normal_draws(num_paths, num_steps, rng, antithetic=antithetic)
    log_returns_paths = (drift - 0.5 * volatility**2) * dt + volatility * np.sqrt(dt) * z
    log_price_paths = np.log(initial_price) + np.cumsum(log_returns_paths, axis=1)
    return np.exp(log_price_paths)


def _distribution_from_terminal(
    terminal: np.ndarray,
    current_price: float,
    horizon_months: int,
) -> ForecastDistribution:
    return ForecastDistribution(
        horizon_months=horizon_months,
        mean=round(float(terminal.mean()), 2),
        median=round(float(np.median(terminal)), 2),
        p5=round(float(np.percentile(terminal, 5)), 2),
        p10=round(float(np.percentile(terminal, 10)), 2),
        p25=round(float(np.percentile(terminal, 25)), 2),
        p75=round(float(np.percentile(terminal, 75)), 2),
        p90=round(float(np.percentile(terminal, 90)), 2),
        p95=round(float(np.percentile(terminal, 95)), 2),
        prob_positive=round(float((terminal > current_price).mean()), 3),
        current_price=current_price,
    )


# ── Monte Carlo ───────────────────────────────────────────────────────────────

def _monte_carlo(
    prices: pd.Series,
    horizons_months: list[int] = MC_HORIZONS_MONTHS,
    num_paths: int = MC_NUM_PATHS,
    lookback_years: int = 3,
) -> tuple[dict[int, ForecastDistribution], np.ndarray, dict[str, float]]:
    """Physical-measure GBM Monte Carlo calibrated to trailing log returns."""
    calibration = _calibrate_lognormal_process(prices, lookback_years)
    S0 = calibration["initial_price"]
    mu = calibration["physical_drift"]
    sigma_adj = calibration["adjusted_volatility"]

    distributions: dict[int, ForecastDistribution] = {}
    max_horizon_days = max(horizons_months) * TRADING_DAYS_PER_MONTH

    rng = np.random.default_rng(42)
    price_paths = _simulate_lognormal_paths(
        initial_price=S0,
        drift=mu,
        volatility=sigma_adj,
        num_paths=num_paths,
        num_steps=max_horizon_days,
        rng=rng,
        antithetic=True,
    )

    for h_months in horizons_months:
        h_days = h_months * TRADING_DAYS_PER_MONTH - 1  # 0-indexed
        h_days = min(h_days, max_horizon_days - 1)
        terminal = price_paths[:, h_days]
        distributions[h_months] = _distribution_from_terminal(terminal, S0, h_months)

    sample_size = min(50, num_paths)
    sample_idx = rng.choice(num_paths, size=sample_size, replace=False)
    sample_paths = price_paths[sample_idx, :]

    return distributions, sample_paths, calibration


def _risk_neutral_monte_carlo(
    prices: pd.Series,
    calibration: dict[str, float],
    risk_free_rate: float,
    dividend_yield: float = 0.0,
    horizons_months: list[int] = MC_HORIZONS_MONTHS,
    num_paths: int = MC_NUM_PATHS,
) -> dict[int, ForecastDistribution]:
    """Risk-neutral Black-Scholes distribution under the money-market numeraire."""
    S0 = calibration["initial_price"]
    sigma_adj = calibration["adjusted_volatility"]
    risk_neutral_drift = risk_free_rate - dividend_yield
    max_horizon_days = max(horizons_months) * TRADING_DAYS_PER_MONTH

    rng = np.random.default_rng(84)
    price_paths = _simulate_lognormal_paths(
        initial_price=S0,
        drift=risk_neutral_drift,
        volatility=sigma_adj,
        num_paths=num_paths,
        num_steps=max_horizon_days,
        rng=rng,
        antithetic=True,
    )

    distributions: dict[int, ForecastDistribution] = {}
    for h_months in horizons_months:
        h_days = min(h_months * TRADING_DAYS_PER_MONTH - 1, max_horizon_days - 1)
        distributions[h_months] = _distribution_from_terminal(
            price_paths[:, h_days],
            S0,
            h_months,
        )
    return distributions


# ── Monte Carlo backtest ──────────────────────────────────────────────────────

def _backtest_mc(prices: pd.Series, horizon_months: int = 12, n_folds: int = 5) -> BacktestMetrics:
    """Walk-forward backtest of GBM Monte Carlo."""
    h_days = horizon_months * TRADING_DAYS_PER_MONTH
    step = max(60, h_days)
    actuals, preds, naive_preds = [], [], []

    for i in range(n_folds):
        end_train = len(prices) - (n_folds - i) * step
        if end_train < 252:
            continue
        train = prices.iloc[:end_train]
        future = prices.iloc[end_train: end_train + h_days]
        if len(future) < h_days // 2:
            continue

        S0 = float(train.iloc[-1])
        actual = float(future.iloc[-1])
        ret_train = _log_returns(train)
        mu, sigma = _annualized_stats(ret_train)
        mc_median = S0 * np.exp((mu - 0.5 * sigma**2) * h_days / 252)
        naive = S0  # buy-and-hold same price

        actuals.append(actual)
        preds.append(mc_median)
        naive_preds.append(naive)

    if not actuals:
        return BacktestMetrics("monte_carlo", mae=0, rmse=0, directional_accuracy=0.5,
                                naive_baseline_mae=0, beats_naive=False)

    actuals_arr = np.array(actuals)
    preds_arr   = np.array(preds)
    naive_arr   = np.array(naive_preds)

    mae  = float(np.mean(np.abs(actuals_arr - preds_arr)))
    rmse = float(np.sqrt(np.mean((actuals_arr - preds_arr)**2)))
    naive_mae = float(np.mean(np.abs(actuals_arr - naive_arr)))

    # Directional: train_end price → actual vs predicted direction
    dir_acc = 0.5  # MC GBM doesn't have strong directional edge
    beats = mae < naive_mae

    return BacktestMetrics(
        method="monte_carlo",
        mae=round(mae, 4),
        rmse=round(rmse, 4),
        directional_accuracy=round(dir_acc, 3),
        naive_baseline_mae=round(naive_mae, 4),
        beats_naive=beats,
    )


# ── Prophet time-series ───────────────────────────────────────────────────────

def _prophet_forecast(
    prices: pd.Series,
    horizons_months: list[int] = MC_HORIZONS_MONTHS,
) -> tuple[Optional[dict[int, ForecastDistribution]], Optional[BacktestMetrics]]:
    try:
        with pywarnings.catch_warnings():
            pywarnings.simplefilter("ignore")
            from prophet import Prophet
    except ImportError:
        logger.warning("prophet not installed — time-series forecast unavailable")
        return None, None

    try:
        S0 = float(prices.iloc[-1])
        df_p = pd.DataFrame({"ds": prices.index, "y": prices.values})
        df_p["ds"] = pd.to_datetime(df_p["ds"])
        df_p = df_p.dropna()

        with pywarnings.catch_warnings():
            pywarnings.simplefilter("ignore")
            m = Prophet(
                daily_seasonality=False,
                weekly_seasonality=True,
                yearly_seasonality=True,
                changepoint_prior_scale=0.05,
                interval_width=0.95,
            )
            m.fit(df_p)

        max_h = max(horizons_months) * TRADING_DAYS_PER_MONTH
        future = m.make_future_dataframe(periods=max_h, freq="B")
        with pywarnings.catch_warnings():
            pywarnings.simplefilter("ignore")
            forecast = m.predict(future)

        results: dict[int, ForecastDistribution] = {}
        today_idx = len(df_p)

        for h_months in horizons_months:
            h_days = h_months * TRADING_DAYS_PER_MONTH
            target_row_idx = today_idx + h_days - 1
            if target_row_idx >= len(forecast):
                target_row_idx = len(forecast) - 1
            row = forecast.iloc[target_row_idx]
            median = float(row["yhat"])
            lower = float(row["yhat_lower"])
            upper = float(row["yhat_upper"])

            std_approx = (upper - lower) / (2 * 1.96)
            dist_arr = np.random.default_rng(42).normal(median, std_approx, 5000)

            results[h_months] = ForecastDistribution(
                horizon_months=h_months,
                mean=round(float(np.mean(dist_arr)), 2),
                median=round(median, 2),
                p5=round(float(np.percentile(dist_arr, 5)), 2),
                p10=round(float(np.percentile(dist_arr, 10)), 2),
                p25=round(float(np.percentile(dist_arr, 25)), 2),
                p75=round(float(np.percentile(dist_arr, 75)), 2),
                p90=round(float(np.percentile(dist_arr, 90)), 2),
                p95=round(float(np.percentile(dist_arr, 95)), 2),
                prob_positive=round(float((dist_arr > S0).mean()), 3),
                current_price=S0,
            )

        # Simple backtest: last year holdout
        backtest = _backtest_prophet(prices, m, df_p)
        return results, backtest

    except Exception as e:
        logger.warning("Prophet forecast failed: %s", e)
        return None, None


def _backtest_prophet(prices: pd.Series, model, df_train: pd.DataFrame) -> BacktestMetrics:
    try:
        cutoff = len(df_train) - 252
        if cutoff < 200:
            raise ValueError("Insufficient data for backtest")
        train_cut = df_train.iloc[:cutoff]
        actuals = prices.iloc[cutoff:]

        with pywarnings.catch_warnings():
            pywarnings.simplefilter("ignore")
            m2 = type(model)(
                daily_seasonality=False,
                weekly_seasonality=True,
                yearly_seasonality=True,
                changepoint_prior_scale=0.05,
                interval_width=0.80,
            )
            m2.fit(train_cut)
            future = m2.make_future_dataframe(periods=252, freq="B")
            fc = m2.predict(future)

        pred_series = fc.set_index("ds")["yhat"]
        aligned = pd.concat([actuals, pred_series], axis=1).dropna()
        if aligned.empty:
            raise ValueError("No alignment")
        mae  = float(np.abs(aligned.iloc[:, 0] - aligned.iloc[:, 1]).mean())
        rmse = float(np.sqrt(((aligned.iloc[:, 0] - aligned.iloc[:, 1])**2).mean()))
        naive_mae = float(np.abs(actuals - float(train_cut["y"].iloc[-1])).mean())
        return BacktestMetrics(
            method="prophet",
            mae=round(mae, 4),
            rmse=round(rmse, 4),
            directional_accuracy=0.5,
            naive_baseline_mae=round(naive_mae, 4),
            beats_naive=(mae < naive_mae),
        )
    except Exception:
        return BacktestMetrics("prophet", 0, 0, 0.5, 0, False)


# ── ML directional model ──────────────────────────────────────────────────────

def _ml_directional(
    prices: pd.Series,
    horizons_months: list[int] = [3, 12],
) -> tuple[dict[int, float], Optional[BacktestMetrics]]:
    """XGBoost directional probability model."""
    try:
        from sklearn.preprocessing import StandardScaler
        from xgboost import XGBClassifier
    except ImportError:
        logger.warning("xgboost/sklearn not installed — ML directional model unavailable")
        return {}, None

    try:
        ret = _log_returns(prices)
        features = _build_ml_features(prices, ret)

        if features is None or len(features) < 200:
            return {}, None

        results: dict[int, float] = {}
        backtest_metrics: Optional[BacktestMetrics] = None

        for h_months in horizons_months:
            h_days = h_months * TRADING_DAYS_PER_MONTH
            X, y = _build_targets(features, prices, h_days)
            if X is None or len(X) < 150:
                continue

            # Chronological split: 80% train, 20% val
            split = int(len(X) * 0.80)
            X_train, X_val = X.iloc[:split], X.iloc[split:]
            y_train, y_val = y.iloc[:split], y.iloc[split:]

            scaler = StandardScaler()
            X_tr_sc = scaler.fit_transform(X_train)
            X_val_sc = scaler.transform(X_val)

            model = XGBClassifier(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="logloss",
                random_state=42,
                verbosity=0,
                use_label_encoder=False,
            )
            model.fit(X_tr_sc, y_train, eval_set=[(X_val_sc, y_val)], verbose=False)

            # Predict on latest features
            X_latest = scaler.transform(X.iloc[[-1]])
            prob_up = float(model.predict_proba(X_latest)[0, 1])
            results[h_months] = round(prob_up, 3)

            # Backtest on validation set
            if backtest_metrics is None and len(X_val) > 0:
                y_pred = model.predict(X_val_sc)
                dir_acc = float((y_pred == y_val).mean())
                naive_dir = float((y_val == 1).mean())
                naive_dir = max(naive_dir, 1 - naive_dir)
                backtest_metrics = BacktestMetrics(
                    method="ml_xgboost",
                    mae=0.0,
                    rmse=0.0,
                    directional_accuracy=round(dir_acc, 3),
                    naive_baseline_mae=round(naive_dir, 3),
                    beats_naive=(dir_acc > naive_dir),
                )

        return results, backtest_metrics

    except Exception as e:
        logger.warning("ML directional model failed: %s", e)
        return {}, None


def _build_ml_features(prices: pd.Series, returns: pd.Series) -> Optional[pd.DataFrame]:
    """Build feature matrix for ML model."""
    try:
        df = pd.DataFrame(index=prices.index)
        # Return features
        for w in [5, 10, 21, 63, 126, 252]:
            df[f"ret_{w}d"] = prices.pct_change(w)
        # Volatility
        for w in [21, 63]:
            df[f"vol_{w}d"] = returns.rolling(w).std() * np.sqrt(252)
        # Technical
        df["rsi_14"] = _rsi_feat(prices)
        for w in [50, 200]:
            df[f"price_vs_sma{w}"] = prices / prices.rolling(w).mean() - 1
        # Volume features omitted (not always available)
        df = df.dropna()
        return df if len(df) > 100 else None
    except Exception:
        return None


def _rsi_feat(prices: pd.Series, window: int = 14) -> pd.Series:
    delta = prices.diff()
    gain = delta.clip(lower=0).ewm(com=window - 1, min_periods=window).mean()
    loss = (-delta).clip(lower=0).ewm(com=window - 1, min_periods=window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _build_targets(features: pd.DataFrame, prices: pd.Series, h_days: int):
    """Binary target: 1 if price is higher in h_days, else 0."""
    future = prices.shift(-h_days)
    target = (future > prices).astype(int)
    aligned = pd.concat([features, target.rename("target")], axis=1).dropna()
    if len(aligned) < 100:
        return None, None
    X = aligned.drop(columns=["target"])
    y = aligned["target"]
    return X, y


# ── Ensemble ──────────────────────────────────────────────────────────────────

def _ensemble(
    mc: dict[int, ForecastDistribution],
    ts: Optional[dict[int, ForecastDistribution]],
    ml: dict[int, float],
    horizons: list[int],
) -> dict[int, ForecastDistribution]:
    """Weighted ensemble of MC + Prophet + ML directional."""
    results: dict[int, ForecastDistribution] = {}

    for h in horizons:
        sources = []
        weights = []

        if h in mc:
            sources.append(mc[h])
            weights.append(0.50)
        if ts and h in ts:
            sources.append(ts[h])
            weights.append(0.35)

        if not sources:
            continue

        # Normalize weights
        w_sum = sum(weights)
        weights = [w / w_sum for w in weights]

        # ML adjusts prob_positive
        ml_prob = ml.get(h) or ml.get(3)  # fallback to 3m ML

        # Blend distributions by weighted percentiles
        def _blend(attr):
            return sum(w * getattr(s, attr) for w, s in zip(weights, sources))

        median = _blend("median")
        mean   = _blend("mean")
        p5     = _blend("p5")
        p10    = _blend("p10")
        p25    = _blend("p25")
        p75    = _blend("p75")
        p90    = _blend("p90")
        p95    = _blend("p95")
        base_prob = _blend("prob_positive")

        # Blend ML into prob_positive with 15% weight if available
        if ml_prob is not None:
            prob_positive = 0.85 * base_prob + 0.15 * ml_prob
        else:
            prob_positive = base_prob

        S0 = sources[0].current_price

        results[h] = ForecastDistribution(
            horizon_months=h,
            mean=round(mean, 2),
            median=round(median, 2),
            p5=round(p5, 2),
            p10=round(p10, 2),
            p25=round(p25, 2),
            p75=round(p75, 2),
            p90=round(p90, 2),
            p95=round(p95, 2),
            prob_positive=round(prob_positive, 3),
            current_price=S0,
        )

    return results


# ── Forecast scoring ──────────────────────────────────────────────────────────

def _score_forecast(
    ensemble: dict[int, ForecastDistribution],
    backtests: list[BacktestMetrics],
) -> tuple[float, dict[str, float]]:
    components: dict[str, float] = {}
    warnings_list: list[str] = []

    # 12-month expected return
    if 12 in ensemble:
        dist = ensemble[12]
        if dist.current_price > 0:
            exp_ret = (dist.median / dist.current_price - 1.0)
            if exp_ret > 0.25:
                components["exp_return"] = 90.0
            elif exp_ret > 0.10:
                components["exp_return"] = 72.0
            elif exp_ret > 0.0:
                components["exp_return"] = 55.0
            elif exp_ret > -0.10:
                components["exp_return"] = 38.0
            else:
                components["exp_return"] = 20.0

        # Prob positive
        components["prob_positive"] = dist.prob_positive * 100.0

    # Backtest quality
    beats_naive = [b.beats_naive for b in backtests if b.beats_naive is not None]
    if beats_naive:
        components["backtest_quality"] = 70.0 if any(beats_naive) else 40.0

    if not components:
        return 50.0, components

    score = float(np.mean(list(components.values())))
    return score, components


# ── Main function ─────────────────────────────────────────────────────────────

def analyze_forecast(
    price_history: PriceHistory,
    risk_free_rate: float | None = None,
    dividend_yield: float = 0.0,
) -> ForecastResult:
    warnings_list: list[str] = []
    col = "adj_close" if "adj_close" in price_history.df.columns else "close"
    prices = price_history.df[col].dropna()

    if len(prices) < 100:
        warnings_list.append("Insufficient price history for forecasting (need 100+ days)")
        return ForecastResult(
            score=ModuleScore(name="forecast", score=50.0, confidence=0.0,
                              warnings=warnings_list)
        )

    horizons = MC_HORIZONS_MONTHS

    # ── Monte Carlo ───────────────────────────────────────────────────────────
    mc_dists, mc_paths, calibration = _monte_carlo(prices, horizons, MC_NUM_PATHS)
    mc_backtest = _backtest_mc(prices, 12)
    backtests: list[BacktestMetrics] = [mc_backtest]

    risk_free = 0.045 if risk_free_rate is None else float(risk_free_rate)
    risk_neutral_dists = _risk_neutral_monte_carlo(
        prices=prices,
        calibration=calibration,
        risk_free_rate=risk_free,
        dividend_yield=dividend_yield,
        horizons_months=horizons,
        num_paths=MC_NUM_PATHS,
    )

    martingale_error_12m = None
    rn_12 = risk_neutral_dists.get(12)
    if rn_12 and rn_12.current_price > 0:
        discounted_mean = rn_12.mean * np.exp(-risk_free * 1.0)
        martingale_error_12m = discounted_mean / rn_12.current_price - 1.0

    # ── Prophet ───────────────────────────────────────────────────────────────
    ts_dists, ts_backtest = _prophet_forecast(prices, horizons)
    if ts_dists is None:
        warnings_list.append("Time-series (Prophet) forecast unavailable")
    if ts_backtest is not None:
        backtests.append(ts_backtest)

    # ── ML directional ────────────────────────────────────────────────────────
    ml_dists, ml_backtest = _ml_directional(prices, [3, 12])
    if not ml_dists:
        warnings_list.append("ML directional model unavailable")
    if ml_backtest is not None:
        backtests.append(ml_backtest)

    # Flag models that don't beat naive baseline
    for bt in backtests:
        if not bt.beats_naive:
            warnings_list.append(
                f"{bt.method}: does not beat naive baseline on held-out validation set"
            )

    # ── Ensemble ──────────────────────────────────────────────────────────────
    ensemble_dists = _ensemble(mc_dists, ts_dists, ml_dists, horizons)

    # ── Score ─────────────────────────────────────────────────────────────────
    methods_available = sum([1, ts_dists is not None, bool(ml_dists)])
    confidence = min(1.0, methods_available / 3)

    overall_score, components = _score_forecast(ensemble_dists, backtests)

    module_score = ModuleScore(
        name="forecast",
        score=round(overall_score, 1),
        confidence=round(confidence, 3),
        components=components,
        warnings=warnings_list,
    )

    return ForecastResult(
        score=module_score,
        monte_carlo=mc_dists,
        risk_neutral=risk_neutral_dists,
        time_series=ts_dists or {},
        ml_directional=ml_dists,
        ensemble=ensemble_dists,
        backtests=backtests,
        mc_sample_paths=mc_paths,
        math_finance=MathematicalFinanceDiagnostics(
            scheme="exact lognormal Black-Scholes discretization",
            physical_drift=calibration["physical_drift"],
            risk_neutral_drift=risk_free - dividend_yield,
            annualized_volatility=calibration["annualized_volatility"],
            adjusted_volatility=calibration["adjusted_volatility"],
            volatility_regime_factor=calibration["volatility_regime_factor"],
            risk_free_rate=risk_free,
            dividend_yield=dividend_yield,
            martingale_error_12m=martingale_error_12m,
            antithetic_variates=True,
            notes=[
                "Price paths evolve the log process, preserving positive prices.",
                "The risk-neutral table uses the money-market numeraire drift.",
                "Antithetic Brownian increments reduce Monte Carlo sampling noise.",
            ],
        ),
    )
