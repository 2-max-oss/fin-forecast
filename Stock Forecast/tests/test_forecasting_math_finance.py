import unittest

import numpy as np
import pandas as pd

from analysis.forecasting import (
    _calibrate_lognormal_process,
    _risk_neutral_monte_carlo,
    _simulate_lognormal_paths,
)


class MathematicalFinanceForecastTests(unittest.TestCase):
    def test_lognormal_path_simulation_preserves_positive_prices(self):
        rng = np.random.default_rng(1)

        paths = _simulate_lognormal_paths(
            initial_price=100.0,
            drift=0.08,
            volatility=0.25,
            num_paths=500,
            num_steps=252,
            rng=rng,
            antithetic=True,
        )

        self.assertEqual(paths.shape, (500, 252))
        self.assertTrue((paths > 0).all())

    def test_risk_neutral_forecast_is_close_to_discounted_martingale(self):
        dates = pd.bdate_range("2024-01-01", periods=756)
        prices = pd.Series(
            100.0 * np.exp(np.linspace(0.0, 0.15, len(dates))),
            index=dates,
        )
        calibration = _calibrate_lognormal_process(prices)

        distributions = _risk_neutral_monte_carlo(
            prices=prices,
            calibration=calibration,
            risk_free_rate=0.05,
            dividend_yield=0.0,
            horizons_months=[12],
            num_paths=10_000,
        )

        dist = distributions[12]
        discounted_mean = dist.mean * np.exp(-0.05)
        martingale_error = discounted_mean / dist.current_price - 1.0

        self.assertLess(abs(martingale_error), 0.02)


if __name__ == "__main__":
    unittest.main()

