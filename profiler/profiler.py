"""
Demand profiling implementation using STL decomposition and statistical metrics.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL

# Import router inside profiler for the inline tests (Task 7)
# Use relative import for the module to support proper packaging
try:
    from .router import route
except ImportError:
    # Fallback for running file directly as __main__
    from router import route


class SeriesDNA:
    """
    Computes statistical and structural characteristics (DNA) of a demand time series.
    """

    def compute(self, sku_id: str, series: pd.Series) -> dict:
        """
        Computes the Series DNA for a given time series.

        Args:
            sku_id: Identifier for the SKU.
            series: Pandas Series with a DatetimeIndex representing demand.

        Returns:
            A dictionary containing the calculated metrics:
                - sku_id: str
                - cv: float
                - intermittency_ratio: float
                - trend_strength: float
                - seasonality_index: float
                - n_periods: int
                - mean_demand: float
                - zero_demand_periods: int
        """
        n_periods = len(series)
        if n_periods == 0:
            return {
                "sku_id": sku_id,
                "cv": 0.0,
                "intermittency_ratio": 0.0,
                "trend_strength": 0.0,
                "seasonality_index": 1.0,
                "n_periods": 0,
                "mean_demand": 0.0,
                "zero_demand_periods": 0,
            }

        mean_demand = series.mean()
        zero_demand_periods = int((series == 0).sum())
        intermittency_ratio = zero_demand_periods / n_periods

        if mean_demand == 0:
            return {
                "sku_id": sku_id,
                "cv": 0.0,
                "intermittency_ratio": float(np.round(intermittency_ratio, 4)),
                "trend_strength": 0.0,
                "seasonality_index": 1.0,
                "n_periods": n_periods,
                "mean_demand": 0.0,
                "zero_demand_periods": zero_demand_periods,
            }

        # Coefficient of variation = std / mean
        std_demand = series.std()
        if pd.isna(std_demand):
            cv = 0.0
        else:
            cv = std_demand / mean_demand

        # Detect STL period from frequency
        period = self._detect_period(series)

        # Minimum series length to attempt STL: 2 * period
        if n_periods < 2 * period:
            trend_strength = 0.0
            seasonality_index = 1.0
        else:
            try:
                # Run STL decomposition
                res = STL(series, period=period).fit()
                trend = res.trend
                seasonal = res.seasonal
                resid = res.resid

                # trend_strength = 1 - var(remainder) / var(trend + remainder)
                var_resid = float(np.var(resid, ddof=0))
                var_trend_resid = float(np.var(trend + resid, ddof=0))

                if var_trend_resid == 0.0:
                    trend_strength = 0.0
                else:
                    trend_strength = 1.0 - (var_resid / var_trend_resid)

                # Clamp to [0, 1]
                trend_strength = max(0.0, min(1.0, trend_strength))

                # seasonality_index = max(seasonal) / (abs(min(seasonal)) + 1e-9)
                max_seasonal = float(np.max(seasonal))
                min_seasonal = float(np.min(seasonal))
                seasonality_index = max_seasonal / (abs(min_seasonal) + 1e-9)

            except Exception:
                trend_strength = 0.0
                seasonality_index = 1.0

        return {
            "sku_id": sku_id,
            "cv": float(np.round(cv, 4)),
            "intermittency_ratio": float(np.round(intermittency_ratio, 4)),
            "trend_strength": float(np.round(trend_strength, 4)),
            "seasonality_index": float(np.round(seasonality_index, 4)),
            "n_periods": int(n_periods),
            "mean_demand": float(np.round(mean_demand, 4)),
            "zero_demand_periods": int(zero_demand_periods),
        }

    def _detect_period(self, series: pd.Series) -> int:
        """
        Detects the seasonal period from the series frequency.
        Returns 52 for weekly, 12 for monthly, 7 for daily, fallback to 52.
        """
        freq = series.index.freqstr or getattr(series.index, "inferred_freq", None)
        if freq:
            freq_upper = freq.upper()
            if "W" in freq_upper:
                return 52
            elif "M" in freq_upper:
                return 12
            elif "D" in freq_upper:
                return 7
        return 52


if __name__ == "__main__":
    # Generate 3 synthetic series for testing routing assertions
    dates = pd.date_range(start="2021-01-03", periods=104, freq="W")
    
    # 1. Stable series (low CV, low IR)
    # Mean ~100, Std ~5, CV ~0.05, IR = 0
    np.random.seed(42)
    stable_data = 100.0 + np.random.normal(loc=0.0, scale=5.0, size=104)
    stable_series = pd.Series(stable_data, index=dates)
    stable_series.index.freq = "W"

    # 2. Intermittent series (IR > 0.3)
    # 50% zeroes
    intermittent_data = np.array([0.0, 100.0] * 52)
    intermittent_series = pd.Series(intermittent_data, index=dates)
    intermittent_series.index.freq = "W"

    # 3. High-volatility series (CV > 1.0)
    # High standard deviation relative to mean, IR = 0
    high_cv_data = np.array([100.0] * 80 + [5000.0] * 24)
    high_cv_series = pd.Series(high_cv_data, index=dates)
    high_cv_series.index.freq = "W"

    # Compute DNAs
    dna_calculator = SeriesDNA()
    stable_dna = dna_calculator.compute("SKU_STABLE", stable_series)
    intermittent_dna = dna_calculator.compute("SKU_INTERMITTENT", intermittent_series)
    high_cv_dna = dna_calculator.compute("SKU_HIGH_CV", high_cv_series)

    # Route DNAs
    stable_route = route(stable_dna)
    intermittent_route = route(intermittent_dna)
    high_cv_route = route(high_cv_dna)

    print("Stable DNA:", stable_dna)
    print("Stable Route:", stable_route)
    print("Intermittent DNA:", intermittent_dna)
    print("Intermittent Route:", intermittent_route)
    print("High CV DNA:", high_cv_dna)
    print("High CV Route:", high_cv_route)

    # Assertions
    assert stable_route.cv_mode == "stable", f"Expected stable, got {stable_route.cv_mode}"
    assert intermittent_route.cv_mode == "intermittent", f"Expected intermittent, got {intermittent_route.cv_mode}"
    assert high_cv_route.cv_mode == "high_cv", f"Expected high_cv, got {high_cv_route.cv_mode}"

    print("All routing assertions passed ✓")
