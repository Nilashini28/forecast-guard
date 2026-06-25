"""
M5 dataset loader and synthetic data generator.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from rich.console import Console

console = Console()


def _get_checkmark() -> str:
    """Returns a green checkmark if supported by stdout, otherwise 'v'."""
    try:
        "✔".encode(sys.stdout.encoding or "utf-8")
        return "✔"
    except Exception:
        return "v"


def load_m5(path: str) -> dict[str, pd.Series]:
    """
    Loads an M5-formatted CSV file where the first column is the date index
    and subsequent columns are demand data for individual SKUs.

    Args:
        path: Path to the CSV file.

    Returns:
        A dictionary mapping SKU ID (string) to a pd.Series with a DatetimeIndex.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(
            f"M5 dataset file not found at: {path_obj.resolve()}"
        )

    try:
        df = pd.read_csv(path_obj)
    except Exception as e:
        raise IOError(f"Failed to read CSV file at {path_obj}: {e}")

    if df.empty:
        raise ValueError(f"The CSV file at {path_obj} is empty.")

    # Treat first column as date/index
    date_col = df.columns[0]
    try:
        df[date_col] = pd.to_datetime(df[date_col])
    except Exception as e:
        raise ValueError(
            f"Could not convert the first column '{date_col}' to DatetimeIndex: {e}"
        )

    df.set_index(date_col, inplace=True)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.DatetimeIndex(df.index)

    # Convert all columns to Series
    skus: dict[str, pd.Series] = {}
    for col in df.columns:
        # Keep name and ensure correct datatype
        skus[str(col)] = df[str(col)].astype(float)

    # Print success message with rich
    console.print(f"[green]{_get_checkmark()}[/green] Loaded {len(skus)} SKUs from '{path_obj.name}'")

    return skus


def generate_synthetic_m5(n_skus: int = 5, n_periods: int = 104) -> dict[str, pd.Series]:
    """
    Generates synthetic weekly demand data for testing.
    Produces realistic demand: base demand 50-500 units, seasonal pattern (period = 52 weeks),
    trend, and random noise.

    Args:
        n_skus: Number of SKUs to generate.
        n_periods: Total number of weekly periods.

    Returns:
        A dictionary mapping SKU ID (string) to a pd.Series with a DatetimeIndex.
    """
    np.random.seed(42)
    dates = pd.date_range(start="2021-01-03", periods=n_periods, freq="W")
    skus: dict[str, pd.Series] = {}

    for i in range(n_skus):
        sku_id = f"SKU_{i+1:03d}"
        # Base demand between 50 and 500
        base_demand = np.random.uniform(50, 500)
        
        # Trend: slight positive or negative trend (up to 20% drift over the entire series)
        trend_drift = np.random.uniform(-0.2, 0.2) * base_demand
        trend = np.linspace(0, trend_drift, n_periods)
        
        # Seasonal pattern: period of 52 weeks (sine wave, amplitude 15% to 35% of base)
        seasonal_amplitude = np.random.uniform(0.15, 0.35) * base_demand
        seasonal = seasonal_amplitude * np.sin(2 * np.pi * np.arange(n_periods) / 52)
        
        # Random noise (10% to 20% of base demand standard deviation)
        noise_std = np.random.uniform(0.10, 0.20) * base_demand
        noise = np.random.normal(loc=0.0, scale=noise_std, size=n_periods)
        
        demand = base_demand + trend + seasonal + noise
        
        # Clamp to 0 to prevent negative demand
        demand = np.clip(demand, 0.0, None)
        
        # Round values to 4 decimal places for consistency
        demand = np.round(demand, 4)
        
        series = pd.Series(demand, index=dates, name=sku_id)
        # Ensure freq is set
        series.index.freq = "W"
        skus[sku_id] = series

    return skus
