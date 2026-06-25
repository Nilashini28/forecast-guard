"""
Data foundation module for loading and generating M5 forecasting datasets.
"""

from .m5_loader import load_m5, generate_synthetic_m5

__all__ = ["load_m5", "generate_synthetic_m5"]
