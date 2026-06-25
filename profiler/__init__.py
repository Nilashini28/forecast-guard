"""
Demand profiler module containing SeriesDNA and routing logic.
"""

from .profiler import SeriesDNA
from .router import route, RoutingConfig

__all__ = ["SeriesDNA", "route", "RoutingConfig"]
