"""
Demand router defining RoutingConfig and model selection rules based on SKU demand profiles.
"""

from dataclasses import dataclass


@dataclass
class RoutingConfig:
    """
    Routing configuration for a specific SKU.
    """
    sku_id: str
    primary_metric: str            # "mase" or "rmse"
    use_mape: bool                 # False if IR > 0.3
    cv_mode: str                   # "stable", "high_cv", or "intermittent"
    split_strategy: str            # "walk_forward" or "regime_aware"
    recommended_models: list[str]  # subset of ["naive_seasonal", "holt_winters", "arima"]


def route(dna: dict) -> RoutingConfig:
    """
    Determines the routing configuration for a SKU based on its demand DNA profile.

    Args:
        dna: A dictionary containing the SKU's profile metrics (cv, intermittency_ratio, etc.).

    Returns:
        A RoutingConfig dataclass instance containing metric and model choices.
    """
    sku_id = dna["sku_id"]
    intermittency_ratio = dna["intermittency_ratio"]
    cv = dna["cv"]

    if intermittency_ratio > 0.3:
        return RoutingConfig(
            sku_id=sku_id,
            primary_metric="mase",
            use_mape=False,
            cv_mode="intermittent",
            split_strategy="walk_forward",
            recommended_models=["naive_seasonal", "arima"],
        )
    elif cv > 1.0:
        return RoutingConfig(
            sku_id=sku_id,
            primary_metric="rmse",
            use_mape=True,
            cv_mode="high_cv",
            split_strategy="regime_aware",
            recommended_models=["holt_winters", "arima"],
        )
    else:
        return RoutingConfig(
            sku_id=sku_id,
            primary_metric="mase",
            use_mape=True,
            cv_mode="stable",
            split_strategy="walk_forward",
            recommended_models=["naive_seasonal", "holt_winters", "arima"],
        )
