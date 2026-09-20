"""
NetForecaster - Multi-Step Forecasting Package (Stage 6)

This package implements autoregressive rollout and multi-step recursive state forecasting
for the Network World Model.
"""

from forecast.rollout import (
    RolloutResult,
    autoregressive_rollout,
    evaluate_rollout_horizons,
    unscale_state_trajectories,
)

__all__ = [
    "RolloutResult",
    "autoregressive_rollout",
    "evaluate_rollout_horizons",
    "unscale_state_trajectories",
]
