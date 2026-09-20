"""
NetForecaster - Forecast Package (Stage 6)

This package implements the multi-step recursive forecasting engine.
It leverages the trained NetworkWorldModel to perform autoregressive
state rollouts, multi-horizon threat risk scoring, and attack category prediction.
"""

from forecast.rollout import (
    ForecastResult,
    RecursiveForecaster,
    evaluate_rollout_metrics,
)
from forecast.lead_time import (
    AttackEpisode,
    EpisodeLeadTimeResult,
    LeadTimeSummary,
    identify_attack_episodes,
    LeadTimeEngine,
)

__all__ = [
    "ForecastResult",
    "RecursiveForecaster",
    "evaluate_rollout_metrics",
    "AttackEpisode",
    "EpisodeLeadTimeResult",
    "LeadTimeSummary",
    "identify_attack_episodes",
    "LeadTimeEngine",
]
