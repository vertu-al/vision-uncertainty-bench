"""Metrics for confidence-threshold selective prediction."""

import math
from dataclasses import dataclass

import torch

from vision_uncertainty_bench.metrics import validate_probabilities_and_targets


@dataclass(frozen=True)
class ConfidenceThresholdMetrics:
    """Evaluation summary for predictions accepted above one confidence threshold."""

    accepted_count: int
    coverage: float
    risk: float


def confidence_threshold_metrics(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    threshold: float,
) -> ConfidenceThresholdMetrics:
    """Measure coverage and risk after abstaining below a confidence threshold."""
    validate_probabilities_and_targets(probabilities, targets)

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must lie in the interval [0, 1].")

    confidences, predictions = probabilities.max(dim=1)
    accepted = confidences >= threshold
    accepted_count = int(accepted.sum().item())
    coverage = accepted_count / len(targets)

    if accepted_count == 0:
        return ConfidenceThresholdMetrics(
            accepted_count=0,
            coverage=coverage,
            risk=math.nan,
        )

    accepted_predictions = predictions[accepted]
    accepted_targets = targets[accepted]
    risk = float((accepted_predictions != accepted_targets).float().mean().item())

    return ConfidenceThresholdMetrics(
        accepted_count=accepted_count,
        coverage=coverage,
        risk=risk,
    )