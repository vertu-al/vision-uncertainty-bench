"""High-level summaries for probabilistic classification predictions."""

from dataclasses import dataclass

import torch

from vision_uncertainty_bench.metrics import (
    expected_calibration_error,
    multiclass_brier_score,
    negative_log_likelihood,
    validate_probabilities_and_targets,
)


@dataclass(frozen=True)
class ProbabilitySummary:
    """Aggregate predictive and calibration metrics for one evaluation condition."""

    n_examples: int
    accuracy: float
    negative_log_likelihood: float
    brier_score: float
    expected_calibration_error: float


def probabilities_from_logits(
    logits: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Convert logits to class probabilities after optional temperature scaling."""
    if logits.ndim != 2:
        raise ValueError("logits must have shape [n_examples, n_classes].")

    if not torch.isfinite(logits).all():
        raise ValueError("logits must contain only finite values.")

    if temperature <= 0.0:
        raise ValueError("temperature must be positive.")

    return torch.softmax(logits / temperature, dim=1)


def summarize_probabilities(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    n_ece_bins: int = 15,
) -> ProbabilitySummary:
    """Return accuracy, NLL, Brier score, and ECE for one condition."""
    validate_probabilities_and_targets(probabilities, targets)

    predictions = probabilities.argmax(dim=1)
    accuracy = (predictions == targets).to(dtype=probabilities.dtype).mean()

    return ProbabilitySummary(
        n_examples=len(targets),
        accuracy=accuracy.item(),
        negative_log_likelihood=negative_log_likelihood(probabilities, targets).item(),
        brier_score=multiclass_brier_score(probabilities, targets).item(),
        expected_calibration_error=expected_calibration_error(
            probabilities,
            targets,
            n_bins=n_ece_bins,
        ).item(),
    )