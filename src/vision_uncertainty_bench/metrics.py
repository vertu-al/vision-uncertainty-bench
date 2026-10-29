"""Metrics for probabilistic image-classification evaluation."""

import torch
import torch.nn.functional as functional


def validate_probabilities_and_targets(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
) -> None:
    """Validate probability vectors and their corresponding class targets."""
    if probabilities.ndim != 2:
        raise ValueError("probabilities must have shape [n_examples, n_classes].")

    if targets.ndim != 1:
        raise ValueError("targets must have shape [n_examples].")

    if probabilities.shape[0] == 0:
        raise ValueError("probabilities and targets cannot be empty.")

    if probabilities.shape[0] != targets.shape[0]:
        raise ValueError("probabilities and targets must contain the same number of examples.")

    if probabilities.shape[1] < 2:
        raise ValueError("probabilities must contain at least two classes.")

    if targets.dtype != torch.long:
        raise ValueError("targets must use the torch.long dtype.")

    if not torch.isfinite(probabilities).all():
        raise ValueError("probabilities must contain only finite values.")

    if (probabilities < 0).any():
        raise ValueError("probabilities cannot be negative.")

    row_sums = probabilities.sum(dim=1)
    if not torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6):
        raise ValueError("each row of probabilities must sum to one.")

    if (targets < 0).any() or (targets >= probabilities.shape[1]).any():
        raise ValueError("targets must be valid class indices.")


def negative_log_likelihood(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    epsilon: float = 1e-12,
) -> torch.Tensor:
    """Return mean negative log-likelihood for true-class probabilities."""
    validate_probabilities_and_targets(probabilities, targets)

    true_class_probabilities = probabilities.gather(1, targets.unsqueeze(1)).squeeze(1)
    return -torch.log(true_class_probabilities.clamp_min(epsilon)).mean()


def multiclass_brier_score(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Return mean per-example sum of squared probability errors across classes."""
    validate_probabilities_and_targets(probabilities, targets)

    one_hot_targets = functional.one_hot(
        targets,
        num_classes=probabilities.shape[1],
    ).to(dtype=probabilities.dtype)

    return ((probabilities - one_hot_targets) ** 2).sum(dim=1).mean()


def expected_calibration_error(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    n_bins: int = 15,
) -> torch.Tensor:
    """Estimate top-label expected calibration error using equally sized bins."""
    validate_probabilities_and_targets(probabilities, targets)

    if n_bins < 1:
        raise ValueError("n_bins must be at least one.")

    confidences, predictions = probabilities.max(dim=1)
    correctness = predictions.eq(targets).to(dtype=probabilities.dtype)

    ece = torch.zeros((), dtype=probabilities.dtype, device=probabilities.device)

    for bin_index in range(n_bins):
        lower = bin_index / n_bins
        upper = (bin_index + 1) / n_bins

        if bin_index == n_bins - 1:
            in_bin = (confidences >= lower) & (confidences <= upper)
        else:
            in_bin = (confidences >= lower) & (confidences < upper)

        if in_bin.any():
            bin_accuracy = correctness[in_bin].mean()
            bin_confidence = confidences[in_bin].mean()
            bin_weight = in_bin.to(dtype=probabilities.dtype).mean()
            ece += bin_weight * (bin_accuracy - bin_confidence).abs()

    return ece