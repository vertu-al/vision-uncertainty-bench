import math

import pytest
import torch

from vision_uncertainty_bench.metrics import (
    expected_calibration_error,
    multiclass_brier_score,
    negative_log_likelihood,
)


def test_negative_log_likelihood_matches_hand_calculation() -> None:
    probabilities = torch.tensor(
        [[0.8, 0.2], [0.4, 0.6]],
        dtype=torch.float64,
    )
    targets = torch.tensor([0, 0])

    expected = (-math.log(0.8) - math.log(0.4)) / 2

    assert negative_log_likelihood(probabilities, targets).item() == pytest.approx(expected)


def test_multiclass_brier_score_matches_hand_calculation() -> None:
    probabilities = torch.tensor(
        [[0.8, 0.2], [0.4, 0.6]],
        dtype=torch.float64,
    )
    targets = torch.tensor([0, 0])

    # Sample 1: (0.8 - 1)^2 + (0.2 - 0)^2 = 0.08
    # Sample 2: (0.4 - 1)^2 + (0.6 - 0)^2 = 0.72
    expected = (0.08 + 0.72) / 2

    assert multiclass_brier_score(probabilities, targets).item() == pytest.approx(expected)


def test_expected_calibration_error_matches_hand_calculation() -> None:
    probabilities = torch.tensor(
        [[0.8, 0.2], [0.4, 0.6]],
        dtype=torch.float64,
    )
    targets = torch.tensor([0, 0])

    # Both predictions fall in the confidence interval [0.5, 1.0].
    # Mean confidence = (0.8 + 0.6) / 2 = 0.7
    # Empirical accuracy = 1 / 2 = 0.5
    # ECE = |0.7 - 0.5| = 0.2
    assert expected_calibration_error(probabilities, targets, n_bins=2).item() == pytest.approx(
        0.2
    )