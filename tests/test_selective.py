import math

import pytest
import torch

from vision_uncertainty_bench.selective import confidence_threshold_metrics


def _probabilities_and_targets() -> tuple[torch.Tensor, torch.Tensor]:
    probabilities = torch.tensor(
        [
            [0.90, 0.10],  # correct, confidence 0.90
            [0.80, 0.20],  # incorrect, confidence 0.80
            [0.60, 0.40],  # correct, confidence 0.60
        ],
        dtype=torch.float64,
    )
    targets = torch.tensor([0, 1, 0])
    return probabilities, targets


def test_confidence_threshold_reports_coverage_and_risk() -> None:
    probabilities, targets = _probabilities_and_targets()

    result = confidence_threshold_metrics(
        probabilities,
        targets,
        threshold=0.70,
    )

    # Two of three predictions are accepted; one accepted prediction is wrong.
    assert result.accepted_count == 2
    assert result.coverage == pytest.approx(2 / 3)
    assert result.risk == pytest.approx(1 / 2)


def test_a_higher_threshold_can_reduce_risk_and_coverage() -> None:
    probabilities, targets = _probabilities_and_targets()

    result = confidence_threshold_metrics(
        probabilities,
        targets,
        threshold=0.85,
    )

    assert result.accepted_count == 1
    assert result.coverage == pytest.approx(1 / 3)
    assert result.risk == pytest.approx(0.0)


def test_no_accepted_predictions_have_zero_coverage_and_undefined_risk() -> None:
    probabilities, targets = _probabilities_and_targets()

    result = confidence_threshold_metrics(
        probabilities,
        targets,
        threshold=0.95,
    )

    assert result.accepted_count == 0
    assert result.coverage == pytest.approx(0.0)
    assert math.isnan(result.risk)


def test_invalid_threshold_raises_a_clear_error() -> None:
    probabilities, targets = _probabilities_and_targets()

    with pytest.raises(ValueError, match="threshold"):
        confidence_threshold_metrics(
            probabilities,
            targets,
            threshold=1.1,
        )