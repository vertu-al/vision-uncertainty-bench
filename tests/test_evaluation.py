import math

import pytest
import torch

from vision_uncertainty_bench.evaluation import (
    probabilities_from_logits,
    summarize_probabilities,
)


def test_probabilities_from_logits_applies_temperature() -> None:
    logits = torch.tensor([[2.0, 0.0]])

    raw_probabilities = probabilities_from_logits(logits, temperature=1.0)
    softened_probabilities = probabilities_from_logits(logits, temperature=2.0)

    assert raw_probabilities.shape == (1, 2)
    assert softened_probabilities.max() < raw_probabilities.max()
    assert torch.allclose(softened_probabilities.sum(dim=1), torch.tensor([1.0]))


def test_probability_summary_matches_hand_calculated_metrics() -> None:
    probabilities = torch.tensor(
        [[0.8, 0.2], [0.4, 0.6]],
        dtype=torch.float64,
    )
    targets = torch.tensor([0, 0], dtype=torch.long)

    summary = summarize_probabilities(probabilities, targets, n_ece_bins=2)

    expected_nll = (-math.log(0.8) - math.log(0.4)) / 2

    assert summary.n_examples == 2
    assert summary.accuracy == pytest.approx(0.5)
    assert summary.negative_log_likelihood == pytest.approx(expected_nll)
    assert summary.brier_score == pytest.approx(0.4)
    assert summary.expected_calibration_error == pytest.approx(0.2)


def test_probabilities_from_logits_rejects_nonpositive_temperature() -> None:
    with pytest.raises(ValueError, match="temperature"):
        probabilities_from_logits(torch.tensor([[1.0, 0.0]]), temperature=0.0)