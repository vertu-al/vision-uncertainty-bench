import pytest
import torch
import torch.nn.functional as functional
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from vision_uncertainty_bench.training import evaluate_classifier, set_global_seed


class IdentityLogitModel(nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs


def test_set_global_seed_reproduces_torch_random_numbers() -> None:
    set_global_seed(2026)
    first = torch.rand(5)

    set_global_seed(2026)
    second = torch.rand(5)

    assert torch.equal(first, second)


def test_evaluate_classifier_matches_known_accuracy_and_cross_entropy() -> None:
    logits = torch.tensor(
        [
            [2.0, 0.0],  # correct class 0
            [0.0, 2.0],  # correct class 1
            [2.0, 0.0],  # incorrect: target is class 1
        ]
    )
    targets = torch.tensor([0, 1, 1], dtype=torch.long)
    loader = DataLoader(TensorDataset(logits, targets), batch_size=2)

    result = evaluate_classifier(
        IdentityLogitModel(),
        loader,
        device=torch.device("cpu"),
    )

    expected_loss = functional.cross_entropy(logits, targets).item()

    assert result.n_examples == 3
    assert result.accuracy == 2 / 3
    assert result.cross_entropy == pytest.approx(expected_loss)