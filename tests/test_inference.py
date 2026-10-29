import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from vision_uncertainty_bench.inference import collect_logits


class IdentityLogitModel(nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs


def test_collect_logits_preserves_order_and_restores_model_mode() -> None:
    logits = torch.tensor(
        [
            [2.0, 0.0],
            [0.0, 2.0],
            [1.0, 1.0],
        ]
    )
    targets = torch.tensor([0, 1, 0], dtype=torch.long)
    loader = DataLoader(TensorDataset(logits, targets), batch_size=2)

    model = IdentityLogitModel()
    model.train()

    collected_logits, collected_targets = collect_logits(
        model,
        loader,
        device=torch.device("cpu"),
    )

    assert model.training
    assert not collected_logits.requires_grad
    assert torch.equal(collected_logits, logits)
    assert torch.equal(collected_targets, targets)