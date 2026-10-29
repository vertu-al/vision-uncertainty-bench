"""Inference utilities that preserve model state and output order."""

import torch
from torch import nn
from torch.utils.data import DataLoader


def collect_logits(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collect CPU logits and targets in data-loader order without gradients."""
    was_training = model.training
    model.eval()

    logits_batches = []
    target_batches = []

    try:
        with torch.no_grad():
            for images, targets in loader:
                logits = model(images.to(device))

                logits_batches.append(logits.detach().cpu())
                target_batches.append(targets.detach().cpu())
    finally:
        if was_training:
            model.train()

    if not logits_batches:
        raise ValueError("loader produced no batches.")

    return torch.cat(logits_batches), torch.cat(target_batches)