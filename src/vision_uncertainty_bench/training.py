"""Reusable training and validation utilities."""

import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as functional
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader


@dataclass(frozen=True)
class ClassificationEvaluation:
    """Aggregate classification metrics for one dataset pass."""

    n_examples: int
    accuracy: float
    cross_entropy: float


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch random-number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_device(requested_device: str = "auto") -> torch.device:
    """Resolve an explicit device or prefer Apple MPS when available."""
    if requested_device == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if requested_device == "cpu":
        return torch.device("cpu")

    if requested_device == "mps":
        if not torch.backends.mps.is_available():
            raise ValueError("MPS was requested but is not available.")
        return torch.device("mps")

    raise ValueError("requested_device must be 'auto', 'cpu', or 'mps'.")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optimizer,
    device: torch.device,
) -> ClassificationEvaluation:
    """Train a classifier for one epoch and return aggregate batch metrics."""
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = functional.cross_entropy(logits, targets)
        loss.backward()
        optimizer.step()

        batch_size = len(targets)
        total_loss += loss.item() * batch_size
        total_correct += int((logits.argmax(dim=1) == targets).sum().item())
        total_examples += batch_size

    return ClassificationEvaluation(
        n_examples=total_examples,
        accuracy=total_correct / total_examples,
        cross_entropy=total_loss / total_examples,
    )


def evaluate_classifier(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> ClassificationEvaluation:
    """Evaluate a classifier without computing gradients."""
    was_training = model.training
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)

            logits = model(images)
            batch_size = len(targets)

            total_loss += functional.cross_entropy(
                logits,
                targets,
                reduction="sum",
            ).item()
            total_correct += int((logits.argmax(dim=1) == targets).sum().item())
            total_examples += batch_size

    if was_training:
        model.train()

    return ClassificationEvaluation(
        n_examples=total_examples,
        accuracy=total_correct / total_examples,
        cross_entropy=total_loss / total_examples,
    )