"""Post-hoc temperature scaling for multiclass classification logits."""

import math

import torch
import torch.nn.functional as functional
from torch import nn


def _validate_logits_and_targets(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> None:
    if logits.ndim != 2:
        raise ValueError("logits must have shape [n_examples, n_classes].")

    if targets.ndim != 1:
        raise ValueError("targets must have shape [n_examples].")

    if logits.shape[0] == 0:
        raise ValueError("logits and targets cannot be empty.")

    if logits.shape[0] != targets.shape[0]:
        raise ValueError("logits and targets must contain the same number of examples.")

    if logits.shape[1] < 2:
        raise ValueError("logits must contain at least two classes.")

    if targets.dtype != torch.long:
        raise ValueError("targets must use the torch.long dtype.")

    if not torch.isfinite(logits).all():
        raise ValueError("logits must contain only finite values.")

    if (targets < 0).any() or (targets >= logits.shape[1]).any():
        raise ValueError("targets must be valid class indices.")


class TemperatureScaler(nn.Module):
    """A positive scalar temperature applied to unnormalised class logits."""

    def __init__(self, initial_temperature: float = 1.0) -> None:
        super().__init__()

        if initial_temperature <= 0.0:
            raise ValueError("initial_temperature must be positive.")

        self.log_temperature = nn.Parameter(
            torch.tensor(math.log(initial_temperature), dtype=torch.float32)
        )

    @property
    def temperature(self) -> torch.Tensor:
        """Return the positive temperature parameter."""
        return self.log_temperature.exp()

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Rescale logits without changing their predicted class order."""
        return logits / self.temperature


def fit_temperature(
    validation_logits: torch.Tensor,
    validation_targets: torch.Tensor,
    max_iterations: int = 50,
) -> TemperatureScaler:
    """Fit one temperature on held-out validation logits by NLL minimisation."""
    _validate_logits_and_targets(validation_logits, validation_targets)

    if max_iterations < 1:
        raise ValueError("max_iterations must be at least one.")

    scaler = TemperatureScaler().to(validation_logits.device)
    optimizer = torch.optim.LBFGS(
        scaler.parameters(),
        lr=0.1,
        max_iter=max_iterations,
        tolerance_grad=1e-7,
        tolerance_change=1e-9,
    )

    logits = validation_logits.detach()
    targets = validation_targets.detach()

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = functional.cross_entropy(scaler(logits), targets)
        loss.backward()
        return loss

    optimizer.step(closure)
    return scaler