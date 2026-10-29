"""Deterministic, stratified train-validation splits."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TrainValidationSplit:
    """Indices for mutually exclusive training and validation subsets."""

    train_indices: torch.Tensor
    validation_indices: torch.Tensor


def stratified_train_validation_split(
    labels: torch.Tensor,
    validation_per_class: int,
    seed: int,
) -> TrainValidationSplit:
    """Create a reproducible split with equal validation count per observed class."""
    if labels.ndim != 1:
        raise ValueError("labels must have shape [n_examples].")

    if labels.dtype != torch.long:
        raise ValueError("labels must use the torch.long dtype.")

    if len(labels) == 0:
        raise ValueError("labels cannot be empty.")

    if validation_per_class < 1:
        raise ValueError("validation_per_class must be at least one.")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    train_parts = []
    validation_parts = []

    for class_label in torch.unique(labels, sorted=True):
        class_indices = torch.nonzero(labels == class_label, as_tuple=False).squeeze(1)

        if len(class_indices) <= validation_per_class:
            raise ValueError(
                f"not enough examples in class {class_label.item()} "
                "to create training and validation subsets."
            )

        shuffled_positions = torch.randperm(len(class_indices), generator=generator)
        shuffled_indices = class_indices[shuffled_positions]

        validation_parts.append(shuffled_indices[:validation_per_class])
        train_parts.append(shuffled_indices[validation_per_class:])

    return TrainValidationSplit(
        train_indices=torch.cat(train_parts).sort().values,
        validation_indices=torch.cat(validation_parts).sort().values,
    )