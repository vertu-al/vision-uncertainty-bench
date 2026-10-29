"""CIFAR-10 loading with a fixed, stratified validation split."""

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Subset
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor

from vision_uncertainty_bench.splits import stratified_train_validation_split


@dataclass(frozen=True)
class Cifar10Datasets:
    """Clean CIFAR-10 datasets for training, validation, and final testing."""

    train: Subset
    validation: Subset
    test: CIFAR10


def load_cifar10_datasets(
    data_directory: str | Path,
    validation_per_class: int = 500,
    split_seed: int = 2026,
) -> Cifar10Datasets:
    """Load CIFAR-10 without downloading and create a reproducible split.

    Images are tensors in [0, 1]. Normalisation is intentionally deferred so
    corruptions can be applied in their natural pixel-value space.
    """
    root = str(Path(data_directory))
    to_tensor = ToTensor()

    clean_training_source = CIFAR10(
        root=root,
        train=True,
        transform=to_tensor,
        download=False,
    )
    test = CIFAR10(
        root=root,
        train=False,
        transform=to_tensor,
        download=False,
    )

    labels = torch.tensor(clean_training_source.targets, dtype=torch.long)
    split = stratified_train_validation_split(
        labels,
        validation_per_class=validation_per_class,
        seed=split_seed,
    )

    return Cifar10Datasets(
        train=Subset(clean_training_source, split.train_indices.tolist()),
        validation=Subset(clean_training_source, split.validation_indices.tolist()),
        test=test,
    )