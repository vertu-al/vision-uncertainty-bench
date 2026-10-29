"""Dataset wrapper that applies deterministic image corruptions per example."""

import torch
from torch.utils.data import Dataset

from vision_uncertainty_bench.corruptions import apply_corruption


class CorruptedDataset(Dataset):
    """Wrap a clean tensor-image dataset in one fixed corruption condition."""

    def __init__(
        self,
        base_dataset: Dataset,
        corruption: str,
        severity: int,
        seed: int,
    ) -> None:
        if seed < 0:
            raise ValueError("seed must be non-negative.")

        self.base_dataset = base_dataset
        self.corruption = corruption
        self.severity = severity
        self.seed = seed

    def __len__(self) -> int:
        """Return the number of examples in the underlying clean dataset."""
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, object]:
        """Return one deterministically corrupted image and its original target."""
        image, target = self.base_dataset[index]

        if not isinstance(image, torch.Tensor):
            raise TypeError("base_dataset must return images as torch.Tensor objects.")

        corrupted_image = apply_corruption(
            image,
            corruption=self.corruption,
            severity=self.severity,
            seed=self.seed + index,
        )
        return corrupted_image, target