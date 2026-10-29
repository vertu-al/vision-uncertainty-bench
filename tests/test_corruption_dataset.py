import torch
from torch.utils.data import TensorDataset

from vision_uncertainty_bench.corruption_dataset import CorruptedDataset


def test_corruption_dataset_is_deterministic_per_example_and_preserves_targets() -> None:
    images = torch.full((2, 3, 32, 32), 0.5)
    targets = torch.tensor([3, 7], dtype=torch.long)
    base_dataset = TensorDataset(images, targets)

    corrupted_dataset = CorruptedDataset(
        base_dataset,
        corruption="gaussian_noise",
        severity=2,
        seed=2026,
    )

    first_image, first_target = corrupted_dataset[0]
    repeated_first_image, repeated_first_target = corrupted_dataset[0]
    second_image, second_target = corrupted_dataset[1]

    assert len(corrupted_dataset) == 2
    assert first_target.item() == 3
    assert repeated_first_target.item() == 3
    assert second_target.item() == 7
    assert torch.equal(first_image, repeated_first_image)
    assert not torch.equal(first_image, second_image)


def test_corruption_dataset_does_not_modify_the_underlying_clean_image() -> None:
    images = torch.full((1, 3, 32, 32), 0.5)
    targets = torch.tensor([3], dtype=torch.long)
    base_dataset = TensorDataset(images, targets)

    corrupted_dataset = CorruptedDataset(
        base_dataset,
        corruption="central_occlusion",
        severity=2,
        seed=2026,
    )
    corrupted_image, _ = corrupted_dataset[0]

    assert torch.all(images == 0.5)
    assert torch.all(corrupted_image[:, 16, 16] == 0.0)