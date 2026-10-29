import pytest
import torch

from vision_uncertainty_bench.splits import stratified_train_validation_split


def _balanced_labels() -> torch.Tensor:
    return torch.tensor([0] * 5 + [1] * 5 + [2] * 5, dtype=torch.long)


def test_stratified_split_preserves_class_counts_and_uses_every_example_once() -> None:
    labels = _balanced_labels()

    split = stratified_train_validation_split(
        labels,
        validation_per_class=2,
        seed=17,
    )

    validation_class_counts = torch.bincount(
        labels[split.validation_indices],
        minlength=3,
    )

    assert validation_class_counts.tolist() == [2, 2, 2]
    assert len(split.train_indices) == 9
    assert len(split.validation_indices) == 6

    all_indices = torch.cat([split.train_indices, split.validation_indices]).sort().values
    assert torch.equal(all_indices, torch.arange(len(labels)))


def test_stratified_split_is_reproducible_for_the_same_seed() -> None:
    labels = _balanced_labels()

    first = stratified_train_validation_split(
        labels,
        validation_per_class=2,
        seed=17,
    )
    second = stratified_train_validation_split(
        labels,
        validation_per_class=2,
        seed=17,
    )

    assert torch.equal(first.train_indices, second.train_indices)
    assert torch.equal(first.validation_indices, second.validation_indices)


def test_stratified_split_rejects_insufficient_examples_in_a_class() -> None:
    labels = torch.tensor([0, 0, 1], dtype=torch.long)

    with pytest.raises(ValueError, match="not enough examples"):
        stratified_train_validation_split(
            labels,
            validation_per_class=2,
            seed=17,
        )