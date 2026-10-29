import pytest
import torch

from vision_uncertainty_bench.corruptions import apply_corruption


def _example_image() -> torch.Tensor:
    return torch.linspace(0.0, 1.0, steps=3 * 32 * 32).reshape(3, 32, 32)


def test_gaussian_noise_is_deterministic_for_a_fixed_seed() -> None:
    image = _example_image()

    first = apply_corruption(
        image,
        corruption="gaussian_noise",
        severity=2,
        seed=123,
    )
    second = apply_corruption(
        image,
        corruption="gaussian_noise",
        severity=2,
        seed=123,
    )
    different_seed = apply_corruption(
        image,
        corruption="gaussian_noise",
        severity=2,
        seed=456,
    )

    assert torch.equal(first, second)
    assert not torch.equal(first, different_seed)
    assert first.shape == image.shape
    assert torch.all((first >= 0.0) & (first <= 1.0))


def test_contrast_reduction_preserves_mean_and_reduces_variation() -> None:
    image = _example_image()

    reduced_contrast = apply_corruption(
        image,
        corruption="contrast",
        severity=2,
        seed=123,
    )

    assert reduced_contrast.mean().item() == pytest.approx(image.mean().item())
    assert reduced_contrast.std().item() < image.std().item()
    assert torch.all((reduced_contrast >= 0.0) & (reduced_contrast <= 1.0))


def test_central_occlusion_removes_evidence_from_the_image_centre() -> None:
    image = torch.ones(3, 32, 32)

    occluded = apply_corruption(
        image,
        corruption="central_occlusion",
        severity=2,
        seed=123,
    )

    assert torch.all(occluded[:, 16, 16] == 0.0)
    assert torch.all(occluded[:, 0, 0] == 1.0)
    assert occluded.shape == image.shape


def test_unknown_corruption_raises_a_clear_error() -> None:
    with pytest.raises(ValueError, match="Unknown corruption"):
        apply_corruption(
            _example_image(),
            corruption="rain",
            severity=1,
            seed=123,
        )