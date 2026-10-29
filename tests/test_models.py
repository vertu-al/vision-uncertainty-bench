import pytest
import torch

from vision_uncertainty_bench.models import SmallCifarCNN


def test_small_cifar_cnn_returns_one_logit_vector_per_image() -> None:
    model = SmallCifarCNN(num_classes=10)
    images = torch.rand(4, 3, 32, 32)

    logits = model(images)

    assert logits.shape == (4, 10)
    assert torch.isfinite(logits).all()


def test_small_cifar_cnn_supports_a_custom_number_of_classes() -> None:
    model = SmallCifarCNN(num_classes=3)
    images = torch.rand(2, 3, 32, 32)

    assert model(images).shape == (2, 3)


def test_small_cifar_cnn_rejects_inputs_without_three_colour_channels() -> None:
    model = SmallCifarCNN()

    with pytest.raises(ValueError, match="three colour channels"):
        model(torch.rand(2, 1, 32, 32))