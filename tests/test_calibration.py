import torch
import torch.nn.functional as functional

from vision_uncertainty_bench.calibration import TemperatureScaler, fit_temperature


def test_temperature_one_leaves_logits_unchanged() -> None:
    logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
    scaler = TemperatureScaler(initial_temperature=1.0)

    assert torch.allclose(scaler(logits), logits)


def test_higher_temperature_makes_probabilities_less_sharp() -> None:
    logits = torch.tensor([[4.0, 0.0]])
    scaler = TemperatureScaler(initial_temperature=2.0)

    raw_maximum_probability = torch.softmax(logits, dim=1).max()
    scaled_maximum_probability = torch.softmax(scaler(logits), dim=1).max()

    assert scaled_maximum_probability < raw_maximum_probability


def test_fitting_temperature_reduces_validation_cross_entropy_for_overconfidence() -> None:
    logits = torch.tensor(
        [
            [4.0, 0.0],  # correct and very confident
            [4.0, 0.0],  # wrong and very confident
            [4.0, 0.0],  # wrong and very confident
            [0.0, 4.0],  # correct and very confident
        ]
    )
    targets = torch.tensor([0, 1, 1, 1], dtype=torch.long)

    fitted_scaler = fit_temperature(logits, targets, max_iterations=50)

    raw_loss = functional.cross_entropy(logits, targets)
    calibrated_loss = functional.cross_entropy(fitted_scaler(logits), targets)

    assert fitted_scaler.temperature.item() > 1.0
    assert calibrated_loss < raw_loss