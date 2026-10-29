"""Fit post-hoc temperature scaling on clean validation logits only."""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader

from vision_uncertainty_bench.calibration import fit_temperature
from vision_uncertainty_bench.data import load_cifar10_datasets
from vision_uncertainty_bench.inference import collect_logits
from vision_uncertainty_bench.models import SmallCifarCNN
from vision_uncertainty_bench.training import resolve_device


def parse_arguments() -> argparse.Namespace:
    """Parse paths for one temperature-scaling fit."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """Fit temperature on validation data and save a readable summary."""
    arguments = parse_arguments()

    with arguments.config.open(encoding="utf-8") as file:
        config = json.load(file)

    device = resolve_device(str(config["device"]))
    datasets = load_cifar10_datasets(
        data_directory=config["data_directory"],
        validation_per_class=int(config["validation_per_class"]),
        split_seed=int(config["split_seed"]),
    )
    validation_loader = DataLoader(
        datasets.validation,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        num_workers=int(config["num_workers"]),
    )

    checkpoint = torch.load(arguments.checkpoint, map_location="cpu")
    model = SmallCifarCNN()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    validation_logits, validation_targets = collect_logits(
        model,
        validation_loader,
        device,
    )
    fitted_scaler = fit_temperature(validation_logits, validation_targets)

    raw_nll = functional.cross_entropy(validation_logits, validation_targets).item()
    calibrated_nll = functional.cross_entropy(
        fitted_scaler(validation_logits),
        validation_targets,
    ).item()

    arguments.output_directory.mkdir(parents=True, exist_ok=True)
    summary = {
        "checkpoint": str(arguments.checkpoint),
        "n_validation_examples": len(validation_targets),
        "temperature": fitted_scaler.temperature.item(),
        "validation_nll_raw": raw_nll,
        "validation_nll_calibrated": calibrated_nll,
    }
    (arguments.output_directory / "temperature.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"device={device.type}")
    print(f"validation_examples={len(validation_targets)}")
    print(f"temperature={summary['temperature']:.6f}")
    print(f"validation_nll_raw={raw_nll:.6f}")
    print(f"validation_nll_calibrated={calibrated_nll:.6f}")
    print(f"saved_to={arguments.output_directory / 'temperature.json'}")


if __name__ == "__main__":
    main()