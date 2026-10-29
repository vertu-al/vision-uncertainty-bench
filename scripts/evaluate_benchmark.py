"""Evaluate clean and corrupted CIFAR-10 test data with fixed conditions."""

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from vision_uncertainty_bench.corruption_dataset import CorruptedDataset
from vision_uncertainty_bench.data import load_cifar10_datasets
from vision_uncertainty_bench.evaluation import (
    probabilities_from_logits,
    summarize_probabilities,
)
from vision_uncertainty_bench.inference import collect_logits
from vision_uncertainty_bench.models import SmallCifarCNN
from vision_uncertainty_bench.selective import confidence_threshold_metrics
from vision_uncertainty_bench.training import resolve_device


def parse_arguments() -> argparse.Namespace:
    """Parse paths for one fixed benchmark evaluation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--temperature", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    """Load one JSON document."""
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def make_conditions(clean_test_dataset, corruption_seed: int) -> list[dict]:
    """Return the pre-specified clean and corrupted evaluation conditions."""
    conditions = [
        {
            "condition": "clean",
            "corruption": "clean",
            "severity": 0,
            "dataset": clean_test_dataset,
        }
    ]

    for corruption in ("gaussian_noise", "contrast", "central_occlusion"):
        for severity in (1, 2, 3):
            conditions.append(
                {
                    "condition": f"{corruption}_severity_{severity}",
                    "corruption": corruption,
                    "severity": severity,
                    "dataset": CorruptedDataset(
                        clean_test_dataset,
                        corruption=corruption,
                        severity=severity,
                        seed=corruption_seed,
                    ),
                }
            )

    return conditions


def main() -> None:
    """Evaluate raw and calibrated probabilities without selecting on test data."""
    arguments = parse_arguments()
    config = load_json(arguments.config)
    temperature_summary = load_json(arguments.temperature)
    temperature = float(temperature_summary["temperature"])

    device = resolve_device(str(config["device"]))
    datasets = load_cifar10_datasets(data_directory=config["data_directory"])

    checkpoint = torch.load(arguments.checkpoint, map_location="cpu")
    model = SmallCifarCNN()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    output_directory = arguments.output_directory
    output_directory.mkdir(parents=True, exist_ok=True)

    probability_conditions = [
        ("raw", 1.0),
        ("temperature_scaled", temperature),
    ]
    summary_records = []
    risk_coverage_records = []

    conditions = make_conditions(
        datasets.test,
        corruption_seed=int(config["corruption_seed"]),
    )

    print(f"device={device.type}")
    print(f"test_examples_per_condition={len(datasets.test)}")
    print(f"temperature={temperature:.6f}")

    for condition in conditions:
        loader = DataLoader(
            condition["dataset"],
            batch_size=int(config["batch_size"]),
            shuffle=False,
            num_workers=int(config["num_workers"]),
        )
        logits, targets = collect_logits(model, loader, device)

        for probability_condition, condition_temperature in probability_conditions:
            probabilities = probabilities_from_logits(
                logits,
                temperature=condition_temperature,
            )
            summary = summarize_probabilities(
                probabilities,
                targets,
                n_ece_bins=int(config["ece_bins"]),
            )

            summary_records.append(
                {
                    "condition": condition["condition"],
                    "corruption": condition["corruption"],
                    "severity": condition["severity"],
                    "probability_condition": probability_condition,
                    "temperature": condition_temperature,
                    "n_examples": summary.n_examples,
                    "accuracy": summary.accuracy,
                    "negative_log_likelihood": summary.negative_log_likelihood,
                    "brier_score": summary.brier_score,
                    "expected_calibration_error": summary.expected_calibration_error,
                }
            )

            for threshold in config["confidence_thresholds"]:
                selective_metrics = confidence_threshold_metrics(
                    probabilities,
                    targets,
                    threshold=float(threshold),
                )
                risk_coverage_records.append(
                    {
                        "condition": condition["condition"],
                        "corruption": condition["corruption"],
                        "severity": condition["severity"],
                        "probability_condition": probability_condition,
                        "temperature": condition_temperature,
                        "threshold": threshold,
                        "accepted_count": selective_metrics.accepted_count,
                        "coverage": selective_metrics.coverage,
                        "risk": selective_metrics.risk,
                    }
                )

            print(
                f"{condition['condition']} | {probability_condition} | "
                f"accuracy={summary.accuracy:.4f} | "
                f"nll={summary.negative_log_likelihood:.4f} | "
                f"ece={summary.expected_calibration_error:.4f}"
            )

    pd.DataFrame(summary_records).to_csv(
        output_directory / "summary_metrics.csv",
        index=False,
    )
    pd.DataFrame(risk_coverage_records).to_csv(
        output_directory / "risk_coverage.csv",
        index=False,
    )

    run_metadata = {
        "checkpoint": str(arguments.checkpoint),
        "temperature_file": str(arguments.temperature),
        "evaluation_config": config,
    }
    (output_directory / "evaluation_metadata.json").write_text(
        json.dumps(run_metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"saved_to={output_directory}")


if __name__ == "__main__":
    main()