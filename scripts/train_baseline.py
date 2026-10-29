"""Train the compact clean-image CIFAR-10 reference classifier."""

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from vision_uncertainty_bench.data import load_cifar10_datasets
from vision_uncertainty_bench.models import SmallCifarCNN
from vision_uncertainty_bench.training import (
    evaluate_classifier,
    resolve_device,
    set_global_seed,
    train_one_epoch,
)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for one baseline training run."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--output-directory", type=Path, default=None)
    return parser.parse_args()


def load_json_config(config_path: Path) -> dict:
    """Load an experiment configuration from JSON."""
    with config_path.open(encoding="utf-8") as file:
        return json.load(file)


def main() -> None:
    """Train and select a model using validation cross-entropy only."""
    arguments = parse_arguments()
    config = load_json_config(arguments.config)

    epochs = arguments.epochs if arguments.epochs is not None else int(config["epochs"])
    if epochs < 1:
        raise ValueError("epochs must be at least one.")

    output_directory = (
        arguments.output_directory
        if arguments.output_directory is not None
        else Path(config["output_directory"])
    )
    output_directory.mkdir(parents=True, exist_ok=True)

    run_config = dict(config)
    run_config["epochs"] = epochs
    run_config["output_directory"] = str(output_directory)

    set_global_seed(int(config["seed"]))
    device = resolve_device(str(config["device"]))

    datasets = load_cifar10_datasets(
        data_directory=config["data_directory"],
        validation_per_class=int(config["validation_per_class"]),
        split_seed=int(config["split_seed"]),
    )

    loader_generator = torch.Generator()
    loader_generator.manual_seed(int(config["seed"]))

    train_loader = DataLoader(
        datasets.train,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=int(config["num_workers"]),
        generator=loader_generator,
    )
    validation_loader = DataLoader(
        datasets.validation,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        num_workers=int(config["num_workers"]),
    )

    model = SmallCifarCNN().to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )

    best_validation_cross_entropy = float("inf")
    best_epoch = 0
    best_model_state: dict[str, torch.Tensor] | None = None
    records = []

    print(f"device={device.type}")
    print(f"training_examples={len(datasets.train)}")
    print(f"validation_examples={len(datasets.validation)}")

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, device)
        validation_metrics = evaluate_classifier(model, validation_loader, device)

        records.append(
            {
                "epoch": epoch,
                "train_cross_entropy": train_metrics.cross_entropy,
                "train_accuracy": train_metrics.accuracy,
                "validation_cross_entropy": validation_metrics.cross_entropy,
                "validation_accuracy": validation_metrics.accuracy,
            }
        )

        print(
            f"epoch={epoch:02d} "
            f"train_loss={train_metrics.cross_entropy:.4f} "
            f"train_accuracy={train_metrics.accuracy:.4f} "
            f"validation_loss={validation_metrics.cross_entropy:.4f} "
            f"validation_accuracy={validation_metrics.accuracy:.4f}"
        )

        if validation_metrics.cross_entropy < best_validation_cross_entropy:
            best_validation_cross_entropy = validation_metrics.cross_entropy
            best_epoch = epoch
            best_model_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }

    if best_model_state is None:
        raise RuntimeError("No model state was selected.")

    pd.DataFrame(records).to_csv(output_directory / "training_history.csv", index=False)
    (output_directory / "config_used.json").write_text(
        json.dumps(run_config, indent=2) + "\n",
        encoding="utf-8",
    )
    torch.save(
        {
            "best_epoch": best_epoch,
            "best_validation_cross_entropy": best_validation_cross_entropy,
            "model_state_dict": best_model_state,
        },
        output_directory / "best_model.pt",
    )

    print(f"best_epoch={best_epoch}")
    print(f"best_validation_loss={best_validation_cross_entropy:.4f}")
    print(f"saved_to={output_directory}")


if __name__ == "__main__":
    main()