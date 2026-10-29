"""Create final benchmark figures and export tracked evaluation results."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

FAMILY_LABELS = {
    "gaussian_noise": "Gaussian noise",
    "contrast": "Contrast",
    "central_occlusion": "Central occlusion",
}

FAMILY_COLORS = {
    "gaussian_noise": "#D55E00",
    "contrast": "#0072B2",
    "central_occlusion": "#009E73",
}

SEVERITY_COLORS = {
    1: "#9ecae1",
    2: "#3182bd",
    3: "#08519c",
}


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Create figures from vision uncertainty benchmark outputs."
    )
    parser.add_argument(
        "--input-directory",
        type=Path,
        default=Path("artifacts/baseline_seed_2026/evaluation"),
        help="Directory containing evaluation CSV files.",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("results"),
        help="Directory for tracked CSVs and figures.",
    )
    return parser.parse_args()


def family_condition(family: str, severity: int) -> str:
    """Return the condition name used in evaluation outputs."""
    return f"{family}_severity_{severity}"


def condition_label(condition: str) -> str:
    """Return a concise human-readable condition label."""
    if condition == "clean":
        return "Clean"

    family, _, severity = condition.rpartition("_severity_")
    return f"{FAMILY_LABELS[family]} {severity}"


def metric_values(
    summary: pd.DataFrame,
    family: str,
    probability_condition: str,
    metric: str,
) -> tuple[list[int], list[float]]:
    """Return clean plus severity-one-to-three values for one metric."""
    clean_value = summary.loc[
        (summary["condition"] == "clean")
        & (summary["probability_condition"] == probability_condition),
        metric,
    ].iloc[0]

    corruption_values = [
        summary.loc[
            (summary["condition"] == family_condition(family, severity))
            & (summary["probability_condition"] == probability_condition),
            metric,
        ].iloc[0]
        for severity in (1, 2, 3)
    ]

    return [0, 1, 2, 3], [clean_value, *corruption_values]


def plot_corruption_summary(summary: pd.DataFrame, output_path: Path) -> None:
    """Plot accuracy, calibration error, and NLL under corruption severity."""
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    for family, label in FAMILY_LABELS.items():
        x_values, accuracy_values = metric_values(
            summary,
            family,
            "raw",
            "accuracy",
        )
        axes[0].plot(
            x_values,
            accuracy_values,
            color=FAMILY_COLORS[family],
            marker="o",
            linewidth=2.2,
            label=label,
        )

    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Corruption severity")
    axes[0].set_ylabel("Test accuracy")
    axes[0].set_xticks([0, 1, 2, 3])
    axes[0].set_ylim(0, 1)
    axes[0].grid(alpha=0.25)

    for metric, axis, title, ylabel in [
        ("expected_calibration_error", axes[1], "Calibration error", "Expected calibration error"),
        (
            "negative_log_likelihood",
            axes[2],
            "Negative log-likelihood",
            "NLL",
        ),
    ]:
        for family in FAMILY_LABELS:
            for probability_condition, line_style in [
                ("raw", "--"),
                ("temperature_scaled", "-"),
            ]:
                x_values, values = metric_values(
                    summary,
                    family,
                    probability_condition,
                    metric,
                )
                axis.plot(
                    x_values,
                    values,
                    color=FAMILY_COLORS[family],
                    linestyle=line_style,
                    marker="o",
                    linewidth=2.0,
                )

        axis.set_title(title)
        axis.set_xlabel("Corruption severity")
        axis.set_ylabel(ylabel)
        axis.set_xticks([0, 1, 2, 3])
        axis.grid(alpha=0.25)

    family_handles = [
        Line2D(
            [0],
            [0],
            color=FAMILY_COLORS[family],
            linewidth=2.5,
            label=label,
        )
        for family, label in FAMILY_LABELS.items()
    ]
    calibration_handles = [
        Line2D([0], [0], color="black", linestyle="--", label="Raw probabilities"),
        Line2D(
            [0],
            [0],
            color="black",
            linestyle="-",
            label="Temperature scaled",
        ),
    ]

    figure.suptitle(
        "CIFAR-10 uncertainty benchmark: performance under controlled corruption",
        fontsize=13,
    )
    figure.legend(
        handles=[*family_handles, *calibration_handles],
        loc="lower center",
        ncol=5,
        frameon=False,
    )
    figure.tight_layout(rect=(0, 0.12, 1, 0.92))
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_threshold_summary(risk_coverage: pd.DataFrame, output_path: Path) -> None:
    """Plot risk and coverage at the preconfigured confidence threshold of 0.80."""
    threshold = 0.80
    selected = risk_coverage.loc[
        np.isclose(risk_coverage["threshold"], threshold)
    ].copy()

    condition_order = [
        "clean",
        *[
            family_condition(family, severity)
            for family in FAMILY_LABELS
            for severity in (1, 2, 3)
        ],
    ]

    risk_table = selected.pivot(
        index="condition",
        columns="probability_condition",
        values="risk",
    ).reindex(condition_order)

    coverage_table = selected.pivot(
        index="condition",
        columns="probability_condition",
        values="coverage",
    ).reindex(condition_order)

    labels = [condition_label(condition) for condition in condition_order]
    positions = np.arange(len(condition_order))
    width = 0.36

    figure, axes = plt.subplots(1, 2, figsize=(16, 5.4), sharex=True)

    for axis, table, title, ylabel in [
        (axes[0], risk_table, "Residual error among accepted predictions", "Risk"),
        (axes[1], coverage_table, "Fraction of predictions retained", "Coverage"),
    ]:
        axis.bar(
            positions - width / 2,
            table["raw"],
            width,
            label="Raw probabilities",
            color="#7F7F7F",
        )
        axis.bar(
            positions + width / 2,
            table["temperature_scaled"],
            width,
            label="Temperature scaled",
            color="#009E73",
        )
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.set_ylim(0, 1)
        axis.set_xticks(positions)
        axis.set_xticklabels(labels, rotation=40, ha="right")
        axis.grid(axis="y", alpha=0.25)

    axes[0].legend(frameon=False)
    figure.suptitle(
        "Selective prediction at a fixed confidence threshold of 0.80",
        fontsize=13,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_risk_coverage_curves(risk_coverage: pd.DataFrame, output_path: Path) -> None:
    """Plot full risk-coverage curves for clean data and each corruption family."""
    figure, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True, sharey=True)

    clean_axis = axes[0, 0]
    clean_data = risk_coverage.loc[risk_coverage["condition"] == "clean"]

    for probability_condition, color, label in [
        ("raw", "#7F7F7F", "Raw probabilities"),
        ("temperature_scaled", "#009E73", "Temperature scaled"),
    ]:
        subset = clean_data.loc[
            clean_data["probability_condition"] == probability_condition
        ].sort_values("coverage")

        clean_axis.plot(
            subset["coverage"],
            subset["risk"],
            color=color,
            marker="o",
            linewidth=2.2,
            label=label,
        )

    clean_axis.set_title("Clean test set")
    clean_axis.legend(frameon=False, fontsize=8)

    for axis, (family, family_label) in zip(
        [axes[0, 1], axes[1, 0], axes[1, 1]],
        FAMILY_LABELS.items(),
        strict=True,
    ):
        for severity in (1, 2, 3):
            for probability_condition, line_style in [
                ("raw", "--"),
                ("temperature_scaled", "-"),
            ]:
                subset = risk_coverage.loc[
                    (risk_coverage["condition"] == family_condition(family, severity))
                    & (
                        risk_coverage["probability_condition"]
                        == probability_condition
                    )
                ].sort_values("coverage")

                axis.plot(
                    subset["coverage"],
                    subset["risk"],
                    color=SEVERITY_COLORS[severity],
                    linestyle=line_style,
                    marker="o",
                    markersize=3,
                    linewidth=1.8,
                    label=f"S{severity} — {probability_condition.replace('_', ' ')}",
                )

        axis.set_title(family_label)
        axis.legend(frameon=False, fontsize=7, ncol=2)

    for axis in axes.flat:
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.set_xlabel("Coverage")
        axis.set_ylabel("Risk among accepted predictions")
        axis.grid(alpha=0.25)

    figure.suptitle(
        "Risk–coverage curves across clean and corrupted test conditions",
        fontsize=13,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    """Create figures and copy final benchmark outputs into results/."""
    arguments = parse_arguments()

    input_directory = arguments.input_directory
    output_directory = arguments.output_directory
    figures_directory = output_directory / "figures"

    output_directory.mkdir(parents=True, exist_ok=True)
    figures_directory.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(input_directory / "summary_metrics.csv")
    risk_coverage = pd.read_csv(input_directory / "risk_coverage.csv")

    summary.to_csv(output_directory / "summary_metrics.csv", index=False)
    risk_coverage.to_csv(output_directory / "risk_coverage.csv", index=False)

    metadata_path = input_directory / "evaluation_metadata.json"
    if metadata_path.exists():
        shutil.copy2(metadata_path, output_directory / "evaluation_metadata.json")

    plot_corruption_summary(
        summary,
        figures_directory / "corruption_benchmark_summary.png",
    )
    plot_threshold_summary(
        risk_coverage,
        figures_directory / "threshold_080_selective_summary.png",
    )
    plot_risk_coverage_curves(
        risk_coverage,
        figures_directory / "risk_coverage_curves.png",
    )

    print(f"saved_results_to={output_directory}")
    print(f"saved_figures_to={figures_directory}")


if __name__ == "__main__":
    main()