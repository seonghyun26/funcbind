"""Render static figures used by gaussian_splat_overfit_report.ipynb."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


NOTEBOOK_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = NOTEBOOK_DIR.parents[1]
RESULT_PATH = (
    PROJECT_ROOT
    / "exps"
    / "decoder_ablation"
    / "gpu2_fullgrid_validation_20260728"
    / "comparison.json"
)
FIGURE_DIR = NOTEBOOK_DIR / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
OCCUPANCY_ROOT = (
    PROJECT_ROOT
    / "exps"
    / "decoder_ablation"
    / "overfit_occupancy_visualization_20260728"
)
OCCUPANCY_PATHS = {
    "reference": (
        OCCUPANCY_ROOT
        / "inr_gpu2"
        / "target_000"
        / "reference_occupancy_points.npz"
    ),
    "inr": (
        OCCUPANCY_ROOT
        / "inr_gpu2"
        / "target_000"
        / "inr_occupancy_points.npz"
    ),
    "gaussian_splat": (
        OCCUPANCY_ROOT
        / "gaussian_splat_gpu3"
        / "target_000"
        / "gaussian_splat_occupancy_points.npz"
    ),
}

COLORS = {"inr": "#64748b", "gaussian_splat": "#e4572e"}
LABELS = {"inr": "INR baseline", "gaussian_splat": "Gaussian splat"}


def _load_results() -> dict:
    with RESULT_PATH.open() as handle:
        report = json.load(handle)
    if report.get("status") != "complete":
        raise RuntimeError(f"Incomplete comparison at {RESULT_PATH}")
    return report["results"]


def render_learning_curves(results: dict) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), constrained_layout=True)
    for name in ("inr", "gaussian_splat"):
        history = results[name]["history"]
        steps = [point["step"] for point in history]
        mse = [point["density_mse"] for point in history]
        miou = [point["miou"] for point in history]
        axes[0].plot(
            steps,
            mse,
            marker="o",
            linewidth=2.2,
            markersize=4,
            color=COLORS[name],
            label=LABELS[name],
        )
        axes[1].plot(
            steps,
            miou,
            marker="o",
            linewidth=2.2,
            markersize=4,
            color=COLORS[name],
            label=LABELS[name],
        )

    axes[0].set_yscale("log")
    axes[0].set_title("Sampled-point reconstruction loss")
    axes[0].set_xlabel("Optimization step")
    axes[0].set_ylabel("Density MSE (log scale)")
    axes[1].set_title("Sampled-point occupancy overlap")
    axes[1].set_xlabel("Optimization step")
    axes[1].set_ylabel("mIoU at threshold 0.5")
    axes[1].set_ylim(-0.03, 1.05)
    for axis in axes:
        axis.grid(alpha=0.22)
        axis.legend(frameon=False)

    path = FIGURE_DIR / "overfit_learning_curves.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def render_full_grid_summary(results: dict) -> Path:
    names = ("inr", "gaussian_splat")
    labels = [LABELS[name] for name in names]
    colors = [COLORS[name] for name in names]
    figure, axes = plt.subplots(
        2, 2, figsize=(12, 9), constrained_layout=True
    )

    full_mse = [
        results[name]["full_grid"]["reconstruction"]["density_mse"]
        for name in names
    ]
    full_miou = [
        results[name]["full_grid"]["reconstruction"]["miou"]
        for name in names
    ]
    axis = axes[0, 0]
    bars = axis.bar(labels, full_mse, color=colors, width=0.62)
    axis.set_yscale("log")
    axis.set_ylabel("Density MSE (log scale)")
    axis.set_title("Full 128³ density reconstruction")
    axis.bar_label(bars, labels=[f"{value:.2e}" for value in full_mse])
    overlap_axis = axis.twinx()
    overlap_axis.plot(
        labels,
        full_miou,
        color="#0f172a",
        marker="D",
        markersize=7,
        linewidth=1.8,
        label="mIoU",
    )
    overlap_axis.set_ylabel("mIoU")
    overlap_axis.set_ylim(0, max(full_miou) * 1.35)
    overlap_axis.legend(frameon=False, loc="upper center")

    recovery_metrics = (
        ("atom_precision", "Precision"),
        ("atom_recall", "Recall"),
        ("atom_f1", "F1"),
        ("element_accuracy", "Element accuracy"),
    )
    positions = np.arange(len(recovery_metrics))
    width = 0.36
    axis = axes[0, 1]
    for decoder_index, name in enumerate(names):
        values = [
            results[name]["full_grid"]["atom_recovery"][key]
            for key, _ in recovery_metrics
        ]
        axis.bar(
            positions + (decoder_index - 0.5) * width,
            values,
            width,
            color=COLORS[name],
            label=LABELS[name],
        )
    axis.set_xticks(
        positions, [label for _, label in recovery_metrics], rotation=14
    )
    axis.set_ylim(0, 1)
    axis.set_ylabel("Score")
    axis.set_title("Atom recovery")
    axis.legend(frameon=False)

    count_metrics = (
        ("n_target_atoms", "Target"),
        ("n_predicted_atoms", "Predicted"),
        ("matched_atoms", "Matched ≤1 Å"),
    )
    positions = np.arange(len(count_metrics))
    axis = axes[1, 0]
    for decoder_index, name in enumerate(names):
        values = [
            results[name]["full_grid"]["atom_recovery"][key]
            for key, _ in count_metrics
        ]
        bars = axis.bar(
            positions + (decoder_index - 0.5) * width,
            values,
            width,
            color=COLORS[name],
            label=LABELS[name],
        )
        axis.bar_label(bars, fontsize=9)
    axis.set_xticks(positions, [label for _, label in count_metrics])
    axis.set_ylabel("Atom/peak count")
    axis.set_title("Peak extraction behavior")
    axis.legend(frameon=False)

    parameter_counts = [results[name]["decoder_parameters"] for name in names]
    peak_memory = [
        results[name]["timing"]["peak_gpu_memory_gib"] for name in names
    ]
    axis = axes[1, 1]
    bars = axis.bar(labels, parameter_counts, color=colors, width=0.62)
    axis.set_yscale("log")
    axis.set_ylabel("Decoder parameters (log scale)")
    axis.set_title("Decoder size and training memory")
    axis.bar_label(
        bars,
        labels=[
            f"{value / 1e6:.3f}M" if value < 1e6 else f"{value / 1e6:.1f}M"
            for value in parameter_counts
        ],
    )
    memory_axis = axis.twinx()
    memory_axis.plot(
        labels,
        peak_memory,
        color="#0f172a",
        marker="s",
        markersize=7,
        linewidth=1.8,
        label="Peak GPU memory",
    )
    memory_axis.set_ylabel("Peak training memory (GiB)")
    memory_axis.set_ylim(0, max(peak_memory) * 1.35)
    memory_axis.legend(frameon=False, loc="upper center")

    figure.suptitle(
        "One-target MCPP overfit: INR versus channel-wise Gaussian splatting",
        fontsize=15,
        fontweight="bold",
    )
    path = FIGURE_DIR / "overfit_fullgrid_summary.png"
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return path


ELEMENTS = ("C", "O", "N", "S", "F", "Cl", "P", "Br")
ELEMENT_COLORS = (
    "#4b5563",
    "#dc2626",
    "#2563eb",
    "#eab308",
    "#22c55e",
    "#16a34a",
    "#f97316",
    "#92400e",
)


def _load_occupancy_cloud(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def _subsample_cloud(
    cloud: dict[str, np.ndarray],
    *,
    max_points_per_channel: int = 5000,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(1234)
    selected = []
    for channel in range(len(ELEMENTS)):
        channel_indices = np.flatnonzero(cloud["channels"] == channel)
        if channel_indices.size > max_points_per_channel:
            channel_indices = rng.choice(
                channel_indices, max_points_per_channel, replace=False
            )
        selected.append(channel_indices)
    indices = np.concatenate(selected) if selected else np.empty(0, dtype=int)
    return {
        key: value[indices]
        for key, value in cloud.items()
        if value.ndim > 0 and value.shape[0] == cloud["channels"].shape[0]
    }


def render_occupancy_3d() -> Path:
    clouds = {
        name: _load_occupancy_cloud(path)
        for name, path in OCCUPANCY_PATHS.items()
    }
    shown_clouds = {name: _subsample_cloud(cloud) for name, cloud in clouds.items()}
    all_coordinates = np.concatenate(
        [cloud["coordinates"] for cloud in shown_clouds.values()], axis=0
    )
    extent = float(np.quantile(np.abs(all_coordinates), 0.995))
    extent = min(max(np.ceil(extent) + 0.5, 4.0), 16.0)

    figure = plt.figure(figsize=(16, 5.4), constrained_layout=True)
    panels = (
        ("reference", "Reference occupancy"),
        ("inr", "INR reconstruction"),
        ("gaussian_splat", "Gaussian-splat reconstruction"),
    )
    for panel_index, (name, title) in enumerate(panels, start=1):
        axis = figure.add_subplot(1, 3, panel_index, projection="3d")
        cloud = shown_clouds[name]
        original_count = clouds[name]["coordinates"].shape[0]
        for channel, (element, color) in enumerate(
            zip(ELEMENTS, ELEMENT_COLORS)
        ):
            mask = cloud["channels"] == channel
            if not np.any(mask):
                continue
            occupancies = cloud["occupancies"][mask].astype(np.float32)
            sizes = 2.0 + 9.0 * occupancies
            coordinates = cloud["coordinates"][mask]
            axis.scatter(
                coordinates[:, 0],
                coordinates[:, 1],
                coordinates[:, 2],
                s=sizes,
                c=color,
                alpha=0.34,
                linewidths=0,
                label=element,
                depthshade=False,
            )
        shown_count = cloud["coordinates"].shape[0]
        axis.set_title(f"{title}\n{shown_count:,}/{original_count:,} points shown")
        axis.set(
            xlim=(-extent, extent),
            ylim=(-extent, extent),
            zlim=(-extent, extent),
            xlabel="x (Å)",
            ylabel="y (Å)",
            zlabel="z (Å)",
        )
        axis.set_box_aspect((1, 1, 1))
        axis.view_init(elev=22, azim=38)
        axis.grid(alpha=0.18)
        if panel_index == 1:
            axis.legend(
                loc="upper left", bbox_to_anchor=(-0.1, 1.0), frameon=False
            )

    threshold = float(clouds["reference"]["threshold"])
    figure.suptitle(
        f"Full-grid occupancy clouds at occupancy ≥ {threshold:.2f}",
        fontsize=15,
        fontweight="bold",
    )
    path = FIGURE_DIR / "overfit_occupancy_3d.png"
    figure.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(figure)
    return path


if __name__ == "__main__":
    decoder_results = _load_results()
    for output in (
        render_learning_curves(decoder_results),
        render_full_grid_summary(decoder_results),
    ):
        print(output)
    if all(path.exists() for path in OCCUPANCY_PATHS.values()):
        print(render_occupancy_3d())
    else:
        print("Occupancy point clouds are not available yet; skipping 3D render.")
