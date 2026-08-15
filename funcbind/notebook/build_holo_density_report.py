"""Build the static figures and notebook for the 5MGL holo-density POC."""

from __future__ import annotations

import base64
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


NOTEBOOK_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = NOTEBOOK_DIR.parents[1]
RESULT_DIR = (
    PROJECT_ROOT
    / "exps"
    / "density_fusion"
    / "holo_5mgl_target69_20260728"
)
RESULT_PATH = RESULT_DIR / "result.json"
FIGURE_DIR = NOTEBOOK_DIR / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
NOTEBOOK_PATH = NOTEBOOK_DIR / "holo_density_funcbind_report.ipynb"

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


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def _subsample_occupancy(
    cloud: dict[str, np.ndarray], max_per_channel: int = 3500
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(1269)
    selected = []
    for channel in range(len(ELEMENTS)):
        indices = np.flatnonzero(cloud["channels"] == channel)
        if indices.size > max_per_channel:
            indices = rng.choice(indices, max_per_channel, replace=False)
        selected.append(indices)
    keep = np.concatenate(selected)
    return {
        "coordinates": cloud["coordinates"][keep],
        "channels": cloud["channels"][keep],
        "occupancies": cloud["occupancies"][keep].astype(np.float32),
    }


def _subsample_density(
    cloud: dict[str, np.ndarray], max_points: int = 24000
) -> dict[str, np.ndarray]:
    if cloud["coordinates"].shape[0] <= max_points:
        return cloud
    rng = np.random.default_rng(1269)
    weights = np.maximum(cloud["values"].astype(np.float32), 0)
    weights = weights / weights.sum()
    keep = rng.choice(
        cloud["coordinates"].shape[0],
        max_points,
        replace=False,
        p=weights,
    )
    return {
        "coordinates": cloud["coordinates"][keep],
        "values": cloud["values"][keep].astype(np.float32),
    }


def render_training(report: dict) -> Path:
    history = json.loads(
        (RESULT_DIR / "training_history.json").read_text(encoding="utf-8")
    )
    steps = [row["step"] for row in history]
    losses = [row["denoise_loss"] for row in history]
    residual = [row["residual_rms"] for row in history]
    figure, axes = plt.subplots(
        1, 2, figsize=(11.5, 4.2), constrained_layout=True
    )
    axes[0].plot(steps, losses, color="#7c3aed", linewidth=2)
    axes[0].set(
        xlabel="Adapter step",
        ylabel="EDM2 denoising objective",
        title="Single-target adapter fit",
        yscale="log",
    )
    axes[1].plot(steps, residual, color="#0f766e", linewidth=2)
    axes[1].set(
        xlabel="Adapter step",
        ylabel="Residual RMS",
        title="Density residual magnitude",
    )
    for axis in axes:
        axis.grid(alpha=0.22)
    figure.suptitle(
        "Frozen FuncBind + frozen density encoder; adapter only",
        fontsize=14,
        fontweight="bold",
    )
    path = FIGURE_DIR / "holo_density_adapter_training.png"
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return path


def render_latent_comparison(report: dict) -> Path:
    original = np.asarray(
        report["sampling"]["original_latent_mse"], dtype=float
    )
    density = np.asarray(
        report["sampling"]["holo_density_latent_mse"], dtype=float
    )
    index = np.arange(original.size)
    figure, axes = plt.subplots(
        1, 2, figsize=(11.5, 4.2), constrained_layout=True
    )
    width = 0.38
    axes[0].bar(
        index - width / 2,
        original,
        width,
        color="#64748b",
        label="Original FuncBind",
    )
    axes[0].bar(
        index + width / 2,
        density,
        width,
        color="#7c3aed",
        label="Holo-density adapter",
    )
    axes[0].set(
        xlabel="Paired diffusion chain",
        ylabel="Latent MSE to reference code",
        title="Same initial latent and sampler RNG",
    )
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.22)
    paired_delta = density - original
    colors = np.where(paired_delta < 0, "#0f766e", "#dc2626")
    axes[1].bar(index, paired_delta, color=colors)
    axes[1].axhline(0, color="#111827", linewidth=1)
    axes[1].set(
        xlabel="Paired diffusion chain",
        ylabel="Density − original latent MSE",
        title="Negative means density helped",
    )
    axes[1].grid(axis="y", alpha=0.22)
    path = FIGURE_DIR / "holo_density_paired_latent_mse.png"
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return path


def render_metric_summary(report: dict) -> Path:
    results = report["results"]
    names = ("original_funcbind", "holo_density_funcbind")
    labels = ("Original FuncBind", "Holo-density adapter")
    colors = ("#64748b", "#7c3aed")
    figure, axes = plt.subplots(
        1, 3, figsize=(14, 4.5), constrained_layout=True
    )

    mse = [
        results[name]["reconstruction"]["density_mse"] for name in names
    ]
    bars = axes[0].bar(labels, mse, color=colors)
    axes[0].bar_label(bars, labels=[f"{value:.2e}" for value in mse])
    axes[0].set(
        yscale="log",
        ylabel="Density MSE",
        title="Full 128³ occupancy error",
    )

    miou = [results[name]["reconstruction"]["miou"] for name in names]
    bars = axes[1].bar(labels, miou, color=colors)
    axes[1].bar_label(bars, labels=[f"{value:.3f}" for value in miou])
    axes[1].set(ylim=(0, max(0.05, max(miou) * 1.25)), title="Occupancy mIoU")

    metric_keys = (
        ("atom_precision", "Precision"),
        ("atom_recall", "Recall"),
        ("atom_f1", "F1"),
    )
    x = np.arange(len(metric_keys))
    width = 0.36
    for method_index, (name, label, color) in enumerate(
        zip(names, labels, colors)
    ):
        values = [
            results[name]["atom_recovery"][key] for key, _ in metric_keys
        ]
        axes[2].bar(
            x + (method_index - 0.5) * width,
            values,
            width,
            color=color,
            label=label,
        )
    axes[2].set_xticks(x, [label for _, label in metric_keys])
    axes[2].set(ylim=(0, 1.05), ylabel="Score", title="Atom recovery ≤1 Å")
    axes[2].legend(frameon=False, fontsize=8)
    for axis in axes:
        axis.tick_params(axis="x", rotation=12)
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle(
        "Oracle-best of 8 paired chains on CrossDocked target 69",
        fontsize=14,
        fontweight="bold",
    )
    path = FIGURE_DIR / "holo_density_metric_summary.png"
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return path


def _plot_atoms(axis, atoms: dict[str, np.ndarray]) -> None:
    for channel, (element, color) in enumerate(
        zip(ELEMENTS, ELEMENT_COLORS)
    ):
        mask = atoms["channels"] == channel
        if not np.any(mask):
            continue
        xyz = atoms["coordinates"][mask]
        axis.scatter(
            xyz[:, 0],
            xyz[:, 1],
            xyz[:, 2],
            s=55,
            color=color,
            edgecolors="white",
            linewidths=0.45,
            alpha=0.95,
            label=element,
            depthshade=False,
        )


def _plot_occupancy(axis, cloud: dict[str, np.ndarray]) -> None:
    shown = _subsample_occupancy(cloud)
    for channel, color in enumerate(ELEMENT_COLORS):
        mask = shown["channels"] == channel
        if not np.any(mask):
            continue
        xyz = shown["coordinates"][mask]
        occupancy = shown["occupancies"][mask]
        axis.scatter(
            xyz[:, 0],
            xyz[:, 1],
            xyz[:, 2],
            s=1.5 + 7 * occupancy,
            color=color,
            alpha=0.28,
            linewidths=0,
            depthshade=False,
        )


def render_occupancy_3d(report: dict) -> Path:
    density = _subsample_density(
        _load_npz(RESULT_DIR / "density_points.npz")
    )
    reference_atoms = _load_npz(RESULT_DIR / "reference_atoms.npz")
    reference = _load_npz(RESULT_DIR / "reference_occupancy_points.npz")
    original = _load_npz(
        RESULT_DIR / "original_funcbind" / "occupancy_points.npz"
    )
    fused = _load_npz(
        RESULT_DIR / "holo_density_funcbind" / "occupancy_points.npz"
    )

    figure = plt.figure(figsize=(18, 5.2), constrained_layout=True)
    panels = (
        ("Real holo map + reference atoms", "density", density),
        ("Reference ligand occupancy", "occupancy", reference),
        ("Original FuncBind", "occupancy", original),
        ("Holo-density-conditioned FuncBind", "occupancy", fused),
    )
    for panel_index, (title, kind, data) in enumerate(panels, start=1):
        axis = figure.add_subplot(1, 4, panel_index, projection="3d")
        if kind == "density":
            xyz = data["coordinates"]
            values = data["values"].astype(np.float32)
            axis.scatter(
                xyz[:, 0],
                xyz[:, 1],
                xyz[:, 2],
                c=values,
                cmap="Purples",
                s=2.5,
                alpha=0.18,
                linewidths=0,
                depthshade=False,
            )
            _plot_atoms(axis, reference_atoms)
        else:
            _plot_occupancy(axis, data)
        axis.set(
            xlim=(-8, 8),
            ylim=(-8, 8),
            zlim=(-8, 8),
            xlabel="x (Å)",
            ylabel="y (Å)",
            zlabel="z (Å)",
        )
        axis.set_box_aspect((1, 1, 1))
        axis.view_init(elev=23, azim=38)
        axis.set_title(title)
        axis.grid(alpha=0.18)
        if panel_index == 1:
            axis.legend(frameon=False, fontsize=7, loc="upper left")
    figure.suptitle(
        "5MGL / 7MU: real holo density and generated ligand occupancies",
        fontsize=15,
        fontweight="bold",
    )
    path = FIGURE_DIR / "holo_density_occupancy_3d.png"
    figure.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(figure)
    return path


def render_density_slices(report: dict) -> Path:
    density = _load_npz(RESULT_DIR / "density_grid.npz")[
        "density"
    ].astype(np.float32)
    mid = density.shape[0] // 2
    slices = (density[mid], density[:, mid], density[:, :, mid])
    labels = ("x = 0 Å", "y = 0 Å", "z = 0 Å")
    figure, axes = plt.subplots(
        1, 3, figsize=(12, 4), constrained_layout=True
    )
    vmax = float(np.quantile(np.abs(density), 0.995))
    for axis, image, label in zip(axes, slices, labels):
        shown = axis.imshow(
            image.T,
            origin="lower",
            cmap="coolwarm",
            vmin=-vmax,
            vmax=vmax,
            extent=(-8, 8, -8, 8),
        )
        axis.set(title=label, xlabel="Å", ylabel="Å")
    figure.colorbar(shown, ax=axes, shrink=0.82, label="Locally normalized density")
    figure.suptitle(
        "Aligned real 2Fo-Fc holo-density crop",
        fontsize=14,
        fontweight="bold",
    )
    path = FIGURE_DIR / "holo_density_slices.png"
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return path


def _markdown_table(report: dict) -> str:
    rows = []
    for key, label in (
        ("original_funcbind", "Original FuncBind"),
        ("holo_density_funcbind", "Holo-density adapter"),
    ):
        result = report["results"][key]
        rec = result["reconstruction"]
        atoms = result["atom_recovery"]
        rows.append(
            f"| {label} | {rec['density_mse']:.3e} | {rec['miou']:.4f} | "
            f"{atoms['n_predicted_atoms']} | {atoms['matched_atoms']} | "
            f"{atoms['atom_precision']:.3f} | {atoms['atom_recall']:.3f} | "
            f"{atoms['atom_f1']:.3f} |"
        )
    return "\n".join(
        [
            "| Method | Occupancy MSE | mIoU | Pred. atoms | Matched | Precision | Recall | F1 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
            *rows,
        ]
    )


def _code_cell(source: str, cell_id: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.rstrip().splitlines()],
    }


def _markdown_cell(source: str, cell_id: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": cell_id,
        "metadata": {},
        "source": [line + "\n" for line in source.rstrip().splitlines()],
    }


def build_notebook(report: dict | None) -> Path:
    cells = [
        _markdown_cell(
            """# Real holo-density fusion into pretrained FuncBind

This report compares **original pretrained FuncBind** against the same frozen
FuncBind model conditioned by a residual derived from an aligned real 2Fo-Fc
holo map. The selected case is CrossDocked test target **69: 5MGL / ligand
7MU**.

> **Interpretation boundary:** this is a single-target proof of concept. The
holo map contains the crystallographic ligand, the small density adapter is
overfit on this target, and the displayed molecule from each method is selected
by oracle latent MSE. It tests whether this density representation can steer the
existing decoder; it does not establish held-out generation performance.""",
            "scope",
        ),
        _code_cell(
            """import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

search_roots = [Path.cwd(), *Path.cwd().parents]
PROJECT_ROOT = next(
    root for root in search_roots
    if (root / 'exps' / 'density_fusion').exists()
)
RESULT_DIR = (
    PROJECT_ROOT / 'exps' / 'density_fusion'
    / 'holo_5mgl_target69_20260728'
)
with (RESULT_DIR / 'result.json').open() as handle:
    report = json.load(handle)
report['status'], report['target']""",
            "load-results",
        ),
    ]

    if report is None:
        cells.append(
            _markdown_cell(
                """## Run pending

The notebook structure is ready, but `result.json` is not present yet because
the four H200 GPUs are occupied by the original full reconstruction run. The
queued experiment will populate this notebook after GPU 0 is empty.""",
                "pending",
            )
        )
    else:
        figures = {
            "training": render_training(report),
            "latent": render_latent_comparison(report),
            "metrics": render_metric_summary(report),
            "occupancy": render_occupancy_3d(report),
            "slices": render_density_slices(report),
        }
        rel = {
            key: path.relative_to(NOTEBOOK_DIR).as_posix()
            for key, path in figures.items()
        }
        cells.extend(
            [
                _markdown_cell(
                    f"""## Quantitative result

{_markdown_table(report)}

The comparison uses {report['sampling']['n_chains']} paired chains, the same
initial latent per pair, and restored sampler RNG state. The table reports the
oracle-best chain for each method.""",
                    "metrics-table",
                ),
                _markdown_cell(
                    f"![Metric summary]({rel['metrics']})",
                    "metrics-figure",
                ),
                _markdown_cell(
                    f"""## What the real density looks like

The crop is a 16 Å cube at 0.25 Å resolution, aligned to FuncBind's centered
pocket frame.

![Density slices]({rel['slices']})""",
                    "density-slices",
                ),
                _markdown_cell(
                    f"""## 3D occupancy comparison

The first panel plots the real holo-density points at ≥1.5 local σ with the
reference atoms. The remaining panels use the same element colors and an
occupancy threshold of 0.1.

![3D occupancy comparison]({rel['occupancy']})""",
                    "occupancy-figure",
                ),
                _markdown_cell(
                    f"""## Paired diffusion behavior

![Paired latent MSE]({rel['latent']})""",
                    "latent-figure",
                ),
                _markdown_cell(
                    f"""## Adapter fit

Only the {report['adapter']['trainable_parameters']:,}-parameter density
adapter was trained; the two FuncBind checkpoints and the VoxBind density
encoder were frozen.

![Adapter training]({rel['training']})""",
                    "training-figure",
                ),
                _code_cell(
                    """def load_cloud(path):
    with np.load(path) as data:
        return {key: data[key] for key in data.files}

clouds = {
    'Reference': load_cloud(
        RESULT_DIR / 'reference_occupancy_points.npz'
    ),
    'Original FuncBind': load_cloud(
        RESULT_DIR / 'original_funcbind' / 'occupancy_points.npz'
    ),
    'Holo-density FuncBind': load_cloud(
        RESULT_DIR / 'holo_density_funcbind' / 'occupancy_points.npz'
    ),
    'Real holo density': load_cloud(
        RESULT_DIR / 'density_points.npz'
    ),
}
{
    name: cloud['coordinates'].shape[0]
    for name, cloud in clouds.items()
}""",
                    "load-clouds",
                ),
                _markdown_cell(
                    """## What this result does—and does not—show

- It directly tests density-side conditioning while leaving the original
  ligand INR decoder intact.
- The unknown ligand atom channels supplied to the density encoder are zero;
  the ligand signal comes from the real holo map.
- A better density-conditioned curve or occupancy fit is evidence that this
  representation can steer FuncBind on this target.
- It cannot distinguish density information from single-target adapter
  memorization. The next valid experiment is to train one adapter on training
  complexes with density and evaluate it, without target fitting or oracle
  selection, on held-out maps including apo maps.

**That control has since been run — see `apo_density_funcbind_report.ipynb`.**
Repeating this experiment on the same map with the ligand erased reproduces the
improvement below (mIoU 0.525 vs 0.571, F1 0.957 vs 0.952), so the gain shown
here comes from fitting the adapter to this one target, not from reading the
ligand out of the density.""",
                    "limitations",
                ),
            ]
        )

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (funcbind .repro-env)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.11",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    NOTEBOOK_PATH.write_text(
        json.dumps(notebook, indent=1) + "\n", encoding="utf-8"
    )
    return NOTEBOOK_PATH


if __name__ == "__main__":
    report = None
    if RESULT_PATH.is_file():
        report = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
        if report.get("status") != "complete":
            report = None
    output = build_notebook(report)
    print(output)

