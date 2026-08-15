"""Build figures and a notebook for a density-fusion run with any number of arms.

`build_holo_density_report.py` is the frozen two-arm builder for the original
holo run. This one takes `--run-dir` and discovers the arms from `result.json`,
so it also covers the apo (ligand-erased) control and the transfer arm, and it
can put a second run side by side with `--compare-dir`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


NOTEBOOK_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = NOTEBOOK_DIR.parents[1]
FIGURE_DIR = NOTEBOOK_DIR / "figures"

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
ARM_COLORS = {
    "original_funcbind": "#64748b",
    "holo_density_funcbind": "#7c3aed",
    "apo_density_funcbind": "#0f766e",
    "transfer_adapter_funcbind": "#b45309",
}
LEGACY_LABELS = {
    "original_funcbind": "Original FuncBind",
    "holo_density_funcbind": "Holo-density adapter",
}


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def arms_of(report: dict) -> list[tuple[str, str]]:
    """(key, label) per arm, tolerating the original two-arm result.json."""
    if "arms" in report:
        return [(arm["key"], arm["label"]) for arm in report["arms"]]
    return [
        (key, LEGACY_LABELS.get(key, key))
        for key in report["results"]
        if key in LEGACY_LABELS
    ]


def latent_mse_of(report: dict, key: str) -> np.ndarray:
    sampling = report["sampling"]
    if "latent_mse" in sampling:
        return np.asarray(sampling["latent_mse"][key], dtype=float)
    legacy = {
        "original_funcbind": "original_latent_mse",
        "holo_density_funcbind": "holo_density_latent_mse",
    }
    return np.asarray(sampling[legacy[key]], dtype=float)


def _color(key: str) -> str:
    return ARM_COLORS.get(key, "#334155")


def render_training(run_dir: Path, prefix: str) -> Path:
    history = json.loads(
        (run_dir / "training_history.json").read_text(encoding="utf-8")
    )
    steps = [row["step"] for row in history]
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), constrained_layout=True)
    axes[0].plot(
        steps, [row["denoise_loss"] for row in history], color="#7c3aed", linewidth=2
    )
    axes[0].set(
        xlabel="Adapter step",
        ylabel="EDM2 denoising objective",
        title="Single-target adapter fit",
        yscale="log",
    )
    axes[1].plot(
        steps, [row["residual_rms"] for row in history], color="#0f766e", linewidth=2
    )
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
    path = FIGURE_DIR / f"{prefix}_adapter_training.png"
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return path


def render_latent_comparison(report: dict, prefix: str) -> Path:
    arms = arms_of(report)
    values = {key: latent_mse_of(report, key) for key, _ in arms}
    index = np.arange(len(values[arms[0][0]]))
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 4.2), constrained_layout=True)

    width = 0.8 / len(arms)
    for position, (key, label) in enumerate(arms):
        offset = (position - (len(arms) - 1) / 2) * width
        axes[0].bar(
            index + offset, values[key], width, color=_color(key), label=label
        )
    axes[0].set(
        xlabel="Paired diffusion chain",
        ylabel="Latent MSE to reference code",
        title="Same initial latent and sampler RNG",
    )
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].grid(axis="y", alpha=0.22)

    baseline = values["original_funcbind"]
    conditioned = [(key, label) for key, label in arms if key != "original_funcbind"]
    width = 0.8 / max(len(conditioned), 1)
    for position, (key, label) in enumerate(conditioned):
        offset = (position - (len(conditioned) - 1) / 2) * width
        axes[1].bar(
            index + offset,
            values[key] - baseline,
            width,
            color=_color(key),
            label=label,
        )
    axes[1].axhline(0, color="#111827", linewidth=1)
    axes[1].set(
        xlabel="Paired diffusion chain",
        ylabel="Conditioned − original latent MSE",
        title="Negative means conditioning helped",
    )
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].grid(axis="y", alpha=0.22)
    path = FIGURE_DIR / f"{prefix}_paired_latent_mse.png"
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return path


def render_metric_summary(report: dict, prefix: str, subtitle: str) -> Path:
    arms = arms_of(report)
    results = report["results"]
    labels = [label for _, label in arms]
    colors = [_color(key) for key, _ in arms]
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)

    mse = [results[key]["reconstruction"]["density_mse"] for key, _ in arms]
    bars = axes[0].bar(labels, mse, color=colors)
    axes[0].bar_label(bars, labels=[f"{value:.2e}" for value in mse], fontsize=8)
    axes[0].set(yscale="log", ylabel="Density MSE", title="Full 128³ occupancy error")

    miou = [results[key]["reconstruction"]["miou"] for key, _ in arms]
    bars = axes[1].bar(labels, miou, color=colors)
    axes[1].bar_label(bars, labels=[f"{value:.3f}" for value in miou], fontsize=8)
    axes[1].set(ylim=(0, max(0.05, max(miou) * 1.25)), title="Occupancy mIoU")

    metric_keys = (
        ("atom_precision", "Precision"),
        ("atom_recall", "Recall"),
        ("atom_f1", "F1"),
    )
    x = np.arange(len(metric_keys))
    width = 0.8 / len(arms)
    for position, (key, label) in enumerate(arms):
        offset = (position - (len(arms) - 1) / 2) * width
        axes[2].bar(
            x + offset,
            [results[key]["atom_recovery"][name] for name, _ in metric_keys],
            width,
            color=_color(key),
            label=label,
        )
    axes[2].set_xticks(x, [label for _, label in metric_keys])
    axes[2].set(ylim=(0, 1.05), ylabel="Score", title="Atom recovery ≤1 Å")
    axes[2].legend(frameon=False, fontsize=8)
    for axis in axes:
        axis.tick_params(axis="x", rotation=14, labelsize=8)
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle(subtitle, fontsize=14, fontweight="bold")
    path = FIGURE_DIR / f"{prefix}_metric_summary.png"
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return path


def _subsample_occupancy(cloud, max_per_channel: int = 3500):
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


def _subsample_density(cloud, max_points: int = 24000):
    if cloud["coordinates"].shape[0] <= max_points:
        return cloud
    rng = np.random.default_rng(1269)
    weights = np.maximum(cloud["values"].astype(np.float32), 0)
    weights = weights / weights.sum()
    keep = rng.choice(
        cloud["coordinates"].shape[0], max_points, replace=False, p=weights
    )
    return {
        "coordinates": cloud["coordinates"][keep],
        "values": cloud["values"][keep].astype(np.float32),
    }


def _plot_atoms(axis, atoms) -> None:
    for channel, (element, color) in enumerate(zip(ELEMENTS, ELEMENT_COLORS)):
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


def _plot_occupancy(axis, cloud) -> None:
    shown = _subsample_occupancy(cloud)
    for channel, color in enumerate(ELEMENT_COLORS):
        mask = shown["channels"] == channel
        if not np.any(mask):
            continue
        xyz = shown["coordinates"][mask]
        axis.scatter(
            xyz[:, 0],
            xyz[:, 1],
            xyz[:, 2],
            s=1.5 + 7 * shown["occupancies"][mask],
            color=color,
            alpha=0.28,
            linewidths=0,
            depthshade=False,
        )


def render_occupancy_3d(
    report: dict, run_dir: Path, prefix: str, density_title: str, subtitle: str
) -> Path:
    reference_atoms = _load_npz(run_dir / "reference_atoms.npz")
    panels = [
        (
            density_title,
            "density",
            _subsample_density(_load_npz(run_dir / "density_points.npz")),
        ),
        (
            "Reference ligand occupancy",
            "occupancy",
            _load_npz(run_dir / "reference_occupancy_points.npz"),
        ),
    ]
    for key, label in arms_of(report):
        panels.append(
            (label, "occupancy", _load_npz(run_dir / key / "occupancy_points.npz"))
        )

    figure = plt.figure(figsize=(4.5 * len(panels), 5.2), constrained_layout=True)
    for panel_index, (title, kind, data) in enumerate(panels, start=1):
        axis = figure.add_subplot(1, len(panels), panel_index, projection="3d")
        if kind == "density":
            xyz = data["coordinates"]
            axis.scatter(
                xyz[:, 0],
                xyz[:, 1],
                xyz[:, 2],
                c=data["values"].astype(np.float32),
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
        axis.set_title(title, fontsize=10)
        axis.grid(alpha=0.18)
        if panel_index == 1:
            axis.legend(frameon=False, fontsize=7, loc="upper left")
    figure.suptitle(subtitle, fontsize=15, fontweight="bold")
    path = FIGURE_DIR / f"{prefix}_occupancy_3d.png"
    figure.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(figure)
    return path


def render_density_slices(run_dir: Path, prefix: str, title: str) -> Path:
    """Holo vs mask vs apo when the run masked, otherwise the fed map alone."""
    mask_path = run_dir / "apo_mask.npz"
    if mask_path.is_file():
        bundle = _load_npz(mask_path)
        volumes = [
            (
                "Measured 2Fo-Fc (ligand present)",
                bundle["holo_density"].astype(np.float32),
                "coolwarm",
            ),
            ("Ligand occupancy mask", bundle["mask"].astype(np.float32), "magma"),
            (
                "Apo density, ligand masked out (fed to encoder)",
                bundle["apo_density"].astype(np.float32),
                "coolwarm",
            ),
        ]
    else:
        density = _load_npz(run_dir / "density_grid.npz")["density"].astype(
            np.float32
        )
        volumes = [("2Fo-Fc map (fed to encoder)", density, "coolwarm")]

    reference = volumes[0][1]
    vmax = float(np.quantile(np.abs(reference), 0.995))
    mid = reference.shape[0] // 2
    views = (
        ("x = 0 Å", lambda v: v[mid]),
        ("y = 0 Å", lambda v: v[:, mid]),
        ("z = 0 Å", lambda v: v[:, :, mid]),
    )
    figure, axes = plt.subplots(
        len(volumes),
        3,
        figsize=(12, 3.9 * len(volumes)),
        constrained_layout=True,
        squeeze=False,
    )
    for row, (name, volume, cmap) in enumerate(volumes):
        for column, (view_name, take) in enumerate(views):
            axis = axes[row][column]
            limits = (
                dict(vmin=0, vmax=1) if cmap == "magma" else dict(vmin=-vmax, vmax=vmax)
            )
            shown = axis.imshow(
                take(volume).T,
                origin="lower",
                cmap=cmap,
                extent=(-8, 8, -8, 8),
                **limits,
            )
            axis.set(xlabel="Å", ylabel="Å")
            axis.set_title(f"{name}\n{view_name}", fontsize=10)
            figure.colorbar(shown, ax=axis, shrink=0.8)
    figure.suptitle(title, fontsize=14, fontweight="bold")
    path = FIGURE_DIR / f"{prefix}_slices.png"
    figure.savefig(path, dpi=175, bbox_inches="tight")
    plt.close(figure)
    return path


def markdown_table(report: dict, compare: dict | None = None) -> str:
    rows = []

    def add(label: str, result: dict) -> None:
        rec = result["reconstruction"]
        atoms = result["atom_recovery"]
        rows.append(
            f"| {label} | {rec['density_mse']:.3e} | {rec['miou']:.4f} | "
            f"{atoms['n_predicted_atoms']} | {atoms['matched_atoms']} | "
            f"{atoms['atom_precision']:.3f} | {atoms['atom_recall']:.3f} | "
            f"{atoms['atom_f1']:.3f} |"
        )

    if compare is not None:
        for key, label in arms_of(compare):
            if key == "original_funcbind":
                continue
            add(f"{label} (holo run)", compare["results"][key])
    for key, label in arms_of(report):
        add(label, report["results"][key])
    return "\n".join(
        [
            "| Method | Occupancy MSE | mIoU | Pred. atoms | Matched | Precision | Recall | F1 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
            *rows,
        ]
    )


def _markdown_cell(source: str, cell_id: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": cell_id,
        "metadata": {},
        "source": [line + "\n" for line in source.rstrip().splitlines()],
    }


def _code_cell(source: str, cell_id: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.rstrip().splitlines()],
    }


def build_notebook(
    report: dict,
    run_dir: Path,
    notebook_path: Path,
    prefix: str,
    compare: dict | None,
) -> Path:
    masked = report.get("apo_mask", {}).get("enabled", False)
    kind = "apo (ligand-erased)" if masked else "holo"
    density_panel = (
        "Apo density (ligand masked out) + reference atoms"
        if masked
        else "Map fed to the encoder + reference atoms"
    )
    slices_title = (
        "Apo density input: the ligand density is masked out of the measured map"
        if masked
        else "Density input fed to the encoder"
    )
    figures = {
        "training": render_training(run_dir, prefix),
        "latent": render_latent_comparison(report, prefix),
        "metrics": render_metric_summary(
            report,
            prefix,
            f"Oracle-best of {report['sampling']['n_chains']} paired chains on "
            f"CrossDocked target {report['target']['crossdocked_test_index']}",
        ),
        "occupancy": render_occupancy_3d(
            report,
            run_dir,
            prefix,
            density_panel,
            f"{report['target']['pdb_id'].upper()} / "
            f"{report['target']['ligand_id'].upper()}: {kind} density and "
            "generated ligand occupancies",
        ),
        "slices": render_density_slices(run_dir, prefix, slices_title),
    }
    rel = {
        key: path.relative_to(NOTEBOOK_DIR).as_posix() for key, path in figures.items()
    }

    if masked:
        mask = report["apo_mask"]
        density_note = f"""> **This run uses apo density: the ligand density is masked out.** The map is
> the measured 2Fo-Fc map of the complex with the ligand's own density removed,
> so the encoder sees the pocket as if nothing were bound. The mask is
> FuncBind's ligand occupancy field (atom radii ×{mask['radius_scale']}) and the
> vacated voxels are refilled with the crop's bulk-solvent level
> ({mask['fill_value']:+.3f}) rather than zeroed — zero is not solvent in this
> arcsinh z-scored map. No separately measured apo crystal structure is
> involved."""
        headline = f"""# Apo-density control for FuncBind density fusion

The holo run showed that a residual derived from the real 2Fo-Fc map of
**{report['target']['pdb_id'].upper()} / {report['target']['ligand_id'].upper()}**
improves a frozen FuncBind. That map contains the crystallographic ligand, so
the improvement could have come from copying the ligand out of the density.

This run repeats the experiment on the same map with the **ligand erased**: the
FuncBind ligand occupancy field (atom radii scaled by
{mask['radius_scale']}) is used as a soft mask and the vacated voxels are
refilled with the crop's bulk-solvent level ({mask['fill_value']:+.3f}).
Mean density at the ligand atom centres drops from
{mask['ligand_atom_density_before']:.2f} to
{mask['ligand_atom_density_after']:.2f}, and the fraction of voxels within
{mask['envelope_radius']} Å of a ligand atom above {mask['leakage_sigma']}σ
drops from {mask['envelope_above_sigma_before']:.3f} to
{mask['envelope_above_sigma_after']:.3f}. Everything outside the mask is
bit-identical to the holo crop.

> **How to read this.** The adapter still fits this one target, so a good apo
> score is *not* evidence that density helped — it is evidence that the adapter
> can reach the target without any ligand density, i.e. that the holo result was
> memorization. The transfer arm makes the same point without retraining: it
> replays the holo-trained adapter on this ligand-free map."""
    else:
        headline = f"""# Density fusion into pretrained FuncBind

Frozen FuncBind conditioned by a residual derived from an aligned real 2Fo-Fc
map of **{report['target']['pdb_id'].upper()} /
{report['target']['ligand_id'].upper()}**, compared against the same model with
no density."""

    try:
        relative = run_dir.relative_to(PROJECT_ROOT).as_posix()
        locate = f"""search_roots = [Path.cwd(), *Path.cwd().parents]
PROJECT_ROOT = next(
    root for root in search_roots
    if (root / 'exps' / 'density_fusion').exists()
)
RESULT_DIR = PROJECT_ROOT / '{relative}'"""
    except ValueError:
        # Run stored outside the repo; fall back to an absolute path.
        locate = f"RESULT_DIR = Path(r'{run_dir.as_posix()}')"

    cells = [
        _markdown_cell(headline, "scope"),
        _code_cell(
            f"""import json
from pathlib import Path

import numpy as np

{locate}
with (RESULT_DIR / 'result.json').open() as handle:
    report = json.load(handle)
report['status'], report['experiment']""",
            "load-results",
        ),
        _markdown_cell(
            f"""## Quantitative result

{markdown_table(report, compare)}

{report['sampling']['n_chains']} paired chains share one initial latent and a
restored sampler RNG state per arm; the table reports the oracle-best chain of
each arm.""",
            "metrics-table",
        ),
        _markdown_cell(f"![Metric summary]({rel['metrics']})", "metrics-figure"),
        _markdown_cell(
            f"""## The density input

{density_note}

The crop is a 16 Å cube at 0.25 Å resolution aligned to FuncBind's centered
pocket frame.

![Density slices]({rel['slices']})""",
            "density-slices",
        ),
        _markdown_cell(
            f"""## 3D occupancy comparison

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
adapter was trained; the FuncBind checkpoints, the INR decoder and the VoxBind
density encoder were frozen.

![Adapter training]({rel['training']})""",
            "training-figure",
        ),
        _markdown_cell(
            "## Limitations\n\n"
            + "\n".join(f"- {item}" for item in report["limitations"]),
            "limitations",
        ),
    ]

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (funcbind .repro-env)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    notebook_path.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
    return notebook_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=PROJECT_ROOT / "exps/density_fusion/apo_5mgl_target69_20260730",
    )
    parser.add_argument(
        "--compare-dir",
        type=Path,
        default=None,
        help="a second run whose conditioned arms are added to the table",
    )
    parser.add_argument("--prefix", default=None)
    parser.add_argument("--notebook", type=Path, default=None)
    arguments = parser.parse_args()

    run_dir = arguments.run_dir.resolve()
    report = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    if report.get("status") != "complete":
        raise SystemExit(f"{run_dir}/result.json is not a completed run")
    compare = None
    if arguments.compare_dir is not None:
        compare = json.loads(
            (arguments.compare_dir.resolve() / "result.json").read_text(
                encoding="utf-8"
            )
        )

    masked = report.get("apo_mask", {}).get("enabled", False)
    prefix = arguments.prefix or ("apo_density" if masked else "holo_density_v2")
    notebook = arguments.notebook or (
        NOTEBOOK_DIR / f"{prefix}_funcbind_report.ipynb"
    )
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    print(build_notebook(report, run_dir, notebook, prefix, compare))


if __name__ == "__main__":
    main()
