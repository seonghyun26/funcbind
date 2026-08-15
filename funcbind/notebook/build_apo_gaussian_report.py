"""Build the notebook for the ten-target apo-density Gaussian-splat run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


NOTEBOOK_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = NOTEBOOK_DIR.parents[1]
FIGURE_DIR = NOTEBOOK_DIR / "figures"
PREFIX = "apo_gaussian"
ARM_COLORS = {
    "original_funcbind": "#64748b",
    "apo_density_funcbind": "#0f766e",
}

# Same-target decoder ablation: the only place the pretrained INR decoder and a
# Gaussian-splat decoder were measured on identical inputs.
ABLATION_DIR = PROJECT_ROOT / "exps/decoder_ablation"
ORIGINAL_INR_RESULT = (
    ABLATION_DIR / "original_pretrained_inr_mcpp_target0_20260728/result.json"
)
SINGLE_TARGET_GAUSSIAN_RESULT = (
    ABLATION_DIR / "gaussian_max_overfit_mcpp_target0_20260728/result.json"
)
# Counted from dec_state_dict / enc_state_dict in nf_unified/model.pt; matches the
# decoder_parameters and encoder_parameters recorded by the joint-overfit ablation.
INR_DECODER_PARAMETERS = 62_258_696
ENCODER_PARAMETERS = 58_921_152


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def _arms(report: dict) -> list[tuple[str, str]]:
    return [(arm["key"], arm["label"]) for arm in report["arms"]]


def render_per_target(report: dict) -> Path:
    """Per-target paired metrics: the spread matters more than the mean."""
    arms = _arms(report)
    targets = [str(index) for index in report["targets"]]
    x = np.arange(len(targets))
    width = 0.8 / len(arms)
    panels = (
        ("miou", "Occupancy mIoU", None),
        ("atom_f1", "Atom recovery F1 (≤1 Å)", None),
        ("density_mse", "Density MSE", "log"),
    )
    figure, axes = plt.subplots(
        len(panels), 1, figsize=(12, 11), constrained_layout=True
    )
    for axis, (metric, title, scale) in zip(axes, panels):
        for position, (key, label) in enumerate(arms):
            values = report["aggregate"][metric][key]["values"]
            axis.bar(
                x + (position - (len(arms) - 1) / 2) * width,
                values,
                width,
                color=ARM_COLORS.get(key, "#334155"),
                label=label,
            )
        axis.set_xticks(x, targets)
        axis.set(xlabel="CrossDocked test target", title=title)
        if scale:
            axis.set_yscale(scale)
        axis.legend(frameon=False, fontsize=9)
        axis.grid(axis="y", alpha=0.22)
    figure.suptitle(
        "Ten targets, one adapter, one Gaussian-splat decoder — per-target results",
        fontsize=14,
        fontweight="bold",
    )
    path = FIGURE_DIR / f"{PREFIX}_per_target.png"
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return path


def render_paired_deltas(report: dict) -> Path:
    """Apo-adapter minus baseline, per target — the sign is the result."""
    key = "apo_density_funcbind"
    deltas = np.asarray(report["paired"][key]["latent_mse_delta"], dtype=float)
    targets = [str(index) for index in report["targets"]]
    miou = report["aggregate"]["miou"]
    miou_delta = np.asarray(miou[key]["values"], dtype=float) - np.asarray(
        miou["original_funcbind"]["values"], dtype=float
    )

    figure, axes = plt.subplots(1, 2, figsize=(12.5, 4.4), constrained_layout=True)
    for axis, values, title, ylabel in (
        (
            axes[0],
            deltas,
            "Latent MSE change (negative = density helped)",
            "Apo adapter − baseline",
        ),
        (
            axes[1],
            miou_delta,
            "Occupancy mIoU change (positive = density helped)",
            "Apo adapter − baseline",
        ),
    ):
        colors = np.where(
            values < 0 if axis is axes[0] else values > 0, "#0f766e", "#dc2626"
        )
        axis.bar(np.arange(len(values)), values, color=colors)
        axis.axhline(0, color="#111827", linewidth=1)
        axis.set_xticks(np.arange(len(values)), targets)
        axis.set(xlabel="CrossDocked test target", ylabel=ylabel, title=title)
        axis.grid(axis="y", alpha=0.22)
    figure.suptitle(
        "Paired per-target effect of apo-density conditioning",
        fontsize=14,
        fontweight="bold",
    )
    path = FIGURE_DIR / f"{PREFIX}_paired_deltas.png"
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return path


def render_fits(run_dir: Path) -> Path:
    adapter = json.loads((run_dir / "adapter_history.json").read_text())
    decoder = json.loads((run_dir / "decoder_history.json").read_text())
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True)
    axes[0].plot(
        [row["step"] for row in adapter],
        [row["denoise_loss"] for row in adapter],
        color="#0f766e",
        linewidth=1.5,
    )
    axes[0].set(
        xlabel="Step",
        ylabel="EDM2 denoising objective",
        title="One density adapter across ten targets",
    )
    axes[1].plot(
        [row["step"] for row in decoder],
        [row["validation_loss"] for row in decoder],
        color="#7c3aed",
        linewidth=2,
        label="mean over 10 targets",
    )
    axes[1].plot(
        [row["step"] for row in decoder],
        [row["validation_worst"] for row in decoder],
        color="#b45309",
        linewidth=1.2,
        linestyle="--",
        label="worst target",
    )
    axes[1].set(
        xlabel="Step",
        ylabel="Validation loss",
        title="One Gaussian-splat decoder across ten targets",
        yscale="log",
    )
    axes[1].legend(frameon=False, fontsize=9)
    for axis in axes:
        axis.grid(alpha=0.22)
    path = FIGURE_DIR / f"{PREFIX}_fits.png"
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return path


def render_masks(report: dict, run_dir: Path) -> Path:
    """One central slice per target: measured map above, apo map below."""
    targets = report["targets"]
    figure, axes = plt.subplots(
        2, len(targets), figsize=(2.05 * len(targets), 4.6), constrained_layout=True
    )
    for column, index in enumerate(targets):
        bundle = _load_npz(run_dir / f"target_{index:03d}" / "apo_mask.npz")
        holo = bundle["holo_density"].astype(np.float32)
        apo = bundle["apo_density"].astype(np.float32)
        mid = holo.shape[0] // 2
        vmax = float(np.quantile(np.abs(holo), 0.995))
        for row, volume in enumerate((holo, apo)):
            axis = axes[row][column]
            axis.imshow(
                volume[mid].T,
                origin="lower",
                cmap="coolwarm",
                vmin=-vmax,
                vmax=vmax,
                extent=(-8, 8, -8, 8),
            )
            axis.set_xticks([])
            axis.set_yticks([])
            if row == 0:
                axis.set_title(f"target {index}", fontsize=9)
            if column == 0:
                axis.set_ylabel(
                    "measured" if row == 0 else "apo (masked)", fontsize=9
                )
    figure.suptitle(
        "Ligand density masked out of every target (x = 0 Å slice)",
        fontsize=14,
        fontweight="bold",
    )
    path = FIGURE_DIR / f"{PREFIX}_masks.png"
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return path


def summary_table(report: dict) -> str:
    arms = _arms(report)
    rows = [
        "| Method | mIoU (mean ± sd) | Atom F1 | Density MSE | Targets improved |",
        "|---|---:|---:|---:|---:|",
    ]
    for key, label in arms:
        miou = report["aggregate"]["miou"][key]
        f1 = report["aggregate"]["atom_f1"][key]
        mse = report["aggregate"]["density_mse"][key]
        improved = (
            f"{report['paired'][key]['targets_improved']}/"
            f"{report['paired'][key]['n_targets']}"
            if key in report.get("paired", {})
            else "—"
        )
        rows.append(
            f"| {label} | {miou['mean']:.3f} ± {miou['std']:.3f} | "
            f"{f1['mean']:.3f} ± {f1['std']:.3f} | {mse['mean']:.3e} | {improved} |"
        )
    return "\n".join(rows)


def per_target_table(report: dict) -> str:
    arms = _arms(report)
    header = "| Target | " + " | ".join(
        f"{label} mIoU / F1" for _, label in arms
    ) + " |"
    rows = [header, "|---" * (len(arms) + 1) + "|"]
    for position, index in enumerate(report["targets"]):
        cells = []
        for key, _ in arms:
            miou = report["aggregate"]["miou"][key]["values"][position]
            f1 = report["aggregate"]["atom_f1"][key]["values"][position]
            cells.append(f"{miou:.3f} / {f1:.3f}")
        rows.append(f"| {index} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


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
GRID_DIM = 128
RESOLUTION = 0.25
LIGAND_RADIUS = 1.0


def _reference_occupancy(
    atoms: dict[str, np.ndarray], threshold: float = 0.1
) -> dict[str, np.ndarray]:
    """FuncBind's reference ligand occupancy, recomputed on the 128³ grid.

    The multi-target run does not dump this cloud, so it is rebuilt here with
    the same formula the dataset uses, 1 - prod(1 - exp(-(d / 0.93r)^2)),
    evaluated only near the ligand so the full 2M-point grid is never held.
    """
    coordinates = atoms["coordinates"].astype(np.float64)
    channels = atoms["channels"].astype(int)
    if coordinates.size == 0:
        return {
            "coordinates": np.zeros((0, 3), np.float32),
            "channels": np.zeros((0,), np.int16),
            "occupancies": np.zeros((0,), np.float32),
        }

    # Occupancy falls below 0.1 beyond ~1.4 A for a lone atom; 2.5 A is safe.
    reach = int(np.ceil(2.5 / RESOLUTION))
    centres = np.rint(coordinates / RESOLUTION + GRID_DIM / 2).astype(int)
    offsets = np.arange(-reach, reach + 1)
    block = np.stack(
        np.meshgrid(offsets, offsets, offsets, indexing="ij"), axis=-1
    ).reshape(-1, 3)
    candidates = np.unique(
        (centres[:, None, :] + block[None, :, :]).reshape(-1, 3), axis=0
    )
    candidates = candidates[
        ((candidates >= 0) & (candidates < GRID_DIM)).all(axis=1)
    ]
    points = (candidates - GRID_DIM / 2) * RESOLUTION

    out_coordinates, out_channels, out_values = [], [], []
    for channel in np.unique(channels):
        members = coordinates[channels == channel]
        distance = np.linalg.norm(
            points[:, None, :] - members[None, :, :], axis=-1
        )
        exponent = np.clip((distance / (LIGAND_RADIUS * 0.93)) ** 2, None, 10.0)
        log_term = np.where(exponent < 10.0, np.log1p(-np.exp(-exponent)), 0.0)
        occupancy = 1.0 - np.exp(log_term.sum(axis=1))
        keep = occupancy >= threshold
        if not keep.any():
            continue
        out_coordinates.append(points[keep])
        out_channels.append(np.full(int(keep.sum()), channel))
        out_values.append(occupancy[keep])

    if not out_coordinates:
        return {
            "coordinates": np.zeros((0, 3), np.float32),
            "channels": np.zeros((0,), np.int16),
            "occupancies": np.zeros((0,), np.float32),
        }
    return {
        "coordinates": np.concatenate(out_coordinates).astype(np.float32),
        "channels": np.concatenate(out_channels).astype(np.int16),
        "occupancies": np.concatenate(out_values).astype(np.float32),
    }


def _subsample_density(cloud: dict[str, np.ndarray], max_points: int = 16000):
    """Thin the density cloud, keeping strong points preferentially."""
    if cloud["coordinates"].shape[0] <= max_points:
        return cloud
    rng = np.random.default_rng(1269)
    weights = np.maximum(cloud["values"].astype(np.float32), 0)
    total = weights.sum()
    probabilities = weights / total if total > 0 else None
    keep = rng.choice(
        cloud["coordinates"].shape[0], max_points, replace=False, p=probabilities
    )
    return {
        "coordinates": cloud["coordinates"][keep],
        "values": cloud["values"][keep].astype(np.float32),
    }


def _plot_atoms(axis, atoms: dict[str, np.ndarray], legend: bool) -> None:
    for channel, (element, color) in enumerate(zip(ELEMENTS, ELEMENT_COLORS)):
        mask = atoms["channels"] == channel
        if not np.any(mask):
            continue
        xyz = atoms["coordinates"][mask]
        axis.scatter(
            xyz[:, 0],
            xyz[:, 1],
            xyz[:, 2],
            s=42,
            color=color,
            edgecolors="white",
            linewidths=0.4,
            alpha=0.95,
            label=element if legend else None,
            depthshade=False,
        )


def _plot_occupancy(axis, cloud: dict[str, np.ndarray], max_per_channel=3000):
    rng = np.random.default_rng(1269)
    for channel, color in enumerate(ELEMENT_COLORS):
        indices = np.flatnonzero(cloud["channels"] == channel)
        if indices.size == 0:
            continue
        if indices.size > max_per_channel:
            indices = rng.choice(indices, max_per_channel, replace=False)
        xyz = cloud["coordinates"][indices]
        axis.scatter(
            xyz[:, 0],
            xyz[:, 1],
            xyz[:, 2],
            s=1.5 + 7 * cloud["occupancies"][indices].astype(np.float32),
            color=color,
            alpha=0.26,
            linewidths=0,
            depthshade=False,
        )


def render_occupancy_3d(report: dict, run_dir: Path, per_figure: int = 5):
    """One row per target: apo density, reference, and both arms."""
    arms = _arms(report)
    columns = 2 + len(arms)
    paths = []
    targets = report["targets"]
    for start in range(0, len(targets), per_figure):
        chunk = targets[start : start + per_figure]
        figure = plt.figure(
            figsize=(4.5 * columns, 4.9 * len(chunk)), constrained_layout=True
        )
        for row, index in enumerate(chunk):
            target_dir = run_dir / f"target_{index:03d}"
            atoms = _load_npz(target_dir / "reference_atoms.npz")
            density = _load_npz(target_dir / "density_points.npz")
            panels = [
                ("Apo density (ligand masked) + reference atoms", "density", density),
                ("Reference ligand occupancy", "occupancy", _reference_occupancy(atoms)),
            ]
            for key, label in arms:
                panels.append(
                    (label, "occupancy", _load_npz(target_dir / key / "occupancy_points.npz"))
                )
            for column, (title, kind, data) in enumerate(panels):
                axis = figure.add_subplot(
                    len(chunk), columns, row * columns + column + 1, projection="3d"
                )
                if kind == "density":
                    shown = _subsample_density(data)
                    xyz = shown["coordinates"]
                    values = shown["values"].astype(np.float32)
                    # Anchor the ramp above the export threshold so the weakest
                    # points are not rendered white against a white panel.
                    axis.scatter(
                        xyz[:, 0],
                        xyz[:, 1],
                        xyz[:, 2],
                        c=values,
                        cmap="Purples",
                        vmin=float(np.quantile(values, 0.05)),
                        vmax=float(np.quantile(values, 0.99)),
                        s=3.0,
                        alpha=0.28,
                        linewidths=0,
                        depthshade=False,
                    )
                    _plot_atoms(axis, atoms, legend=(row == 0))
                else:
                    _plot_occupancy(axis, data)
                    _plot_atoms(axis, atoms, legend=False)
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
                axis.set_title(
                    f"target {index} — {title}" if column == 0 else title, fontsize=9
                )
                axis.grid(alpha=0.18)
                if row == 0 and column == 0:
                    axis.legend(frameon=False, fontsize=7, loc="upper left")
        figure.suptitle(
            "Apo density and generated ligand occupancies — targets "
            + ", ".join(str(index) for index in chunk),
            fontsize=15,
            fontweight="bold",
        )
        path = FIGURE_DIR / f"{PREFIX}_occupancy_3d_{start // per_figure + 1}.png"
        figure.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(figure)
        paths.append(path)
    return paths


def mask_table(report: dict) -> str:
    """Per-target mask quality — the spread here qualifies the result."""
    rows = [
        "| Target | Density at ligand atoms (before → after) | Voxels ≥1.5σ "
        "within 2 Å (before → after) |",
        "|---|---:|---:|",
    ]
    for entry in report["per_target"]:
        mask = entry["apo_mask"]
        rows.append(
            f"| {entry['target_index']} | "
            f"{mask['ligand_atom_density_before']:.2f} → "
            f"{mask['ligand_atom_density_after']:.2f} | "
            f"{mask['envelope_above_sigma_before']:.3f} → "
            f"{mask['envelope_above_sigma_after']:.4f} |"
        )
    return "\n".join(rows)


def decoder_reference_section(report: dict, run_dir: Path) -> str | None:
    """The existing FuncBind decoder against the one Gaussian-splat decoder.

    Returns None when the same-target ablation is absent, so the notebook still
    builds from the ten-target run alone.
    """
    if not (
        ORIGINAL_INR_RESULT.exists() and SINGLE_TARGET_GAUSSIAN_RESULT.exists()
    ):
        return None
    inr = json.loads(ORIGINAL_INR_RESULT.read_text(encoding="utf-8"))
    single = json.loads(SINGLE_TARGET_GAUSSIAN_RESULT.read_text(encoding="utf-8"))

    entries = [
        (
            f"Original FuncBind INR · posterior {mode}",
            INR_DECODER_PARAMETERS,
            inr["results"][mode]["reconstruction"],
            inr["results"][mode]["atom_recovery"],
        )
        for mode in ("sample", "mean")
    ]
    entries.append(
        (
            "Gaussian splat · single-target overfit",
            single["decoder_parameters"],
            single["full_grid"]["reconstruction"],
            single["full_grid"]["atom_recovery"],
        )
    )
    quality_rows = [
        "| Decoder | Parameters | Density MSE | Density MAE | mIoU | "
        "Atom P / R / F1 | Coord RMSD (Å) | Element acc. |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, parameters, reconstruction, atoms in entries:
        quality_rows.append(
            f"| {label} | {parameters:,} | "
            f"{reconstruction['density_mse']:.3e} | "
            f"{reconstruction['density_mae']:.3e} | "
            f"{reconstruction['miou']:.3f} | "
            f"{atoms['atom_precision']:.3f} / {atoms['atom_recall']:.3f} / "
            f"{atoms['atom_f1']:.3f} | {atoms['coordinate_rmsd']:.3f} | "
            f"{atoms['element_accuracy']:.3f} |"
        )

    gaussian_parameters = single["decoder_parameters"]
    shrink = INR_DECODER_PARAMETERS / gaussian_parameters
    pipeline_inr = ENCODER_PARAMETERS + INR_DECODER_PARAMETERS
    pipeline_gaussian = ENCODER_PARAMETERS + gaussian_parameters
    gaussian_mse = single["full_grid"]["reconstruction"]["density_mse"]
    mse_ratios = sorted(
        gaussian_mse / inr["results"][mode]["reconstruction"]["density_mse"]
        for mode in ("sample", "mean")
    )
    miou_gap = sorted(
        inr["results"][mode]["reconstruction"]["miou"]
        - single["full_grid"]["reconstruction"]["miou"]
        for mode in ("sample", "mean")
    )
    rmsds = [atoms["coordinate_rmsd"] for _, _, _, atoms in entries]

    history = json.loads(
        (run_dir / "decoder_history.json").read_text(encoding="utf-8")
    )
    decoder = report["decoder"]
    best = next(
        (record for record in history if record["step"] == decoder["best_step"]),
        None,
    )
    worst = f"{best['validation_worst']:.2e}" if best else "—"
    fit_rows = [
        "| Gaussian decoder fit | Targets | Parameters | Steps | Best step | "
        "Best val loss | Worst single target |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| Single target (MCPP target 0) | 1 | "
        f"{gaussian_parameters:,} | {single['steps_completed']:,} | "
        f"{single['best_step']:,} | {single['best_validation_loss']:.2e} | — |",
        f"| Joint, this run ({len(report['targets'])} CrossDocked targets) | "
        f"{len(decoder['fitted_jointly_on'])} | "
        f"{decoder['trainable_parameters']:,} | {decoder['steps']:,} | "
        f"{decoder['best_step']:,} | {decoder['best_validation_loss']:.2e} | "
        f"{worst} |",
    ]

    ablation_target = Path(inr["target_id"][0]).name
    return f"""## The existing FuncBind decoder versus the Gaussian-splat decoder

Neither table below is from this run's ten CrossDocked targets. Both come from
the same-target decoder ablation on **MCPP target 0** (`{ablation_target}`),
which is the only place the pretrained INR decoder and a Gaussian-splat decoder
were measured on identical inputs. They are reported here because this notebook
replaces the INR decoder, and that substitution needs its own reference point.

### Reconstructing one reference code (full 128³ grid)

{chr(10).join(quality_rows)}

The Gaussian decoder is **{shrink:.0f}× smaller** — {gaussian_parameters:,}
parameters against {INR_DECODER_PARAMETERS:,}. The encoder
({ENCODER_PARAMETERS:,} parameters) is shared and frozen in both cases, so the
whole pipeline goes from {pipeline_inr / 1e6:.1f}M to
{pipeline_gaussian / 1e6:.1f}M parameters — the decoder falls from
{100 * INR_DECODER_PARAMETERS / pipeline_inr:.0f}% of the pipeline to
{100 * gaussian_parameters / pipeline_gaussian:.2f}%.

What that costs is density fidelity: the INR decoder reaches
{mse_ratios[0]:.0f}–{mse_ratios[1]:.0f}× lower MSE and about
{miou_gap[0]:.2f} higher mIoU. What it does not cost is atom placement — atom
precision, recall, F1, and element accuracy all saturate at 1.000 for every row,
so they do not separate the decoders at all, and coordinate RMSD spans only
{max(rmsds) - min(rmsds):.3f} Å across the three. The Gaussian decoder's deficit
is in the density field it paints, not in where it puts the atoms.

### What each Gaussian decoder fit cost

{chr(10).join(fit_rows)}

Ten codes cost the same {gaussian_parameters:,}-parameter decoder almost
nothing: {decoder['best_validation_loss']:.2e} jointly against
{single['best_validation_loss']:.2e} on one target, and even its worst single
target ({worst}) stays within a factor of two of the single-target fit. The
single-target run is the easier problem and is the reference, not a baseline to
beat."""


def _locate_result_dir(run_dir: Path) -> str:
    """Notebook-side path resolution, repo-relative when the run lives inside it."""
    try:
        relative = run_dir.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return f"RESULT_DIR = Path(r'{run_dir.as_posix()}')"
    return f"""search_roots = [Path.cwd(), *Path.cwd().parents]
PROJECT_ROOT = next(
    root for root in search_roots
    if (root / 'exps' / 'density_fusion').exists()
)
RESULT_DIR = PROJECT_ROOT / '{relative}'"""


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


def build(report: dict, run_dir: Path, notebook_path: Path) -> Path:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    figures = {
        "per_target": render_per_target(report),
        "deltas": render_paired_deltas(report),
        "fits": render_fits(run_dir),
        "masks": render_masks(report, run_dir),
    }
    rel = {
        key: path.relative_to(NOTEBOOK_DIR).as_posix()
        for key, path in figures.items()
    }
    occupancy_3d = [
        path.relative_to(NOTEBOOK_DIR).as_posix()
        for path in render_occupancy_3d(report, run_dir)
    ]
    leakage = max(
        entry["apo_mask"]["envelope_above_sigma_after"]
        for entry in report["per_target"]
    )
    decoder = report["decoder"]
    adapter = report["adapter"]
    reference_section = decoder_reference_section(report, run_dir)
    single_target_loss = "the single-target fit"
    if reference_section:
        single_target = json.loads(
            SINGLE_TARGET_GAUSSIAN_RESULT.read_text(encoding="utf-8")
        )
        single_target_loss = f"{single_target['best_validation_loss']:.2e}"

    cells = [
        _markdown_cell(
            f"""# Ten-target apo density with a Gaussian-splat decoder

Overfitting on **{len(report['targets'])} CrossDocked test targets**
({', '.join(str(index) for index in report['targets'])}) with **apo density**
and a **Gaussian-splat decoder**.

> **This run uses apo density: the ligand density is masked out.** Each input is
> the measured 2Fo-Fc map of the complex with the ligand's own density removed
> using FuncBind's ligand occupancy field, and the vacated voxels refilled with
> the crop's bulk-solvent level. The encoder sees each pocket as if nothing were
> bound. Across all ten targets, at most {leakage:.4f} of the voxels within 2 Å
> of a ligand atom still exceed 1.5σ. No separately measured apo crystal
> structure is involved.

Two things differ from the earlier single-target run, and both remove a way that
result could have been explained by memorization:

1. **One density adapter across all ten targets**
   ({adapter['trainable_parameters']:,} parameters, {adapter['steps']:,} steps).
   A single-target adapter sees a constant density feature and can reach its one
   reference code by memorizing a constant residual. An adapter shared by ten
   targets cannot.
2. **One Gaussian-splat decoder across all ten targets**
   ({decoder['trainable_parameters']:,} parameters, best step
   {decoder['best_step']:,}). There is no pretrained Gaussian decoder, so it is
   fitted here on the frozen encoder's codes for these ten targets — replacing
   the INR decoder used in the earlier reports.""",
            "scope",
        ),
        _code_cell(
            f"""import json
from pathlib import Path

import numpy as np

{_locate_result_dir(run_dir)}
with (RESULT_DIR / 'result.json').open() as handle:
    report = json.load(handle)
report['experiment'], report['targets']""",
            "load-results",
        ),
        _markdown_cell(
            f"""## Result across the ten targets

{summary_table(report)}

Every target is sampled with {report['sampling']['n_chains']} chains that share
one initial latent and a restored sampler RNG state between the two arms, so the
comparison is paired per target. "Targets improved" counts how many of the ten
had a lower latent MSE with the apo-density adapter than without it.

![Per-target results]({rel['per_target']})""",
            "summary",
        ),
        _markdown_cell(
            f"""## Per-target detail

{per_target_table(report)}

![Paired deltas]({rel['deltas']})""",
            "per-target",
        ),
        _markdown_cell(
            f"""## The apo density inputs

![Masked density per target]({rel['masks']})

{mask_table(report)}

Two caveats this table makes visible, both of which qualify how much any single
target is worth:

- **How much ligand density there was to remove varies widely.** Where the
  "before" value is low the ligand sits in weak density — poorly ordered in the
  crystal, or imperfectly aligned to the pocket crop — so masking changes little
  and that target is weak evidence in either direction.
- **The leakage column is an upper bound, not pure ligand density.** It counts
  every voxel above 1.5σ within 2 Å of a ligand atom, which includes
  neighbouring *receptor* density that is deliberately kept. It therefore reads
  high in tight pockets even when the ligand was fully erased.""",
            "masks",
        ),
        _markdown_cell(
            """## 3D occupancy comparison

One row per target. The first panel shows the **apo density actually fed to the
encoder** — points at ≥1.5 local σ, with the reference ligand atoms drawn on top
so the emptied ligand site is visible. The remaining panels use the same element
colours at an occupancy threshold of 0.1, with the reference atoms repeated in
every panel as the common ground truth.

Both generated arms are decoded by the shared Gaussian-splat decoder.

"""
            + "\n\n".join(
                f"![3D occupancy comparison {position}]({path})"
                for position, path in enumerate(occupancy_3d, start=1)
            ),
            "occupancy-3d",
        ),
        _markdown_cell(
            f"""## The two joint fits

The adapter and the Gaussian decoder each serve all ten targets. The decoder
panel shows the mean validation loss and the worst single target, so a decoder
that fits nine targets and fails one is visible rather than averaged away.

![Joint fits]({rel['fits']})""",
            "fits",
        ),
        _markdown_cell(
            f"""## How to read this

The single-target run had an escape hatch: with one target the density feature
is a constant, so the adapter could reach its one reference code by learning a
constant residual, and a map with the ligand erased scored just as well as the
map with it. That is why the earlier result was read as memorization.

Sharing one adapter across {len(report['targets'])} targets closes that
particular hatch. A constant residual cannot serve ten different reference
codes, so any per-target effect has to come from the only per-target input the
adapter receives — the density feature.

**But this is still an in-sample result, and one specific loophole remains
open.** The adapter was fitted on these same ten targets, and each target's
density feature is effectively a unique fingerprint. A shared adapter can
therefore still memorize *per-target* residuals by keying off that fingerprint —
a lookup table rather than a constant. That is memorization too, and nothing in
this notebook can distinguish it from the adapter genuinely reading pocket
density.

Only a held-out target — one whose density the adapter never saw during
fitting — separates the two. That experiment is the necessary next step, and it
is not what was run here.

One thing this run does establish on its own terms: the Gaussian-splat decoder
holds up. A single {decoder['trainable_parameters']:,}-parameter decoder fitted
across all ten codes reached a validation loss of
{decoder['best_validation_loss']:.2e}, against {single_target_loss} for the same
decoder overfit on a *single* target, and it does so with
{decoder['trainable_parameters'] / INR_DECODER_PARAMETERS:.2%} of the original
INR decoder's parameters. Ten codes cost it almost nothing.

## Limitations

"""
            + "\n".join(f"- {item}" for item in report["limitations"])
            + "\n- The shared adapter was fitted on these same ten targets, so a "
            "per-target density feature can act as a target fingerprint and be "
            "memorized. Held-out targets are required to rule this out.",
            "limitations",
        ),
    ]
    if reference_section:
        cells.insert(-1, _markdown_cell(reference_section, "decoder-reference"))

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
        default=PROJECT_ROOT
        / "exps/density_fusion/apo_gaussian_10targets_20260730",
    )
    parser.add_argument(
        "--notebook",
        type=Path,
        default=NOTEBOOK_DIR / "apo_gaussian_10target_report.ipynb",
    )
    arguments = parser.parse_args()
    run_dir = arguments.run_dir.resolve()
    report = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    if report.get("status") != "complete":
        raise SystemExit(f"{run_dir}/result.json is not a completed run")
    print(build(report, run_dir, arguments.notebook))


if __name__ == "__main__":
    main()
