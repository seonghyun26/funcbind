"""Sample one CrossDocked target with a model-zoo density encoder.

The model-zoo checkpoints are MAE encoders, not complete VoxBind generators.
This script performs an explicit zero-shot encoder swap:

* architecture + WJS weights: a trained density-conditioned VoxBind checkpoint;
* frozen density encoder: the selected model-zoo checkpoint;
* density: the aligned real holo map for the requested CrossDocked target.

No target coordinates are passed to the encoder's ligand atom channels. They
are zeroed by VoxBind's full-voxel density input. The optional apo/leak mask is
disabled because this experiment intentionally tests holo-density generation.
"""

from __future__ import annotations

import os
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf


VOXBIND_PYTHON_ROOT = Path(os.environ.get("VOXBIND_PYTHON_ROOT", "/home1/irteam/VoxBind"))
VOXBIND_ROOT = VOXBIND_PYTHON_ROOT / "voxbind"
if str(VOXBIND_PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(VOXBIND_PYTHON_ROOT))

from voxbind.dataset import create_sampling_dataloader
from voxbind.models import create_model
from voxbind.utils.sampling_utils import sample_molecules
from voxbind.voxelizer import Voxelizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--generator-dir",
        type=Path,
        default=(
            VOXBIND_ROOT
            / "exps"
            / "voxbind_frozenenc_atomblob7_v2p1_sig0.9"
        ),
    )
    parser.add_argument("--encoder-config", type=Path, required=True)
    parser.add_argument("--encoder-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-index", type=int, default=69)
    parser.add_argument("--n-samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1269)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def make_config(args: argparse.Namespace):
    generator_config_path = args.generator_dir / "cfg.yaml"
    generator_checkpoint = args.generator_dir / "checkpoint.pth.tar"
    for required in (
        generator_config_path,
        generator_checkpoint,
        args.encoder_config,
        args.encoder_checkpoint,
    ):
        if not required.is_file():
            raise FileNotFoundError(required)

    config = OmegaConf.load(generator_config_path)
    zoo = OmegaConf.load(args.encoder_config)
    model_zoo = zoo.model

    # Preserve the trained generator/fusion recipe, replacing only the frozen
    # density encoder architecture and weights with the model-zoo entry.
    config.model.with_density = True
    config.model.density_encoder_type = "vit"
    config.model.density_pretrained_path = str(
        args.encoder_checkpoint.resolve()
    )
    config.model.density_freeze = True
    config.model.density_mask_ligand = False
    config.model.density_vit.patch = int(model_zoo.patch_size)
    config.model.density_vit.dim = int(model_zoo.dim)
    config.model.density_vit.depth = int(model_zoo.depth)
    config.model.density_vit.heads = int(model_zoo.heads)
    config.model.density_vit.mlp_ratio = int(model_zoo.mlp_ratio)
    config.model.density_vit.dropout = float(model_zoo.dropout)
    config.model.density_vit.n_in_channels = int(
        model_zoo.n_in_channels
    )
    config.model.density_vit.patch_embed_mode = str(
        model_zoo.patch_embed_mode
    )
    config.model.density_vit.channel_groups = list(
        model_zoo.channel_groups
    )

    # Full-voxel mode derives gradmag internally from the one density channel.
    config.with_gradmag = False
    config.dset.dset_name = "crossdocked_xray"
    config.dset.data_dir = str((VOXBIND_ROOT / "dataset/data").resolve())
    config.dset.crops_dir = str(
        (
            VOXBIND_ROOT
            / "dataset/data/xray_crops_aligned_v5"
        ).resolve()
    )
    config.dset.use_xray = True
    config.dset.normalize = False
    config.dset.ligand_radius = 0.5
    config.dset.pocket_radius = -1
    config.dset.subset_n = None
    config.dset.subset_xray_only = False
    config.dset.subset_val_n = None
    config.num_workers = 0
    config.prefetch_factor = 2
    config.debug = False
    config.seed = int(args.seed)
    config.wjs.split = "test"
    config.wjs.start = int(args.target_index)
    config.wjs.end = int(args.target_index)
    config.wjs.n_targets = 100
    config.wjs.n_samples_per_pocket = int(args.n_samples)
    config.wjs.chain_init = "denovo"
    config.output_dir = str(args.output_dir.resolve())
    config.save_dir = str(args.output_dir.resolve())

    return config, zoo, generator_checkpoint


def clean_state_dict(state_dict: dict[str, torch.Tensor]):
    cleaned = {}
    for key, value in state_dict.items():
        key = key.replace("_orig_mod.", "").replace("module.", "")
        cleaned[key] = value
    return cleaned


def load_generator_except_density(
    model: torch.nn.Module, checkpoint_path: Path
) -> dict:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    state_key = (
        "state_dict_ema"
        if "state_dict_ema" in checkpoint
        else "state_dict"
    )
    full_state = clean_state_dict(checkpoint[state_key])
    generator_state = {
        key: value
        for key, value in full_state.items()
        if not key.startswith("density_encoder.")
    }
    missing, unexpected = model.load_state_dict(
        generator_state, strict=False
    )
    expected_missing = {
        key
        for key in model.state_dict()
        if key.startswith("density_encoder.")
    }
    if set(missing) != expected_missing:
        extra = sorted(set(missing) - expected_missing)
        absent = sorted(expected_missing - set(missing))
        raise RuntimeError(
            "generator swap had unexpected missing keys: "
            f"extra={extra[:10]}, density_not_missing={absent[:10]}"
        )
    if unexpected:
        raise RuntimeError(f"unexpected generator keys: {unexpected[:20]}")
    return {
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "checkpoint_state_key": state_key,
        "loaded_non_density_tensors": len(generator_state),
        "kept_model_zoo_density_tensors": len(expected_missing),
    }


def target_ids(batch: dict) -> tuple[str, str]:
    return str(batch["pocket"]["id"][0]), str(batch["ligand"]["id"][0])


def main() -> None:
    args = parse_args()
    config, zoo_config, generator_checkpoint = make_config(args)
    recipe = {
        "status": "dry_run" if args.dry_run else "starting",
        "experiment": "VoxBind zero-shot model-zoo encoder swap",
        "target_index": int(args.target_index),
        "n_samples_requested": int(args.n_samples),
        "holo_density": True,
        "density_ligand_mask": False,
        "generator": {
            "directory": str(args.generator_dir.resolve()),
            "checkpoint": str(generator_checkpoint.resolve()),
            "smooth_sigma": float(config.smooth_sigma),
            "fusion": str(config.model.get("fusion", "default")),
            "wjs": OmegaConf.to_container(config.wjs, resolve=True),
        },
        "model_zoo": {
            "config": str(args.encoder_config.resolve()),
            "checkpoint": str(args.encoder_checkpoint.resolve()),
            "source_exp_name": str(zoo_config.exp_name),
            "architecture": OmegaConf.to_container(
                zoo_config.model, resolve=True
            ),
            "selection": (
                "pareto-best entry: tied test Spearman, best Pearson and RMSE"
            ),
        },
        "interpretation": (
            "The generator was trained with a compatible earlier frozen "
            "ChannelViT. Replacing it with the Pareto model-zoo encoder is a "
            "zero-shot encoder swap; no downstream retraining was performed."
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output_dir / "run_metadata.json", recipe)
    if args.dry_run:
        print(OmegaConf.to_yaml(config))
        print(json.dumps(recipe, indent=2))
        return
    if not torch.cuda.is_available():
        raise RuntimeError("VoxBind sampling requires CUDA")

    seed_everything(int(args.seed))
    device = torch.device("cuda:0")
    started = time.time()
    model = create_model(config, device=device)
    load_info = load_generator_except_density(model, generator_checkpoint)
    model.eval()

    voxelizer = Voxelizer(
        grid_dim=int(config.vox.grid_dim),
        resolution=float(config.vox.resolution),
        cubes_around=int(config.vox.cubes_around),
        device=device,
    )
    loader = create_sampling_dataloader(config, split="test")
    if not 0 <= args.target_index < len(loader.dataset):
        raise IndexError(
            f"target {args.target_index} outside test set of "
            f"size {len(loader.dataset)}"
        )

    selected = None
    for index, batch in enumerate(loader):
        if index == args.target_index:
            selected = batch
            break
    if selected is None:
        raise RuntimeError(f"could not load target {args.target_index}")
    pocket_id, ligand_id = target_ids(selected)
    if "5mgl" not in pocket_id.lower() or "7mu" not in ligand_id.lower():
        raise RuntimeError(
            "target identity mismatch: "
            f"expected 5MGL/7MU, got {pocket_id} / {ligand_id}"
        )
    available = bool(selected["xray_available"].flatten()[0].item())
    if not available:
        raise RuntimeError("target 69 unexpectedly has no X-ray density")
    density = selected["xray_density"]

    target_dir = args.output_dir / f"target_{args.target_index:02d}"
    n_valid = sample_molecules(
        model,
        selected["pocket"],
        selected["ligand"],
        voxelizer,
        str(target_dir),
        config,
        density=density,
    )
    sample_path = target_dir / "samples.sdf"
    result = {
        **recipe,
        "status": "complete",
        "target": {
            "index": int(args.target_index),
            "pocket_id": pocket_id,
            "ligand_id": ligand_id,
            "xray_available": available,
        },
        "load": load_info,
        "n_valid_molecules": int(n_valid),
        "samples_sdf": str(sample_path.resolve())
        if sample_path.is_file()
        else None,
        "elapsed_seconds": time.time() - started,
        "peak_gpu_memory_gib": (
            torch.cuda.max_memory_allocated() / (1024**3)
        ),
    }
    atomic_json(args.output_dir / "run_metadata.json", result)
    if n_valid <= 0 or not sample_path.is_file():
        raise RuntimeError("VoxBind produced no valid molecule SDF")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

