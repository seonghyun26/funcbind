#!/usr/bin/env python
"""CPU-only data-path smoke for the MCP density fine-tune. No GPU, no model.

The companion to mcp_default_fusion_smoke.py: that one checks the model wiring on a
GPU, this one checks that data actually reaches it, on a box whose GPUs are busy.
It walks exactly the steps train_fb.main() takes before the denoiser exists --

  1. compose the real training config (not a toy one);
  2. load the neural-field checkpoint and derive config_nf, which is what the loaders
     actually read -- update_config_nf forwards only a whitelist of dset keys, so a
     knob set in the run config is not necessarily a knob the loader sees;
  3. build the train loader WITH workers, the branch that needs DataLoader(in_order=),
     an argument that exists only in torch >= 2.6. utils_dataset guards it by signature;
     this is the test that fails if that guard regresses;
  4. pull a real batch and check density came with it;
  5. cross the repo seam -- voxbind.voxelizer + build_density_input -- and confirm the
     13-channel 64^3 volume matches the encoder's channel_groups.

Step 3 is why this exists: a TypeError on a DataLoader kwarg is invisible to import
tests and to static version scans, and it kills the run before batch one.

    CUDA_VISIBLE_DEVICES="" <env>/bin/python test/mcp_datapath_cpu_smoke.py
"""
import os
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "2")   # this box is capped at 32 cores and a
os.environ.setdefault("MKL_NUM_THREADS", "2")   # 4-GPU train run is usually holding them
os.environ.setdefault("WANDB_MODE", "disabled")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)                                  # dset.data_dir resolves relative to cwd

import torch
torch.set_num_threads(2)
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from lightning.fabric import Fabric

CONFIG_NAME = os.environ.get("CONFIG_NAME", "train_fb_mcpp_holo_density_default")
fails = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{'  ' + detail if detail else ''}")
    if not ok:
        fails.append(name)


print("== 0. env ==")
import lightning
import numpy
print(f"  python {sys.version.split()[0]} | torch {torch.__version__} "
      f"| numpy {numpy.__version__} | lightning {lightning.__version__}")
check("no GPU is visible to this process", not torch.cuda.is_available())

print(f"\n== 1. hydra config ({CONFIG_NAME}) ==")
with initialize_config_dir(config_dir=os.path.join(REPO, "funcbind", "configs"), version_base=None):
    cfg = compose(
        config_name=CONFIG_NAME,
        # workers > 0 on purpose: that is the branch where the loader wants `in_order=`.
        overrides=["dset.batch_size=2", "dset.val_batch_size=2", "dset.num_workers=1",
                   "dset.prefetch_factor=2", "wandb=false"],
    )
config = OmegaConf.to_container(cfg, resolve=True)
check("with_density is on", bool(config["denoiser"]["with_density"]))
check("fusion is the winning recipe", config["denoiser"]["density"]["fusion"] == "default",
      f'fusion={config["denoiser"]["density"]["fusion"]}')

print("\n== 2. neural-field checkpoint -> loader config ==")
fabric = Fabric(accelerator="cpu", devices=1, precision="32-true")
fabric.launch()
t0 = time.time()
nf_ckpt = fabric.load(os.path.join(config["nf_pretrained_path"], "model.pt"))
from funcbind.utils.utils_nf import update_config_nf
config_nf = update_config_nf(nf_ckpt["config"], config)
check("nf checkpoint loads on CPU", "config" in nf_ckpt, f"{time.time() - t0:.1f}s")

print("\n== 3. loaders, with workers ==")
from funcbind.utils.utils_dataset import create_field_loaders
t0 = time.time()
loader = create_field_loaders(config_nf, split="train", fabric=fabric,
                              sample_points=False, n_samples=config_nf["n_samples"])
check("train loader built", loader is not None, f"{len(loader)} batches, {time.time() - t0:.1f}s")

import inspect
has_in_order = "in_order" in inspect.signature(torch.utils.data.DataLoader.__init__).parameters
inner = getattr(loader, "_dataloader", loader)
applied = getattr(inner, "in_order", "<absent>")
# What create_field_loaders reads is config_nf, and update_config_nf does not forward
# dset.in_order -- so the effective value is the default True regardless of the run config.
want = bool(config_nf["dset"].get("in_order", True))
check("multi-worker loader builds (num_workers=1)", loader is not None,
      f"torch {torch.__version__}: DataLoader in_order supported={has_in_order}, applied={applied!r}")
check("in_order is passed iff torch supports it, with the value the loader computes",
      (applied == want) if has_in_order else (applied == "<absent>"),
      f'config_nf says {config_nf["dset"].get("in_order", "<absent>")} -> effective {want}')

print("\n== 4. a real batch comes through ==")
t0 = time.time()
batch = next(iter(loader))
dt = time.time() - t0
keys = sorted(batch) if isinstance(batch, dict) else None
check("batch is a dict", keys is not None, f"keys={keys}")
check("batch carries density", "density" in batch and "density_available" in batch,
      f"{dt:.1f}s to fetch")
if "density" in batch:
    d, avail = batch["density"], batch["density_available"]
    print(f"    density {tuple(d.shape)} {d.dtype}  min={d.min():.3f} max={d.max():.3f} mean={d.mean():.3f}")
    print(f"    density_available {avail.tolist()}")
    check("density is finite", bool(torch.isfinite(d).all()))
    check("density is not all-zero", float(d.abs().sum()) > 0)

print("\n== 5. the VoxBind seam: voxelizer + density input ==")
from funcbind.models.density_condition import build_density_input, make_density_voxelizer
voxbind_root = config["denoiser"]["density"]["voxbind_python_root"]
vox = make_density_voxelizer("cpu", voxbind_root, backend="torch")
check("voxbind.voxelizer.Voxelizer constructed on CPU", vox is not None, type(vox).__name__)
t0 = time.time()
dens_in = build_density_input(batch["receptor"], batch["density"], vox, voxbind_root)
check("build_density_input ran", dens_in is not None, f"{time.time() - t0:.1f}s")
print(f"    density_input {tuple(dens_in.shape)} {dens_in.dtype}  "
      f"min={dens_in.min():.3f} max={dens_in.max():.3f}")
groups = config["denoiser"]["density"]["channel_groups"]
check("channel count matches the encoder's channel_groups",
      dens_in.shape[1] == sum(groups), f"{dens_in.shape[1]} == sum({groups})")
check("density_input is finite", bool(torch.isfinite(dens_in).all()))

print()
if fails:
    print(f"{len(fails)} CHECK(S) FAILED: {fails}")
    sys.exit(1)
print("ALL CHECKS PASSED")
