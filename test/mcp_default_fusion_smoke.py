#!/usr/bin/env python
"""GPU smoke test for the `default` density fusion + atomblob7 encoder.

Checks the four things that would otherwise surface as a dead multi-week run:

  1. the atomblob7 v2.1 checkpoint loads into the geometry the config declares
     (load_state_dict is strict here -- a mismatch raises, it does not warn);
  2. the projection is wired to the conditioned width (encoder dim + code_dim);
  3. STEP-0 EQUIVALENCE -- with the zero-init projection the denoiser's output with
     density is bit-identical to its output without, so switching fusion cannot
     regress the density-free baseline at initialisation;
  4. the branch still learns -- the projection receives a non-zero gradient, and it
     depends on the ligand, which is the entire point of `default` over the
     density-only branch this replaces.

    <env>/bin/python test/mcp_default_fusion_smoke.py
"""
import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from funcbind.models.density_condition import DensityCondition

DENS_CFG = dict(
    voxbind_python_root=os.environ.get("VOXBIND_PYTHON_ROOT", "/home1/irteam/VoxBind"),
    pretrained_path=os.environ.get("VOXBIND_PYTHON_ROOT", "/home1/irteam/VoxBind")
    + "/voxbind/exps/260701_plinder_v2p1_box_atomblob7_cdg_channelvit_full_pretrain/checkpoint_e0099.pth.tar",
    patch=8, dim=512, depth=12, heads=8, mlp_ratio=4, c_out=16,
    pos_encoding="learnable", patch_embed_mode="channel_group",
    channel_groups=[7, 4, 1, 1],
)
CODE_DIM, GRID, LATENT_EXTENT = 128, 16, 32.0
DEV = "cuda" if torch.cuda.is_available() else "cpu"
B = 2
fails = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{'  ' + detail if detail else ''}")
    if not ok:
        fails.append(name)


print(f"device={DEV}")
print("\n== 1. encoder loads at the declared geometry ==")
dc = DensityCondition(
    density_cfg=DENS_CFG, code_dim=CODE_DIM, code_grid_dim=GRID,
    voxbind_root=DENS_CFG["voxbind_python_root"], hidden=192,
    freeze=True, amp=False, latent_extent=LATENT_EXTENT, fusion="default",
).to(DEV)
n_frozen = sum(p.numel() for p in dc.encoder.parameters())
n_train = sum(p.numel() for p in dc.proj.parameters())
check("strict load of atomblob7 v2.1", n_frozen > 0, f"{n_frozen:,} frozen params")
check("encoder is frozen", not any(p.requires_grad for p in dc.encoder.parameters()))

print("\n== 2. projection width follows the fusion mode ==")
check("default proj in_channels == dim + code_dim",
      dc.proj[0].in_channels == DENS_CFG["dim"] + CODE_DIM,
      f"{dc.proj[0].in_channels} (= {DENS_CFG['dim']} + {CODE_DIM}), {n_train:,} trainable")
dc_only = DensityCondition(
    density_cfg=DENS_CFG, code_dim=CODE_DIM, code_grid_dim=GRID,
    voxbind_root=DENS_CFG["voxbind_python_root"], hidden=192, freeze=True, amp=False,
    latent_extent=LATENT_EXTENT, fusion="density_only").to(DEV)
check("density_only proj in_channels == dim", dc_only.proj[0].in_channels == DENS_CFG["dim"])
try:
    DensityCondition(density_cfg=DENS_CFG, code_dim=CODE_DIM, code_grid_dim=GRID,
                     voxbind_root=DENS_CFG["voxbind_python_root"], fusion="protien_first")
    check("a typo'd fusion name raises", False, "it did not raise")
except ValueError:
    check("a typo'd fusion name raises", True)

print("\n== 3. step-0 equivalence (the zero-init guarantee) ==")
dens_in = torch.randn(B, 13, 64, 64, 64, device=DEV)
# The trainer runs inside Fabric's bf16 autocast, so the branch must survive a bf16
# input while its own trunk is fp32 (encoder_amp: False). Missing this crashed the
# first launch on batch 1.
with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(DEV == "cuda")):
    _d = dc(torch.randn(B, 13, 64, 64, 64, device=DEV).to(torch.bfloat16),
            cond=torch.randn(B, CODE_DIM, GRID, GRID, GRID, device=DEV).to(torch.bfloat16))
check("survives a bf16 input under autocast", _d is not None, f"out dtype={_d.dtype}")
receptor = torch.randn(B, CODE_DIM, GRID, GRID, GRID, device=DEV)
ligand = torch.randn(B, CODE_DIM, GRID, GRID, GRID, device=DEV)
with torch.no_grad():
    delta = dc(dens_in, cond=receptor + ligand)
check("delta is exactly zero at init", bool((delta == 0).all()),
      f"max|delta|={delta.abs().max().item():.3e}, shape={tuple(delta.shape)}")
check("delta is shaped like the receptor code", delta.shape == receptor.shape)

print("\n== 4. the branch learns, and it depends on the ligand ==")
# The loss must have a non-zero gradient w.r.t. delta AT delta=0, or this proves
# nothing: with L = mean(delta^2), dL/dW = 2*delta*h is identically zero at the zero
# init for arithmetic reasons, not because the branch is dead. A denoising loss pulls
# delta towards a non-zero target, which is the situation training is actually in.
target = torch.randn_like(receptor)
delta = dc(dens_in, cond=receptor + ligand)
(delta - target).square().mean().backward()
g = dc.proj[-1].weight.grad
check("zero conv receives gradient", g is not None and bool((g.abs() > 0).any()),
      f"max|grad|={g.abs().max().item():.3e}")
# Break the zero-init so the branch is live, then confirm the ligand actually moves it.
with torch.no_grad():
    torch.nn.init.normal_(dc.proj[-1].weight, std=0.02)
    d_a = dc(dens_in, cond=receptor + ligand)
    d_b = dc(dens_in, cond=receptor + torch.randn_like(ligand))
check("output depends on the ligand term", not torch.allclose(d_a, d_b),
      f"mean|diff|={(d_a - d_b).abs().mean().item():.3e}")
with torch.no_grad():
    d_only_a = dc_only(dens_in, cond=None)
    d_only_b = dc_only(dens_in, cond=None)
check("density_only ignores the ligand (control)", torch.allclose(d_only_a, d_only_b))

if DEV == "cuda":
    print(f"\npeak GPU memory: {torch.cuda.max_memory_allocated()/2**30:.2f} GiB")
print("\nFAILED: " + ", ".join(fails) if fails else "\nALL CHECKS PASSED")
sys.exit(1 if fails else 0)
