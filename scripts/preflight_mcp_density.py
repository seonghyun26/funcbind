#!/usr/bin/env python
"""Preflight for the MCP density-conditioning recipe: can this box actually run it?

Checks, in order, the things that otherwise fail hours in:

  1. inputs  -- MCP dataset, holo-density memmap, base weights, frozen encoder;
  2. geometry -- the density block's ViT dims against the encoder checkpoint it names
                 (load_state_dict is strict; a mismatch is a hard error at step 0);
  3. MEMORY  -- the per-GPU budget for plain DDP against the VRAM actually present.

(3) is the one that decides whether a port is possible at all. FuncBind's denoiser is
5.14B parameters and Fabric runs `bf16-mixed`, which keeps fp32 master weights:

    params 4B/p + grads 4B/p + AdamW(exp_avg, exp_avg_sq) 8B/p + EMA 4B/p = 20 B/param

DDP replicates all of it on every rank, so adding GPUs does not help -- 8x80GB has the
same per-rank budget as 1x80GB. Activations sit on top.

    python scripts/preflight_mcp_density.py [--config train_fb_mcpp_holo_density_default]
"""
import argparse, json, os, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
GIB = 1024 ** 3
# From the run log: ">> FuncBind has 5141.25M parameters". Overridable for a resized model.
DEFAULT_PARAMS = 5_141_250_000
BYTES_PER_PARAM = dict(params=4, grads=4, adamw=8, ema=4)   # bf16-mixed => fp32 master

ok = True


def check(name, good, detail=""):
    global ok
    print(f"  [{'OK  ' if good else 'FAIL'}] {name}{'  — ' + detail if detail else ''}")
    ok = ok and good
    return good


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="train_fb_mcpp_holo_density_default")
    ap.add_argument("--params", type=int, default=DEFAULT_PARAMS)
    ap.add_argument("--data-root", type=Path, default=REPO / "funcbind/dataset/data")
    ap.add_argument("--skip-memory", action="store_true")
    a = ap.parse_args()

    import yaml
    cfg_path = REPO / "funcbind/configs" / f"{a.config}.yaml"
    print(f"config: {cfg_path}")
    cfg = yaml.safe_load(cfg_path.read_text()) if cfg_path.exists() else {}
    dfile = next((d for d in (cfg.get("defaults") or []) if isinstance(d, dict) and "denoiser" in d), None)
    dname = dfile["denoiser"] if dfile else "unet_edm2_XXXL_density_atomblob7_default"
    dcfg = yaml.safe_load((REPO / "funcbind/configs/denoiser" / f"{dname}.yaml").read_text())
    dens = dcfg.get("density", {})

    print("\n== 1. inputs ==")
    mcpp = a.data_root / "mcpp_dataset"
    check("MCP dataset", (mcpp / "train_data.pt").exists(), str(mcpp))
    holo = a.data_root / "mcpp_holo_xray_v1"
    complete = holo / ".complete"
    if check("holo-density set built", complete.exists(), str(holo)) :
        n = json.loads(complete.read_text())
        check("density maps available", n.get("n_available", 0) > 0,
              f"{n.get('n_available')} of {n.get('n_targets')} targets")
    check("neural-field weights", (REPO / "exps/neural_field/nf_unified/model.pt").exists())
    check("base FuncBind weights", (REPO / "exps/funcbind/fb_unified/checkpoint.pth.tar").exists())
    enc = Path(dens.get("pretrained_path", ""))
    check("frozen density encoder", enc.is_file(), str(enc))
    vb = Path(dens.get("voxbind_python_root", ""))
    check("VoxBind checkout (density_vit)", (vb / "voxbind/models/density_vit.py").is_file(), str(vb))

    print("\n== 2. encoder geometry vs its checkpoint ==")
    if enc.is_file():
        import torch
        ck = torch.load(str(enc), map_location="meta", weights_only=False, mmap=True)
        raw = ck.get("encoder_state_dict_ema") or ck.get("encoder_state_dict") or {}
        raw = {k[len("encoder."):] if k.startswith("encoder.") else k: v for k, v in raw.items()}
        blocks = {k.split(".")[1] for k in raw if k.startswith("blocks.")}
        dim = next((tuple(v.shape)[-1] for k, v in raw.items() if k.endswith("norm.weight")), None)
        check("depth matches", len(blocks) == int(dens.get("depth", -1)),
              f"checkpoint {len(blocks)} vs config {dens.get('depth')}")
        if dim:
            check("dim matches", dim == int(dens.get("dim", -1)),
                  f"checkpoint {dim} vs config {dens.get('dim')}")
        print(f"         channel_groups (config): {dens.get('channel_groups')} — "
              "must match the pretrain run; [7,4,2] and [7,4,1,1] are different patch embeddings")

    if a.skip_memory:
        return 0 if ok else 1

    print("\n== 3. per-GPU memory, plain DDP (the decisive one) ==")
    per = {k: a.params * b / GIB for k, b in BYTES_PER_PARAM.items()}
    static = sum(per.values())
    print(f"         {a.params/1e9:.2f}B params × 20 B/param (fp32 master + grads + AdamW + EMA)")
    for k, v in per.items():
        print(f"           {k:8s} {v:6.1f} GiB")
    print(f"           {'TOTAL':8s} {static:6.1f} GiB static, before activations")

    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            vram = props.total_memory / GIB
            print(f"         device: {props.name}, {vram:.1f} GiB, count={torch.cuda.device_count()}")
            headroom = vram - static
            check("static state fits one GPU", headroom > 0,
                  f"{'%.1f GiB spare' % headroom if headroom > 0 else 'short by %.1f GiB' % -headroom}")
            if headroom <= 0:
                print("\n         DDP replicates every byte on every rank, so more GPUs do NOT help.")
                print("         Options, cheapest first:")
                print(f"           a) precision='bf16-true'  -> {static/2:5.1f} GiB static (halves all four)")
                print( "                                        one-line change in utils_base.setup_fabric;")
                print( "                                        changes numerics -- bf16 AdamW moments on a 5B model")
                print(f"           b) FSDP/ZeRO-3 over N ranks -> {static:.1f}/N GiB  (8 ranks: {static/8:5.1f} GiB)")
                print( "                                        setup_fabric hardcodes DDPStrategy; a real port,")
                print( "                                        and the manual accumulation loop + PowerFunctionEMA")
                print( "                                        + custom AdamW(eps=0) all need re-checking under it")
                print( "           c) a smaller denoiser       -> only unet_edm2_XXXL exists today")
        else:
            print("         (no CUDA here — run this on the target box for the device check)")
    except Exception as exc:  # noqa: BLE001
        print(f"         (torch unavailable: {exc})")

    print("\n" + ("PREFLIGHT PASSED" if ok else "PREFLIGHT FAILED — fix the items above"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
