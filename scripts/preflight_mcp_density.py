#!/usr/bin/env python
"""Preflight for the MCP density-conditioning recipe: can this box actually run it?

Checks, in order, the things that otherwise fail hours in:

  1. inputs  -- MCP dataset, holo-density memmap, base weights, frozen encoder;
  2. geometry -- the density block's ViT dims against the encoder checkpoint it names
                 (load_state_dict is strict; a mismatch is a hard error at step 0);
  3. MEMORY  -- the per-GPU budget for plain DDP against the VRAM actually present.

(3) is the one that decides whether a port is possible at all. FuncBind's denoiser is
5.14B parameters. The original `bf16-mixed` run keeps fp32 state:

    params 4B/p + grads 4B/p + AdamW(exp_avg, exp_avg_sq) 8B/p + EMA 4B/p = 20 B/param

DDP replicates all of it on every rank, so adding GPUs does not lower that per-rank
budget. The H100 config preserves bf16-mixed, batch 1, and non-foreach AdamW.
It cannot fit 80 GB under plain DDP; sharding/offload support is still required.

    python scripts/preflight_mcp_density.py [--config train_fb_mcpp_holo_density_h100]
"""
import argparse, json, os, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
GIB = 1024 ** 3
# From the run log: ">> FuncBind has 5141.25M parameters". Overridable for a resized model.
DEFAULT_PARAMS = 5_141_250_000
MIXED_BYTES_PER_PARAM = dict(params=4, grads=4, adamw=8, ema=4)
TRUE_BF16_BYTES_PER_PARAM = dict(params=2, grads=2, adamw=4, ema=2)
ACTIVATION_GIB_PER_SAMPLE = 12.0
RUNTIME_RESERVE_GIB = 4.0

ok = True


def check(name, good, detail=""):
    global ok
    print(f"  [{'OK  ' if good else 'FAIL'}] {name}{'  — ' + detail if detail else ''}")
    ok = ok and good
    return good


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="train_fb_mcpp_holo_density_h100")
    ap.add_argument("--params", type=int, default=DEFAULT_PARAMS)
    ap.add_argument("--data-root", type=Path)
    ap.add_argument("--expected-gpus", type=int)
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--skip-memory", action="store_true")
    a = ap.parse_args()

    cfg_path = REPO / "funcbind/configs" / f"{a.config}.yaml"
    print(f"config: {cfg_path}")
    if not cfg_path.exists():
        raise FileNotFoundError(cfg_path)
    # Compose the real Hydra tree so defaults and oc.env interpolations resolve exactly
    # as they do in train_fb.py. Parsing one YAML file with PyYAML left strings such as
    # "${oc.env:VOXBIND_PYTHON_ROOT,...}" unresolved and made valid paths fail preflight.
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    overrides = [] if a.batch_size is None else [f"dset.batch_size={a.batch_size}"]
    with initialize_config_dir(config_dir=str(REPO / "funcbind/configs"), version_base=None):
        cfg = OmegaConf.to_container(
            compose(config_name=a.config, overrides=overrides), resolve=True
        )
    dens = cfg.get("denoiser", {}).get("density", {})
    data_root = a.data_root or Path(cfg["dset"]["data_dir"])

    print("\n== 1. inputs ==")
    mcpp = data_root / "mcpp_dataset"
    check("MCP dataset", (mcpp / "train_data.pt").exists(), str(mcpp))
    holo = Path(cfg["dset"].get("mcpp_holo_density_dir") or data_root / "mcpp_holo_xray_v1")
    complete = holo / ".complete"
    if check("holo-density set built", complete.exists(), str(holo)) :
        n = json.loads(complete.read_text())
        check("density maps available", n.get("n_available", 0) > 0,
              f"{n.get('n_available')} of {n.get('n_targets')} targets")
    nf = Path(cfg["nf_pretrained_path"]) / "model.pt"
    fb = Path(cfg["fb_pretrained_path"]) / "checkpoint.pth.tar"
    check("neural-field weights", nf.is_file(), str(nf))
    check("base FuncBind weights", fb.is_file(), str(fb))
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
    precision = str(cfg.get("precision", "bf16-mixed"))
    true_bf16 = precision == "bf16-true"
    bytes_per_param = TRUE_BF16_BYTES_PER_PARAM if true_bf16 else MIXED_BYTES_PER_PARAM
    per = {k: a.params * b / GIB for k, b in bytes_per_param.items()}
    static = sum(per.values())
    total_bytes = sum(bytes_per_param.values())
    print(f"         precision={precision}, {a.params/1e9:.2f}B params × {total_bytes} B/param")
    for k, v in per.items():
        print(f"           {k:8s} {v:6.1f} GiB")
    print(f"           {'TOTAL':8s} {static:6.1f} GiB static, before activations")

    optimizer_foreach = bool(cfg.get("performance", {}).get("optimizer_foreach", True))
    # The custom foreach AdamW creates full-size sqrt and denominator tensor lists.
    optimizer_transient = a.params * (4 if true_bf16 else 8) / GIB if optimizer_foreach else 0.0
    batch_size = int(cfg["dset"]["batch_size"])
    activation_estimate = ACTIVATION_GIB_PER_SAMPLE * batch_size
    # gradient_as_bucket_view removes the steady-state gradient duplicate, but DDP may
    # still reach one extra gradient-sized bucket allocation during the first backward.
    # Include it for multi-rank launches so the H100 go/no-go number is deliberately
    # stricter than the eventual steady-state footprint.
    requested_ranks = a.expected_gpus if a.expected_gpus is not None else 1
    ddp_first_step_transient = per["grads"] if requested_ranks > 1 else 0.0
    estimated_peak = (
        static
        + optimizer_transient
        + ddp_first_step_transient
        + activation_estimate
        + RUNTIME_RESERVE_GIB
    )
    print(f"         optimizer_foreach={optimizer_foreach}: +{optimizer_transient:.1f} GiB transient")
    print(f"         first-step DDP bucket: +{ddp_first_step_transient:.1f} GiB transient")
    print(
        f"         conservative peak estimate={estimated_peak:.1f} GiB "
        f"(batch={batch_size}, activations~{activation_estimate:.1f}, "
        f"reserve={RUNTIME_RESERVE_GIB:.1f})"
    )

    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            vram = props.total_memory / GIB
            device_count = torch.cuda.device_count()
            print(f"         device: {props.name}, {vram:.1f} GiB, count={device_count}")
            if a.expected_gpus is not None:
                check(
                    "requested GPU count is visible",
                    device_count == a.expected_gpus,
                    f"expected {a.expected_gpus}, found {device_count}",
                )
            headroom = vram - estimated_peak
            fits = headroom > 0
            check("estimated peak fits one GPU", fits,
                  f"{'%.1f GiB spare' % headroom if fits else 'short by %.1f GiB' % -headroom}")
            if not fits:
                print("\n         DDP replicates every byte on every rank, so more GPUs do NOT help.")
                if static >= vram:
                    print("         Static state alone exceeds VRAM; lowering batch size cannot fix it.")
                    print("         To preserve bf16-mixed, implement FSDP/ZeRO sharding or CPU offload.")
                else:
                    print("         Lower batch size, disable optimizer foreach, or add sharding/offload.")
        else:
            check("CUDA device available", False, "run this inside the GPU container")
    except Exception as exc:  # noqa: BLE001
        print(f"         (torch unavailable: {exc})")

    print("\n" + ("PREFLIGHT PASSED" if ok else "PREFLIGHT FAILED — fix the items above"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
