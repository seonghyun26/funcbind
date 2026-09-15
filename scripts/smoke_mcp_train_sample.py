#!/usr/bin/env python
"""One-target download -> X-ray processing -> tiny training -> latent sampling.

Uses real MCP data and pretrained NF/CDG v2 encoders, but a randomly initialized,
reduced FuncBind denoiser. This checks plumbing, not the 5.14B model's memory or
MCP generation quality. The test-split example is used only for this isolated
smoke run and must never become a production training/evaluation checkpoint.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch
import torchmetrics
from hydra import compose, initialize_config_dir
from lightning import Fabric
from lightning.fabric.strategies import DDPStrategy
from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def check(name, condition, detail=""):
    print(f"[{'PASS' if condition else 'FAIL'}] {name} {detail}", flush=True)
    if not condition:
        raise RuntimeError(name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--target", default="1bm2")
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--nf-checkpoint", type=Path, required=True)
    parser.add_argument("--fb-checkpoint", type=Path, required=True,
                        help="Only the latent normalization statistics are read.")
    parser.add_argument("--cdg-checkpoint", type=Path, required=True)
    parser.add_argument("--voxbind-root", type=Path, required=True)
    parser.add_argument("--train-steps", type=int, default=3)
    parser.add_argument("--sample-steps", type=int, default=4)
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--accum-steps", type=int, default=1)
    parser.add_argument("--prepared-root", type=Path,
                        help="Reuse data/ and density/ from an earlier one-target smoke; no downloads.")
    args = parser.parse_args()
    # The random denoiser and density projection both have zero-init gates;
    # three updates are needed to observe a gradient through the whole branch.
    if args.train_steps < 3 or args.sample_steps < 2:
        parser.error("at least three training steps and two sampling steps are required")
    if args.accum_steps < 1:
        parser.error("--accum-steps must be positive")
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    target = args.target.lower()
    if len(target) != 4 or not target.isalnum():
        parser.error("target must be a four-character PDB identifier")
    check("disk reserve", shutil.disk_usage(args.out).free > 30 * 1024**3)
    check("CUDA available", torch.cuda.is_available())
    started = time.monotonic()
    report = dict(status="running", target=target, precision="bf16-mixed",
                  scope="reduced random FuncBind; real pretrained NF and CDG v2",
                  production_training_validated=False, chemistry_validated=False,
                  source_split="test (isolated plumbing smoke only)")
    report_path = args.out / "report.json"
    if int(os.environ.get("LOCAL_RANK", "0")) == 0:
        report_path.write_text(json.dumps(report, indent=2) + "\n")

    # Fresh output directories force real public downloads. The original MCP
    # reference files are reused for local-to-deposited coordinate alignment.
    from download_mcpp_data import DEFAULT_REPO_ID, dataset_files, download_file
    from funcbind.dataset.prepare_mcpp_holo_density import resolve_structure_root
    if args.prepared_root:
        split_root = args.prepared_root / "data/mcpp_dataset"
        density_root = args.prepared_root / "density"
        check("prepared test split exists", (split_root / "test_data.pt").is_file())
        report["downloads"] = dict(reused_prepared_root=str(args.prepared_root))
    else:
        if args.devices > 1:
            parser.error("multi-GPU smoke requires --prepared-root to avoid concurrent downloads")
        metadata = dataset_files(DEFAULT_REPO_ID)
        split_root = args.out / "data/mcpp_dataset"
        download_file(DEFAULT_REPO_ID, "test_data.pt", split_root / "test_data.pt",
                      metadata["test_data.pt"])
        raw_root, targets = resolve_structure_root(args.raw_root)
        check("original MCP reference files exist", target in targets)
        density_root = args.out / "density"
        subprocess.run([
            sys.executable, str(REPO / "funcbind/dataset/prepare_mcpp_holo_density.py"),
            "--data-root", str(raw_root), "--out-dir", str(density_root),
            "--target", target, "--download-workers", "1", "--build-workers", "1",
            "--normalization-recipe",
            str(REPO / "funcbind/dataset/recipes/xray_resample_plinder_v2_perelem.json"),
        ], check=True)
        report["downloads"] = dict(test_split=metadata["test_data.pt"],
                                   reused_original_reference_directory=str(raw_root / target))
    complete = json.loads((density_root / ".complete").read_text())
    check("one real X-ray map processed", complete["n_available"] == 1)
    report["downloads"]["xray"] = json.loads((density_root / "download_status.json").read_text())

    os.environ["FUNCBIND_ROOT"] = str(REPO)
    os.environ["VOXBIND_PYTHON_ROOT"] = str(args.voxbind_root.resolve())
    os.environ["FUNCBIND_DENSITY_ENCODER"] = str(args.cdg_checkpoint.resolve())
    with initialize_config_dir(config_dir=str(REPO / "funcbind/configs"), version_base=None):
        cfg = compose(config_name="train_fb_mcpp_holo_density_h100", overrides=[
            "wandb=false", "dset.data_aug=false", "dset.num_workers=0", f"accum_steps={args.accum_steps}",
            "performance.gpu_prefetch=false", "denoiser.model_channels=32",
            "denoiser.ch_mults=[1,2]", "denoiser.n_blocks=1",
            "denoiser.attn_resolutions=[]", "denoiser.cfg_dropout=0",
        ])
    config = OmegaConf.to_container(cfg, resolve=True)
    config["dset"]["data_dir"] = str(split_root.parent)
    config["dset"]["mcpp_holo_density_dir"] = str(density_root)
    config["sampler"]["modality_cond"] = False
    strategy = DDPStrategy(start_method="popen", gradient_as_bucket_view=True) if args.devices > 1 else "auto"
    fabric = Fabric(accelerator="cuda", devices=args.devices, precision=config["precision"], strategy=strategy)
    fabric.launch()
    # Bound this reduced-model regression even when sharing a GPU with other jobs.
    torch.cuda.set_per_process_memory_fraction(.20, fabric.device)
    fabric.seed_everything(1234)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")

    from funcbind.utils.utils_nf import create_nf_encoder, create_nf_decoder, update_config_nf, FrozenInferenceModule
    from funcbind.utils.utils_fb import create_field_makers, create_optimizer, create_funcbind, load_funcbind
    from funcbind.dataset.dataset_omni import DatasetOmni, collate_fn
    from funcbind.dataset.mcpp_holo_density import mcpp_target_id
    from funcbind.models.denoiser import FuncBind
    from funcbind.models.phema import create_ema, ema_evaluation
    from funcbind.models.density_condition import make_density_voxelizer
    from funcbind.train_fb import train_denoiser, _prepare_train_batch

    nf = torch.load(args.nf_checkpoint, map_location="cpu", mmap=True, weights_only=False)
    nf_config = update_config_nf(copy.deepcopy(nf["config"]), config)
    config["encoder"] = nf_config["encoder"]
    config["decoder"] = nf_config["decoder"]
    enc = create_nf_encoder(nf_config, fabric)
    dec = create_nf_decoder(nf_config, fabric)
    enc.load_state_dict(nf["enc_state_dict"], strict=True)
    dec.load_state_dict(nf["dec_state_dict"], strict=True)
    del nf
    enc.eval().requires_grad_(False)
    dec.eval().requires_grad_(False)
    enc = FrozenInferenceModule(enc, fabric)
    dec = FrozenInferenceModule(dec, fabric)
    dec_module = dec.module
    base = torch.load(args.fb_checkpoint, map_location="cpu", mmap=True, weights_only=False)
    code_stats = {k: torch.as_tensor(v, device=fabric.device).clone()
                  for k, v in base["code_stats"].items()}
    del base
    dec_module.code_stats = code_stats

    dataset = DatasetOmni(nf_config, split="test", sample_points=False,
                          sample_full_grid=False, rebalance=False)
    index = next(i for i, sample in enumerate(dataset.data)
                 if mcpp_target_id(sample[2]) == target)
    batch = fabric.to_device(collate_fn([dataset[index]]))
    check("real density reaches dataset batch", bool(batch["density_available"].all()))
    check("density crop shape", list(batch["density"].shape) == [1, 64, 64, 64])
    check("density finite and nonconstant", bool(torch.isfinite(batch["density"]).all())
          and batch["density"].std().item() > 0)
    field, receptor_field = create_field_makers(config, nf_config, fabric)
    voxelizer = make_density_voxelizer(fabric.device, args.voxbind_root, backend="torch")
    model = create_funcbind(config, code_stats=code_stats, fabric=fabric, num_classes=0)
    ema = create_ema(model, config, fabric)
    optimizer = create_optimizer(model, config, fabric)
    wrapped, wrapped_optimizer = fabric.setup(model, optimizer)
    report["trainable_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    report["frozen_cdg_parameters"] = sum(p.numel() for p in model.density_condition.encoder.parameters())

    # Use the production training loop and check gradients before it clears them.
    gradient_peaks = []
    def before_step(opt, positional, keyword):
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        check("finite FP32 gradients", bool(grads) and all(
            g.dtype == torch.float32 and bool(torch.isfinite(g).all()) for g in grads))
        grad = model.density_condition.proj[-1].weight.grad
        gradient_peaks.append(0.0 if grad is None else grad.abs().max().item())
    handle = optimizer.register_step_pre_hook(before_step)
    loss, samples_seen, steps = train_denoiser(
        [batch] * (args.train_steps * args.accum_steps), enc, dec_module, wrapped, wrapped_optimizer,
        torchmetrics.MeanMetric().to(fabric.device), config, nf_config,
        model_ema=ema, fabric=fabric, field_maker=field,
        field_maker_receptor=receptor_field, num_classes=0, density_voxelizer=voxelizer,
    )
    handle.remove()
    check("requested optimizer steps completed", steps == args.train_steps)
    check("training loss is finite", math.isfinite(loss), f"loss={loss:.6g}")
    check("density projection learns", max(gradient_peaks) > 0, str(gradient_peaks))
    check("model remains finite FP32", all(
        p.dtype == torch.float32 and bool(torch.isfinite(p).all())
        for p in model.parameters()))
    if ema is not None:
        check("EMA remains finite on CPU", all(
            p.device.type == "cpu" and bool(torch.isfinite(p).all())
            for p in ema.state_dict()["emas"][0].values()))
    check("CDG encoder stays frozen", all(p.grad is None and not p.requires_grad
          for p in model.density_condition.encoder.parameters()))
    local_optimizer = getattr(optimizer, "optim", optimizer)
    check("Adam moments remain FP32", bool(local_optimizer.state) and all(
        s[key].dtype == torch.float32 and bool(torch.isfinite(s[key]).all())
        for s in local_optimizer.state.values() for key in ("exp_avg", "exp_avg_sq")))
    report["training"] = dict(steps=steps, samples_seen=samples_seen, loss=loss,
                              density_gradient_max=gradient_peaks, devices=fabric.world_size,
                              optimizer_sharding=config["performance"]["optimizer_sharding"],
                              cpu_ema=True, activation_checkpointing=True,
                              accum_steps=args.accum_steps,
                              zero_grad_set_to_none=config["performance"]["zero_grad_set_to_none"])

    # All ranks save their local moments; main checkpoint remains sampler-readable.
    from funcbind.utils.training_state import save_training_state, load_cpu_checkpoint, restore_optimizer_state
    saved_path, _ = save_training_state(fabric, wrapped, wrapped_optimizer, ema,
        dict(epoch=1, config=config, code_stats=code_stats, acc_iter=samples_seen,
             global_step=steps, best_res=loss), args.out / "checkpoint", reserve_gib=30)
    resume_config = copy.deepcopy(config)
    resume_config["resume_optimizer"] = True
    (model, ema, saved_optimizer, restored_stats, restored_samples,
     restored_step, _, restored_epoch) = load_funcbind(
        str(args.out / "checkpoint"), fabric=fabric, config=resume_config,
        num_classes=0, train=True, return_global_step=True)
    check("checkpoint progress restored", restored_step == steps and restored_epoch == 1
          and restored_samples == samples_seen)
    optimizer = create_optimizer(model, config, fabric)
    wrapped, wrapped_optimizer = fabric.setup(model, optimizer)
    restore_optimizer_state(wrapped_optimizer, saved_optimizer, wrapped, fabric,
                            args.out / "checkpoint")
    del saved_optimizer, local_optimizer
    dec_module.code_stats = restored_stats
    resumed_loss, _, resumed_step = train_denoiser(
        [batch], enc, dec_module, wrapped, wrapped_optimizer,
        torchmetrics.MeanMetric().to(fabric.device), config, nf_config,
        model_ema=ema, acc_iter=restored_samples, global_step=restored_step,
        epoch=restored_epoch, fabric=fabric, field_maker=field,
        field_maker_receptor=receptor_field, num_classes=0, density_voxelizer=voxelizer)
    check("training continues after full reload", resumed_step == steps + 1 and math.isfinite(resumed_loss))
    report["checkpoint_roundtrip"] = dict(restored_epoch=restored_epoch,
        restored_step=restored_step, resumed_step=resumed_step, resumed_loss=resumed_loss)
    if fabric.global_rank != 0:
        fabric.barrier()
        return

    from funcbind.sampling import DiffusionSampler
    from funcbind.models.decoder import get_code_spatial
    with ema_evaluation(ema) as sample_model, torch.no_grad(), fabric.autocast():
        prepared = _prepare_train_batch(batch, enc, dec_module, wrapped, config, nf_config,
                                        field, receptor_field, fabric, 0, voxelizer)
        codes, receptor, _, _, _, density_input, available = prepared
        receptor_code = sample_model.receptor_encoder(receptor)
        sampler = DiffusionSampler(N=args.sample_steps, sigma_min=0.1, sigma_max=10.0,
                                   S_churn=0, save_trajectory=True)
        sampler.t_steps = sampler.t_steps.to(fabric.device)
        result = sampler.sample(
            lambda y, sigma: sample_model.score(y, sigma, receptor_encoding=receptor_code,
                                               density_input=density_input, density_available=available),
            y_init=torch.randn_like(codes) * sampler.sigma_max,
        )
        check("sample and trajectory finite", bool(torch.isfinite(result["sample"]).all())
              and bool(torch.isfinite(result["xhat_traj"]).all()))
        # Decode a bounded 32^3 query grid with the actual pretrained NF decoder.
        # This is a field/shape check, not atom extraction or molecular validity.
        axis = torch.linspace(-1, 1, 32, device=fabric.device)
        xs = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), dim=-1).reshape(1, -1, 3)
        unnormalized = dec_module.unnormalize_code(result["sample"].to(fabric.device))
        field_codes = get_code_spatial(xs, unnormalized)
        decoded = dec_module.forward_batched(xs, field_codes, batch_size_render=256).float()
        check("NF decoded field finite", bool(torch.isfinite(decoded).all()))
    report["sampling"] = dict(steps=args.sample_steps, chains=1,
                              latent_shape=list(result["sample"].shape),
                              trajectory_shape=list(result["xhat_traj"].shape),
                              decoded_shape=list(decoded.shape),
                              decoded_min=decoded.min().item(), decoded_max=decoded.max().item(),
                              scope="latent diffusion and NF field decoding; no molecule validity test")
    torch.save({**result, "decoded_field": decoded, "query_grid_dim": 32,
                "scope": report["scope"]}, args.out / "sample.pt")
    report.update(status="passed", gpu=torch.cuda.get_device_name(),
                  gpu_peak_allocated_gib=torch.cuda.max_memory_allocated() / 1024**3,
                  elapsed_seconds=time.monotonic() - started)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    print("ALL SMOKE CHECKS PASSED", flush=True)
    fabric.barrier()


if __name__ == "__main__":
    main()
