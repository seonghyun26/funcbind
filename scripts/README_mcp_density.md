# MCP density conditioning — porting the recipe to another box

Two scripts, run in order:

```bash
bash scripts/01_setup_mcp_density_data.sh        # data + weights + holo-density build
bash scripts/preflight_mcp_density.py            # will it run here?  ← read this one
bash scripts/02_train_mcp_density_conditioned.sh # the conditioned fine-tune
```

## Read this before provisioning: 8×H100 80 GB does not fit

FuncBind's denoiser is **5.14 B parameters**, and Fabric runs `bf16-mixed`, which keeps
fp32 master weights. Per rank, before a single activation:

| | bytes/param | GiB |
|---|---:|---:|
| params (fp32) | 4 | 19.2 |
| grads (fp32) | 4 | 19.2 |
| AdamW `exp_avg` + `exp_avg_sq` | 8 | 38.3 |
| `PowerFunctionEMA` copy | 4 | 19.2 |
| **static total** | **20** | **95.8** |

`setup_fabric` hardcodes **DDP**, which replicates every byte on every rank — so **more
GPUs do not lower the per-GPU requirement**. 8×80 GB has exactly the same per-rank budget
as 1×80 GB, and 95.8 GiB does not fit in 79.6 GiB. It runs here only because an H200 has
139.8 GiB: measured usage is ~138 GiB at `batch_size=5`, i.e. 95.8 static + ~42 activations.
Dropping the EMA saves 19.2 GiB and still leaves 76.6 GiB static — not enough.

Three ways to make 80 GB work, cheapest first:

1. **`precision="bf16-true"`** → 47.9 GiB static. One line in
   `funcbind/utils/utils_base.py:setup_fabric`. It halves all four rows, but it also puts
   the AdamW moments of a 5 B model in bf16; treat it as an experiment, not a free win.
2. **FSDP / ZeRO-3 across the ranks** → 95.8/N GiB (8 ranks: **12.0 GiB**), the right
   answer for this shape. It is a real port: `setup_fabric` builds a `DDPStrategy`
   unconditionally, and the manual gradient-accumulation loop, `PowerFunctionEMA`, and the
   custom `AdamW(eps=0)` all need re-checking under sharding.
3. **A smaller denoiser** — only `unet_edm2_XXXL` exists today, so this means authoring a
   config and giving up comparability with every number measured so far.

The preflight prints this table for the card it actually finds and fails rather than
letting you discover it 20 minutes into a run. `FORCE=1` overrides it once you have made
one of the changes above.

## What step 1 fetches, and from where

Public, no credentials — handled by `prepare_mcpp_holo_density.py` itself:
coordinates from **RCSB**, 2Fo-Fc CCP4 maps from **PDBe**. Resumable; hours on a cold cache.

Private — from `ASSETS_SRC=<dir>` you staged, else the lab Dropbox over `rclone`
(`notebook/html/dropbox-sync.md`):

| asset | why |
|---|---|
| `mcpp_dataset/` | MCP structures + `{train,val,test}_data.pt` |
| `nf_unified/` | neural-field encoder/decoder |
| `fb_unified/` | density-free MCP FuncBind — the fine-tune start point |
| `atomblob7_v2p1_e0099.pth.tar` | the FROZEN density encoder |
| VoxBind checkout | the branch imports `voxbind.models.density_vit` at runtime |

The normalization constants live in `funcbind/dataset/recipes/xray_resample_plinder_v2p1.json`,
vendored into this repo because the original sits under a gitignored `data/` dir and does
not exist on a fresh checkout. They must match the encoder's pretraining — a differently
normalized map is a different input distribution to a frozen trunk.

## Things that bite

* **Encoder geometry is not a preference.** `load_state_dict` is strict. atomblob7 v2.1 is
  dim 512 / depth 12 / heads 8 / groups **[7,4,1,1]**; the 100 M champion is 640 / 18 / 10 /
  **[7,4,2]**. Different groupings are different patch embeddings — the checkpoints are not
  interchangeable at fixed geometry. The preflight checks dim and depth against the file.
* **`encoder_amp: False` needs the input cast.** Fabric's outer bf16 autocast hands the
  fp32 trunk a bf16 volume; without the cast it dies on batch 1 with
  `Input type (c10::BFloat16) and bias type (float) should be the same`. Handled in
  `density_condition._encode`.
* **CPU quota starves the loaders.** A 52-core job sharing a 32-core container dropped
  training from 14.2 to 6.6 samples/s with the GPUs at 4–23%. Check `nproc` against the
  cgroup quota, not the host.
* **Effective batch is part of the experiment.** The LR decay onset is
  `ref_batches × effective_batch` in `acc_iter`, so step 2 derives `accum_steps` from your
  batch size and GPU count to hold it at 760. On 8 GPUs, `BATCH_SIZE=5` gives exactly
  760 (accum 19); other sizes may not divide evenly and the script says so.
* **`fb_unified` carries `acc_iter = 87,214,080`.** Anything that ranks checkpoints by
  `acc_iter` (the watchdog) will treat the base as permanently furthest along and restart
  the fine-tune from scratch. Keep the base out of any resume-source ranking.
