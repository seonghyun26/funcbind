# MCP density-conditioned FuncBind

Run the four entrypoints in order. Density fusion is `default`; the frozen
encoder is CDG v2 epoch 25.

## 0. Build the Docker image

From the VoxBind repository root:

```bash
git submodule update --init --recursive FuncBind
bash FuncBind/scripts/0_env_setup.sh
```

The host `.repro-env` is not used. The Dockerfile creates the FuncBind Conda
environment and its own `.repro-env` compatibility link, includes the current
VoxBind CDG/voxelizer source, and runs `pip check` during the build. No host source
mount is required. PyRosetta is an optional antibody dependency, not needed for MCP.

The VoxBind base image must already exist (`voxbind:allinone` by default).
To select another existing base, set `VOXBIND_BASE=<image:tag>` when running
`0_env_setup.sh`. Data and checkpoints are mounted separately as shown below.

## Start the container

Prepare persistent directories and model URLs. Copy
[`mcp_assets.env.example`](mcp_assets.env.example) to `scripts/mcp_assets.env`,
fill `CDG_MODEL_URL` with the separately supplied **file** link, and source it
before starting Docker. The filled file is ignored by Git and the VoxBind Docker
build context; do not commit private links or credentials.

| Variable | Required checkpoint | Source | Size |
|---|---|---|---|
| `FB_MODEL_URL` | Density-free FuncBind fine-tuning start point | [Official FuncBind checkpoint](https://huggingface.co/mkirchmeyer/funcbind/resolve/f42d3daeb6e7c1fa2b20096f9a147aa3b1f8814f/fb_unified/checkpoint.pth.tar) | 61.68 GB |
| `NF_MODEL_URL` | Pretrained neural-field encoder/decoder | [Official NF checkpoint](https://huggingface.co/mkirchmeyer/funcbind/resolve/f42d3daeb6e7c1fa2b20096f9a147aa3b1f8814f/nf_unified/model.pt) | 1.45 GB |
| `CDG_MODEL_URL` | Frozen CDG v2 epoch-25 density encoder | Dropbox file link supplied separately; `VoxBind/results/task1-affinity/CDG-v2/checkpoint_e0025.pth.tar` | 1.19 GB |

The example includes SHA-256 checksums for all three files. Public URLs were
checked on 2026-09-15. The CDG file is uploaded; its link/access must be supplied
separately. No new density-conditioned MCP checkpoint is required to start
fine-tuning. `MCP_MODEL_URL` is optional and only downloads a separately trained
checkpoint for generation, not the density-free starting model.

```bash
# On the host, from the VoxBind repository root, after filling the local copy:
source FuncBind/scripts/mcp_assets.env

docker run --rm -it --gpus all --shm-size=32g --cpus=32 --memory=256g \
  -e NF_MODEL_URL -e FB_MODEL_URL -e CDG_MODEL_URL \
  -e NF_MODEL_SHA256 -e FB_MODEL_SHA256 -e CDG_MODEL_SHA256 \
  -e MCP_MODEL_URL -e MCP_MODEL_SHA256 \
  -v /path/to/funcbind-data:/workspace/FuncBind/funcbind/dataset/data \
  -v /path/to/funcbind-exps:/workspace/FuncBind/exps \
  -v /path/to/funcbind-artifacts:/workspace/FuncBind/artifacts \
  -v /path/to/voxbind-exps:/workspace/VoxBind/voxbind/exps \
  voxbind-funcbind:sb
```

Public Hugging Face URLs and downloadable public Dropbox file links need no
rclone configuration or account login. Team-only/password-protected links will
not work for this unattended download. An existing destination is reused first;
otherwise `ASSETS_SRC` takes priority over URLs, then a configured `rclone`
remote is the final fallback. Leave `ASSETS_SRC` unset when using the URLs.
Checksums are verified on new URL downloads; existing files are reused as-is.

H100 target: one node with **8 x H100 80GB**, at least 32 CPU cores and preferably
256 GiB host RAM allocated to the job (the example sets limits, not reservations).
Keep at least 200 GiB free on the checkpoint volume after staging the input data
and weights to allow a checkpoint replacement plus the safety reserve. Retaining
older best checkpoints/shards needs additional space. Mount outputs on persistent
storage, not the container layer. Ensure the container UID can write those mounts;
the image also supports `--user UID:GID` with a writable working/output directory.

## 1. Download and process data

```bash
bash scripts/1_data_process.sh
```

This downloads the public MCP splits and original structures, then downloads
RCSB coordinates and PDBe 2Fo-Fc maps and builds the X-ray density cache. It is
resumable and refuses downloads that violate its free-space safety margin.
The public source is [Willete3/mcpp-dataset](https://huggingface.co/datasets/Willete3/mcpp-dataset/tree/main):
`train_data.pt`, `val_data.pt`, `test_data.pt`, and `mcpp_dataset.tar.gz` (~32.7 GB
compressed). X-ray files are fetched per target automatically; no separate
private data link is required. Plan additional space for extraction and density
caches, beyond the checkpoint-volume reserve above.

## 2. Fine-tune

```bash
SMOKE=1 bash scripts/2_train.sh
bash scripts/2_train.sh
```

The default uses `bf16-mixed` on GPUs `0-7`, with train and validation batch
size 1, accumulation 95 (effective batch 760), eager execution, and non-foreach
AdamW. Model weights, gradients, optimizer moments, and EMA remain FP32.
The H100 profile keeps DDP gradient bucket views across updates, uses two loader
workers/GPU with CPU prefetch factor 2, and disables GPU prefetch. Validation
renders one code at a time in chunks of 256 queries without the legacy 10x
render-batch multiplier. All-rank peak allocated/reserved VRAM is printed in the
training log even with W&B disabled. CUDA reserved memory includes allocator cache.

The H100 profile wraps the existing AdamW in PyTorch ZeRO-1, keeps one FP32 EMA
on rank-zero CPU, and recomputes UNet/receptor blocks during backward. Forced
weight normalization is skipped on recomputation so weights are not changed twice.
Static trainable state is approximately **43.1 GiB/GPU on 8 GPUs**, compared with
95.8 GiB under ordinary DDP. This excludes activations, communication buffers,
frozen models, and temporary allocations; it does not prove H100 capacity.

Use `SMOKE=1 bash scripts/2_train.sh` on the target H100 node first. It uses the
full model on a small data subset without saving a large checkpoint. CPU EMA
needs ~19.2 GiB on rank zero and another ~19.2 GiB temporarily during evaluation,
in addition to other host-memory requirements. EMA transfers may reduce speed.

For a bounded accumulation check on the real H100 node, before the longer run:

```bash
SMOKE=1 N_SAMPLES=48 EFFECTIVE_BATCH=16 bash scripts/2_train.sh
```

This intentionally changes the effective batch only for the smoke (micro-batch 1,
8 GPUs, accumulation 2). The regular command keeps effective batch 760. Inspect
`exps/funcbind/fb_mcpp_holo_density_h100_smoke/run.log` for training AND validation
completion and peak memory; then run the default `SMOKE=1` before a full job.
Do not set `FORCE=1` to bypass an unresolved preflight failure.

Checkpoints retain the standard model/EMA keys, so `3_generate.sh` reads them
without optimizer shards. Resuming training requires `checkpoint.pth.tar` AND
the referenced `optimizer_shards/` directory on shared storage, with the same
GPU count and parameter partition. Set `resume_optimizer=true` and point
`fb_pretrained_path` at that run when resuming. Initial fine-tuning keeps a fresh
optimizer. Loader-worker augmentation RNG is not restored exactly.

A full training checkpoint is ~77 GiB or more across its files. Saving checks
for the new checkpoint size plus a 30 GiB free-space reserve; previous latest/
best checkpoints and optimizer shards are not automatically deleted. Keep enough
space for old and new checkpoints, and review obsolete shards before removal.

Small distributed regression (CPU by default; use `--accelerator cuda` for GPUs):

```bash
python scripts/smoke_h100_memory.py --devices 2 --out artifacts/h100_memory_smoke
```

This checks AdamW/reference parity, CPU EMA evaluation, sharded save/resume and
the disk guard. It is not a full-model/H100 capacity or generation-quality test.

## 3. Generate

Run one target first, then all 100 targets over eight GPUs:

```bash
SMOKE=1 bash scripts/3_generate.sh
bash scripts/3_generate.sh
```

If using a separately distributed trained checkpoint, set `MCP_MODEL_URL`; it
will be downloaded to the persistent FuncBind `exps` mount automatically.

Useful overrides: `GPUS`, `EXP_NAME`, `FB_PATH`, `CHUNKS`, `NPR`, and `DRY_RUN`.

## Small GPU smoke test

Inside the container, with the original MCP reference files and checkpoints mounted:

```bash
python scripts/smoke_mcp_train_sample.py \
  --out artifacts/mcp_train_sample_smoke --target 1bm2 \
  --raw-root funcbind/dataset/data/mcpp_dataset \
  --nf-checkpoint exps/neural_field/nf_unified/model.pt \
  --fb-checkpoint exps/funcbind/fb_unified/checkpoint.pth.tar \
  --cdg-checkpoint "$FUNCBIND_DENSITY_ENCODER" \
  --voxbind-root "$VOXBIND_PYTHON_ROOT"
```

This downloads the public test split and deposited PDB/map, processes one density
box, runs three production-loop training steps in `bf16-mixed`, reloads the
model/EMA/optimizer checkpoint for one more training step, and samples one
latent with four diffusion steps before NF field decoding. It reuses the original
MCP reference files and pretrained NF/CDG weights. Outputs are `report.json`,
`sample.pt`, and a small-model checkpoint with optimizer shards. It also tests
CPU EMA and activation checkpointing. Use a fresh output directory to test fresh
downloads, or `--prepared-root <previous-smoke-output>` to reuse prepared data
without network access. Add `--devices 2` with prepared data to exercise ZeRO-1
and distributed training; the one-GPU variant cannot shard optimizer states.

The denoiser is reduced and randomly initialized. This smoke uses a test example
only to check the code path; it does not validate full-model training, atom/SDF
generation, or molecular quality, and its outputs are not evaluation results.

Verification (2026-09-15): 16 regression tests passed; 4 separate apo-fixture tests
were skipped. Two-process CPU and two-GPU AdamW/EMA/save-resume regressions passed.
On two RTX 3090s, the reduced MCP model completed 3 training updates, checkpoint
reload and update 4, then EMA sampling and NF decoding with real CDG v2/1bm2 data.
These used the existing SB environment with source mounts; rebuild the child
image for delivery. Full 5.14B H100 x8 training and molecular quality remain unverified.
