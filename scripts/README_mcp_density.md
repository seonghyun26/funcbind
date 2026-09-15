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

Prepare persistent directories and optional Dropbox shared-file URLs:

```bash
export NF_MODEL_URL='.../model.pt?dl=0'
export FB_MODEL_URL='.../checkpoint.pth.tar?dl=0'
export CDG_MODEL_URL='.../checkpoint_e0025.pth.tar?dl=0'

docker run --rm -it --gpus all --shm-size=32g \
  -e NF_MODEL_URL -e FB_MODEL_URL -e CDG_MODEL_URL \
  -e MCP_MODEL_URL -e MCP_MODEL_SHA256 \
  -v /path/to/funcbind-data:/workspace/FuncBind/funcbind/dataset/data \
  -v /path/to/funcbind-exps:/workspace/FuncBind/exps \
  -v /path/to/funcbind-artifacts:/workspace/FuncBind/artifacts \
  -v /path/to/voxbind-exps:/workspace/VoxBind/voxbind/exps \
  voxbind-funcbind:sb
```

Dropbox shared links need no rclone configuration. If a URL is omitted, the
data script falls back to `ASSETS_SRC`, then a configured `rclone` remote.

## 1. Download and process data

```bash
bash scripts/1_data_process.sh
```

This downloads the public MCP splits and original structures, then downloads
RCSB coordinates and PDBe 2Fo-Fc maps and builds the X-ray density cache. It is
resumable and refuses downloads that violate its free-space safety margin.

## 2. Fine-tune

```bash
SMOKE=1 bash scripts/2_train.sh
bash scripts/2_train.sh
```

The default uses `bf16-mixed` on GPUs `0-7`, with train and validation batch
size 1, accumulation 95 (effective batch 760), eager execution, and non-foreach
AdamW. Model weights, gradients, optimizer moments, and EMA remain FP32.

**H100 80GB is blocked with the current plain DDP strategy.** Static state alone
requires about 95.8 GiB per GPU before activations. DDP replicates that state on
all eight GPUs, so lowering batch size cannot make it fit. Keeping mixed
precision requires implementing FSDP/ZeRO sharding or CPU offload; this workflow
does not yet provide either. The preflight will reject an H100 launch.

On hardware with enough memory, use `SMOKE=1 bash scripts/2_train.sh` first;
it runs one tiny epoch without writing the large training checkpoint.

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
box, runs three production-loop training steps in `bf16-mixed`, and samples one
latent with four diffusion steps before NF field decoding. It reuses the original
MCP reference files and pretrained NF/CDG weights. Outputs are `report.json` and
`sample.pt`. Use a fresh output directory to test fresh downloads.

The denoiser is reduced and randomly initialized. This smoke uses a test example
only to check the code path; it does not validate full-model training, atom/SDF
generation, or molecular quality, and its outputs are not evaluation results.
