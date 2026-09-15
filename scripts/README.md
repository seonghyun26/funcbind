# VoxBind launch scripts

For MCP density-conditioned FuncBind, use only these public entrypoints:

1. `0_env_setup.sh`
2. `1_data_process.sh`
3. `2_train.sh`
4. `3_generate.sh`

See [`README_mcp_density.md`](README_mcp_density.md) for the short Docker workflow.

## VoxBind training helpers

Use two public launchers:

- `train_voxbind.sh` validates and launches frozen-density VoxBind training.
  It accepts a model-zoo folder name or explicit encoder paths, and supports
  new runs, exact resume segments, AdamW, and the opt-in Muon implementation.
- `wait_for_gpus.sh` waits for configurable physical GPUs and then executes any
  command as an argument array.

List the locally downloaded model-zoo entries:

```bash
scripts/train_voxbind.sh --list-models
```

Validate a training command without allocating a GPU:

```bash
scripts/train_voxbind.sh \
  --model-zoo champion_100m_v2_mask075 \
  --exp-name voxbind_champion_holo \
  --gpus 0,1,2,3 \
  --preflight
```

Wait and launch:

```bash
scripts/wait_for_gpus.sh --gpus 0,1,2,3 -- \
  scripts/train_voxbind.sh \
    --model-zoo champion_100m_v2_mask075 \
    --exp-name voxbind_champion_holo \
    --gpus 0,1,2,3
```

The coordinate-only model-zoo control has 11 input channels and is rejected
for holo-density VoxBind, which requires the full 13-channel encoder.

`supervise_voxbind_optimized_to_epoch500_v2.sh`,
`run_voxbind_optimized_segment_v2.sh`, and
`sample_voxbind_pareto_epoch500.sh` remain temporarily because the active
epoch-500 run depends on them. The segment script delegates to the generic
trainer.
