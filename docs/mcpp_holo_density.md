# MCP receptor-level holo X-ray density

This pipeline conditions FuncBind MCP generation on the deposited receptor's original
2Fo-Fc density. It is intentionally a **holo** experiment: the density of the bound
reference CP ligand remains in the map. No ligand mask, apo-like erasure, or synthetic
density is used.

## Availability and coordinate frame

For the current MCP corpus:

- raw targets: 643 total; 593 X-ray and 50 cryo-EM;
- official PDBe 2Fo-Fc maps: 559/643 targets;
- alignment-validated density boxes: 557/643 targets;
- FuncBind split targets: 555/641 have usable boxes;
- train/val/test target coverage: 377/456, 83/85, and 95/100;
- train conformer-weighted coverage: 147,709/186,528 (79.19%).

Most MCP receptor PDBs use the deposited crystallographic frame. The builder fits
and records a target-level heavy-atom Kabsch transform, rejects alignments above 1 Å
RMSD, and records raw density sampled at reference-ligand atoms as a diagnostic. The
validated transform RMSD is 0.000 Å median, 0.462 Å at p95, and 0.925 Å maximum;
99.82% of usable targets have positive mean raw density at reference-ligand atoms.
Two map-bearing train targets (`4uog`, `7p2g`) do not admit a reliable rigid receptor
alignment and are deliberately unavailable rather than being force-aligned.

## Build data

From the repository root:

```bash
python -m funcbind.dataset.prepare_mcpp_holo_density
```

The default output is `funcbind/dataset/data/mcpp_holo_xray_v1`. The job is resumable
and produces:

- `pdb/` and `ccp4/`: official deposited coordinate/map files;
- `boxes_float16.dat`: one raw 144³ × 0.25 Å box per receptor target;
- `manifest.json`: target-to-box mapping, CP reference, alignment, and diagnostics;
- `meta.json`: density semantics, normalization, source URLs, and coverage.

The 36 Å target box is cropped to the pretrained encoder's 64³ × 0.25 Å field at read
time. This avoids duplicating a crop for roughly 148k train conformers. A conformer whose
requested rotated crop does not fit completely in the cache is marked unavailable; its
density residual is exactly zero rather than receiving a partly padded volume.

A small end-to-end build can be checked first:

```bash
python -m funcbind.dataset.prepare_mcpp_holo_density \
  --out-dir /tmp/mcpp_holo_xray_smoke --limit 2
```

## Train and sample

```bash
python funcbind/train_fb.py --config-name train_fb_mcpp_holo_density
python funcbind/sample_fb.py --config-name sample_fb_mcpp_holo_density \
  fb_pretrained_path=exps/funcbind/<density-checkpoint-dir>
```

The model input is the same 13-channel representation as the selected VoxBind encoder:
seven zero ligand channels, four receptor atom-blob channels, normalized 2Fo-Fc density,
and density gradient magnitude. The encoder is frozen. Its output reaches only the
FuncBind receptor latent through a zero-initialized final 1×1×1 convolution. Therefore
the first density-conditioned forward pass is exactly equal to the pretrained baseline.
The per-sample availability mask remains after the projection, so missing maps remain an
exact no-op even after the projection learns a bias.

Sampling precomputes the fused receptor latent once, outside the diffusion loop.
`sampling.rotate_receptor` must remain false because the dataset crop is already aligned
to the receptor frame.

## Show that generation is not copying the visible reference ligand

Use the deposited CP ligand recorded in the manifest, not the mutant/conformer used as a
FuncBind label:

```bash
python -m funcbind.metrics.evaluate_mcpp_reference_similarity \
  --manifest funcbind/dataset/data/mcpp_holo_xray_v1/manifest.json \
  --target-id 6xif \
  --generated exps/funcbind/<run>/samples/target_*/molecules*.pdb \
  --output-prefix exps/funcbind/<run>/reference_similarity_6xif
```

Report the full distributions (and paired density-free baseline), including Morgan
Tanimoto mean/median/max, fraction below 0.4, best forward/reverse residue identity, and
fraction below 0.5. These are novelty/leakage diagnostics; binding, interface, validity,
and diversity metrics should still be reported separately.
