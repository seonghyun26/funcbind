#!/usr/bin/env bash
# Step 1 of the MCP density-conditioning recipe: get the data and weights onto a fresh box.
#
#   bash scripts/01_setup_mcp_density_data.sh
#   ASSETS_SRC=/mnt/share/funcbind_assets bash scripts/01_setup_mcp_density_data.sh   # from a local copy
#   SKIP_BUILD=1 bash scripts/01_setup_mcp_density_data.sh                            # fetch only
#
# WHAT IT PRODUCES
#   funcbind/dataset/data/mcpp_dataset/         MCP structures + {train,val,test}_data.pt
#   funcbind/dataset/data/mcpp_holo_xray_v1/    deposited 2Fo-Fc density, float16 memmap + manifest
#   exps/neural_field/nf_unified/               neural-field weights (decoder/encoder)
#   exps/funcbind/fb_unified/                   density-free MCP FuncBind, the fine-tune start point
#   $ENCODER_DST                                frozen atomblob7 v2.1 density encoder
#   $VOXBIND_ROOT                               VoxBind checkout -- the density branch imports
#                                               voxbind.models.density_vit at runtime
#
# WHERE THE BYTES COME FROM
#   * the maps themselves are PUBLIC and fetched by the build step: coordinates from RCSB,
#     2Fo-Fc CCP4 from PDBe. That part needs only outbound HTTPS, no credentials.
#   * the MCP dataset and the three checkpoints are NOT public. They come either from a
#     directory you staged yourself ($ASSETS_SRC) or from the lab Dropbox via rclone
#     (see notebook/html/dropbox-sync.md for configuring the `dropbox` remote).
#
# The build step is RESUMABLE and incremental: re-running skips targets already downloaded
# and built, so an interrupted run costs only the target in flight.
set -uo pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PY="${PY:-$REPO/.repro-env/bin/python}"
# The density build needs gemmi (+scipy/numpy) and NO torch, and gemmi is deliberately not
# in the training venv on every box -- on this one `.repro-env` has lightning+torch but no
# gemmi, while the voxbind conda env has gemmi but no lightning. So the build step gets its
# own interpreter knob. Point it at whichever env has gemmi, or pip install gemmi into $PY.
PREP_PY="${PREP_PY:-$PY}"
DATA="$REPO/funcbind/dataset/data"
VOXBIND_ROOT="${VOXBIND_ROOT:-$(dirname "$REPO")/VoxBind}"
VOXBIND_GIT="${VOXBIND_GIT:-}"                 # optional: git URL to clone if missing
ENCODER_DST="${ENCODER_DST:-$REPO/assets/density_encoder/atomblob7_v2p1_e0099.pth.tar}"
RECIPE="${RECIPE:-$REPO/funcbind/dataset/recipes/xray_resample_plinder_v2p1.json}"

# Staging source. Either a local/NFS directory holding the private assets, or empty to use
# rclone. Layout expected under $ASSETS_SRC (same names as the Dropbox folders):
#   mcpp_dataset/  nf_unified/  fb_unified/  atomblob7_v2p1_e0099.pth.tar
ASSETS_SRC="${ASSETS_SRC:-}"
RCLONE_REMOTE="${RCLONE_REMOTE:-dropbox:박성현/VoxBind}"

DOWNLOAD_WORKERS="${DOWNLOAD_WORKERS:-16}"
BUILD_WORKERS="${BUILD_WORKERS:-8}"
G_BOX="${G_BOX:-144}"
LIMIT="${LIMIT:-}"                             # e.g. LIMIT=5 for a smoke run

say() { echo "[$(date --iso-8601=seconds)] [01-setup] $*"; }
die() { say "ABORT: $*"; exit 1; }

say "repo=$REPO  voxbind=$VOXBIND_ROOT"
say "train py=$PY"
say "prep  py=$PREP_PY"
[ -x "$PY" ] || die "no python at $PY (set PY=... to your env's interpreter)"

# ── 0. python deps the build step needs ───────────────────────────────────────
say "checking python deps"
"$PY" - <<'EOF' || die "training interpreter is missing deps: $PY"
import importlib.util as u, sys
missing = [m for m in ("torch", "lightning") if not u.find_spec(m)]
print("  train env:", "ok" if not missing else f"MISSING {missing}")
sys.exit(1 if missing else 0)
EOF
"$PREP_PY" - <<'EOF' || die "the density-build interpreter is missing deps.
     Either  PREP_PY=/path/to/env/bin/python  (an env that has gemmi), or
             $PY -m pip install gemmi scipy numpy"
import importlib.util as u, sys
missing = [m for m in ("gemmi", "scipy", "numpy") if not u.find_spec(m)]
print("  prep env: ", "ok" if not missing else f"MISSING {missing}")
sys.exit(1 if missing else 0)
EOF

# ── 1. VoxBind checkout (density_vit lives there) ─────────────────────────────
if [ -f "$VOXBIND_ROOT/voxbind/models/density_vit.py" ]; then
    say "VoxBind checkout present"
elif [ -n "$VOXBIND_GIT" ]; then
    say "cloning VoxBind into $VOXBIND_ROOT"
    git clone "$VOXBIND_GIT" "$VOXBIND_ROOT" || die "clone failed"
else
    die "no VoxBind at $VOXBIND_ROOT — set VOXBIND_ROOT, or VOXBIND_GIT to clone it.
     The frozen density encoder is a voxbind.models.density_vit.DensityViT, imported at
     runtime through denoiser.density.voxbind_python_root; without it training cannot start."
fi

# ── 2. private assets ─────────────────────────────────────────────────────────
fetch() {  # fetch <name> <dest>
    local name="$1" dest="$2"
    if [ -e "$dest" ]; then say "have $name"; return 0; fi
    mkdir -p "$(dirname "$dest")"
    if [ -n "$ASSETS_SRC" ]; then
        [ -e "$ASSETS_SRC/$name" ] || die "$ASSETS_SRC/$name not found"
        say "copying $name from $ASSETS_SRC"
        cp -r "$ASSETS_SRC/$name" "$dest"
    else
        command -v rclone >/dev/null || die "rclone not installed and ASSETS_SRC unset — see notebook/html/dropbox-sync.md"
        say "rclone copy $name"
        if [ -d "$dest" ] || [ "${name%.tar}" != "$name" ] || [[ "$name" != *.* ]]; then
            rclone copy "$RCLONE_REMOTE/$name" "$dest" --progress || die "rclone failed for $name"
        else
            rclone copyto "$RCLONE_REMOTE/$name" "$dest" --progress || die "rclone failed for $name"
        fi
    fi
}

fetch mcpp_dataset "$DATA/mcpp_dataset"
fetch nf_unified   "$REPO/exps/neural_field/nf_unified"
fetch fb_unified   "$REPO/exps/funcbind/fb_unified"
fetch model_zoo/atomblob7_v2p1_e0099.pth.tar "$ENCODER_DST"

[ -f "$DATA/mcpp_dataset/train_data.pt" ] || die "mcpp_dataset looks wrong: no train_data.pt"
[ -f "$REPO/exps/funcbind/fb_unified/checkpoint.pth.tar" ] || die "fb_unified has no checkpoint.pth.tar"
[ -f "$ENCODER_DST" ] || die "no density encoder at $ENCODER_DST"

# ── 3. normalization recipe ───────────────────────────────────────────────────
# These constants must be the ones the encoder was pretrained with; a differently
# normalized map is a different input distribution to a frozen trunk.
[ -f "$RECIPE" ] || die "normalization recipe missing: $RECIPE"
say "normalization: $("$PREP_PY" -c "import json;n=json.load(open('$RECIPE'))['normalization'];print(n['scheme'])")"

# ── 4. build the holo-density set ─────────────────────────────────────────────
if [ -n "${SKIP_BUILD:-}" ]; then
    say "SKIP_BUILD set — stopping after the fetch"; exit 0
fi
OUT="$DATA/mcpp_holo_xray_v1"
if [ -f "$OUT/.complete" ]; then
    say "holo-density already built: $("$PREP_PY" -c "import json;d=json.load(open('$OUT/.complete'));print(f\"{d['n_available']} of {d['n_targets']} targets\")")"
    say "delete $OUT/.complete to force a rebuild"
else
    say "building holo-density (RCSB coords + PDBe 2Fo-Fc; resumable, hours on a cold cache)"
    "$PREP_PY" "$REPO/funcbind/dataset/prepare_mcpp_holo_density.py" \
        --data-root "$DATA/mcpp_dataset" \
        --out-dir "$OUT" \
        --normalization-recipe "$RECIPE" \
        --g-box "$G_BOX" \
        --download-workers "$DOWNLOAD_WORKERS" \
        --build-workers "$BUILD_WORKERS" \
        ${LIMIT:+--limit "$LIMIT"} || die "density build failed"
fi

say "done. next: scripts/02_train_mcp_density_conditioned.sh"
say "run scripts/preflight_mcp_density.py first — on 80 GB cards it will tell you the recipe does not fit as-is"
