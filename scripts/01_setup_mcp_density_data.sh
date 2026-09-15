#!/usr/bin/env bash
# Step 1 of the MCP density-conditioning recipe: get the data and weights onto a fresh box.
#
#   bash scripts/01_setup_mcp_density_data.sh
#   ASSETS_SRC=/mnt/share/funcbind_assets bash scripts/01_setup_mcp_density_data.sh   # from a local copy
#   SKIP_BUILD=1 bash scripts/01_setup_mcp_density_data.sh                            # fetch only
#   CHECK_LINKS_ONLY=1 bash scripts/1_data_process.sh   # 32 bytes/model; no files saved
#   MCPP_INCLUDE_ORIGINAL=1 MCPP_EXTRACT_ORIGINAL=1 bash scripts/01_setup_mcp_density_data.sh
#       # also fetch/extract the 32.7 GB public raw archive (requires much more free space)
#
# WHAT IT PRODUCES
#   funcbind/dataset/data/mcpp_dataset/         MCP structures + {train,val,test}_data.pt
#   funcbind/dataset/data/mcpp_holo_xray_v1/    deposited 2Fo-Fc density, float16 memmap + manifest
#   exps/neural_field/nf_unified/               neural-field weights (decoder/encoder)
#   exps/funcbind/fb_unified/                   density-free MCP FuncBind, the fine-tune start point
#   $ENCODER_DST                                frozen CDG v2 epoch-25 density encoder
#   $VOXBIND_ROOT                               VoxBind checkout -- the density branch imports
#                                               voxbind.models.density_vit at runtime
#
# WHERE THE BYTES COME FROM
#   * the maps themselves are PUBLIC and fetched by the build step: coordinates from RCSB,
#     2Fo-Fc CCP4 from PDBe. That part needs only outbound HTTPS, no credentials.
#   * the prepared MCP splits and original structure archive are public on Hugging Face.
#     The three checkpoints come from individual shared links (NF_MODEL_URL,
#     FB_MODEL_URL, CDG_MODEL_URL), a directory staged in ASSETS_SRC, or the lab
#     Dropbox through rclone as a final fallback.
#
# The build step is RESUMABLE and incremental: re-running skips targets already downloaded
# and built, so an interrupted run costs only the target in flight.
set -uo pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PY="${PY:-$REPO/.repro-env/bin/python}"
# The Docker environment contains both training and density-build dependencies. PREP_PY
# remains overridable for older local environments that keep gemmi in a separate env.
PREP_PY="${PREP_PY:-$PY}"
DATA="$REPO/funcbind/dataset/data"
VOXBIND_ROOT="${VOXBIND_ROOT:-$(dirname "$REPO")/VoxBind}"
VOXBIND_GIT="${VOXBIND_GIT:-}"                 # optional: git URL to clone if missing
ENCODER_DST="${ENCODER_DST:-$VOXBIND_ROOT/voxbind/exps/260806_cdg_100m_v2_ep100/checkpoint_e0025.pth.tar}"
RECIPE="${RECIPE:-$REPO/funcbind/dataset/recipes/xray_resample_plinder_v2_perelem.json}"

# Staging source. Layout expected under $ASSETS_SRC (same names as the Dropbox folders):
#   nf_unified/model.pt
#   fb_unified/checkpoint.pth.tar
#   results/task1-affinity/CDG-v2/checkpoint_e0025.pth.tar
ASSETS_SRC="${ASSETS_SRC:-}"
RCLONE_REMOTE="${RCLONE_REMOTE:-dropbox:박성현/VoxBind}"

DOWNLOAD_WORKERS="${DOWNLOAD_WORKERS:-16}"
BUILD_WORKERS="${BUILD_WORKERS:-8}"
G_BOX="${G_BOX:-144}"
LIMIT="${LIMIT:-}"                             # e.g. LIMIT=5 for a smoke run

say() { echo "[$(date --iso-8601=seconds)] [01-setup] $*"; }
die() { say "ABORT: $*"; exit 1; }

# Exit before dependency checks, dataset downloads, directory creation, or
# existing-file shortcuts: this mode verifies the actual configured remote URLs.
case "${CHECK_LINKS_ONLY:-0}" in
    1|true|yes)
        exec "$PY" "$REPO/scripts/check_mcp_asset_links.py"
        ;;
    0|false|no|'') ;;
    *) die "CHECK_LINKS_ONLY must be 0 or 1" ;;
esac

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

# ── 2. public splits + private weights ────────────────────────────────────────
download_shared_file() {  # download_shared_file <url> <dest> [sha256]
    local url="$1" dest="$2" expected_sha="${3:-}" part
    command -v curl >/dev/null || die "curl is required for shared-link downloads"
    if [[ "$url" == *dropbox.com* ]]; then
        if [[ "$url" == *dl=0* ]]; then
            url="${url//dl=0/dl=1}"
        elif [[ "$url" != *dl=1* ]]; then
            [[ "$url" == *\?* ]] && url="${url}&dl=1" || url="${url}?dl=1"
        fi
    fi
    part="$dest.part"
    say "downloading $(basename "$dest") from shared link (URL hidden)"
    curl --fail --location --continue-at - --retry 5 --retry-all-errors \
        --retry-delay 5 --output "$part" "$url" || die "shared-link download failed: $dest"
    [ -s "$part" ] || die "downloaded an empty file: $part"
    if [ -n "$expected_sha" ]; then
        printf '%s  %s\n' "$expected_sha" "$part" | sha256sum --check - \
            || die "SHA-256 verification failed: $dest"
    fi
    mv "$part" "$dest"
}

fetch_file() {  # fetch_file <relative-name> <dest-file> [shared-url] [sha256]
    local name="$1" dest="$2" url="${3:-}" expected_sha="${4:-}"
    if [ -s "$dest" ]; then say "have $name"; return 0; fi
    mkdir -p "$(dirname "$dest")"
    if [ -n "$ASSETS_SRC" ]; then
        [ -f "$ASSETS_SRC/$name" ] || die "$ASSETS_SRC/$name not found"
        say "copying $name from $ASSETS_SRC"
        cp "$ASSETS_SRC/$name" "$dest"
    elif [ -n "$url" ]; then
        download_shared_file "$url" "$dest" "$expected_sha"
    else
        command -v rclone >/dev/null || die "no shared URL/ASSETS_SRC and rclone is not configured for $name"
        say "rclone copyto $name"
        rclone copyto "$RCLONE_REMOTE/$name" "$dest" --progress || die "rclone failed for $name"
    fi
    [ -s "$dest" ] || die "asset is missing or empty after fetch: $dest"
}

download_args=(--dest "$DATA/mcpp_dataset")
case "${MCPP_INCLUDE_ORIGINAL:-0}" in
    0|false|no|'') ;;
    1|true|yes) download_args+=(--include-original) ;;
    *) die "MCPP_INCLUDE_ORIGINAL must be 0 or 1" ;;
esac
case "${MCPP_EXTRACT_ORIGINAL:-0}" in
    0|false|no|'') ;;
    1|true|yes) download_args+=(--extract-original) ;;
    *) die "MCPP_EXTRACT_ORIGINAL must be 0 or 1" ;;
esac
say "checking/downloading public MCP data from Hugging Face"
"$PY" "$REPO/scripts/download_mcpp_data.py" "${download_args[@]}" || die "MCP download failed"

fetch_file nf_unified/model.pt \
    "$REPO/exps/neural_field/nf_unified/model.pt" \
    "${NF_MODEL_URL:-}" "${NF_MODEL_SHA256:-}"
fetch_file fb_unified/checkpoint.pth.tar \
    "$REPO/exps/funcbind/fb_unified/checkpoint.pth.tar" \
    "${FB_MODEL_URL:-}" "${FB_MODEL_SHA256:-}"
fetch_file results/task1-affinity/CDG-v2/checkpoint_e0025.pth.tar \
    "$ENCODER_DST" "${CDG_MODEL_URL:-}" "${CDG_MODEL_SHA256:-}"

[ -f "$DATA/mcpp_dataset/train_data.pt" ] || die "mcpp_dataset looks wrong: no train_data.pt"
[ -f "$REPO/exps/funcbind/fb_unified/checkpoint.pth.tar" ] || die "fb_unified has no checkpoint.pth.tar"
[ -f "$ENCODER_DST" ] || die "no density encoder at $ENCODER_DST"
export FUNCBIND_DENSITY_ENCODER="$ENCODER_DST"
export VOXBIND_PYTHON_ROOT="$VOXBIND_ROOT"

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

say "done. next: scripts/2_train.sh"
say "run scripts/preflight_mcp_density.py and the H100 smoke test before starting the full job"
