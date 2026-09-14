#!/usr/bin/env bash
# Wire a checkout to the heavy directories it cannot carry: the dataset (173 GB) and the
# experiment tree (1.6 TB). Link them if they already exist on this box, fetch them if not.
# Idempotent, and it never touches a real directory it finds in place.
#
#   bash scripts/00_link_or_fetch_paths.sh                 # link what exists, fetch what doesn't
#   DRY_RUN=1 bash scripts/00_link_or_fetch_paths.sh       # say what it would do, change nothing
#   NO_FETCH=1 bash scripts/00_link_or_fetch_paths.sh      # link only; report gaps and exit 1
#   FORCE=1 bash scripts/00_link_or_fetch_paths.sh         # repoint a symlink that points elsewhere
#   FB_SRC_ROOT=/path/to/old/checkout bash scripts/00_link_or_fetch_paths.sh
#
# WHY
#   Experiments are written to $EXPS_DIR and read back by name on resume, so a checkout that
#   resolves it to a different place silently starts a fresh run instead of continuing one.
#   This script makes that link explicit and checkable, rather than implicit in a cwd.
#
# LAYOUT
#   data   <repo>/funcbind/dataset/data     <- the mcpp_dataset + mcpp_holo_xray_v1 trees
#   exps   <repo>/funcbind/exps             <- training runs, mirroring voxbind/exps
#          (a legacy checkout keeps exps at <repo>/exps; that is auto-detected and kept)
#
# FETCHING
#   Missing pieces are handed to scripts/01_setup_mcp_density_data.sh, which pulls the public
#   maps (RCSB + PDBe) itself and the private assets from $ASSETS_SRC or the lab Dropbox via
#   rclone. Every knob that script takes (ASSETS_SRC, PY, PREP_PY, LIMIT, ...) passes through.
set -uo pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DRY_RUN="${DRY_RUN:-}"
NO_FETCH="${NO_FETCH:-}"
FORCE="${FORCE:-}"

# Where the bytes already are on this box (the checkout that owns them today).
FB_SRC_ROOT="${FB_SRC_ROOT:-/home1/irteam/funcbind}"
DATA_SRC="${FB_DATA_SRC:-$FB_SRC_ROOT/funcbind/dataset/data}"
EXPS_SRC="${FB_EXPS_SRC:-$FB_SRC_ROOT/exps}"

DATA_DIR="${FB_DATA_DIR:-$REPO/funcbind/dataset/data}"
# A legacy checkout keeps runs in <repo>/exps. Honour that rather than splitting the tree.
if [ -z "${FB_EXPS_DIR:-}" ] && [ -e "$REPO/exps" ]; then
    EXPS_DIR="$REPO/exps"
else
    EXPS_DIR="${FB_EXPS_DIR:-$REPO/funcbind/exps}"
fi

say()  { echo "[$(date --iso-8601=seconds)] [00-paths] $*"; }
die()  { say "ABORT: $*"; exit 1; }
run()  { if [ -n "$DRY_RUN" ]; then say "DRY_RUN: $*"; else "$@"; fi; }

need_fetch=0

# link_or_gap <label> <target> <source>
#   returns 0 if the target is usable afterwards, 1 if it still has to be fetched
link_or_gap() {
    local label="$1" target="$2" src="$3"

    if [ -L "$target" ]; then
        local cur; cur="$(readlink -f "$target" 2>/dev/null || true)"
        if [ -d "$cur" ]; then
            if [ -n "$FORCE" ] && [ -d "$src" ] && [ "$cur" != "$(readlink -f "$src")" ]; then
                say "$label: repointing symlink ($cur -> $src)"
                run rm -f "$target"
            else
                say "$label: symlink OK -> $cur"
                return 0
            fi
        else
            say "$label: BROKEN symlink -> ${cur:-?}; replacing"
            run rm -f "$target"
        fi
    elif [ -d "$target" ]; then
        # A real directory is someone's data. Never replace it, even with FORCE.
        say "$label: real directory in place, leaving it ($(du -sh "$target" 2>/dev/null | cut -f1))"
        return 0
    fi

    if [ -d "$src" ]; then
        say "$label: linking -> $src"
        run mkdir -p "$(dirname "$target")"
        run ln -s "$src" "$target"
        return 0
    fi

    say "$label: MISSING and no source at $src"
    return 1
}

say "repo=$REPO"
say "data: $DATA_DIR"
say "exps: $EXPS_DIR"

link_or_gap "data" "$DATA_DIR" "$DATA_SRC" || need_fetch=1
link_or_gap "exps" "$EXPS_DIR" "$EXPS_SRC" || need_fetch=1

if [ "$need_fetch" -eq 0 ]; then
    say "both paths are in place"
    [ -n "$DRY_RUN" ] || {
        # Resolve through the link so a later run fails loudly here rather than mid-epoch.
        [ -d "$DATA_DIR/mcpp_dataset" ] || say "NOTE: $DATA_DIR has no mcpp_dataset/ — run 01 to build it"
    }
    exit 0
fi

if [ -n "$NO_FETCH" ]; then
    say "NO_FETCH set — stopping with the gaps above"
    exit 1
fi

# Nothing to link: create the destinations and let step 01 fill them. It fetches the public
# maps itself and the private assets from ASSETS_SRC/rclone, and it is resumable.
say "fetching the missing pieces via scripts/01_setup_mcp_density_data.sh"
run mkdir -p "$DATA_DIR" "$EXPS_DIR"
if [ -n "$DRY_RUN" ]; then
    say "DRY_RUN: REPO=$REPO bash $REPO/scripts/01_setup_mcp_density_data.sh"
    exit 0
fi
REPO="$REPO" bash "$REPO/scripts/01_setup_mcp_density_data.sh" || die "01_setup_mcp_density_data.sh failed"
say "done"
