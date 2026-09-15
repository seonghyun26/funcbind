#!/usr/bin/env bash
# Download MCP data/weights and build deposited X-ray density boxes.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"

export REPO
export FUNCBIND_ROOT="${FUNCBIND_ROOT:-$REPO}"
export VOXBIND_ROOT="${VOXBIND_ROOT:-$(cd "$REPO/.." && pwd)}"
# A full data preparation needs the public original-structure archive. Set both
# values to 0 only when reusing an already-extracted structure tree.
export MCPP_INCLUDE_ORIGINAL="${MCPP_INCLUDE_ORIGINAL:-1}"
export MCPP_EXTRACT_ORIGINAL="${MCPP_EXTRACT_ORIGINAL:-$MCPP_INCLUDE_ORIGINAL}"

if [ -n "${DRY_RUN:-}" ]; then
    printf 'DRY_RUN: include_original=%s extract_original=%s\n' \
        "$MCPP_INCLUDE_ORIGINAL" "$MCPP_EXTRACT_ORIGINAL"
    printf 'DRY_RUN: %q\n' "$HERE/01_setup_mcp_density_data.sh"
    exit 0
fi

exec "$HERE/01_setup_mcp_density_data.sh"
