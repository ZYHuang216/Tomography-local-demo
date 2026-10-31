#!/usr/bin/env bash
# 3D model of the rat skull: full-volume reconstruction -> bone surface.
# Opens an interactive 3D viewer and saves the mesh (STL, mm) and volume (TIFF stack).
#   ./model_skull.sh                          165 um voxels (--sample 2, all views)
#   ./model_skull.sh --sample 1 --voxel 0.1   finest detail (more RAM/time)
#   ./model_skull.sh --iso 0.2                set the bone threshold yourself
set -euo pipefail
cd "$(dirname "$0")"
dir="${SKULL_DIR:-skull_image_downloaded/skull}"
mkdir -p results/model
exec .venv/bin/python -m minirecast model flexray "$dir" \
     --sample 2 --skip 1 --filter shepp-logan \
     --save-mesh results/model/skull.stl --save-volume results/model/skull_volume.tif \
     --snapshot results/model/skull_model.png "$@"
