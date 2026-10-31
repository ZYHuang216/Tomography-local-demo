#!/usr/bin/env bash
# Reconstruct slices of the FleX-ray rat skull on demand (cone-beam FDK, CPU).
#   ./run_skull.sh                          3D viewer, binning 4, every 2nd projection
#   ./run_skull.sh --sample 2 --skip 1      sharper, slower, more RAM
#   ./run_skull.sh --viewer none --save-slices results --snapshot results/skull_3d.png
#   ./run_skull.sh --auto-cor               refine the rotation centre (+-3 px)
#   ./run_skull.sh --full-box               whole field of view instead of a box around the skull
# Data are searched in ./skull_image_downloaded/skull and ./data/skull
# (or set SKULL_DIR=/path/to/folder/with/scan_000000.tif).
set -euo pipefail
cd "$(dirname "$0")"
dir="${SKULL_DIR:-}"
if [ -z "$dir" ]; then
  for d in skull_image_downloaded/skull data/skull; do
    if ls "$d"/scan_0*.tif >/dev/null 2>&1; then dir="$d"; break; fi
  done
fi
if [ -z "$dir" ]; then
  first=$(find . -path ./.venv -prune -o -name 'scan_000000.tif' -print 2>/dev/null | head -n 1 || true)
  [ -n "$first" ] && dir=$(dirname "$first")
fi
if [ -z "$dir" ]; then
  echo "No FleX-ray projections (scan_000000.tif) found - download doi:10.5281/zenodo.1164088 into skull_image_downloaded/skull or set SKULL_DIR." >&2
  exit 1
fi
echo ">> data folder: $dir"
exec .venv/bin/python -m minirecast flexray "$dir" "$@"
