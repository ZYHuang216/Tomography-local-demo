#!/usr/bin/env bash
# Full-resolution reconstruction of the rat skull: no software binning (82.5 um
# voxels), all 1200 projections, sharp Ram-Lak filter, 640x640 slices.
# Needs ~3 GB RAM. Add --gaussian 0.7 if it looks too noisy.
set -euo pipefail
cd "$(dirname "$0")"
exec ./run_skull.sh --sample 1 --skip 1 --filter ram-lak --res 640 --preview-res 160 "$@"
