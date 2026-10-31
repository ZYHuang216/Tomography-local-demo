#!/usr/bin/env bash
# Simulated live scan of the rat skull: projections arrive in groups and the
# slices update as data comes in (RECAST3D's main use case).
#   ./live_skull.sh                          interleaved order, 100 projections/s
#   ./live_skull.sh --order sequential       continuous rotation order
#   ./live_skull.sh --rate 30 --group 10     slower, smaller groups
set -euo pipefail
cd "$(dirname "$0")"
exec ./run_skull.sh --stream --order interleaved --rate 100 --group 20 "$@"
