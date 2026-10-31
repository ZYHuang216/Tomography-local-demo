#!/usr/bin/env bash
# RECAST3D-style pipeline in three processes on this Mac:
#   adapter (push) --projections--> server (serve) <--slices--> viewer (view)
# On a real setup run the server next to the data/GPU and tunnel the port:
#   ssh -L 5558:localhost:5558 user@compute-node
set -euo pipefail
cd "$(dirname "$0")"
PORT="${PORT:-5558}"
PY=.venv/bin/python
dir="${SKULL_DIR:-skull_image_downloaded/skull}"
export MINIRECAST_KEY="${MINIRECAST_KEY:-minirecast-$RANDOM-$RANDOM}"   # shared secret

$PY -m minirecast serve --port "$PORT" &
SERVER=$!
trap 'kill $SERVER 2>/dev/null || true' EXIT
sleep 1
$PY -m minirecast push --server "localhost:$PORT" flexray "$dir" \
    --order interleaved --rate 100 --group 20 &
$PY -m minirecast view --server "localhost:$PORT" "$@"
