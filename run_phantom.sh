#!/usr/bin/env bash
# Optional: synthetic phantom (no files written). The main demo is ./run_skull.sh
#   ./run_phantom.sh                      parallel beam (synchrotron-like)
#   ./run_phantom.sh --geometry cone      cone beam (lab micro-CT-like)
#   ./run_phantom.sh --angles 21          few-projection test (cf. limited-tilt ET)
#   ./run_phantom.sh --photons 2000       add Poisson noise
#   ./run_phantom.sh --viewer mpl         2D matplotlib viewer instead of 3D
set -euo pipefail
cd "$(dirname "$0")"
exec .venv/bin/python -m minirecast phantom "$@"
