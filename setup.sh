#!/usr/bin/env bash
# Create a local Python environment for the macOS (CPU-only) tomography demo.
set -euo pipefail
cd "$(dirname "$0")"

pick_python() {
  for v in 3.12 3.11 3.13 3.10 3.14; do
    if command -v "python$v" >/dev/null 2>&1; then command -v "python$v"; return; fi
    for p in "$HOME"/.pyenv/versions/"$v"*/bin/python; do
      if [ -x "$p" ]; then echo "$p"; return; fi
    done
  done
  command -v python3
}

PY="${PYTHON:-$(pick_python)}"
echo ">> using $PY ($("$PY" --version 2>&1))"
"$PY" - <<'EOF'
import sys
if sys.version_info < (3, 10):
    sys.exit("Python >= 3.10 is needed (e.g. `pyenv install 3.12`), then rerun ./setup.sh "
             "or PYTHON=/path/to/python3.12 ./setup.sh")
EOF

[ -d .venv ] || "$PY" -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt

echo ">> self-test"
.venv/bin/python -m minirecast selftest
echo
echo "Done. Next:  ./run_skull.sh        (FleX-ray rat skull, 3D slice viewer)"
