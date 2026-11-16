#!/usr/bin/env bash
# Run a command inside the FEniCSx container with this folder mounted at /work.
# Usage: ./run.sh python3 scripts/single_cut_buckling.py
# The FFCx JIT cache is kept in .cache/ so compiled forms survive between runs.
cd "$(dirname "$0")"
HERE="$(pwd -W 2>/dev/null || pwd)"
mkdir -p .cache
MSYS_NO_PATHCONV=1 exec docker run --rm --init --label kirigami=1 \
  -v "$HERE:/work" -v "$HERE/.cache:/root/.cache/fenics" -w /work \
  -e PYTHONPATH=/work:/usr/local/dolfinx-real/lib/python3.12/dist-packages:/usr/local/lib \
  -e OMP_NUM_THREADS=1 dolfinx/dolfinx:v0.9.0 "$@"
