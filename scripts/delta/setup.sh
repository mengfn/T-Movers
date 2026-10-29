#!/usr/bin/env bash
# Sets up the project inside a Delta GPU sandbox (run from the bundle root).
# Picks the JAX CUDA build the GPU driver supports and installs the exact
# versions from uv.lock into .venv. Downloads go through the sandbox's pip
# index when one is configured: some providers (e.g. AutoDL) reach PyPI only
# through a mirror.
set -euo pipefail
cd "$(dirname "$0")/../.."

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
driver_major=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader |
               head -1 | cut -d. -f1)
extra=cuda12
if [ "$driver_major" -ge 580 ]; then extra=cuda13; fi
echo "driver $driver_major -> jax[$extra]"

if ! command -v uv >/dev/null; then
  python3 -m pip install --quiet uv || curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
# TMOVERS_INDEX overrides the image's index (its default mirror can be slow).
index=${TMOVERS_INDEX:-$(python3 -m pip config get global.index-url 2>/dev/null || true)}
echo "package index: ${index:-PyPI}"

uv export --frozen --no-dev --extra "$extra" --no-hashes --no-emit-project \
  -o /tmp/requirements.txt
# Some images ship an older python3; uv fetches 3.12 when it is missing.
[ -x .venv/bin/python ] || uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r /tmp/requirements.txt \
  ${index:+--index-url "$index"}
uv pip install --python .venv/bin/python --no-deps -e .
.venv/bin/python -c "import jax; print('jax', jax.__version__, jax.devices())"
