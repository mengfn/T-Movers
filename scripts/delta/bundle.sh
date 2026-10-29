#!/usr/bin/env bash
# Copies the files a training sandbox needs into dist/delta-bundle, for
#   delta-cli sandbox upload <id> --source dist/delta-bundle --target project
set -euo pipefail
cd "$(dirname "$0")/../.."
rm -rf dist/delta-bundle
mkdir -p dist/delta-bundle
cp -R pyproject.toml uv.lock README.md src scripts tests dist/delta-bundle/
find dist/delta-bundle -name __pycache__ -type d -prune -exec rm -rf {} +
du -sh dist/delta-bundle
