#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [[ ! -f "README.md" || ! -f "frontend/app.py" ]]; then
  echo "ERROR: Run this script from the repository root (missing README.md or frontend/app.py)." >&2
  exit 1
fi

echo "Cleaning legacy/generated files from: $ROOT_DIR"

# Legacy folders from old architecture (safe to remove if present)
rm -rf "dashboard" "notebooks" "results" 2>/dev/null || true

# Python caches
find . -type d -name "__pycache__" -prune -exec rm -rf {} + 2>/dev/null || true
rm -rf ".pytest_cache" ".mypy_cache" ".ruff_cache" 2>/dev/null || true

# Common generated artifacts
rm -f ./.coverage 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true

echo "Done."
