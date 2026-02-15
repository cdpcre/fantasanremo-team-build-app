#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${1:-/tmp/fantasanremo_team_builder_public}"
COMMIT_MESSAGE="${2:-chore: initial public release}"

if [ "$OUTPUT_DIR" = "$ROOT_DIR" ]; then
  echo "Output directory must be different from project root." >&2
  exit 1
fi

echo "Preparing clean public repo in: $OUTPUT_DIR"
rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

rsync -a \
  --exclude '.git/' \
  --exclude '.claude/' \
  --exclude '.playwright-mcp/' \
  --exclude '.venv/' \
  --exclude '.uv_cache/' \
  --exclude '.ruff_cache/' \
  --exclude '.pytest_cache/' \
  --exclude '.cache/' \
  --exclude 'node_modules/' \
  --exclude 'frontend/node_modules/' \
  --exclude 'frontend/dist/' \
  --exclude 'frontend/build/' \
  --exclude 'frontend/public/data/artist_images/' \
  --exclude 'vercel_app/' \
  --exclude 'data/backups/' \
  --exclude 'logs/' \
  --exclude '*.log' \
  --exclude '*.pid' \
  --exclude '.env' \
  --exclude '.env.local' \
  --exclude '.env.*.local' \
  --exclude '*.env' \
  --exclude 'CLAUDE.md' \
  --exclude '.note' \
  --exclude '.backend.pid' \
  --exclude '.frontend.pid' \
  --exclude '.DS_Store' \
  --exclude 'db/*.db' \
  --exclude 'db/*.sqlite' \
  --exclude 'db/*.sqlite3' \
  "$ROOT_DIR/" "$OUTPUT_DIR/"

git -C "$OUTPUT_DIR" init -b main >/dev/null 2>&1 || {
  git -C "$OUTPUT_DIR" init >/dev/null
  git -C "$OUTPUT_DIR" branch -M main
}

git -C "$OUTPUT_DIR" add .
git -C "$OUTPUT_DIR" commit -m "$COMMIT_MESSAGE"

echo ""
echo "Public repository ready."
echo "Path: $OUTPUT_DIR"
echo ""
echo "Next steps:"
echo "  cd \"$OUTPUT_DIR\""
echo "  git remote add origin <GITHUB_REPO_URL>"
echo "  git push -u origin main --force"
