#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_DEPLOY_SCRIPT="$HOME/.codex/skills/vercel-deploy/scripts/deploy.sh"
DEPLOY_SCRIPT="${DEPLOY_SCRIPT_PATH:-$DEFAULT_DEPLOY_SCRIPT}"
BUNDLE_DIR="$ROOT_DIR/vercel_app"

"$ROOT_DIR/scripts/prepare_vercel_bundle.sh"

echo "Deploying standalone bundle to Vercel..."
if [ -f "$DEPLOY_SCRIPT" ]; then
  bash "$DEPLOY_SCRIPT" "$BUNDLE_DIR"
elif command -v vercel >/dev/null 2>&1; then
  (
    cd "$BUNDLE_DIR"
    vercel --yes
  )
else
  echo "No deploy tool available." >&2
  echo "Install Vercel CLI ('npm i -g vercel') or set DEPLOY_SCRIPT_PATH." >&2
  exit 1
fi
