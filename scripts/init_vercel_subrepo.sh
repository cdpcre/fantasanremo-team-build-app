#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUNDLE_DIR="$ROOT_DIR/vercel_app"

mkdir -p "$BUNDLE_DIR"

if [ -d "$BUNDLE_DIR/.git" ]; then
  echo "Nested repo already initialized in: $BUNDLE_DIR"
  exit 0
fi

git -C "$BUNDLE_DIR" init -b main >/dev/null 2>&1 || {
  git -C "$BUNDLE_DIR" init >/dev/null
  git -C "$BUNDLE_DIR" branch -M main
}

echo "Nested repo initialized in: $BUNDLE_DIR"
