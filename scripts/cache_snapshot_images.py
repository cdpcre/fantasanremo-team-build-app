#!/usr/bin/env python3
"""
Download remote artist images referenced in snapshot and rewrite to local paths.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.request import urlopen


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cache remote snapshot image URLs into local static files"
    )
    parser.add_argument(
        "--snapshot",
        type=Path,
        required=True,
        help="Path to vercel_snapshot.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where downloaded images are stored",
    )
    parser.add_argument(
        "--output-prefix",
        default="/data/artist_images",
        help="URL prefix used in rewritten image_url values",
    )
    return parser.parse_args()


def detect_extension(payload: bytes) -> str:
    if payload.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if payload.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if payload.startswith(b"RIFF") and payload[8:12] == b"WEBP":
        return ".webp"
    return ".img"


def download_bytes(url: str) -> bytes:
    with urlopen(url, timeout=30) as response:
        return response.read()


def main() -> int:
    args = parse_args()
    snapshot_path = args.snapshot.resolve()
    output_dir = args.output_dir.resolve()
    output_prefix = args.output_prefix.rstrip("/")

    if not snapshot_path.exists():
        raise FileNotFoundError(f"Snapshot not found: {snapshot_path}")

    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    artists = snapshot.get("artisti", [])
    if not isinstance(artists, list):
        raise SystemExit("Invalid snapshot format: `artisti` must be a list")

    output_dir.mkdir(parents=True, exist_ok=True)

    downloaded = 0
    reused = 0
    skipped = 0
    failed = 0

    for artist in artists:
        if not isinstance(artist, dict):
            continue
        artist_id = artist.get("id")
        image_url = artist.get("image_url")

        if not artist_id or not isinstance(image_url, str) or not image_url.startswith("http"):
            skipped += 1
            continue

        try:
            payload = download_bytes(image_url)
            extension = detect_extension(payload)
            filename = f"{artist_id}{extension}"
            target_path = output_dir / filename
            if target_path.exists():
                reused += 1
            else:
                target_path.write_bytes(payload)
                downloaded += 1
            artist["image_url"] = f"{output_prefix}/{filename}"
        except Exception:
            failed += 1

    snapshot_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "snapshot": str(snapshot_path),
                "output_dir": str(output_dir),
                "downloaded": downloaded,
                "reused": reused,
                "skipped": skipped,
                "failed": failed,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
