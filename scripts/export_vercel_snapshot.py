#!/usr/bin/env python3
"""Export artists, predictions, and history from SQLite for Vercel standalone deploy."""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = ROOT_DIR / "db" / "fantasanremo.db"
DEFAULT_OUTPUT_PATH = ROOT_DIR / "frontend" / "public" / "data" / "vercel_snapshot.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export a static JSON snapshot consumed by the frontend when "
            "VITE_API_MODE=local."
        )
    )
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH, help="SQLite DB path")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output JSON file path",
    )
    return parser.parse_args()


def row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {key: row[key] for key in row.keys()}


def main() -> int:
    args = parse_args()
    db_path = args.db.resolve()
    output_path = args.output.resolve()

    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    try:
        artists_rows = conn.execute(
            """
            SELECT
              a.id,
              a.nome,
              a.quotazione_2026,
              a.genere_musicale,
              a.anno_nascita,
              a.prima_partecipazione,
              a.debuttante_2026,
              a.image_url,
              p.id AS predizione_id,
              p.artista_id AS predizione_artista_id,
              p.punteggio_predetto,
              p.confidence,
              p.livello_performer
            FROM artisti a
            LEFT JOIN predizioni_2026 p ON p.artista_id = a.id
            ORDER BY a.nome COLLATE NOCASE ASC
            """
        ).fetchall()

        history_rows = conn.execute(
            """
            SELECT
              id,
              artista_id,
              anno,
              punteggio_finale,
              posizione,
              quotazione_baudi
            FROM edizioni_fantasanremo
            ORDER BY artista_id ASC, anno ASC
            """
        ).fetchall()
    finally:
        conn.close()

    history_by_artist: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in history_rows:
        entry = row_to_dict(row)
        artist_id = int(entry["artista_id"])
        history_by_artist[artist_id].append(entry)

    artists: list[dict[str, Any]] = []
    for row in artists_rows:
        row_data = row_to_dict(row)

        predizione_2026 = None
        if row_data["predizione_id"] is not None:
            predizione_2026 = {
                "id": row_data["predizione_id"],
                "artista_id": row_data["predizione_artista_id"],
                "punteggio_predetto": row_data["punteggio_predetto"],
                "confidence": row_data["confidence"],
                "livello_performer": row_data["livello_performer"],
            }

        artist_id = int(row_data["id"])
        artists.append(
            {
                "id": artist_id,
                "nome": row_data["nome"],
                "quotazione_2026": row_data["quotazione_2026"],
                "genere_musicale": row_data["genere_musicale"],
                "anno_nascita": row_data["anno_nascita"],
                "prima_partecipazione": row_data["prima_partecipazione"],
                "debuttante_2026": bool(row_data["debuttante_2026"]),
                "image_url": row_data["image_url"],
                "predizione_2026": predizione_2026,
                "edizioni_fantasanremo": history_by_artist.get(artist_id, []),
            }
        )

    snapshot = {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "source_db": db_path.name,
        "artisti": artists,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Snapshot exported: {output_path}")
    print(f"Artists: {len(artists)}")
    print(f"History rows: {len(history_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
