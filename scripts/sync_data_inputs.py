#!/usr/bin/env python3
"""
Sync and enrich input JSON data used by pipeline and ML.

Actions:
- fill missing prima_partecipazione in artisti_2026_enriched.json (default 2026)
- sync quotazione from artisti_2026_enriched.json to storico_fantasanremo_unified.json/artisti_2026
- sync artist image_url from official Fantasanremo artists feed
- sync missing historical punteggio_finale in artisti_2026_enriched.json from unified storico
- fill known missing quotazione_baudi in storico_fantasanremo_unified.json
  using documented web sources
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path
from urllib.request import urlopen

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


ENRICHED_PATH = PROJECT_ROOT / "data" / "artisti_2026_enriched.json"
UNIFIED_PATH = PROJECT_ROOT / "data" / "storico_fantasanremo_unified.json"
FANTASANREMO_GAME_CONFIG_URL = "https://api-v2.fantasanremo.com/game-config/current"
FANTASANREMO_DATA_URL_TEMPLATE = "https://fantasanremo.com/data/{version}/artists.json"
FANTASANREMO_UPLOADS_BASE_URL = "https://fantasanremo.com/uploads"

# Local key -> remote key aliases for known naming mismatches.
ARTIST_IMAGE_ALIASES: dict[str, str] = {
    "fedez and marco masini": "fedez and masini",
}

# Historical quotations recoverable from public archived pages / media lists.
# Sources:
# - Deejay (2023 quotazioni): https://www.deejay.it/articoli/fantasanremo-le-quotazioni-dei-cantanti-in-gara/
# - Sky TG24 (2025 quotazioni): https://tg24.sky.it/spettacolo/musica/2025/02/10/fantasanremo-quotazioni-artisti
HISTORICAL_QUOTES: dict[tuple[str, int], dict[str, object]] = {
    ("Levante", 2023): {"quotazione_baudi": 20, "source": "deejay_2023"},
    ("Mara Sattei", 2023): {"quotazione_baudi": 21, "source": "deejay_2023"},
    ("Serena Brancale", 2025): {"quotazione_baudi": 13, "source": "sky_tg24_2025"},
    # Inferred from same 2025 list where Fedez appears as solo act.
    ("Fedez & Marco Masini", 2025): {
        "quotazione_baudi": 17,
        "source": "sky_tg24_2025_inferred_fedez",
    },
}


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def fetch_fantasanremo_artists_feed() -> tuple[int, list[dict]]:
    with urlopen(FANTASANREMO_GAME_CONFIG_URL, timeout=30) as response:
        game_config = json.load(response)

    latest_update = game_config.get("latestArtistsUpdate")
    if not isinstance(latest_update, int):
        raise SystemExit("Fantasanremo game config missing latestArtistsUpdate")

    artists_url = FANTASANREMO_DATA_URL_TEMPLATE.format(version=latest_update)
    with urlopen(artists_url, timeout=30) as response:
        artists_payload = json.load(response)

    if not isinstance(artists_payload, list):
        raise SystemExit("Unexpected Fantasanremo artists feed format")
    return latest_update, artists_payload


def main() -> int:
    from backend.utils.name_normalization import normalize_artist_name

    enriched = load_json(ENRICHED_PATH)
    unified = load_json(UNIFIED_PATH)
    latest_artists_update, remote_artists = fetch_fantasanremo_artists_feed()

    artists = enriched.get("artisti", [])
    if not isinstance(artists, list) or not artists:
        raise SystemExit("Invalid or empty artisti_2026_enriched.json")

    unified_2026 = unified.get("artisti_2026", [])
    if not isinstance(unified_2026, list):
        raise SystemExit("Invalid storico_fantasanremo_unified.json: artisti_2026 not a list")
    historical = unified.get("artisti_storici", {})
    if not isinstance(historical, dict):
        historical = {}

    normalized_quote_map: dict[str, int] = {}
    normalized_image_map: dict[str, str] = {}
    normalized_history_scores: dict[str, dict[int, int]] = {}
    for remote_artist in remote_artists:
        if not isinstance(remote_artist, dict):
            continue
        remote_name = remote_artist.get("name")
        remote_avatar = remote_artist.get("avatar")
        remote_key = normalize_artist_name(remote_name)
        if not remote_key or not remote_avatar:
            continue
        normalized_image_map[remote_key] = f"{FANTASANREMO_UPLOADS_BASE_URL}/{remote_avatar}"

    if not normalized_image_map:
        raise SystemExit("Fantasanremo artists feed returned no avatar URLs")

    for artist_name, payload in historical.items():
        if not isinstance(payload, dict):
            continue
        key = normalize_artist_name(artist_name)
        if not key:
            continue
        year_scores = normalized_history_scores.setdefault(key, {})
        for ed in payload.get("edizioni", []):
            if not isinstance(ed, dict):
                continue
            anno = ed.get("anno")
            score = ed.get("punteggio_finale")
            if isinstance(anno, int) and score is not None:
                year_scores[anno] = int(score)

    updated_prima = 0
    synced_images = 0
    missing_image_matches = 0
    filled_historical_scores = 0
    for artist in artists:
        if not isinstance(artist, dict):
            continue
        nome = artist.get("nome")
        key = normalize_artist_name(nome)
        quote = artist.get("quotazione")
        if key and isinstance(quote, int):
            normalized_quote_map[key] = quote

        if not artist.get("prima_partecipazione"):
            artist["prima_partecipazione"] = 2026
            updated_prima += 1

        if artist.get("debuttante_2026") is None:
            storico = artist.get("storico_fantasanremo", [])
            artist["debuttante_2026"] = not bool(storico)

        storico_entries = artist.get("storico_fantasanremo", [])
        if isinstance(storico_entries, list):
            known_scores = normalized_history_scores.get(key, {})
            for ed in storico_entries:
                if not isinstance(ed, dict):
                    continue
                anno = ed.get("anno")
                if not isinstance(anno, int):
                    continue
                if ed.get("punteggio_finale") is not None:
                    continue
                score = known_scores.get(anno)
                if score is not None:
                    ed["punteggio_finale"] = score
                    filled_historical_scores += 1

        remote_key = key
        if remote_key not in normalized_image_map:
            remote_key = ARTIST_IMAGE_ALIASES.get(key, key)

        remote_image_url = normalized_image_map.get(remote_key)
        if remote_image_url:
            if artist.get("image_url") != remote_image_url:
                artist["image_url"] = remote_image_url
                synced_images += 1
        else:
            missing_image_matches += 1

    synced_unified_quotes = 0
    for artist in unified_2026:
        if not isinstance(artist, dict):
            continue
        key = normalize_artist_name(artist.get("nome"))
        quote = normalized_quote_map.get(key)
        if quote is not None and artist.get("quotazione") != quote:
            artist["quotazione"] = quote
            synced_unified_quotes += 1

    filled_hist_quotes = 0
    for artist_name, payload in historical.items():
        if not isinstance(payload, dict):
            continue
        for ed in payload.get("edizioni", []):
            if not isinstance(ed, dict):
                continue
            anno = ed.get("anno")
            key = (artist_name, anno)
            if ed.get("quotazione_baudi") is None and key in HISTORICAL_QUOTES:
                quote_info = HISTORICAL_QUOTES[key]
                ed["quotazione_baudi"] = int(quote_info["quotazione_baudi"])
                existing_note = ed.get("nota")
                source_note = f"source:{quote_info['source']}"
                if existing_note:
                    if source_note not in existing_note:
                        ed["nota"] = f"{existing_note}; {source_note}"
                else:
                    ed["nota"] = source_note
                filled_hist_quotes += 1

    unified.setdefault("metadata", {})["data_last_updated"] = date.today().isoformat()

    save_json(ENRICHED_PATH, enriched)
    save_json(UNIFIED_PATH, unified)

    print(
        json.dumps(
            {
                "updated_prima_partecipazione": updated_prima,
                "synced_artist_image_urls": synced_images,
                "missing_artist_image_matches": missing_image_matches,
                "filled_historical_scores": filled_historical_scores,
                "synced_unified_quotazioni_2026": synced_unified_quotes,
                "filled_historical_quotazioni": filled_hist_quotes,
                "fantasanremo_latest_artists_update": latest_artists_update,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
