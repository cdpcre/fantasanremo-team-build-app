from __future__ import annotations

from typing import Any

from .name_normalization import normalize_artist_name

BIO_FIELDS = {
    "anno_nascita",
    "genere_musicale",
    "prima_partecipazione",
    "note",
}

CAR_FIELDS = {
    "viralita_social",
    "social_followers_total",
    "social_followers_by_platform",
    "social_followers_last_updated",
    "storia_bonus_ottenuti",
    "ad_personam_bonus_count",
    "ad_personam_bonus_points",
}


def merge_artisti_enriched(
    artisti_data: dict,
    biografico_data: dict | None = None,
    caratteristiche_data: dict | None = None,
) -> dict:
    """
    Merge artisti with biografico + caratteristiche into a single enriched structure.

    Keeps the top-level structure of artisti_2026_enriched.json and injects bio/car fields
    into each artist entry.
    """
    if not isinstance(artisti_data, dict):
        return {}

    base_artists = artisti_data.get("artisti", [])
    if not isinstance(base_artists, list):
        return {}

    bio_map: dict[str, dict[str, Any]] = {}
    if biografico_data and isinstance(biografico_data, dict):
        for entry in biografico_data.get("artisti_2026_biografico", []):
            if not isinstance(entry, dict):
                continue
            key = normalize_artist_name(entry.get("nome"))
            if key:
                bio_map[key] = {k: entry.get(k) for k in BIO_FIELDS}

    car_map: dict[str, dict[str, Any]] = {}
    if caratteristiche_data and isinstance(caratteristiche_data, dict):
        for entry in caratteristiche_data.get("caratteristiche_artisti_2026", []):
            if not isinstance(entry, dict):
                continue
            key = normalize_artist_name(entry.get("nome"))
            if key:
                car_map[key] = {k: entry.get(k) for k in CAR_FIELDS}

    enriched_artists = []
    for artist in base_artists:
        if not isinstance(artist, dict):
            continue
        merged = dict(artist)
        key = normalize_artist_name(artist.get("nome"))
        if key in bio_map:
            merged.update({k: v for k, v in bio_map[key].items() if v is not None})
        if key in car_map:
            merged.update({k: v for k, v in car_map[key].items() if v is not None})
        enriched_artists.append(merged)

    enriched = dict(artisti_data)
    enriched["artisti"] = enriched_artists
    return enriched


def build_biografico_from_enriched(enriched_data: dict) -> dict:
    """Create biografico-like payload from enriched artist data."""
    import re

    # Pattern per estrarre generi musicali dalle note
    genre_patterns = [
        r"hip[- ]?hop",
        r"rap",
        r"pop(?:\s+(?:latino|rap))?",
        r"rock(?:\s+indie)?",
        r"indie(?:\s+pop)?",
        r"r&b",
        r"soul",
        r"dance",
        r"electronic",
        r"folk",
    ]

    entries: list[dict[str, Any]] = []
    for artist in enriched_data.get("artisti", []):
        if not isinstance(artist, dict):
            continue
        nome = artist.get("nome")
        if not nome:
            continue
        entry = {"nome": nome}

        # Copia campi esistenti
        for field in BIO_FIELDS:
            if field in artist:
                entry[field] = artist.get(field)

        # CALCOLA prima_partecipazione da storico_fantasanremo
        storico = artist.get("storico_fantasanremo", [])
        if storico and "prima_partecipazione" not in entry:
            anni = [ed.get("anno") for ed in storico if isinstance(ed.get("anno"), int)]
            if anni:
                entry["prima_partecipazione"] = min(anni)

        # Se assente e senza storico noto, assume debutto nell'edizione corrente.
        if "prima_partecipazione" not in entry:
            entry["prima_partecipazione"] = 2026

        # ESTRAI genere_musicale dalle note se mancante
        if "genere_musicale" not in entry:
            note = artist.get("note", "")
            if note and isinstance(note, str):
                note_lower = note.lower()
                for pattern in genre_patterns:
                    match = re.search(pattern, note_lower)
                    if match:
                        # Normalizza il nome del genere
                        genre = match.group()
                        genre = genre.replace("-", " ").replace("  ", " ").title()
                        entry["genere_musicale"] = genre
                        break

        entries.append(entry)
    return {"artisti_2026_biografico": entries}


def build_caratteristiche_from_enriched(enriched_data: dict) -> dict:
    """Create caratteristiche-like payload from enriched artist data."""
    entries: list[dict[str, Any]] = []
    for artist in enriched_data.get("artisti", []):
        if not isinstance(artist, dict):
            continue
        nome = artist.get("nome")
        if not nome:
            continue
        entry = {"nome": nome}
        for field in CAR_FIELDS:
            if field in artist:
                entry[field] = artist.get(field)
        entries.append(entry)
    return {"caratteristiche_artisti_2026": entries}
