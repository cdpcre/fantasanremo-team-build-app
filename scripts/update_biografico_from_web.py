#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import requests
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Missing dependency: requests. Install with `uv add requests`.") from exc

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.ml.genre_features import normalize_genre  # noqa: E402
from backend.utils.name_normalization import normalize_artist_name  # noqa: E402

WIKIDATA_API = "https://www.wikidata.org/w/api.php"
SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"

USER_AGENT = "fantasanremo-team-builder/1.0 (data-update script)"

COLLAB_SPLIT = re.compile(r"\s*(?:&|,|/|\+|feat\.?|ft\.?|featuring)\s*", re.IGNORECASE)

FOLLOWERS_PROP = "P8687"
POINT_IN_TIME_PROP = "P585"

QUALIFIER_PLATFORM_MAP = {
    "P2003": "instagram",
    "P2002": "x_twitter",
    "P6552": "x_twitter_numeric",
    "P2397": "youtube_channel",
    "P11245": "youtube_handle",
    "P2013": "facebook",
    "P7085": "tiktok",
}


def _request_json(url: str, params: dict[str, Any]) -> dict[str, Any]:
    resp = requests.get(url, params=params, headers={"User-Agent": USER_AGENT}, timeout=30)
    resp.raise_for_status()
    return resp.json()


def _request_json_raw(url: str) -> dict[str, Any]:
    resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
    resp.raise_for_status()
    return resp.json()


def search_wikidata_entity(name: str, language: str = "it", limit: int = 5) -> list[dict]:
    params = {
        "action": "wbsearchentities",
        "search": name,
        "language": language,
        "format": "json",
        "limit": limit,
    }
    data = _request_json(WIKIDATA_API, params)
    return data.get("search", [])


def _score_search_result(name: str, result: dict) -> int:
    label = result.get("label", "")
    description = (result.get("description") or "").lower()
    name_key = normalize_artist_name(name)
    label_key = normalize_artist_name(label)

    score = 0
    if name_key == label_key:
        score += 5
    if result.get("match", {}).get("type") == "label":
        score += 2
    if any(
        kw in description
        for kw in [
            "cantante",
            "musicista",
            "rapper",
            "gruppo musicale",
            "band",
            "singer",
            "musician",
        ]
    ):
        score += 3
    if "disambiguazione" in description or "disambiguation" in description:
        score -= 4
    return score


def pick_best_entity(name: str, results: list[dict]) -> dict | None:
    if not results:
        return None
    ranked = sorted(results, key=lambda r: _score_search_result(name, r), reverse=True)
    return ranked[0]


def fetch_wikidata_details(qid: str) -> dict[str, Any]:
    query = f"""
    SELECT ?birth ?genreLabel WHERE {{
      OPTIONAL {{ wd:{qid} wdt:P569 ?birth. }}
      OPTIONAL {{ wd:{qid} wdt:P136 ?genre. }}
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "it,en". }}
    }}
    """
    params = {"query": query, "format": "json"}
    data = _request_json(SPARQL_ENDPOINT, params)
    bindings = data.get("results", {}).get("bindings", [])

    birth_year = None
    genres: list[str] = []
    for row in bindings:
        birth = row.get("birth", {}).get("value")
        if birth and birth_year is None:
            birth_year = int(birth[:4])
        genre_label = row.get("genreLabel", {}).get("value")
        if genre_label:
            genres.append(genre_label)

    return {"birth_year": birth_year, "genres": sorted(set(genres))}


def fetch_wikidata_entity_json(qid: str) -> dict[str, Any]:
    url = f"https://www.wikidata.org/wiki/Special:EntityData/{qid}.json"
    data = _request_json_raw(url)
    return data.get("entities", {}).get(qid, {})


def _parse_wikidata_time(value: str | None) -> str | None:
    if not value:
        return None
    # format: +YYYY-MM-DDT00:00:00Z
    if value.startswith("+") and len(value) >= 11:
        return value[1:11]
    return None


def _parse_wikidata_amount(value: str | None) -> int | None:
    if not value:
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def extract_social_followers(entity: dict) -> tuple[dict[str, int], int | None, str | None]:
    """
    Extract follower counts from Wikidata entity claims.

    Returns:
        (followers_by_platform, total_followers, latest_date)
    """
    claims = entity.get("claims", {})
    follower_claims = claims.get(FOLLOWERS_PROP, [])
    if not follower_claims:
        return {}, None, None

    platform_entries: dict[str, dict] = {}
    for claim in follower_claims:
        mainsnak = claim.get("mainsnak", {})
        datavalue = mainsnak.get("datavalue", {})
        amount = datavalue.get("value", {}).get("amount")
        followers = _parse_wikidata_amount(amount)
        if followers is None:
            continue

        qualifiers = claim.get("qualifiers", {})
        platform = None
        for prop_id, platform_name in QUALIFIER_PLATFORM_MAP.items():
            if prop_id in qualifiers:
                platform = platform_name
                break
        if platform is None:
            platform = "unknown"

        date_val = None
        if POINT_IN_TIME_PROP in qualifiers:
            date_raw = (
                qualifiers[POINT_IN_TIME_PROP][0].get("datavalue", {}).get("value", {}).get("time")
            )
            date_val = _parse_wikidata_time(date_raw)

        existing = platform_entries.get(platform)
        if not existing:
            platform_entries[platform] = {"value": followers, "date": date_val}
        else:
            # Prefer latest date if available, else max value
            if date_val and (not existing["date"] or date_val > existing["date"]):
                platform_entries[platform] = {"value": followers, "date": date_val}
            elif not date_val and followers > existing["value"]:
                platform_entries[platform] = {"value": followers, "date": existing["date"]}

    followers_by_platform = {k: v["value"] for k, v in platform_entries.items()}
    total_followers = sum(followers_by_platform.values()) if followers_by_platform else None
    dates = [v["date"] for v in platform_entries.values() if v.get("date")]
    latest_date = max(dates) if dates else None
    return followers_by_platform, total_followers, latest_date


def followers_to_score(total_followers: int | None) -> int | None:
    if not total_followers or total_followers <= 0:
        return None
    # Log-scaled score: 10k -> ~60, 100k -> ~75, 1M -> ~90
    score = int(round(15 * math.log10(total_followers + 1)))
    return max(1, min(100, score))


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open() as f:
        return json.load(f)


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_overrides(path: Path) -> dict[str, dict]:
    """Load manual overrides for all fields (biographical and social)."""
    data = load_json(path)
    if not data:
        return {}
    raw = data.get("overrides") if isinstance(data, dict) and "overrides" in data else data
    overrides: dict[str, dict] = {}
    if isinstance(raw, dict):
        for name, details in raw.items():
            key = normalize_artist_name(name)
            if key and isinstance(details, dict):
                overrides[key] = details
    return overrides


def apply_overrides(artist_entry: dict, override: dict, force: bool) -> None:
    """Apply manual overrides to artist entry (all fields: biographical and social)."""
    if not override:
        return

    # Apply biographical fields
    for field in ["genere_musicale", "anno_nascita", "note"]:
        value = override.get(field)
        if value is not None and value not in ["null", ""]:
            if field == "anno_nascita":
                artist_entry[field] = int(value) if value else value
            else:
                artist_entry[field] = value

    # Apply social fields (only if force or no existing data)
    if not force and artist_entry.get("social_followers_total"):
        return

    by_platform = override.get("followers_by_platform") or {}
    total = override.get("followers_total")
    if total is None and by_platform:
        total = sum(v for v in by_platform.values() if isinstance(v, (int, float)))
    last_updated = override.get("followers_last_updated") or override.get("date")

    if by_platform:
        artist_entry["followers_by_platform"] = by_platform
    if total is not None:
        artist_entry["social_followers_total"] = int(total)
    if last_updated:
        artist_entry["social_followers_last_updated"] = last_updated


def split_collaboration_name(name: str) -> list[str]:
    parts = [p.strip() for p in COLLAB_SPLIT.split(name) if p.strip()]
    return parts if parts else [name]


def infer_prima_partecipazione(artista: dict) -> int | None:
    storico = artista.get("storico_fantasanremo", [])
    years = [entry.get("anno") for entry in storico if entry.get("anno")]
    if years:
        return min(years)
    # If an artist is in the 2026 roster and has no prior historical entries,
    # treat first participation as 2026.
    return 2026


def _weighted_average(values: list[int], weights: list[float]) -> int | None:
    if not values or not weights or len(values) != len(weights):
        return None
    total_weight = sum(weights)
    if total_weight <= 0:
        return None
    avg = sum(v * w for v, w in zip(values, weights)) / total_weight
    return int(round(avg))


def _weighted_top_choice(values: list[str], weights: list[float]) -> str | None:
    if not values or not weights or len(values) != len(weights):
        return None
    totals: dict[str, float] = {}
    for value, weight in zip(values, weights):
        if not value:
            continue
        totals[value] = totals.get(value, 0.0) + weight
    if not totals:
        return None
    return max(totals.items(), key=lambda item: item[1])[0]


def aggregate_collab_social(
    member_details: list[dict],
) -> tuple[dict[str, int], int | None, str | None]:
    aggregated_by_platform: dict[str, int] = {}
    dates: list[str] = []

    for details in member_details:
        followers_by_platform = details.get("followers_by_platform") or {}
        for platform, count in followers_by_platform.items():
            if isinstance(count, (int, float)) and count >= 0:
                aggregated_by_platform[platform] = aggregated_by_platform.get(platform, 0) + int(
                    count
                )
        date_val = details.get("followers_last_updated")
        if date_val:
            dates.append(date_val)

    total = sum(aggregated_by_platform.values()) if aggregated_by_platform else None
    latest_date = max(dates) if dates else None
    return aggregated_by_platform, total, latest_date


def build_ad_personam_map(regolamento_data: dict, artisti_list: list[dict]) -> dict[str, dict]:
    """
    Build map of normalized artist name -> ad personam bonus stats.
    """
    if not regolamento_data or not artisti_list:
        return {}

    ad_personam = regolamento_data.get("bonus", {}).get("ad_personam", [])
    if not ad_personam:
        return {}

    name_keys = {
        normalize_artist_name(a.get("nome")): a.get("nome") for a in artisti_list if a.get("nome")
    }
    out: dict[str, dict] = {k: {"count": 0, "points": 0} for k in name_keys.keys()}

    for bonus in ad_personam:
        desc = bonus.get("descrizione", "")
        points = bonus.get("punti", 0)
        desc_key = normalize_artist_name(desc)
        for artist_key in name_keys.keys():
            if artist_key and artist_key in desc_key:
                out[artist_key]["count"] += 1
                out[artist_key]["points"] += points

    return out


def build_bonus_history_map(storico_data: dict) -> dict[str, int]:
    artisti_storici = storico_data.get("artisti_storici")
    if not isinstance(artisti_storici, dict):
        return {}

    result: dict[str, int] = {}
    for nome, info in artisti_storici.items():
        if not nome:
            continue
        total_points = 0
        for edizione in info.get("edizioni", []):
            punti = edizione.get("punteggio_finale")
            if isinstance(punti, (int, float)):
                total_points += int(punti)
            elif isinstance(punti, str):
                punti_clean = punti.replace("\xa0", "").replace(" ", "")
                if punti_clean.isdigit():
                    total_points += int(punti_clean)

        result[normalize_artist_name(nome)] = total_points

    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Update enriched artist data from Wikidata.")
    parser.add_argument(
        "--enriched",
        default=str(PROJECT_ROOT / "data" / "artisti_2026_enriched.json"),
        help="Path to artisti_2026_enriched.json",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Output path (defaults to overwrite enriched file)",
    )
    parser.add_argument(
        "--regolamento",
        default=str(PROJECT_ROOT / "data" / "regolamento_2026.json"),
        help="Path to regolamento_2026.json",
    )
    parser.add_argument(
        "--storico",
        default=str(PROJECT_ROOT / "data" / "storico_fantasanremo_unified.json"),
        help="Path to storico_fantasanremo_unified.json",
    )
    parser.add_argument(
        "--overrides",
        default=str(PROJECT_ROOT / "data" / "overrides.json"),
        help="Path to overrides.json for manual data (biographical and social)",
    )
    parser.add_argument(
        "--skip-caratteristiche",
        action="store_true",
        help="Skip updating caratteristiche (social followers)",
    )
    parser.add_argument("--limit", type=int, default=0, help="Limit number of updates")
    parser.add_argument("--dry-run", action="store_true", help="Do not write output files")
    parser.add_argument("--sleep", type=float, default=0.5, help="Sleep between requests")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing fields when web data is found",
    )
    parser.add_argument(
        "--cache",
        default=str(PROJECT_ROOT / "data" / ".cache" / "wikidata_cache.json"),
        help="Cache file path",
    )
    args = parser.parse_args()

    enriched_path = Path(args.enriched)
    output_path = Path(args.output) if args.output else enriched_path
    regolamento_path = Path(args.regolamento)
    storico_path = Path(args.storico)
    overrides_path = Path(args.overrides)
    cache_path = Path(args.cache)

    enriched_data = load_json(enriched_path)
    regolamento_data = load_json(regolamento_path)
    storico_data = load_json(storico_path)
    overrides = load_overrides(overrides_path)

    artisti_list = enriched_data.get("artisti", [])
    if not artisti_list:
        raise SystemExit(f"Missing or empty artists list in {enriched_path}")

    cache = load_json(cache_path) or {"entities": {}}
    cache_entities = cache.get("entities", {})

    ad_personam_map = build_ad_personam_map(regolamento_data, artisti_list)
    bonus_history_map = build_bonus_history_map(storico_data)
    bonus_history_available = bool(bonus_history_map)

    updated = 0
    unresolved = 0
    updated_social = 0
    missing_social_count = 0
    missing_bonus_history = 0

    for artista in artisti_list:
        nome = artista.get("nome")
        if not nome:
            continue

        key = normalize_artist_name(nome)
        artist_entry = artista

        missing_birth = not artist_entry.get("anno_nascita")
        missing_genre = not artist_entry.get("genere_musicale")
        missing_prima = not artist_entry.get("prima_partecipazione")

        missing_social = False
        if not args.skip_caratteristiche:
            missing_social = args.force or (
                not artist_entry.get("viralita_social")
                or not artist_entry.get("social_followers_total")
            )

        if missing_prima:
            prima = infer_prima_partecipazione(artista)
            if prima:
                artist_entry["prima_partecipazione"] = prima

        # Apply manual overrides (genres, birth year, notes, social)
        override = overrides.get(key)
        if override:
            apply_overrides(artist_entry, override, args.force)

        if not (missing_birth or missing_genre or missing_social):
            continue

        if args.limit and updated >= args.limit:
            break

        collab_parts = split_collaboration_name(nome)
        is_collab = len(collab_parts) > 1

        # Use cache if available
        cached = cache_entities.get(key)
        if cached:
            details = cached
        else:
            details = {}

            # Search best entity using full name
            results = search_wikidata_entity(nome, language="it", limit=5)
            if not results:
                results = search_wikidata_entity(nome, language="en", limit=5)
            best = pick_best_entity(nome, results)

            if best and best.get("id"):
                details = {"qid": best["id"]}
            cache_entities[key] = details

        qid = details.get("qid")

        if overrides.get(key):
            # Apply social overrides to details (for non-collaboration case)
            override = overrides[key]
            by_platform = override.get("followers_by_platform") or {}
            total = override.get("followers_total")
            if total is None and by_platform:
                total = sum(v for v in by_platform.values() if isinstance(v, (int, float)))
            last_updated = override.get("followers_last_updated") or override.get("date")

            if by_platform:
                details["followers_by_platform"] = by_platform
            if total is not None:
                details["followers_total"] = int(total)
            if last_updated:
                details["followers_last_updated"] = last_updated

        # Fetch birth/genre if needed
        if qid and (missing_birth or missing_genre):
            if "birth_year" not in details or "genres" not in details:
                fetched = fetch_wikidata_details(qid)
                details.update(fetched)
                cache_entities[key] = details

        # Fetch followers if needed
        if qid and missing_social:
            if "followers_total" not in details:
                entity_json = fetch_wikidata_entity_json(qid)
                followers_by_platform, total_followers, latest_date = extract_social_followers(
                    entity_json
                )
                details["followers_by_platform"] = followers_by_platform
                details["followers_total"] = total_followers
                details["followers_last_updated"] = latest_date
                cache_entities[key] = details

        if not details and not is_collab:
            unresolved += 1
            continue

        member_details: list[dict] = []

        if is_collab:
            # For collaborations, avoid setting a single birth year
            note_parts = []
            for part in collab_parts:
                part_key = normalize_artist_name(part)
                part_details = cache_entities.get(part_key)
                if not part_details:
                    results = search_wikidata_entity(part, language="it", limit=5)
                    if not results:
                        results = search_wikidata_entity(part, language="en", limit=5)
                    best = pick_best_entity(part, results)
                    if best and best.get("id"):
                        part_details = {"qid": best["id"]}
                        fetched = fetch_wikidata_details(best["id"])
                        part_details.update(fetched)

                        if missing_social:
                            entity_json = fetch_wikidata_entity_json(best["id"])
                            followers_by_platform, total_followers, latest_date = (
                                extract_social_followers(entity_json)
                            )
                            part_details["followers_by_platform"] = followers_by_platform
                            part_details["followers_total"] = total_followers
                            part_details["followers_last_updated"] = latest_date

                        cache_entities[part_key] = part_details
                if part_details:
                    override = overrides.get(part_key)
                    if override:
                        # Apply biographical overrides for collaboration
                        for field in ["genere_musicale"]:
                            value = override.get(field)
                            if value and value not in part_details.get("genres", []):
                                if "genres" not in part_details:
                                    part_details["genres"] = []
                                part_details["genres"].append(value)
                        # Apply social overrides
                        by_platform = override.get("followers_by_platform") or {}
                        total = override.get("followers_total")
                        if total is None and by_platform:
                            total = sum(
                                v for v in by_platform.values() if isinstance(v, (int, float))
                            )
                        last_updated = override.get("followers_last_updated") or override.get(
                            "date"
                        )

                        if by_platform:
                            part_details["followers_by_platform"] = by_platform
                        if total is not None:
                            part_details["followers_total"] = int(total)
                        if last_updated:
                            part_details["followers_last_updated"] = last_updated
                    note_parts.append(
                        f"{part}: {part_details.get('birth_year')} - "
                        f"{', '.join(part_details.get('genres', []))}"
                    )
                    member_details.append(part_details)
            if note_parts:
                artist_entry["note"] = "; ".join(note_parts)

            if member_details:
                weights = []
                birth_years = []
                genres = []
                for member in member_details:
                    followers = member.get("followers_total")
                    weight = (
                        float(followers)
                        if isinstance(followers, (int, float)) and followers > 0
                        else 1.0
                    )
                    weights.append(weight)

                    birth_year = member.get("birth_year")
                    if isinstance(birth_year, int):
                        birth_years.append(birth_year)
                    else:
                        birth_years.append(None)

                    member_genres = member.get("genres") or []
                    normalized = normalize_genre(member_genres[0]) if member_genres else None
                    genres.append(normalized)

                if (missing_birth or args.force) and any(isinstance(y, int) for y in birth_years):
                    filtered_years = [y for y in birth_years if isinstance(y, int)]
                    filtered_weights = [
                        w for y, w in zip(birth_years, weights) if isinstance(y, int)
                    ]
                    avg_birth = _weighted_average(filtered_years, filtered_weights)
                    if avg_birth:
                        artist_entry["anno_nascita"] = avg_birth

                if (missing_genre or args.force) and any(genres):
                    filtered_genres = [g for g in genres if g]
                    filtered_weights = [w for g, w in zip(genres, weights) if g]
                    chosen_genre = _weighted_top_choice(filtered_genres, filtered_weights)
                    if chosen_genre:
                        artist_entry["genere_musicale"] = chosen_genre
            else:
                unresolved += 1
                continue
        else:
            birth_year = details.get("birth_year")
            if birth_year and (missing_birth or args.force):
                artist_entry["anno_nascita"] = birth_year

        genres = details.get("genres", [])
        if genres and (missing_genre or args.force):
            normalized = normalize_genre(genres[0])
            if normalized:
                artist_entry["genere_musicale"] = normalized

        # Update caratteristiche with social followers
        if not args.skip_caratteristiche and missing_social:
            if is_collab and member_details:
                followers_by_platform, total_followers, latest_date = aggregate_collab_social(
                    member_details
                )
            else:
                followers_by_platform = details.get("followers_by_platform") or {}
                total_followers = details.get("followers_total")
                latest_date = details.get("followers_last_updated")

            if total_followers:
                artist_entry["social_followers_total"] = total_followers
                artist_entry["social_followers_by_platform"] = followers_by_platform
                artist_entry["social_followers_last_updated"] = (
                    latest_date or datetime.utcnow().date().isoformat()
                )
                updated_social += 1

            score = followers_to_score(total_followers)
            if score is not None:
                artist_entry["viralita_social"] = score
            else:
                artist_entry["viralita_social"] = None
                missing_social_count += 1

            # Remove legacy mock fields if present
            artist_entry.pop("carisma", None)
            artist_entry.pop("presenza_scenica", None)

        # Update historical bonus/points and regolamento-based bonus fields
        if bonus_history_available:
            history_points = bonus_history_map.get(key)
            if history_points is None:
                history_points = 0
                missing_bonus_history += 1
            artist_entry["storia_bonus_ottenuti"] = history_points

            bonus_info = ad_personam_map.get(key, {"count": 0, "points": 0})
            artist_entry["ad_personam_bonus_count"] = bonus_info["count"]
            artist_entry["ad_personam_bonus_points"] = bonus_info["points"]

        updated += 1
        time.sleep(args.sleep)

    cache["entities"] = cache_entities

    if not args.dry_run:
        save_json(output_path, enriched_data)
        save_json(cache_path, cache)

    report = {
        "updated": updated,
        "unresolved": unresolved,
        "total_artists": len(artisti_list),
        "output": str(output_path),
        "updated_social": updated_social,
        "missing_social_followers": missing_social_count,
        "missing_bonus_history": missing_bonus_history if bonus_history_available else None,
        "cache": str(cache_path),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
