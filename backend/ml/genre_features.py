"""
Genre-Based Features Module

Crea feature basate sul genere musicale degli artisti.
"""

import sys
from pathlib import Path

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from collections import defaultdict

import numpy as np
import pandas as pd

from backend.data_pipeline.config import get_logger
from backend.utils.name_normalization import normalize_artist_name

# Genre mappings and groupings
GENRE_GROUPS = {
    "mainstream_pop": ["Pop", "Pop/Rock", "Dance"],
    "rap_urban": ["Rap", "Hip-hop", "Urban", "Pop Rap", "Trap"],
    "rock_indie": ["Rock", "Indie", "Indie Pop", "Pop-punk", "Alternative"],
    "veteran": ["Napoletana", "Blues", "Reggaeton", "Pop/Rock"],
}

# Normalize genre names
GENRE_NORMALIZATION = {
    "pop": "Pop",
    "rap": "Rap",
    "hip-hop": "Hip-hop",
    "hip hop": "Hip-hop",
    "rock": "Rock",
    "indie": "Indie",
    "indie pop": "Indie Pop",
    "urban": "Urban",
    "pop rap": "Pop Rap",
    "pop/rock": "Pop/Rock",
    "pop-rock": "Pop/Rock",
    "reggaeton": "Reggaeton",
    "napoletana": "Napoletana",
    "blues": "Blues",
    "dance": "Dance",
    "elettronica": "Elettronica",
}


def create_genre_features(
    artisti_data: list[dict],
    biografico_data: dict | None = None,
    storico_performance: dict | None = None,
    as_of_year: int = 2026,
) -> pd.DataFrame:
    """
    Crea feature basate sul genere musicale.

    Features:
    - genre_avg_performance: Media storica per genere
    - genre_win_rate: % top 5 per genere
    - genre_trend: Popolarità nel tempo (2020-2025)
    - genre_mainstream_pop: Flag per Pop mainstream
    - genre_rap_urban: Flag per Rap/Urban
    - genre_rock_indie: Flag per Rock/Indie
    - genre_veteran: Flag per veteran-associated genres

    Args:
        artisti_data: Lista di dict artisti
        biografico_data: Dict con dati biografici
        storico_performance: Dict con performance storiche per genere

    Returns:
        DataFrame con feature genere per artista
    """
    get_logger("genre_features")

    # Build genre mapping from biografico data
    genre_map = {}
    if biografico_data:
        for artista in biografico_data.get("artisti_2026_biografico", []):
            nome = artista.get("nome")
            genere = artista.get("genere_musicale")
            key = normalize_artist_name(nome)
            if key and genere:
                genre_map[key] = normalize_genre(genere)

    # Calculate genre performance stats from historical data
    genre_stats = calculate_genre_performance(storico_performance, genre_map, as_of_year=as_of_year)

    features = []
    for artista in artisti_data:
        nome = artista.get("nome")
        artista_id = artista.get("id")

        # Get genre for this artist
        key = normalize_artist_name(nome)
        genre = genre_map.get(key, "Unknown")

        feat = {
            "artista_id": artista_id,
            "artista_nome": nome,
            "genre": genre,
            "genre_mainstream_pop": is_mainstream_pop(genre),
            "genre_rap_urban": is_rap_urban(genre),
            "genre_rock_indie": is_rock_indie(genre),
            "genre_veteran": is_veteran_genre(genre),
        }

        # Add performance stats if available
        if genre in genre_stats:
            stats = genre_stats[genre]
            feat["genre_avg_performance"] = stats.get("avg_score", 50)
            feat["genre_win_rate"] = stats.get("win_rate", 0.1)
            feat["genre_trend"] = stats.get("trend", 0)
        else:
            # Default values for unknown genres
            feat["genre_avg_performance"] = 50
            feat["genre_win_rate"] = 0.1
            feat["genre_trend"] = 0

        features.append(feat)

    df = pd.DataFrame(features)

    # Normalize numeric features to 0-1
    if not df.empty:
        df["genre_avg_performance"] = df["genre_avg_performance"] / 100
        df["genre_win_rate"] = df["genre_win_rate"]
        df["genre_trend"] = (df["genre_trend"] + 1) / 2  # Scale -1,1 to 0,1

    return df


def normalize_genre(genere: str) -> str:
    """
    Normalizza il nome del genere.

    Args:
        genere: Nome genere grezzo

    Returns:
        Genere normalizzato
    """
    if not genere:
        return "Unknown"

    genere_lower = genere.lower().strip()

    # Check for multi-word genres first
    for key, value in GENRE_NORMALIZATION.items():
        if key.lower() == genere_lower:
            return value

    # Check for partial matches
    if "rap" in genere_lower or "hip" in genere_lower:
        if "pop" in genere_lower:
            return "Pop Rap"
        return "Rap"
    elif "rock" in genere_lower:
        if "pop" in genere_lower:
            return "Pop/Rock"
        if "indie" in genere_lower:
            return "Indie"
        return "Rock"
    elif "indie" in genere_lower or "alt" in genere_lower:
        return "Indie"
    elif "urban" in genere_lower or "trap" in genere_lower:
        return "Urban"
    elif "pop" in genere_lower:
        return "Pop"

    return "Unknown"  # Default for unrecognized genres


def is_mainstream_pop(genre: str) -> bool:
    """Check if genre is mainstream pop."""
    return genre in GENRE_GROUPS["mainstream_pop"]


def is_rap_urban(genre: str) -> bool:
    """Check if genre is rap/urban."""
    return genre in GENRE_GROUPS["rap_urban"]


def is_rock_indie(genre: str) -> bool:
    """Check if genre is rock/indie."""
    return genre in GENRE_GROUPS["rock_indie"]


def is_veteran_genre(genre: str) -> bool:
    """Check if genre is associated with veteran artists."""
    return genre in GENRE_GROUPS["veteran"]


def calculate_genre_performance(
    storico_data: dict | None, genre_map: dict[str, str], as_of_year: int = 2026
) -> dict[str, dict]:
    """
    Calcola statistiche performance per genere.

    Args:
        storico_data: Dati storici FantaSanremo
        genre_map: Mapping artista -> genere

    Returns:
        Dict genere -> {avg_score, win_rate, trend}
    """
    if not storico_data:
        # Return default stats
        return {
            "Pop": {"avg_score": 55, "win_rate": 0.15, "trend": 0.1},
            "Rap": {"avg_score": 60, "win_rate": 0.20, "trend": 0.3},
            "Rock": {"avg_score": 45, "win_rate": 0.10, "trend": -0.1},
            "Indie": {"avg_score": 50, "win_rate": 0.12, "trend": 0.2},
            "Urban": {"avg_score": 58, "win_rate": 0.18, "trend": 0.4},
            "Hip-hop": {"avg_score": 55, "win_rate": 0.15, "trend": 0.2},
            "Pop/Rock": {"avg_score": 52, "win_rate": 0.12, "trend": 0.0},
        }

    # Calculate from actual historical data
    genre_scores = defaultdict(list)
    genre_positions = defaultdict(list)
    genre_by_year = defaultdict(lambda: defaultdict(list))

    for entry in storico_data:
        artista_nome = entry.get("artista")
        if not artista_nome:
            continue

        genre = genre_map.get(normalize_artist_name(artista_nome))
        if not genre:
            continue

        posizioni = entry.get("posizioni", {})
        for anno_str, pos in posizioni.items():
            if pos == "NP" or pos is None:
                continue

            try:
                anno = int(anno_str)
                posizione = int(pos)
                if anno >= as_of_year:
                    continue

                # Calculate score (inverted)
                score = max(1, 31 - posizione)

                genre_scores[genre].append(score)
                genre_positions[genre].append(posizione)
                genre_by_year[genre][anno].append(posizione)

            except (ValueError, TypeError):
                continue

    # Calculate stats per genre
    stats = {}
    for genre in genre_scores.keys():
        scores = genre_scores[genre]
        positions = genre_positions[genre]

        avg_score = np.mean(scores) if scores else 50
        win_rate = sum(1 for p in positions if p <= 5) / len(positions) if positions else 0.1

        # Calculate trend (improvement over time)
        years_data = genre_by_year[genre]
        if len(years_data) >= 2:
            avg_pos_by_year = {
                year: (np.mean(positions) if positions else 15)
                for year, positions in years_data.items()
            }
            sorted_years = sorted(avg_pos_by_year.keys())
            recent_avg = np.mean([avg_pos_by_year[y] for y in sorted_years[-2:]])
            older_years = sorted_years[:-2]
            older_avg = (
                np.mean([avg_pos_by_year[y] for y in older_years]) if older_years else recent_avg
            )

            # Positive trend = improving (lower positions)
            trend = (older_avg - recent_avg) / 30  # Normalize to -1,1
        else:
            trend = 0

        stats[genre] = {"avg_score": avg_score, "win_rate": win_rate, "trend": trend}

    return stats


def get_genre_similarity(genre1: str, genre2: str) -> float:
    """
    Calcola similarità tra due generi.

    Args:
        genre1: Primo genere
        genre2: Secondo genere

    Returns:
        Similarità 0-1
    """
    if genre1 == genre2:
        return 1.0

    # Check if in same group
    for group, genres in GENRE_GROUPS.items():
        if genre1 in genres and genre2 in genres:
            return 0.7

    # Check for partial name matches
    if "rap" in genre1.lower() and "rap" in genre2.lower():
        return 0.5
    if "pop" in genre1.lower() and "pop" in genre2.lower():
        return 0.5
    if "rock" in genre1.lower() and "rock" in genre2.lower():
        return 0.5

    return 0.0


def get_genre_distribution(artisti_data: list[dict], biografico_data: dict) -> dict[str, int]:
    """
    Calcola distribuzione generi negli artisti.

    Args:
        artisti_data: Lista artisti
        biografico_data: Dati biografici

    Returns:
        Dict genere -> count
    """
    genre_map = {}
    if biografico_data:
        for artista in biografico_data.get("artisti_2026_biografico", []):
            nome = artista.get("nome")
            genere = artista.get("genere_musicale")
            key = normalize_artist_name(nome)
            if key and genere:
                genre_map[key] = normalize_genre(genere)

    distribution = defaultdict(int)
    for artista in artisti_data:
        nome = artista.get("nome")
        key = normalize_artist_name(nome)
        genere = genre_map.get(key, "Unknown")
        distribution[genere] += 1

    return dict(distribution)
