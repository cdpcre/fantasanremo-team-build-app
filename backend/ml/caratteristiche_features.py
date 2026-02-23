"""
Characteristics-Based Features Module

Crea feature dai dati caratteristiche basate su viralità social reale e bonus regolamentari.
"""

import sys
from pathlib import Path

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


import math

import pandas as pd

from backend.data_pipeline.config import get_logger
from backend.utils.name_normalization import normalize_artist_name


def create_caratteristiche_features(
    artisti_data: list[dict], caratteristiche_data: dict | None = None
) -> pd.DataFrame:
    """
    Crea feature dai dati caratteristiche artisti.

    Features da caratteristiche.json:
    - viral_potential: viralita_social / 100 (reale)
    - social_followers_score: log-scaled follower score (0-1)
    - social_followers_total: follower totali (raw)
    - has_bonus_history: storia_bonus_ottenuti > 0 (binary)
    - bonus_count: storia_bonus_ottenuti (somma punti storici)
    - ad_personam_bonus_count: count bonus ad personam da regolamento
    - ad_personam_bonus_points: somma punti bonus ad personam da regolamento

    Args:
        artisti_data: Lista di dict artisti
        caratteristiche_data: Dict con caratteristiche_artisti_2026

    Returns:
        DataFrame con feature caratteristiche
    """
    get_logger("caratteristiche_features")

    # Build caratteristiche map
    car_map = {}
    if caratteristiche_data:
        for artista in caratteristiche_data.get("caratteristiche_artisti_2026", []):
            nome = artista.get("nome")
            key = normalize_artist_name(nome)
            if key:
                car_map[key] = {
                    "viralita_social": artista.get("viralita_social"),
                    "social_followers_total": artista.get("social_followers_total"),
                    "storia_bonus_ottenuti": artista.get("storia_bonus_ottenuti", 0),
                    "ad_personam_bonus_count": artista.get("ad_personam_bonus_count"),
                    "ad_personam_bonus_points": artista.get("ad_personam_bonus_points"),
                }

    features = []
    for artista in artisti_data:
        nome = artista.get("nome")
        artista_id = artista.get("id")

        # Get characteristics for this artist
        key = normalize_artist_name(nome)
        car = car_map.get(key, {})

        viralita = car.get("viralita_social")
        followers_total = car.get("social_followers_total")
        bonus_history = car.get("storia_bonus_ottenuti", 0)
        ad_personam_count = car.get("ad_personam_bonus_count", 0)
        ad_personam_points = car.get("ad_personam_bonus_points", 0)

        viral_potential = None
        if isinstance(viralita, (int, float)):
            viral_potential = max(0.0, min(1.0, float(viralita) / 100))

        followers_score = None
        if isinstance(followers_total, (int, float)) and followers_total > 0:
            followers_score = min(1.0, math.log10(followers_total + 1) / 7)

        # Calculate features
        feat = {
            "artista_id": artista_id,
            "artista_nome": nome,
            "viral_potential": viral_potential,
            "social_followers_score": followers_score,
            "social_followers_total": followers_total,
            "has_bonus_history": 1 if bonus_history > 0 else 0,
            "bonus_count": bonus_history,
            "ad_personam_bonus_count": ad_personam_count,
            "ad_personam_bonus_points": ad_personam_points,
        }

        # Clip normalized values to 0-1
        for key, value in feat.items():
            if key in ["viral_potential", "social_followers_score"] and isinstance(
                value, (int, float)
            ):
                feat[key] = max(0.0, min(1.0, float(value)))

        features.append(feat)

    df = pd.DataFrame(features)

    if not df.empty:
        fill_cols = [
            "has_bonus_history",
            "bonus_count",
            "ad_personam_bonus_count",
            "ad_personam_bonus_points",
        ]
        for col in fill_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0)

    return df


def create_viral_features(caratteristiche_data: dict) -> pd.DataFrame:
    """
    Crea feature specifiche per viralità.

    Args:
        caratteristiche_data: Dict caratteristiche

    Returns:
        DataFrame con feature viralità
    """
    if not caratteristiche_data:
        return pd.DataFrame()

    artisti = caratteristiche_data.get("caratteristiche_artisti_2026", [])

    features = []
    for artista in artisti:
        nome = artista.get("nome")
        if not nome:
            continue

        viralita = artista.get("viralita_social")
        if viralita is None:
            continue

        # Viral categories
        if viralita >= 80:
            viral_category = "viral_superstar"
        elif viralita >= 60:
            viral_category = "viral_high"
        elif viralita >= 40:
            viral_category = "viral_medium"
        else:
            viral_category = "viral_low"

        features.append(
            {
                "nome": nome,
                "viral_category": viral_category,
                "viral_score_normalized": viralita / 100,
                "is_viral_phenomenon": 1 if viralita >= 80 else 0,
            }
        )

    return pd.DataFrame(features)


def create_performance_features(caratteristiche_data: dict) -> pd.DataFrame:
    """
    Crea feature basate su performance attesa.

    Args:
        caratteristiche_data: Dict caratteristiche

    Returns:
        DataFrame con feature performance
    """
    if not caratteristiche_data:
        return pd.DataFrame()

    artisti = caratteristiche_data.get("caratteristiche_artisti_2026", [])

    features = []
    for artista in artisti:
        nome = artista.get("nome")
        if not nome:
            continue

        viralita = artista.get("viralita_social")
        bonus_history = artista.get("storia_bonus_ottenuti", 0)

        if viralita is None:
            continue

        # Performance potential score derived from real viralita_social
        performance_potential = viralita / 100

        # Bonus affinity (high for artists with bonus history)
        bonus_affinity = min(bonus_history / 5, 1.0)  # Cap at 5 bonuses

        features.append(
            {
                "nome": nome,
                "performance_potential": performance_potential,
                "bonus_affinity": bonus_affinity,
                "is_viral": 1 if viralita >= 70 else 0,
            }
        )

    df = pd.DataFrame(features)

    # Normalize numeric columns
    numeric_cols = ["performance_potential", "bonus_affinity"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].clip(0, 1)

    return df


def get_top_viral_artists(caratteristiche_data: dict, top_n: int = 10) -> list[dict]:
    """
    Restituisce gli artisti più virali.

    Args:
        caratteristiche_data: Dict caratteristiche
        top_n: Numero top artisti

    Returns:
        Lista di dict con nome e viralita
    """
    if not caratteristiche_data:
        return []

    artisti = caratteristiche_data.get("caratteristiche_artisti_2026", [])

    # Sort by viralita
    sorted_artists = sorted(artisti, key=lambda x: x.get("viralita_social", 0), reverse=True)

    return [
        {
            "nome": a.get("nome"),
            "viralita_social": a.get("viralita_social"),
        }
        for a in sorted_artists[:top_n]
    ]


def get_bonus_affinity_scores(caratteristiche_data: dict) -> dict[str, float]:
    """
    Calcola punteggi affinità bonus per artista.

    Args:
        caratteristiche_data: Dict caratteristiche

    Returns:
        Dict nome -> bonus_affinity_score (0-1)
    """
    if not caratteristiche_data:
        return {}

    artisti = caratteristiche_data.get("caratteristiche_artisti_2026", [])

    scores = {}
    for artista in artisti:
        nome = artista.get("nome")
        if not nome:
            continue

        bonus_history = artista.get("storia_bonus_ottenuti", 0)
        viralita = artista.get("viralita_social")

        if viralita is None:
            continue

        # Bonus affinity: combines history with viral potential
        historical_component = min(bonus_history / 5, 1.0) * 0.6
        potential_component = (viralita / 100) * 0.4

        scores[nome] = historical_component + potential_component

    return scores
