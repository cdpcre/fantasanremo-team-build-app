"""
Regulatory and Bonus Features Module

Crea feature basate su bonus regolamentari ufficiali (ad personam).
"""

import sys
from pathlib import Path

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


import pandas as pd

from backend.data_pipeline.config import get_logger
from backend.utils.name_normalization import normalize_artist_name


def create_regulatory_features(
    artisti_data: list[dict],
    biografico_data: dict | None = None,
    caratteristiche_data: dict | None = None,
    regolamento_data: dict | None = None,
    as_of_year: int = 2026,
) -> pd.DataFrame:
    """
    Crea feature basate sul regolamento 2026.

    Feature categories:
    1. Ad Personam Bonus (individual, da regolamento ufficiale)

    Args:
        artisti_data: Lista artisti
        biografico_data: Dati biografici
        caratteristiche_data: Dati caratteristiche
        regolamento_data: Regolamento 2026
        as_of_year: Anno di riferimento per età e carriera

    Returns:
        DataFrame con feature regolatorie
    """
    get_logger("regulatory_features")

    # Build maps from data
    car_map = _build_caratteristiche_map(caratteristiche_data)
    ad_personam_map = _build_ad_personam_map(regolamento_data, artisti_data)

    features = []
    for artista in artisti_data:
        nome = artista.get("nome")
        artista_id = artista.get("id")
        key = normalize_artist_name(nome)
        car = car_map.get(key, {})

        feat = {"artista_id": artista_id, "artista_nome": nome}

        ad_personam = ad_personam_map.get(key, {})
        count = ad_personam.get("count")
        points = ad_personam.get("points")

        if count is None:
            count = car.get("ad_personam_bonus_count")
        if points is None:
            points = car.get("ad_personam_bonus_points")

        feat.update(
            {
                "has_ad_personam_bonus": 1 if count and count > 0 else 0,
                "ad_personam_bonus_count": count or 0,
                "ad_personam_bonus_points": points or 0,
            }
        )

        features.append(feat)

    df = pd.DataFrame(features)

    return df


def _build_caratteristiche_map(caratteristiche_data: dict | None) -> dict:
    """Build map from artist name to characteristics."""
    if not caratteristiche_data:
        return {}

    car_map = {}
    for artista in caratteristiche_data.get("caratteristiche_artisti_2026", []):
        nome = artista.get("nome")
        key = normalize_artist_name(nome)
        if key:
            car_map[key] = {
                "viralita_social": artista.get("viralita_social"),
                "storia_bonus_ottenuti": artista.get("storia_bonus_ottenuti", 0),
                "ad_personam_bonus_count": artista.get("ad_personam_bonus_count"),
                "ad_personam_bonus_points": artista.get("ad_personam_bonus_points"),
            }
    return car_map


def _build_ad_personam_map(regolamento_data: dict | None, artisti_data: list[dict]) -> dict:
    """Build a map of normalized artist name -> ad personam bonus info."""
    if not regolamento_data or not artisti_data:
        return {}

    ad_personam = regolamento_data.get("bonus", {}).get("ad_personam", [])
    if not ad_personam:
        return {}

    name_keys = {
        normalize_artist_name(a.get("nome")): a.get("nome") for a in artisti_data if a.get("nome")
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


def get_regional_bonus_potential(bio: dict, regolamento_data: dict | None = None) -> dict:
    """
    Calcola potenziale bonus regionali (Nord, Centro, Sud, Isole).

    Args:
        bio: Dati biografici artista
        regolamento_data: Regolamento

    Returns:
        Dict con punteggi regionali
    """
    # Simplified regional classification based on genre and associations
    genere = bio.get("genere_musicale", "")

    regional_potential = {
        "nord_score": 0.2,  # Generic default
        "centro_score": 0.2,
        "sud_score": 0.2,
        "isole_score": 0.2,
    }

    # Napoletana -> Sud high
    if "Napoletana" in genere or "napoletana" in genere.lower():
        regional_potential["sud_score"] = 0.8

    # Urban/Rap -> Nord high (Milan scene)
    if "Urban" in genere or "Rap" in genere:
        regional_potential["nord_score"] = 0.6

    # Indie/Pop -> Centro (Rome scene)
    if "Indie" in genere or "Pop" in genere:
        regional_potential["centro_score"] = 0.5

    return regional_potential


def get_bonus_summary(regulatory_df: pd.DataFrame) -> dict:
    """
    Genera summary delle feature regolatorie.

    Args:
        regulatory_df: DataFrame feature regolatorie

    Returns:
        Dict con statistiche
    """
    if regulatory_df.empty:
        return {}

    return {
        "avg_ad_personam_points": regulatory_df["ad_personam_bonus_points"].mean(),
        "artists_with_ad_personam": (regulatory_df["has_ad_personam_bonus"] == 1).sum(),
    }
