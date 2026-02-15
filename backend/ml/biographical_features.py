"""
Biographical Features Module

Crea feature dai dati biografici (età, esperienza, generazioni).
"""

import sys
from pathlib import Path

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


import pandas as pd

from backend.data_pipeline.config import get_logger
from backend.utils.name_normalization import normalize_artist_name

# Generational cohorts definition
GENERATIONAL_COHORTS = {
    "gen_z": {"range": (1997, 2012), "age_2026": (14, 29), "description": "Gen Z (14-29)"},
    "millennial": {
        "range": (1981, 1996),
        "age_2026": (30, 45),
        "description": "Millennial (30-45)",
    },
    "gen_x": {"range": (1965, 1980), "age_2026": (46, 61), "description": "Gen X (46-61)"},
    "boomer": {"range": (1946, 1964), "age_2026": (62, 80), "description": "Boomer (62-80)"},
}


def create_biographical_features(
    artisti_data: list[dict],
    biografico_data: dict | None = None,
    storico_data: list | None = None,
    as_of_year: int = 2026,
) -> pd.DataFrame:
    """
    Crea feature biografiche per artisti.

    Features:
    - artist_age: Età nel as_of_year
    - career_length: Anni di carriera (as_of_year - prima_partecipazione)
    - is_veteran: career_length >= 10 (binary)
    - is_debuttante: prima_partecipazione == 2026 (binary)
    - experience_score: career_length * 10 (normalized 0-1)
    - sanremo_veteran_bonus: >=3 partecipazioni (binary)

    Generational cohorts (one-hot):
    - gen_z: 1997-2012 (18-29 anni)
    - millennial: 1981-1996 (30-45)
    - gen_x: 1965-1980 (46-61)
    - boomer: 1946-1964 (62-80)

    Args:
        artisti_data: Lista artisti
        biografico_data: Dati biografici
        storico_data: Dati storici
        as_of_year: Anno di riferimento per età e carriera

    Returns:
        DataFrame con feature biografiche
    """
    get_logger("biographical_features")

    # Build biografico map
    bio_map = {}
    if biografico_data:
        for artista in biografico_data.get("artisti_2026_biografico", []):
            nome = artista.get("nome")
            key = normalize_artist_name(nome)
            if key:
                bio_map[key] = {
                    "genere_musicale": artista.get("genere_musicale"),
                    "anno_nascita": artista.get("anno_nascita"),
                    "prima_partecipazione": artista.get("prima_partecipazione", 2026),
                }

    # Build storico map for participation count
    storico_map = {}
    if storico_data:
        for entry in storico_data:
            nome = entry.get("artista")
            key = normalize_artist_name(nome)
            if key:
                posizioni = entry.get("posizioni", {})
                # Count actual participations (not NP)
                partecipazioni = len([p for p in posizioni.values() if p != "NP"])
                storico_map[key] = partecipazioni

    features = []
    for artista in artisti_data:
        nome = artista.get("nome")
        artista_id = artista.get("id")

        # Get biographical data
        key = normalize_artist_name(nome)
        bio = bio_map.get(key, {})
        anno_nascita = bio.get("anno_nascita")
        prima_part = bio.get("prima_partecipazione", 2026)

        # Calculate age (handle missing data)
        if anno_nascita:
            artist_age = as_of_year - anno_nascita
        else:
            artist_age = 35  # Default median age

        # Calculate career length
        if prima_part and prima_part < as_of_year:
            career_length = as_of_year - prima_part
        else:
            career_length = 0

        # Get participations from storico or calculate from career
        if key in storico_map:
            partecipazioni = storico_map[key]
        else:
            partecipazioni = career_length if career_length > 0 else 0

        # Determine generational cohort
        cohort = get_generational_cohort(anno_nascita)

        # Build features
        feat = {
            "artista_id": artista_id,
            "artista_nome": nome,
            "artist_age": min(artist_age / 50, 1.0),  # Normalize age 0-1 (cap at 50)
            "career_length": min(career_length / 20, 1.0),  # Normalize 0-1 (cap at 20 years)
            "is_veteran": 1 if career_length >= 10 else 0,
            "is_debuttante": 1 if partecipazioni == 0 else 0,
            "experience_score": min(career_length * 10 / 100, 1.0),  # Normalize 0-1
            "sanremo_veteran_bonus": 1 if partecipazioni >= 3 else 0,
            # Generational cohorts (one-hot)
            "gen_z": 1 if cohort == "gen_z" else 0,
            "millennial": 1 if cohort == "millennial" else 0,
            "gen_x": 1 if cohort == "gen_x" else 0,
            "boomer": 1 if cohort == "boomer" else 0,
        }

        features.append(feat)

    df = pd.DataFrame(features)

    # Fill missing values
    if not df.empty:
        df["artist_age"] = df["artist_age"].fillna(0.7)  # Default to ~35yo
        df["career_length"] = df["career_length"].fillna(0)
        df["is_veteran"] = df["is_veteran"].fillna(0).astype(int)
        df["is_debuttante"] = df["is_debuttante"].fillna(1).astype(int)
        df["experience_score"] = df["experience_score"].fillna(0)
        df["sanremo_veteran_bonus"] = df["sanremo_veteran_bonus"].fillna(0).astype(int)

    return df


def get_generational_cohort(anno_nascita: int | None) -> str:
    """
    Determina la generazione di appartenenza.

    Args:
        anno_nascita: Anno di nascita

    Returns:
        Generazione (gen_z, millennial, gen_x, boomer, unknown)
    """
    if not anno_nascita:
        return "millennial"  # Default assumption

    for cohort, info in GENERATIONAL_COHORTS.items():
        min_year, max_year = info["range"]
        if min_year <= anno_nascita <= max_year:
            return cohort

    # Before boomers
    if anno_nascita < 1946:
        return "boomer"

    # After Gen Z (too young for Sanremo)
    if anno_nascita > 2012:
        return "gen_z"

    return "millennial"  # Default


def calculate_age_specific_features(eta: int) -> dict:
    """
    Calcola feature specifiche per età.

    Args:
        eta: Età dell'artista

    Returns:
        Dict con feature età-specifiche
    """
    # Age-based categories
    if eta < 25:
        age_category = "very_young"
        peak_potential = 0.9
        experience_factor = 0.2
    elif eta < 35:
        age_category = "young"
        peak_potential = 1.0
        experience_factor = 0.5
    elif eta < 45:
        age_category = "mid_career"
        peak_potential = 0.8
        experience_factor = 0.7
    elif eta < 55:
        age_category = "veteran"
        peak_potential = 0.6
        experience_factor = 0.9
    else:
        age_category = "legend"
        peak_potential = 0.4
        experience_factor = 1.0

    return {
        "age_category": age_category,
        "peak_potential": peak_potential,
        "experience_factor": experience_factor,
    }


def estimate_missing_birth_year(
    artista_nome: str, genere: str, prima_part: int | None = None
) -> int:
    """
    Stima anno di nascita mancante.

    Args:
        artista_nome: Nome artista
        genere: Genere musicale
        prima_part: Prima partecipazione

    Returns:
        Anno di nascita stimato
    """
    # Use prima partecipazione as base
    if prima_part and prima_part < 2026:
        # Assume artist was ~25 when first participated
        return prima_part - 25

    # Use genre-based assumptions
    if "Rap" in genere or "Urban" in genere or "Trap" in genere:
        return 1995  # Rappers typically younger
    elif "Indie" in genere or "Rock" in genere:
        return 1990  # Indie/rock in 90s
    elif "Napoletana" in genere:
        return 1970  # Neapolitan veterans
    else:
        return 1985  # Generic pop assumption


def get_experience_metrics(career_length: int, partecipazioni: int) -> dict:
    """
    Calcola metriche esperienza.

    Args:
        career_length: Anni di carriera
        partecipazioni: Numero partecipazioni

    Returns:
        Dict con metriche esperienza
    """
    # Participation rate (per year)
    if career_length > 0:
        participation_rate = partecipazioni / career_length
    else:
        participation_rate = 0

    # Consistency (high participation rate = consistent)
    consistency = min(participation_rate, 1.0)

    # Experience level
    if career_length == 0:
        experience_level = "debuttante"
    elif career_length < 5:
        experience_level = "early_career"
    elif career_length < 10:
        experience_level = "mid_career"
    elif career_length < 20:
        experience_level = "veteran"
    else:
        experience_level = "legend"

    return {
        "participation_rate": participation_rate,
        "consistency": consistency,
        "experience_level": experience_level,
    }


def get_biographical_summary(bio_df: pd.DataFrame) -> dict:
    """
    Genera summary feature biografiche.

    Args:
        bio_df: DataFrame feature biografiche

    Returns:
        Dict con statistiche
    """
    if bio_df.empty:
        return {}

    return {
        "avg_age": bio_df["artist_age"].mean() * 50,  # Denormalize
        "avg_career_length": bio_df["career_length"].mean() * 20,
        "veteran_count": int(bio_df["is_veteran"].sum()),
        "debuttante_count": int(bio_df["is_debuttante"].sum()),
        "sanremo_veteran_count": int(bio_df["sanremo_veteran_bonus"].sum()),
        "generational_distribution": {
            "gen_z": int(bio_df["gen_z"].sum()),
            "millennial": int(bio_df["millennial"].sum()),
            "gen_x": int(bio_df["gen_x"].sum()),
            "boomer": int(bio_df["boomer"].sum()),
        },
    }


def interpolate_age_for_missing(
    bio_df: pd.DataFrame, reference_median: float | None = None
) -> pd.DataFrame:
    """
    Interpola età mancante con median.

    Args:
        bio_df: DataFrame feature biografiche
        reference_median: Mediana di riferimento (usa median df se None)

    Returns:
        DataFrame con età interpolata
    """
    if bio_df.empty:
        return bio_df

    # Use provided median or calculate from data
    if reference_median is None:
        # Calculate median from non-zero ages
        non_zero_ages = bio_df[bio_df["artist_age"] > 0]["artist_age"]
        if not non_zero_ages.empty:
            reference_median = non_zero_ages.median()
        else:
            reference_median = 0.7  # Default to ~35yo

    # Fill zero ages with median
    bio_df["artist_age"] = bio_df["artist_age"].replace(0, reference_median)

    return bio_df
