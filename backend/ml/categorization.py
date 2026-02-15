"""
Artist Categorization Module

Sistema di categorizzazione artisti in 7 archetipi.
"""

import sys
from pathlib import Path

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


import pandas as pd

from backend.utils.name_normalization import normalize_artist_name

# Archetipi definiti
ARCHETYPES = {
    "VIRAL_PHENOMENON": {
        "description": "Alta viralità social, giovani",
        "criteria": {"viralita_min": 80, "eta_max": 35},
    },
    "VETERAN_PERFORMER": {
        "description": "Veterani con almeno 3 partecipazioni, over 40",
        "criteria": {"partecipazioni_min": 3, "eta_min": 40},
    },
    "INDIE_DARLING": {
        "description": "Artisti Indie/Rock con bassa viralità",
        "criteria": {"generi": ["Indie", "Rock", "Indie Pop"], "viralita_max": 50},
    },
    "RAP_TRAP_STAR": {
        "description": "Artisti Rap/Hip-hop/Under 30",
        "criteria": {"generi": ["Rap", "Hip-hop", "Urban", "Trap"], "eta_max": 30},
    },
    "POP_MAINSTREAM": {
        "description": "Pop mainstream con quotazione alta",
        "criteria": {"generi": ["Pop", "Pop/Rock", "Dance"], "quotazione_min": 15},
    },
    "LEGENDARY_STATUS": {
        "description": "Leggende over 60, prima partecipazione pre-1990",
        "criteria": {"eta_min": 60, "prima_partecipazione_max": 1990},
    },
    "DEBUTTANTE_POTENTIAL": {
        "description": "Debuttanti con quotazione alta",
        "criteria": {"partecipazioni": 0, "quotazione_min": 14},
    },
}


def categorize_artist(
    artista: dict,
    biografico: dict | None = None,
    caratteristiche: dict | None = None,
    storico: dict | None = None,
    as_of_year: int = 2026,
) -> dict[str, any]:
    """
    Categorizza un artista negli archetipi.

    Args:
        artista: Dict artista (id, nome, quotazione)
        biografico: Dati biografici
        caratteristiche: Dati caratteristiche
        storico: Dati storici

    Returns:
        Dict con archetipi (one-hot + label)
    """
    nome = artista.get("nome")
    quotazione = artista.get("quotazione", 15)

    # Extract data from sources
    bio_data = _find_artist_data(nome, biografico, "artisti_2026_biografico")
    car_data = _find_artist_data(nome, caratteristiche, "caratteristiche_artisti_2026")
    storico_data = _find_artist_data(nome, storico, None) if storico else None

    # Get artist attributes
    genere = bio_data.get("genere_musicale", "Pop") if bio_data else "Pop"
    anno_nascita = bio_data.get("anno_nascita") if bio_data else None
    prima_part = bio_data.get("prima_partecipazione", 2026) if bio_data else 2026

    viralita = car_data.get("viralita_social", 50) if car_data else 50

    # Calculate age
    eta = as_of_year - anno_nascita if anno_nascita else 35
    career_length = as_of_year - prima_part if prima_part else 0

    # Count participations from storico
    partecipazioni = 0
    if storico_data:
        posizioni = storico_data.get("posizioni", {})
        partecipazioni = len([p for p in posizioni.values() if p != "NP"])
    else:
        partecipazioni = career_length if career_length > 0 else 0

    # Check each archetype
    archetypes = {
        "VIRAL_PHENOMENON": _is_viral_phenomenon(viralita, eta),
        "VETERAN_PERFORMER": _is_veteran_performer(partecipazioni, eta),
        "INDIE_DARLING": _is_indie_darling(genere, viralita),
        "RAP_TRAP_STAR": _is_rap_trap_star(genere, eta),
        "POP_MAINSTREAM": _is_pop_mainstream(genere, quotazione),
        "LEGENDARY_STATUS": _is_legendary_status(eta, prima_part),
        "DEBUTTANTE_POTENTIAL": _is_debuttante_potential(partecipazioni, quotazione),
    }

    # Determine primary archetype (first match)
    primary_archetype = "UNCATEGORIZED"
    for arch, is_match in archetypes.items():
        if is_match:
            primary_archetype = arch
            break

    return {
        "artista_nome": nome,
        "primary_archetype": primary_archetype,
        **archetypes,  # One-hot encoded
    }


def categorize_all_artists(
    artisti_data: list[dict],
    biografico_data: dict | None = None,
    caratteristiche_data: dict | None = None,
    storico_data: list | None = None,
    as_of_year: int = 2026,
) -> pd.DataFrame:
    """
    Categorizza tutti gli artisti.

    Args:
        artisti_data: Lista artisti
        biografico_data: Dati biografici
        caratteristiche_data: Dati caratteristiche
        storico_data: Lista dati storici

    Returns:
        DataFrame con categorizzazioni
    """
    # Build lookup maps
    bio_map = {}
    if biografico_data:
        for a in biografico_data.get("artisti_2026_biografico", []):
            key = normalize_artist_name(a.get("nome"))
            if key:
                bio_map[key] = a

    car_map = {}
    if caratteristiche_data:
        for a in caratteristiche_data.get("caratteristiche_artisti_2026", []):
            key = normalize_artist_name(a.get("nome"))
            if key:
                car_map[key] = a

    storico_map = {}
    if storico_data:
        for a in storico_data:
            key = normalize_artist_name(a.get("artista"))
            if key:
                storico_map[key] = a

    categories = []
    for artista in artisti_data:
        cat = categorize_artist(
            artista,
            biografico=bio_map.get(normalize_artist_name(artista.get("nome"))),
            caratteristiche=car_map.get(normalize_artist_name(artista.get("nome"))),
            storico=storico_map.get(normalize_artist_name(artista.get("nome"))),
            as_of_year=as_of_year,
        )
        cat["artista_id"] = artista.get("id")
        categories.append(cat)

    return pd.DataFrame(categories)


def _find_artist_data(nome: str, source_data: dict | None, list_key: str | None) -> dict:
    """Trova dati artista nella sorgente."""
    if not source_data or not nome:
        return {}

    key = normalize_artist_name(nome)
    if list_key:
        for a in source_data.get(list_key, []):
            if normalize_artist_name(a.get("nome")) == key:
                return a
    else:
        if normalize_artist_name(source_data.get("artista")) == key:
            return source_data

    return {}


# Archetype checker functions
def _is_viral_phenomenon(viralita: int, eta: int) -> bool:
    """VIRAL_PHENOMENON: viralita >= 80 AND eta < 35"""
    return viralita >= 80 and eta < 35


def _is_veteran_performer(partecipazioni: int, eta: int) -> bool:
    """VETERAN_PERFORMER: partecipazioni >= 3 AND eta > 40"""
    return partecipazioni >= 3 and eta > 40


def _is_indie_darling(genere: str, viralita: int) -> bool:
    """INDIE_DARLING: Indie/Rock AND viralita < 50"""
    indie_genres = ["Indie", "Rock", "Indie Pop", "Alternative", "Pop-punk"]
    return any(g in genere for g in indie_genres) and viralita < 50


def _is_rap_trap_star(genere: str, eta: int) -> bool:
    """RAP_TRAP_STAR: Rap/Hip-hop/Urban AND eta < 30"""
    rap_genres = ["Rap", "Hip-hop", "Urban", "Trap", "Pop Rap"]
    return any(g in genere for g in rap_genres) and eta < 30


def _is_pop_mainstream(genere: str, quotazione: int) -> bool:
    """POP_MAINSTREAM: Pop AND quotazione >= 15"""
    pop_genres = ["Pop", "Pop/Rock", "Dance", "Elettronica"]
    return any(g in genere for g in pop_genres) and quotazione >= 15


def _is_legendary_status(eta: int, prima_part: int) -> bool:
    """LEGENDARY_STATUS: eta > 60 AND prima_partecipazione < 1990"""
    return eta > 60 and prima_part < 1990


def _is_debuttante_potential(partecipazioni: int, quotazione: int) -> bool:
    """DEBUTTANTE_POTENTIAL: partecipazioni == 0 AND quotazione >= 14"""
    return partecipazioni == 0 and quotazione >= 14


def get_archetype_features(categorization_df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea feature one-hot encoded dagli archetipi.

    Args:
        categorization_df: DataFrame da categorize_all_artists

    Returns:
        DataFrame con feature archetipi
    """
    if categorization_df.empty:
        return pd.DataFrame()

    # Select archetype columns (all except ID and name columns)
    archetype_cols = [
        c
        for c in categorization_df.columns
        if c not in ["artista_id", "artista_nome", "primary_archetype"]
    ]

    features = categorization_df[["artista_id"] + archetype_cols].copy()

    # Convert boolean to int
    for col in archetype_cols:
        if col in features.columns:
            features[col] = features[col].astype(int)

    return features


def get_archetype_summary(categorization_df: pd.DataFrame) -> dict:
    """
    Genera summary della categorizzazione.

    Args:
        categorization_df: DataFrame categorizzazioni

    Returns:
        Dict con statistiche
    """
    if categorization_df.empty:
        return {}

    archetype_cols = [
        c
        for c in categorization_df.columns
        if c not in ["artista_id", "artista_nome", "primary_archetype"]
    ]

    summary = {
        "total_artists": len(categorization_df),
        "archetype_distribution": {},
        "multi_archetype_artists": 0,
    }

    # Count per archetype
    for col in archetype_cols:
        count = categorization_df[col].sum()
        summary["archetype_distribution"][col] = int(count)

    # Count artists with multiple archetypes
    archetype_counts = categorization_df[archetype_cols].sum(axis=1)
    summary["multi_archetype_artists"] = int((archetype_counts > 1).sum())

    # Primary archetype distribution
    if "primary_archetype" in categorization_df.columns:
        summary["primary_distribution"] = (
            categorization_df["primary_archetype"].value_counts().to_dict()
        )

    return summary


def get_archetype_description(archetype: str) -> str:
    """Restituisce descrizione archetipo."""
    return ARCHETYPES.get(archetype, {}).get("description", "Unknown archetype")


def get_archetype_criteria(archetype: str) -> dict:
    """Restituisce criteri archetipo."""
    return ARCHETYPES.get(archetype, {}).get("criteria", {})
