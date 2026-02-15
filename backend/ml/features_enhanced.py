"""
Enhanced Features Integration Module

Orchestra tutti i moduli feature engineering in una unified interface.
"""

import numpy as np
import pandas as pd

from backend.data_pipeline.config import get_logger

from .feature_builder import FeatureBuilder


def generate_all_features(
    artisti_data: list[dict],
    storico_data: list[dict],
    biografico_data: dict | None = None,
    caratteristiche_data: dict | None = None,
    regolamento_data: dict | None = None,
    as_of_year: int = 2026,
) -> pd.DataFrame:
    """
    Genera tutte le feature per il ML.

    Combines:
    - Historical features (from features.py)
    - Genre features
    - Characteristics features
    - Regulatory features
    - Biographical features
    - Categorization features

    Args:
        artisti_data: Lista artisti 2026
        storico_data: Lista dati storici
        biografico_data: Dati biografici
        caratteristiche_data: Dati caratteristiche
        regolamento_data: Regolamento 2026

    Returns:
        DataFrame con tutte le feature
    """
    logger = get_logger("features_enhanced")
    logger.info("Generating all features...")

    builder = FeatureBuilder()
    sources = builder.build_sources_from_inputs(
        artisti_data=artisti_data,
        storico_data=storico_data,
        biografico_data=biografico_data,
        caratteristiche_data=caratteristiche_data,
        regolamento_data=regolamento_data,
    )

    features_df = builder.build_prediction_frame(sources, as_of_year=as_of_year)

    logger.info(f"Generated {len(features_df.columns)} features for {len(features_df)} artists")

    return features_df


def fill_missing_features(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing values in features DataFrame.

    Args:
        features_df: DataFrame con feature

    Returns:
        DataFrame con missing values filled
    """
    df = features_df.copy()

    # Binary columns - fill with 0
    binary_cols = [c for c in df.columns if c.startswith(("is_", "has_", "genre_", "gen_"))]
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int)

    # Numeric columns - fill with median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col not in binary_cols and col != "artista_id":
            median_val = df[col].median()
            if pd.isna(median_val):
                median_val = 0
            df[col] = df[col].fillna(median_val)

    return df


def normalize_all_features(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizza tutte le feature numeriche 0-1.

    Args:
        features_df: DataFrame feature

    Returns:
        DataFrame con feature normalizzate
    """
    df = features_df.copy()

    # Skip ID and string columns
    skip_cols = ["artista_id", "artista_nome", "primary_archetype", "artista_nome_y", "nome"]

    for col in df.columns:
        if col in skip_cols:
            continue

        # Skip already normalized columns (binary 0/1)
        if df[col].dtype in [int, bool] and df[col].nunique() <= 2:
            continue

        # Normalize numeric columns
        if pd.api.types.is_numeric_dtype(df[col]):
            min_val = df[col].min()
            max_val = df[col].max()

            if max_val > min_val:
                df[col] = (df[col] - min_val) / (max_val - min_val)
            else:
                df[col] = 0.5  # Neutral value if no variance

            # Clip to 0-1
            df[col] = df[col].clip(0, 1)

    return df


def get_feature_groups() -> dict[str, list[str]]:
    """
    Restituisce i gruppi di feature per analisi.

    Returns:
        Dict con nomi gruppi -> lista feature
    """
    return {
        "historical": [
            "avg_position",
            "position_variance",
            "position_trend",
            "participations",
            "best_position",
            "recent_avg",
            "consistency_score",
            "momentum_score",
            "peak_performance",
            "longevity_bonus",
            "top10_finishes",
            "top5_finishes",
            "median_position",
            "volatility_index",
            "years_since_last",
        ],
        "quotation": ["quotazione_2026", "is_top_quoted", "is_mid_quoted", "is_low_quoted"],
        "genre": [
            "genre_avg_performance",
            "genre_win_rate",
            "genre_trend",
            "genre_mainstream_pop",
            "genre_rap_urban",
            "genre_rock_indie",
        ],
        "characteristics": [
            "viral_potential",
            "social_followers_score",
            "social_followers_total",
            "has_bonus_history",
            "bonus_count",
            "ad_personam_bonus_count",
            "ad_personam_bonus_points",
        ],
        "regulatory": [
            "has_ad_personam_bonus",
            "ad_personam_bonus_count",
            "ad_personam_bonus_points",
        ],
        "biographical": [
            "artist_age",
            "career_length",
            "is_veteran",
            "is_debuttante",
            "experience_score",
            "sanremo_veteran_bonus",
            "gen_z",
            "millennial",
            "gen_x",
            "boomer",
        ],
        "archetypes": [
            "VIRAL_PHENOMENON",
            "VETERAN_PERFORMER",
            "INDIE_DARLING",
            "RAP_TRAP_STAR",
            "POP_MAINSTREAM",
            "LEGENDARY_STATUS",
            "DEBUTTANTE_POTENTIAL",
        ],
    }


def get_all_feature_names() -> list[str]:
    """
    Restituisce tutti i nomi delle feature.

    Returns:
        Lista completa nomi feature
    """
    groups = get_feature_groups()
    all_features = []
    for group_features in groups.values():
        all_features.extend(group_features)
    return sorted(set(all_features))


def filter_features_by_importance(
    features_df: pd.DataFrame,
    importances: dict[str, float],
    top_n: int | None = None,
    min_importance: float | None = None,
) -> pd.DataFrame:
    """
    Filtra feature per importanza.

    Args:
        features_df: DataFrame feature
        importances: Dict feature -> importanza
        top_n: Mantieni top N feature
        min_importance: Mantieni feature con importanza >= min

    Returns:
        DataFrame filtrato + artista_id
    """
    # Always keep artist_id
    keep_cols = ["artista_id"]

    # Filter by importance
    sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)

    if top_n:
        keep_cols.extend([f for f, _ in sorted_features[:top_n] if f in features_df.columns])

    if min_importance:
        keep_cols.extend(
            [f for f, imp in sorted_features if imp >= min_importance and f in features_df.columns]
        )

    # Remove duplicates
    keep_cols = list(dict.fromkeys(keep_cols))

    return features_df[keep_cols]


def get_feature_statistics(features_df: pd.DataFrame) -> dict:
    """
    Calcola statistiche sulle feature.

    Args:
        features_df: DataFrame feature

    Returns:
        Dict con statistiche
    """
    stats = {
        "total_features": len(features_df.columns),
        "total_artists": len(features_df),
        "missing_values": features_df.isnull().sum().to_dict(),
        "feature_types": features_df.dtypes.value_counts().to_dict(),
    }

    # Feature group counts
    groups = get_feature_groups()
    group_counts = {}
    for group_name, group_features in groups.items():
        count = sum(1 for f in group_features if f in features_df.columns)
        group_counts[group_name] = count

    stats["features_by_group"] = group_counts

    return stats
