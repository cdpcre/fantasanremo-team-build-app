"""
Interaction Features Module

Crea feature di interazione per catturare relazioni complesse tra feature.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from backend.data_pipeline.config import get_logger


def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create domain-informed interaction features.

    Interactions:
    - Age × Genre (captures genre-specific age effects)
    - Experience × Archetype (archetype-specific experience benefits)
    - Viral × Genre (viral potential by genre)
    - Bonus × Experience (experience moderates bonus impact)

    Args:
        df: DataFrame with base features to create interactions from

    Returns:
        DataFrame with interaction features added
    """
    logger = get_logger("interaction_features")

    if df.empty:
        return df

    df = df.copy()

    # Age × Genre interactions
    # Age affects different genres differently (e.g., pop vs rock)
    if "artist_age" in df.columns:
        genre_cols = [
            "genre_mainstream_pop",
            "genre_rap_urban",
            "genre_rock_indie",
            "genre_veteran",
        ]
        for genre_col in genre_cols:
            if genre_col in df.columns:
                interaction_name = f"age_{genre_col}"
                df[interaction_name] = (df["artist_age"] * df[genre_col]).astype(float)
                logger.debug(f"Created interaction: {interaction_name}")

    # Experience × Archetype interactions
    # Experience benefits different archetypes differently
    experience_col = "experience_score"
    if experience_col in df.columns:
        for archetype_col in [
            "VIRAL_PHENOMENON",
            "VETERAN_PERFORMER",
            "INDIE_DARLING",
            "RAP_TRAP_STAR",
            "POP_MAINSTREAM",
            "LEGENDARY_STATUS",
            "DEBUTTANTE_POTENTIAL",
        ]:
            if archetype_col in df.columns:
                interaction_name = f"experience_{archetype_col.lower()}"
                df[interaction_name] = (df[experience_col] * df[archetype_col]).astype(float)
                logger.debug(f"Created interaction: {interaction_name}")

    # Viral × Genre interactions
    # Viral potential varies by genre (e.g., viral pop vs viral rock)
    viral_col = "viral_potential"
    if viral_col in df.columns:
        genre_cols = [
            "genre_mainstream_pop",
            "genre_rap_urban",
            "genre_rock_indie",
            "genre_veteran",
        ]
        for genre_col in genre_cols:
            if genre_col in df.columns:
                interaction_name = f"viral_{genre_col}"
                df[interaction_name] = (df[viral_col] * df[genre_col]).astype(float)
                logger.debug(f"Created interaction: {interaction_name}")

    # Bonus × Experience interactions
    # Veterans may handle bonus pressure differently than newcomers
    bonus_col = "ad_personam_bonus_points"
    # Handle column name variations from merge operations
    if bonus_col not in df.columns:
        for col in df.columns:
            if "ad_personam_bonus_points" in col:
                bonus_col = col
                break

    if bonus_col in df.columns and experience_col in df.columns:
        # Normalize bonus points to 0-1 range for interaction
        bonus_normalized = df[bonus_col] / (df[bonus_col].max() + 1e-6)
        df["bonus_experience_interaction"] = (bonus_normalized * df[experience_col]).astype(float)
        logger.debug("Created interaction: bonus_experience_interaction")

    # Additional: Age × Viral interaction (young viral artists)
    if "artist_age" in df.columns and viral_col in df.columns:
        df["age_viral_interaction"] = (df["artist_age"] * df[viral_col]).astype(float)
        logger.debug("Created interaction: age_viral_interaction")

    # Additional: Career Length × Genre (genre-specific career patterns)
    if "career_length" in df.columns:
        for genre_col in ["genre_mainstream_pop", "genre_rap_urban", "genre_rock_indie"]:
            if genre_col in df.columns:
                interaction_name = f"career_{genre_col}"
                df[interaction_name] = (df["career_length"] * df[genre_col]).astype(float)
                logger.debug(f"Created interaction: {interaction_name}")

    return df


def get_interaction_feature_names() -> list[str]:
    """
    Get list of all interaction feature names.

    Returns:
        List of interaction feature names
    """
    features = []

    # Age × Genre
    for genre in ["genre_mainstream_pop", "genre_rap_urban", "genre_rock_indie", "genre_veteran"]:
        features.append(f"age_{genre}")

    # Experience × Archetype
    for archetype in [
        "viral_phenomenon",
        "veteran_performer",
        "indie_darling",
        "rap_trap_star",
        "pop_mainstream",
        "legendary_status",
        "debuttante_potential",
    ]:
        features.append(f"experience_{archetype}")

    # Viral × Genre
    for genre in ["genre_mainstream_pop", "genre_rap_urban", "genre_rock_indie", "genre_veteran"]:
        features.append(f"viral_{genre}")

    # Bonus × Experience
    features.append("bonus_experience_interaction")

    # Age × Viral
    features.append("age_viral_interaction")

    # Career Length × Genre
    for genre in ["genre_mainstream_pop", "genre_rap_urban", "genre_rock_indie"]:
        features.append(f"career_{genre}")

    return features


def validate_interaction_features(df: pd.DataFrame) -> dict[str, Any]:
    """
    Validate interaction features for data quality.

    Args:
        df: DataFrame with interaction features

    Returns:
        Dict with validation results
    """
    logger = get_logger("interaction_features")

    interaction_names = get_interaction_feature_names()
    existing_interactions = [col for col in interaction_names if col in df.columns]

    results = {
        "total_interactions_expected": len(interaction_names),
        "total_interactions_created": len(existing_interactions),
        "missing_interactions": [col for col in interaction_names if col not in df.columns],
        "null_counts": {},
        "stats": {},
    }

    # Check for null values
    for col in existing_interactions:
        null_count = df[col].isna().sum()
        results["null_counts"][col] = int(null_count)

        # Calculate stats if no nulls
        if null_count == 0:
            results["stats"][col] = {
                "min": float(df[col].min()),
                "max": float(df[col].max()),
                "mean": float(df[col].mean()),
                "std": float(df[col].std()),
            }

    # Log validation results
    msg = (
        f"Interaction validation: {results['total_interactions_created']}/"
        f"{results['total_interactions_expected']} features created"
    )
    logger.info(msg)

    if results["missing_interactions"]:
        logger.warning(f"Missing interactions: {results['missing_interactions']}")

    return results


def get_interaction_summary(df: pd.DataFrame) -> dict[str, Any]:
    """
    Generate summary of interaction features.

    Args:
        df: DataFrame with interaction features

    Returns:
        Dict with summary statistics
    """
    interaction_names = get_interaction_feature_names()
    existing_interactions = [col for col in interaction_names if col in df.columns]

    if not existing_interactions:
        return {"error": "No interaction features found"}

    summary = {
        "feature_count": len(existing_interactions),
        "features": existing_interactions,
        "correlations": {},
    }

    # Calculate correlation matrix for interaction features
    if len(existing_interactions) > 1:
        corr_matrix = df[existing_interactions].corr()
        # Find high correlations (>0.8)
        high_corr = []
        for i, col1 in enumerate(existing_interactions):
            for col2 in existing_interactions[i + 1 :]:
                corr_val = corr_matrix.loc[col1, col2]
                if abs(corr_val) > 0.8:
                    high_corr.append({"pair": (col1, col2), "correlation": float(corr_val)})

        summary["high_correlations"] = high_corr

    return summary
