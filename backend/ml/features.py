import numpy as np
import pandas as pd


def create_historical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features from historical FantaSanremo data.

    Features:
    - avg_position: Average position (inverted, higher is better)
    - position_variance: Variance in positions
    - position_trend: Improvement trend over years
    - participations: Number of FantaSanremo participations
    - best_position: Best position achieved
    - recent_avg: Average of recent 2 participations

    Enhanced features:
    - consistency_score: Performance stability (1 - CV)
    - momentum_score: Exponential decay average of recent positions
    - peak_performance: Max inverted position achieved
    - longevity_bonus: Bonus based on total participations
    - top10_finishes: Count of top 10 positions
    - top5_finishes: Count of top 5 positions
    - median_position: Median of inverted positions
    - volatility_index: Std dev of inverted positions
    """
    features = []

    # Handle empty dataframe
    if df.empty:
        return pd.DataFrame()

    for artista_id in df["artista_id"].unique():
        artista_df = df[df["artista_id"] == artista_id].copy()

        # Only consider editions where artist participated (posizione is not None)
        participations = artista_df[artista_df["posizione"].notna()].copy()

        if len(participations) == 0:
            # New artist - use default features
            features.append(
                {
                    "artista_id": artista_id,
                    "avg_position": 0,
                    "position_variance": 0,
                    "position_trend": 0,
                    "participations": 0,
                    "best_position": 0,
                    "recent_avg": 0,
                    "is_debuttante": True,
                    # Enhanced features - default values
                    "consistency_score": 0,
                    "momentum_score": 0,
                    "peak_performance": 0,
                    "longevity_bonus": 0,
                    "top10_finishes": 0,
                    "top5_finishes": 0,
                    "median_position": 0,
                    "volatility_index": 0,
                }
            )
            continue

        # Invert position (1st = 30 points, 30th = 1 point)
        max_pos = 30
        participations["inverted_pos"] = max_pos - participations["posizione"] + 1

        inverted_positions = participations["inverted_pos"].values

        # Basic features
        feat = {
            "artista_id": artista_id,
            "avg_position": participations["inverted_pos"].mean(),
            "position_variance": participations["inverted_pos"].var()
            if len(participations) > 1
            else 0,
            "participations": len(participations),
            "best_position": participations["inverted_pos"].max(),
            "is_debuttante": False,
        }

        # Trend: positive if improving, negative if declining
        if len(participations) >= 2:
            sorted_years = sorted(participations["anno"].values)
            recent = participations[participations["anno"].isin(sorted_years[-2:])]
            feat["recent_avg"] = recent["inverted_pos"].mean()
            feat["position_trend"] = feat["recent_avg"] - feat["avg_position"]
        else:
            feat["recent_avg"] = feat["avg_position"]
            feat["position_trend"] = 0

        # Enhanced features

        # Consistency score: 1 - coefficient of variation (higher = more consistent)
        if len(inverted_positions) > 1 and np.std(inverted_positions) > 0:
            cv = np.std(inverted_positions) / np.mean(inverted_positions)
            feat["consistency_score"] = max(0, 1 - cv)
        else:
            feat["consistency_score"] = 0.5  # Neutral for single participation

        # Momentum score: exponential decay average (recent years weighted more)
        if len(participations) >= 2:
            sorted_years = sorted(participations["anno"].values, reverse=True)
            weights = np.exp(-np.arange(len(sorted_years)) * 0.3)  # Decay factor
            weights = weights / weights.sum()

            momentum_values = []
            for i, anno in enumerate(sorted_years):
                pos = participations[participations["anno"] == anno]["inverted_pos"].values[0]
                momentum_values.append(pos * weights[i])

            feat["momentum_score"] = sum(momentum_values)
        else:
            feat["momentum_score"] = feat["avg_position"]

        # Peak performance
        feat["peak_performance"] = np.max(inverted_positions)

        # Longevity bonus: 5 points per participation
        feat["longevity_bonus"] = len(participations) * 5

        # Top finishes
        feat["top10_finishes"] = np.sum(inverted_positions >= 21)  # Top 10 = position 21-30
        feat["top5_finishes"] = np.sum(inverted_positions >= 26)  # Top 5 = position 26-30

        # Median position
        feat["median_position"] = np.median(inverted_positions)

        # Volatility index (std dev)
        feat["volatility_index"] = np.std(inverted_positions) if len(inverted_positions) > 1 else 0

        features.append(feat)

    return pd.DataFrame(features)


def create_quotation_features(artisti_df: pd.DataFrame) -> pd.DataFrame:
    """Create features from artist quotations"""
    features = []

    for _, row in artisti_df.iterrows():
        features.append(
            {
                "artista_id": row["id"],
                "quotazione_2026": row["quotazione_2026"],
                # High quotation artists tend to perform better
                "is_top_quoted": row["quotazione_2026"] >= 16,
                "is_mid_quoted": row["quotazione_2026"] == 15,
                "is_low_quoted": row["quotazione_2026"] <= 14,
            }
        )

    return pd.DataFrame(features)


def calculate_performer_level(avg_position: float, participations: int) -> str:
    """
    Calculate performer level based on historical data.
    - HIGH: avg inverted position >= 20 (top 10 historically)
    - MEDIUM: avg inverted position 10-19
    - LOW: avg inverted position < 10
    - DEBUTTANTE: participations == 0
    """
    if participations == 0:
        return "DEBUTTANTE"

    if avg_position >= 20:
        return "HIGH"
    elif avg_position >= 10:
        return "MEDIUM"
    else:
        return "LOW"


def prepare_training_data(edizioni_df: pd.DataFrame, artisti_df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare full feature set for training.

    Combines historical features with quotation features.
    Ensures all artists are included, even those without historical data.
    """
    hist_features = create_historical_features(edizioni_df)
    quot_features = create_quotation_features(artisti_df)

    # Merge features - use right join to ensure all artists are included
    features = quot_features.merge(hist_features, on="artista_id", how="left")

    # Fill missing values for new artists (from hist_features)
    historical_cols = [
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
    ]

    for col in historical_cols:
        if col in features.columns:
            features[col] = features[col].fillna(0)

    features["is_debuttante"] = (
        features["is_debuttante"].astype("boolean").fillna(True).astype(bool)
    )

    return features


def normalize_features(
    features_df: pd.DataFrame, mode: str = "train", scaler_path: str = None
) -> tuple[pd.DataFrame, any]:
    """
    Normalize numerical features for ML model with proper train/test separation.

    Args:
        features_df: DataFrame with features to normalize
        mode: 'train' or 'predict'
            - 'train': Fit scaler on data and transform (for training data)
            - 'predict': Load fitted scaler and only transform (for prediction data)
        scaler_path: Path to save/load the fitted scaler (uses default if None)

    Returns:
        Tuple of (normalized DataFrame, scaler object)

    Raises:
        ValueError: If mode is not 'train' or 'predict'
        FileNotFoundError: If scaler file not found in predict mode
    """
    from pathlib import Path

    import joblib
    from sklearn.preprocessing import StandardScaler

    if mode not in ["train", "predict"]:
        raise ValueError(f"mode must be 'train' or 'predict', got '{mode}'")

    numeric_cols = [
        "avg_position",
        "position_variance",
        "position_trend",
        "participations",
        "best_position",
        "recent_avg",
        "quotazione_2026",
        "consistency_score",
        "momentum_score",
        "peak_performance",
        "longevity_bonus",
        "top10_finishes",
        "top5_finishes",
        "median_position",
        "volatility_index",
        "years_since_last",
    ]

    # Default scaler path
    if scaler_path is None:
        scaler_path = Path(__file__).parent / "models" / "feature_scaler.pkl"

    # Handle empty dataframe
    if features_df.empty:
        if mode == "train":
            return pd.DataFrame(), StandardScaler()
        else:
            # Load scaler even for empty DF in predict mode
            scaler = joblib.load(scaler_path)
            return pd.DataFrame(), scaler

    normalized = features_df.copy()

    # Filter to only existing columns
    existing_cols = [col for col in numeric_cols if col in normalized.columns]

    if not existing_cols:
        # No numeric columns to normalize
        return normalized, None

    if mode == "train":
        # TRAINING MODE: Fit scaler on training data and transform
        print("Fitting scaler on training data...")
        scaler = StandardScaler()
        normalized[existing_cols] = scaler.fit_transform(normalized[existing_cols].fillna(0))

        # Save fitted scaler for later use in predictions
        scaler_path = Path(scaler_path)
        scaler_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, scaler_path)
        print(f"Saved fitted scaler to: {scaler_path}")

    else:  # mode == 'predict'
        # PREDICTION MODE: Load fitted scaler and only transform
        print("Loading fitted scaler for prediction...")
        scaler_path = Path(scaler_path)

        if not scaler_path.exists():
            raise FileNotFoundError(
                f"Scaler file not found at {scaler_path}. "
                "Train the model first to generate the scaler."
            )

        scaler = joblib.load(scaler_path)
        normalized[existing_cols] = scaler.transform(normalized[existing_cols].fillna(0))
        print(f"Loaded scaler from: {scaler_path}")

    return normalized, scaler
