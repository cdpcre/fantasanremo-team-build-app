#!/usr/bin/env python3
"""
Example usage of the fixed normalize_features() function
Demonstrates proper separation of training and prediction normalization
"""

import numpy as np
import pandas as pd
from ml.features import normalize_features

# Create sample training data
print("=" * 80)
print("EXAMPLE 1: Training Mode - Fit scaler on training data")
print("=" * 80)

train_data = pd.DataFrame(
    {
        "avg_position": [15.5, 22.3, 8.7, 18.2, 12.4],
        "position_variance": [25.3, 18.7, 32.1, 15.6, 28.9],
        "position_trend": [2.1, -1.5, 3.2, 0.8, -2.3],
        "participations": [3, 5, 2, 4, 3],
        "best_position": [20, 25, 15, 22, 18],
        "recent_avg": [16.2, 21.5, 9.3, 17.8, 11.2],
        "quotazione_2026": [15, 17, 14, 16, 15],
    }
)

print("\nOriginal training data:")
print(train_data.describe())

# Normalize in TRAIN mode
normalized_train, scaler_train = normalize_features(train_data, mode="train")

print("\nNormalized training data:")
print(normalized_train.describe())

print("\n✓ Scaler fitted and saved to: backend/ml/models/feature_scaler.pkl")


print("\n" + "=" * 80)
print("EXAMPLE 2: Prediction Mode - Use fitted scaler for new data")
print("=" * 80)

# Create sample prediction data (NEW artists for 2026)
prediction_data = pd.DataFrame(
    {
        "avg_position": [19.8, 14.2, 21.5],  # Different from training
        "position_variance": [22.1, 30.5, 17.8],
        "position_trend": [1.2, -0.8, 2.5],
        "participations": [4, 2, 5],
        "best_position": [23, 17, 24],
        "recent_avg": [18.5, 13.8, 20.2],
        "quotazione_2026": [16, 14, 17],
    }
)

print("\nOriginal prediction data:")
print(prediction_data.describe())

# Normalize in PREDICT mode
normalized_pred, scaler_pred = normalize_features(prediction_data, mode="predict")

print("\nNormalized prediction data:")
print(normalized_pred.describe())

print("\n✓ Used saved scaler from training - NO refitting!")


print("\n" + "=" * 80)
print("EXAMPLE 3: Verify No Data Leakage")
print("=" * 80)

# The key check: prediction data should NOT influence scaler parameters
print("\nTraining scaler means:")
print(f"  {scaler_train.mean_}")

print("\nPrediction scaler means (should be SAME as training):")
print(f"  {scaler_pred.mean_}")

if np.allclose(scaler_train.mean_, scaler_pred.mean_):
    print("\n✅ SUCCESS: Scaler parameters unchanged - no data leakage!")
else:
    print("\n❌ FAILURE: Scaler was refitted - data leakage detected!")

# Check n_samples_seen_ - should be same as training set size
print(f"\nTraining samples seen: {scaler_train.n_samples_seen_[0]}")
print(f"Prediction samples seen: {scaler_pred.n_samples_seen_[0]}")

if scaler_train.n_samples_seen_[0] == scaler_pred.n_samples_seen_[0]:
    print("✅ Correct: Scaler remembers training set size, not prediction data")
else:
    print("❌ Error: Scaler was refitted on prediction data")


print("\n" + "=" * 80)
print("EXAMPLE 4: What Happens With Missing Scaler")
print("=" * 80)

try:
    # Try to predict with non-existent scaler path
    normalized, scaler = normalize_features(
        prediction_data, mode="predict", scaler_path="nonexistent/path/scaler.pkl"
    )
except FileNotFoundError as e:
    print(f"\n✅ Correctly raised error: {e}")
    print("This prevents using uninitialized scalers in production!")


print("\n" + "=" * 80)
print("EXAMPLE 5: Edge Cases")
print("=" * 80)

# Empty DataFrame
empty_df = pd.DataFrame()
normalized_empty, scaler_empty = normalize_features(empty_df, mode="train")
print(f"\n✓ Empty DataFrame handled: {len(normalized_empty)} rows")

# Invalid mode
try:
    normalize_features(train_data, mode="invalid")
except ValueError as e:
    print(f"✓ Invalid mode rejected: {e}")


print("\n" + "=" * 80)
print("SUMMARY: Key Benefits of the Fix")
print("=" * 80)

print("""
1. ✅ TRAINING MODE:
   - Fits scaler ONLY on training data
   - Saves scaler to disk for reuse
   - Returns normalized training features

2. ✅ PREDICTION MODE:
   - Loads previously fitted scaler
   - Transforms prediction data WITHOUT refitting
   - Prevents data leakage from future data

3. ✅ PERSISTENCE:
   - Scaler saved at: backend/ml/models/feature_scaler.pkl
   - Uses joblib for efficient sklearn object serialization
   - Can be loaded months later for consistent predictions

4. ✅ VALIDATION:
   - Checks for missing scaler files
   - Validates mode parameter
   - Handles edge cases gracefully

5. ✅ REPRODUCIBILITY:
   - Same normalization applied to all predictions
   - Training parameters preserved indefinitely
   - No hidden state or side effects
""")

print("\n" + "=" * 80)
print("USAGE IN PRODUCTION")
print("=" * 80)

print("""
# In your training pipeline (train.py or scripts/run_pipeline.py --ml-training):

# After preparing features
X = df_features[feature_cols].fillna(0)

# Normalize in TRAIN mode
X_normalized, scaler = normalize_features(
    pd.DataFrame(X),
    mode='train'
)

# Train model
model.fit(X_normalized, y)

---

# In your prediction pipeline (predict.py):

# After preparing features
X = df_features[feature_cols].fillna(0)

# Normalize in PREDICT mode (uses saved scaler)
X_normalized, scaler = normalize_features(
    pd.DataFrame(X),
    mode='predict'
)

# Generate predictions
predictions = model.predict(X_normalized)
""")
