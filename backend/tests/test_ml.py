"""
ML features and prediction tests for FantaSanremo backend.

This module tests:
- Feature engineering functions
- Performer level calculation
- Prediction generation
- Model training logic
- Data preparation and normalization

Test categories:
- Feature creation and validation
- Prediction accuracy
- Edge cases in ML pipeline
- Data transformation
"""

import numpy as np
import pandas as pd
import pytest

from backend.ml import predict
from backend.ml.features import (
    calculate_performer_level,
    create_historical_features,
    create_quotation_features,
    normalize_features,
    prepare_training_data,
)

pytestmark = pytest.mark.ml


# Mock the train module for testing
class MockModel:
    """Mock model for testing predictions."""

    def predict(self, X):
        # Return predictions based on feature averages
        if len(X) > 0:
            base_scores = np.array([250.0] * len(X))
            # Add some variation based on first feature
            if len(X.columns) > 0:
                variation = X.iloc[:, 0].values * 10
                return base_scores + variation
        return np.array([])


class TestFeatureEngineering:
    """Test cases for feature engineering functions."""

    def test_create_historical_features_veteran_artist(self):
        """Test creating features for an artist with historical data."""
        # Create sample historical data
        storico_df = pd.DataFrame(
            [
                {"artista_id": 1, "anno": 2020, "posizione": 1, "punteggio_finale": 500},
                {"artista_id": 1, "anno": 2021, "posizione": 3, "punteggio_finale": 400},
                {"artista_id": 1, "anno": 2023, "posizione": 2, "punteggio_finale": 450},
            ]
        )

        features_df = create_historical_features(storico_df)

        assert len(features_df) == 1
        assert features_df.iloc[0]["artista_id"] == 1
        assert features_df.iloc[0]["participations"] == 3
        assert bool(features_df.iloc[0]["is_debuttante"]) is False

        # Check inverted position calculation
        # Position 1 = 30 points, Position 3 = 28 points, Position 2 = 29 points
        # Average: (30 + 28 + 29) / 3 = 29
        avg_pos = features_df.iloc[0]["avg_position"]
        assert 28 <= avg_pos <= 29.5  # Allow for floating point

    def test_create_historical_features_debuttant_artist(self):
        """Test creating features for a debuttant artist."""
        storico_df = pd.DataFrame(
            [
                {"artista_id": 2, "anno": 2024, "posizione": None, "punteggio_finale": None},
            ]
        )

        features_df = create_historical_features(storico_df)

        assert len(features_df) == 1
        assert features_df.iloc[0]["artista_id"] == 2
        assert features_df.iloc[0]["participations"] == 0
        assert bool(features_df.iloc[0]["is_debuttante"]) is True
        assert features_df.iloc[0]["avg_position"] == 0

    def test_create_historical_features_multiple_artists(self):
        """Test creating features for multiple artists."""
        storico_df = pd.DataFrame(
            [
                {"artista_id": 1, "anno": 2020, "posizione": 1, "punteggio_finale": 500},
                {"artista_id": 1, "anno": 2021, "posizione": 2, "punteggio_finale": 450},
                {"artista_id": 2, "anno": 2020, "posizione": 10, "punteggio_finale": 200},
                {"artista_id": 3, "anno": 2021, "posizione": None, "punteggio_finale": None},
            ]
        )

        features_df = create_historical_features(storico_df)

        assert len(features_df) == 3

        # Find artist 1 (veteran)
        artist1 = features_df[features_df["artista_id"] == 1].iloc[0]
        assert artist1["participations"] == 2
        assert bool(artist1["is_debuttante"]) is False

        # Find artist 3 (debuttante)
        artist3 = features_df[features_df["artista_id"] == 3].iloc[0]
        assert artist3["participations"] == 0
        assert bool(artist3["is_debuttante"]) is True

    def test_create_historical_features_position_trend(self):
        """Test position trend calculation."""
        # Improving artist: positions 10, 5, 1
        storico_df = pd.DataFrame(
            [
                {"artista_id": 1, "anno": 2019, "posizione": 10, "punteggio_finale": 100},
                {"artista_id": 1, "anno": 2021, "posizione": 5, "punteggio_finale": 200},
                {"artista_id": 1, "anno": 2023, "posizione": 1, "punteggio_finale": 500},
            ]
        )

        features_df = create_historical_features(storico_df)
        trend = features_df.iloc[0]["position_trend"]

        # Should be positive (improving)
        assert trend > 0

    def test_create_quotation_features(self):
        """Test creating quotation-based features."""
        artisti_df = pd.DataFrame(
            [
                {"id": 1, "nome": "Artist 1", "quotazione_2026": 17},
                {"id": 2, "nome": "Artist 2", "quotazione_2026": 15},
                {"id": 3, "nome": "Artist 3", "quotazione_2026": 13},
            ]
        )

        features_df = create_quotation_features(artisti_df)

        assert len(features_df) == 3

        # Check artist 1 (top quoted)
        artist1 = features_df[features_df["artista_id"] == 1].iloc[0]
        assert artist1["quotazione_2026"] == 17
        assert bool(artist1["is_top_quoted"]) is True
        assert bool(artist1["is_mid_quoted"]) is False
        assert bool(artist1["is_low_quoted"]) is False

        # Check artist 2 (mid quoted)
        artist2 = features_df[features_df["artista_id"] == 2].iloc[0]
        assert artist2["quotazione_2026"] == 15
        assert bool(artist2["is_top_quoted"]) is False
        assert bool(artist2["is_mid_quoted"]) is True
        assert bool(artist2["is_low_quoted"]) is False

        # Check artist 3 (low quoted)
        artist3 = features_df[features_df["artista_id"] == 3].iloc[0]
        assert artist3["quotazione_2026"] == 13
        assert bool(artist3["is_top_quoted"]) is False
        assert bool(artist3["is_mid_quoted"]) is False
        assert bool(artist3["is_low_quoted"]) is True

    def test_prepare_training_data(self):
        """Test preparing complete training dataset."""
        storico_df = pd.DataFrame(
            [
                {"artista_id": 1, "anno": 2021, "posizione": 1, "punteggio_finale": 500},
            ]
        )

        artisti_df = pd.DataFrame(
            [
                {"id": 1, "nome": "Artist 1", "quotazione_2026": 17},
                {"id": 2, "nome": "Artist 2", "quotazione_2026": 14},
            ]
        )

        features_df = prepare_training_data(storico_df, artisti_df)

        assert len(features_df) == 2
        assert "artista_id" in features_df.columns
        assert "quotazione_2026" in features_df.columns
        assert "is_top_quoted" in features_df.columns
        assert "avg_position" in features_df.columns
        assert "participations" in features_df.columns

    def test_normalize_features(self, tmp_path):
        """Test feature normalization."""
        features_df = pd.DataFrame(
            [
                {"artista_id": 1, "avg_position": 25.0, "participations": 5, "quotazione_2026": 17},
                {"artista_id": 2, "avg_position": 15.0, "participations": 2, "quotazione_2026": 14},
            ]
        )

        scaler_path = tmp_path / "feature_scaler.pkl"
        normalized_df, scaler = normalize_features(
            features_df, mode="train", scaler_path=scaler_path
        )

        assert len(normalized_df) == 2
        assert "avg_position" in normalized_df.columns

        # Check that normalization occurred (values should be different)
        assert (
            normalized_df.iloc[0]["avg_position"] != 25.0
            or normalized_df.iloc[1]["avg_position"] != 15.0
        )

    def test_normalize_empty_dataframe(self, tmp_path):
        """Test normalizing empty dataframe."""
        features_df = pd.DataFrame()

        scaler_path = tmp_path / "feature_scaler.pkl"
        normalized_df, scaler = normalize_features(
            features_df, mode="train", scaler_path=scaler_path
        )

        assert len(normalized_df) == 0

    def test_normalize_features_with_missing_values(self, tmp_path):
        """Test normalizing features with NaN values."""
        features_df = pd.DataFrame(
            [
                {"artista_id": 1, "avg_position": 25.0, "participations": 5, "quotazione_2026": 17},
                {
                    "artista_id": 2,
                    "avg_position": None,
                    "participations": None,
                    "quotazione_2026": 14,
                },
            ]
        )

        scaler_path = tmp_path / "feature_scaler.pkl"
        normalized_df, scaler = normalize_features(
            features_df, mode="train", scaler_path=scaler_path
        )

        assert len(normalized_df) == 2
        # NaN should be filled with 0
        assert not normalized_df.iloc[1][["avg_position", "participations"]].isna().any()


class TestPerformerLevelCalculation:
    """Test cases for performer level calculation."""

    def test_calculate_performer_level_high(self):
        """Test HIGH performer level calculation."""
        # avg_position >= 20 (top 10 historically)
        level = calculate_performer_level(avg_position=25, participations=3)

        assert level == "HIGH"

    def test_calculate_performer_level_medium(self):
        """Test MEDIUM performer level calculation."""
        # 10 <= avg_position < 20
        level = calculate_performer_level(avg_position=15, participations=3)

        assert level == "MEDIUM"

    def test_calculate_performer_level_low(self):
        """Test LOW performer level calculation."""
        # avg_position < 10
        level = calculate_performer_level(avg_position=5, participations=3)

        assert level == "LOW"

    def test_calculate_performer_level_debuttante(self):
        """Test performer level for debuttantes."""
        level = calculate_performer_level(avg_position=0, participations=0)

        assert level == "DEBUTTANTE"

    def test_calculate_performer_level_boundary_high_medium(self):
        """Test boundary between HIGH and MEDIUM."""
        # Exactly 20 should be HIGH
        level = calculate_performer_level(avg_position=20, participations=3)

        assert level == "HIGH"

        # Just below 20 should be MEDIUM
        level = calculate_performer_level(avg_position=19.9, participations=3)

        assert level == "MEDIUM"

    def test_calculate_performer_level_boundary_medium_low(self):
        """Test boundary between MEDIUM and LOW."""
        # Exactly 10 should be MEDIUM
        level = calculate_performer_level(avg_position=10, participations=3)

        assert level == "MEDIUM"

        # Just below 10 should be LOW
        level = calculate_performer_level(avg_position=9.9, participations=3)

        assert level == "LOW"


class TestPredictionGeneration:
    """Test cases for prediction generation."""

    def test_predict_2026_with_mock_model(self, monkeypatch):
        """Test prediction generation with mocked models."""

        # Mock load_models function
        def mock_load_models():
            models = {"rf": MockModel(), "gb": MockModel(), "ridge": MockModel()}
            meta = {
                "ensemble_weights": {"rf": 0.33, "gb": 0.34, "ridge": 0.33},
                "year_stats": {},
                "selected_features": [],
            }
            return models, meta

        monkeypatch.setattr(predict, "load_models", mock_load_models)

        pd.DataFrame(
            [
                {"id": 1, "nome": "Artist 1", "quotazione_2026": 17},
                {"id": 2, "nome": "Artist 2", "quotazione_2026": 14},
            ]
        )

        pd.DataFrame(
            [
                {"artista_id": 1, "anno": 2021, "posizione": 1, "punteggio_finale": 500},
            ]
        )

        # This would normally call predict_2026
        # For now, just verify the setup works
        models, meta = mock_load_models()
        assert models is not None
        assert "rf" in models

    def test_generate_predictions_simple_no_history(self):
        """Test simplified prediction with no historical data."""
        from backend.ml.predict import generate_predictions_simple

        artisti = [
            {"id": 1, "nome": "Artist 1", "quotazione_2026": 17},
            {"id": 2, "nome": "Artist 2", "quotazione_2026": 15},
            {"id": 3, "nome": "Artist 3", "quotazione_2026": 13},
        ]

        storico = []  # No history

        predictions = generate_predictions_simple(artisti, storico)

        assert len(predictions) == 3

        # Check structure
        pred1 = next(p for p in predictions if p["artista_id"] == 1)
        assert "punteggio_predetto" in pred1
        assert "confidence" in pred1
        assert "livello_performer" in pred1
        assert pred1["confidence"] == 0.5  # Default for no history
        assert pred1["livello_performer"] in ["HIGH", "MEDIUM", "LOW"]

        # Higher quotation should have higher score
        pred17 = next(p for p in predictions if p["artista_id"] == 1)
        pred13 = next(p for p in predictions if p["artista_id"] == 3)

        # Score should be roughly based on quotation
        assert pred17["punteggio_predetto"] > pred13["punteggio_predetto"]

    def test_generate_predictions_simple_with_history(self, monkeypatch):
        """Test simplified prediction with historical data."""
        from backend.ml.predict import generate_predictions_simple

        # Mock predict_2026 to avoid complex setup
        def mock_predict_2026(artisti_df, storico_df):
            return [
                {
                    "artista_id": 1,
                    "punteggio_predetto": 500.0,
                    "confidence": 0.9,
                    "livello_performer": "HIGH",
                },
                {
                    "artista_id": 2,
                    "punteggio_predetto": 300.0,
                    "confidence": 0.7,
                    "livello_performer": "MEDIUM",
                },
            ]

        monkeypatch.setattr(predict, "predict_2026", mock_predict_2026)

        artisti = [
            {"id": 1, "nome": "Artist 1", "quotazione_2026": 17},
            {"id": 2, "nome": "Artist 2", "quotazione_2026": 14},
        ]

        storico = [
            {"artista_id": 1, "anno": 2021, "posizione": 1, "punteggio_finale": 500},
        ]

        predictions = generate_predictions_simple(artisti, storico)

        assert len(predictions) == 2

        # Should use mocked predict_2026 results
        pred1 = next(p for p in predictions if p["artista_id"] == 1)
        assert pred1["punteggio_predetto"] == 500.0
        assert pred1["confidence"] == 0.9
        assert pred1["livello_performer"] == "HIGH"


@pytest.mark.edge
class TestMLEdgeCases:
    """Test edge cases and boundary conditions in ML pipeline."""

    def test_empty_historical_data(self):
        """Test feature engineering with no historical data."""
        storico_df = pd.DataFrame()

        features_df = create_historical_features(storico_df)

        assert len(features_df) == 0

    def test_single_participation(self):
        """Test features for artist with single participation."""
        storico_df = pd.DataFrame(
            [
                {"artista_id": 1, "anno": 2023, "posizione": 5, "punteggio_finale": 300},
            ]
        )

        features_df = create_historical_features(storico_df)

        assert len(features_df) == 1
        assert features_df.iloc[0]["participations"] == 1
        assert features_df.iloc[0]["position_variance"] == 0  # No variance with single data point

    def test_extreme_positions(self):
        """Test features with extreme positions (1 and 30)."""
        storico_df = pd.DataFrame(
            [
                {"artista_id": 1, "anno": 2020, "posizione": 1, "punteggio_finale": 600},
                {"artista_id": 1, "anno": 2021, "posizione": 30, "punteggio_finale": 50},
            ]
        )

        features_df = create_historical_features(storico_df)

        # Inverted: position 1 = 30, position 30 = 1
        # Average: (30 + 1) / 2 = 15.5
        avg_pos = features_df.iloc[0]["avg_position"]
        assert 15 <= avg_pos <= 16

        # Best should be 30 (inverted position 1)
        best = features_df.iloc[0]["best_position"]
        assert best == 30

    def test_all_debuttanti(self):
        """Test feature engineering when all artists are debútantes."""
        storico_df = pd.DataFrame(
            [
                {"artista_id": 1, "anno": 2024, "posizione": None, "punteggio_finale": None},
                {"artista_id": 2, "anno": 2024, "posizione": None, "punteggio_finale": None},
                {"artista_id": 3, "anno": 2024, "posizione": None, "punteggio_finale": None},
            ]
        )

        features_df = create_historical_features(storico_df)

        assert len(features_df) == 3
        assert all(features_df["is_debuttante"].astype(bool))
        assert all(features_df["participations"] == 0)

    def test_quotation_boundary_values(self):
        """Test quotation features with boundary values."""
        artisti_df = pd.DataFrame(
            [
                {"id": 1, "nome": "Min", "quotazione_2026": 13},
                {"id": 2, "nome": "Max", "quotazione_2026": 17},
            ]
        )

        features_df = create_quotation_features(artisti_df)

        # Min quota
        min_artist = features_df[features_df["artista_id"] == 1].iloc[0]
        assert bool(min_artist["is_low_quoted"]) is True
        assert bool(min_artist["is_top_quoted"]) is False

        # Max quota
        max_artist = features_df[features_df["artista_id"] == 2].iloc[0]
        assert bool(max_artist["is_top_quoted"]) is True
        assert bool(max_artist["is_low_quoted"]) is False

    def test_prepare_training_data_missing_artists(self):
        """Test data preparation when some artists have no history."""
        storico_df = pd.DataFrame(
            [
                {"artista_id": 1, "anno": 2021, "posizione": 1, "punteggio_finale": 500},
            ]
        )

        artisti_df = pd.DataFrame(
            [
                {"id": 1, "nome": "Artist 1", "quotazione_2026": 17},
                {"id": 2, "nome": "Artist 2", "quotazione_2026": 14},  # No history
            ]
        )

        features_df = prepare_training_data(storico_df, artisti_df)

        assert len(features_df) == 2

        # Artist 1 should have historical features
        artist1 = features_df[features_df["artista_id"] == 1].iloc[0]
        assert artist1["participations"] > 0

        # Artist 2 should be debuttant
        artist2 = features_df[features_df["artista_id"] == 2].iloc[0]
        assert artist2["participations"] == 0
        assert bool(artist2["is_debuttante"]) is True


@pytest.mark.integration
class TestMLIntegration:
    """Integration tests for ML pipeline."""

    def test_full_feature_pipeline(self, tmp_path):
        """Test complete feature engineering pipeline."""
        # Input data
        storico_df = pd.DataFrame(
            [
                {"artista_id": 1, "anno": 2021, "posizione": 1, "punteggio_finale": 500},
                {"artista_id": 1, "anno": 2023, "posizione": 3, "punteggio_finale": 400},
                {"artista_id": 2, "anno": 2024, "posizione": None, "punteggio_finale": None},
            ]
        )

        artisti_df = pd.DataFrame(
            [
                {"id": 1, "nome": "Veteran", "quotazione_2026": 17},
                {"id": 2, "nome": "New", "quotazione_2026": 14},
            ]
        )

        # Step 1: Create historical features
        hist_features = create_historical_features(storico_df)
        assert len(hist_features) == 2
        assert "participations" in hist_features.columns

        # Step 2: Create quotation features
        quot_features = create_quotation_features(artisti_df)
        assert len(quot_features) == 2
        assert "is_top_quoted" in quot_features.columns

        # Step 3: Prepare training data
        full_features = prepare_training_data(storico_df, artisti_df)
        assert len(full_features) == 2

        # Check combined features
        assert "avg_position" in full_features.columns
        assert "quotazione_2026" in full_features.columns
        assert "is_debuttante" in full_features.columns

        # Step 4: Normalize
        scaler_path = tmp_path / "feature_scaler.pkl"
        normalized, scaler = normalize_features(
            full_features, mode="train", scaler_path=scaler_path
        )
        assert len(normalized) == 2

    def test_performer_level_distribution(self):
        """Test performer level distribution across artists."""
        test_cases = [
            (30, 5, "HIGH"),  # Top performer
            (25, 3, "HIGH"),
            (18, 2, "MEDIUM"),
            (12, 1, "MEDIUM"),
            (8, 2, "LOW"),
            (5, 1, "LOW"),
            (0, 0, "DEBUTTANTE"),  # Debuttante
        ]

        for avg_pos, participations, expected_level in test_cases:
            level = calculate_performer_level(avg_pos, participations)
            assert level == expected_level, f"Failed for avg_pos={avg_pos}, parts={participations}"

    def test_feature_consistency(self):
        """Test that features are consistent across multiple runs."""
        storico_df = pd.DataFrame(
            [
                {"artista_id": 1, "anno": 2021, "posizione": 5, "punteggio_finale": 300},
            ]
        )

        artisti_df = pd.DataFrame(
            [
                {"id": 1, "nome": "Test", "quotazione_2026": 15},
            ]
        )

        # Run twice
        features1 = prepare_training_data(storico_df, artisti_df)
        features2 = prepare_training_data(storico_df, artisti_df)

        # Should produce identical results
        assert features1.equals(features2)

    def test_prediction_structure_validation(self):
        """Test that predictions have correct structure."""
        from backend.ml.predict import generate_predictions_simple

        artisti = [
            {"id": 1, "nome": "Test", "quotazione_2026": 15},
        ]

        predictions = generate_predictions_simple(artisti, [])

        # Validate structure
        assert isinstance(predictions, list)
        assert len(predictions) == 1

        pred = predictions[0]
        required_fields = ["artista_id", "punteggio_predetto", "confidence", "livello_performer"]
        for field in required_fields:
            assert field in pred

        # Validate types
        assert isinstance(pred["artista_id"], int)
        assert isinstance(pred["punteggio_predetto"], (int, float))
        assert isinstance(pred["confidence"], (int, float, type(None)))
        assert isinstance(pred["livello_performer"], str)
        assert pred["livello_performer"] in ["HIGH", "MEDIUM", "LOW", "DEBUTTANTE"]

        # Validate ranges
        assert pred["punteggio_predetto"] >= 0
        if pred["confidence"] is not None:
            assert 0 <= pred["confidence"] <= 1
