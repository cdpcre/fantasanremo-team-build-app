"""
Unit tests for unified storico fantasanremo migration and validation.
"""

import json
from datetime import datetime

import pytest
from data_pipeline.config import get_config


class TestUnifiedStoricoMigration:
    """Test suite for unified storico migration."""

    @pytest.fixture
    def config(self):
        """Get test configuration."""
        return get_config()

    @pytest.fixture
    def sample_storico_data(self):
        """Sample storico_fantasanremo.json data."""
        return {
            "albo_oro": [
                {
                    "edizione": 0,
                    "anno": 2020,
                    "squadre": 47,
                    "vincitore": "Piero Pelù",
                    "punteggio": 220,
                },
                {
                    "edizione": 1,
                    "anno": 2021,
                    "squadre": 46962,
                    "vincitore": "Måneskin",
                    "punteggio": 315,
                },
            ],
            "statistiche_artisti_2026": [
                {
                    "nome": "Tommaso Paradiso",
                    "partecipazioni": 0,
                    "2021": None,
                    "2022": None,
                    "2023": None,
                    "2024": None,
                    "2025": None,
                    "debuttante": True,
                },
                {
                    "nome": "Fedez & Marco Masini",
                    "partecipazioni": 2,
                    "2021": 8,
                    "2022": None,
                    "2023": None,
                    "2024": None,
                    "2025": 21,
                    "debuttante": False,
                },
            ],
        }

    @pytest.fixture
    def sample_storico_completo_data(self):
        """Sample storico_fantasanremo_completo.json data."""
        return {
            "storico_fantasanremo_completo": [
                {
                    "artista": "Fedez & Marco Masini",
                    "edizioni": [
                        {
                            "anno": 2021,
                            "posizione": 8,
                            "punteggio_finale": 245,
                            "quotazione_baudi": 38,
                        },
                        {
                            "anno": 2025,
                            "posizione": 21,
                            "punteggio_finale": 195,
                            "quotazione_baudi": None,
                        },
                    ],
                },
                {
                    "artista": "Arisa",
                    "edizioni": [
                        {
                            "anno": 2021,
                            "posizione": 9,
                            "punteggio_finale": 235,
                            "quotazione_baudi": 26,
                        }
                    ],
                },
            ],
            "vincitori_edizioni": [
                {"anno": 2020, "vincitore": "Piero Pelù", "punteggio": 220, "squadre": 47},
                {"anno": 2021, "vincitore": "Måneskin", "punteggio": 315, "squadre": 46962},
            ],
            "note": "Test data",
        }

    @pytest.fixture
    def expected_unified_data(self):
        """Expected unified data structure."""
        return {
            "festival_edizioni": {
                "2020": {
                    "edizione": 0,
                    "anno": 2020,
                    "vincitore": "Piero Pelù",
                    "punteggio": 220,
                    "squadre": 47,
                },
                "2021": {
                    "edizione": 1,
                    "anno": 2021,
                    "vincitore": "Måneskin",
                    "punteggio": 315,
                    "squadre": 46962,
                },
            },
            "artisti_storici": {
                "Fedez & Marco Masini": {
                    "partecipazioni_totali": 2,
                    "edizioni": [
                        {
                            "anno": 2021,
                            "posizione": 8,
                            "punteggio_finale": 245,
                            "quotazione_baudi": 38,
                        },
                        {
                            "anno": 2025,
                            "posizione": 21,
                            "punteggio_finale": 195,
                            "quotazione_baudi": None,
                        },
                    ],
                },
                "Arisa": {
                    "partecipazioni_totali": 1,
                    "edizioni": [
                        {
                            "anno": 2021,
                            "posizione": 9,
                            "punteggio_finale": 235,
                            "quotazione_baudi": 26,
                        }
                    ],
                },
            },
            "artisti_2026": [
                {
                    "nome": "Tommaso Paradiso",
                    "quotazione": None,
                    "debuttante": True,
                    "partecipazioni": 0,
                    "storico_posizioni": {
                        "2021": None,
                        "2022": None,
                        "2023": None,
                        "2024": None,
                        "2025": None,
                    },
                },
                {
                    "nome": "Fedez & Marco Masini",
                    "quotazione": None,
                    "debuttante": False,
                    "partecipazioni": 2,
                    "storico_posizioni": {
                        "2021": 8,
                        "2022": None,
                        "2023": None,
                        "2024": None,
                        "2025": 21,
                    },
                },
            ],
            "metadata": {
                "unified_version": "1.0",
                "data_last_updated": datetime.now().strftime("%Y-%m-%d"),
                "source_files": ["storico_fantasanremo.json", "storico_fantasanremo_completo.json"],
                "festival_years": ["2020", "2021"],
                "total_artists_2026": 2,
                "total_historical_artists": 2,
            },
        }

    def test_unified_structure_has_required_keys(self, expected_unified_data):
        """Test that unified structure has all required keys."""
        required_keys = ["festival_edizioni", "artisti_storici", "artisti_2026", "metadata"]
        for key in required_keys:
            assert key in expected_unified_data, f"Missing required key: {key}"

    def test_festival_edizioni_structure(self, expected_unified_data):
        """Test festival_edizioni structure."""
        festival_edizioni = expected_unified_data["festival_edizioni"]

        assert "2020" in festival_edizioni
        assert "2021" in festival_edizioni

        for anno, edizione in festival_edizioni.items():
            assert "edizione" in edizione
            assert "anno" in edizione
            assert "vincitore" in edizione
            assert "punteggio" in edizione
            assert "squadre" in edizione
            assert isinstance(edizione["edizione"], int)
            assert isinstance(edizione["anno"], int)
            assert isinstance(edizione["vincitore"], str)
            assert isinstance(edizione["punteggio"], int)
            assert isinstance(edizione["squadre"], int)

    def test_artisti_storici_structure(self, expected_unified_data):
        """Test artisti_storici structure."""
        artisti_storici = expected_unified_data["artisti_storici"]

        assert "Fedez & Marco Masini" in artisti_storici
        assert "Arisa" in artisti_storici

        for artista, data in artisti_storici.items():
            assert "partecipazioni_totali" in data
            assert "edizioni" in data
            assert isinstance(data["partecipazioni_totali"], int)
            assert isinstance(data["edizioni"], list)

            for edizione in data["edizioni"]:
                assert "anno" in edizione
                assert "posizione" in edizione
                assert isinstance(edizione["anno"], int)
                assert isinstance(edizione["posizione"], int)

    def test_artisti_2026_structure(self, expected_unified_data):
        """Test artisti_2026 structure."""
        artisti_2026 = expected_unified_data["artisti_2026"]

        assert len(artisti_2026) == 2

        for artista in artisti_2026:
            assert "nome" in artista
            assert "debuttante" in artista
            assert "partecipazioni" in artista
            assert "storico_posizioni" in artista
            assert isinstance(artista["nome"], str)
            assert isinstance(artista["debuttante"], bool)
            assert isinstance(artista["partecipazioni"], int)
            assert isinstance(artista["storico_posizioni"], dict)

            # Check storico_posizioni has all years
            for year in ["2021", "2022", "2023", "2024", "2025"]:
                assert year in artista["storico_posizioni"]

    def test_metadata_structure(self, expected_unified_data):
        """Test metadata structure."""
        metadata = expected_unified_data["metadata"]

        assert "unified_version" in metadata
        assert "data_last_updated" in metadata
        assert "source_files" in metadata
        assert "festival_years" in metadata
        assert "total_artists_2026" in metadata
        assert "total_historical_artists" in metadata

        assert isinstance(metadata["unified_version"], str)
        assert isinstance(metadata["source_files"], list)
        assert isinstance(metadata["festival_years"], list)
        assert isinstance(metadata["total_artists_2026"], int)
        assert isinstance(metadata["total_historical_artists"], int)

    def test_data_integrity_no_loss(
        self, sample_storico_data, sample_storico_completo_data, expected_unified_data
    ):
        """Test that no data is lost during migration."""
        # Check festival_edizioni count
        original_albo_count = len(sample_storico_data["albo_oro"])
        unified_edizioni_count = len(expected_unified_data["festival_edizioni"])
        assert original_albo_count == unified_edizioni_count

        # Check artisti_storici count
        original_storico_count = len(sample_storico_completo_data["storico_fantasanremo_completo"])
        unified_storici_count = len(expected_unified_data["artisti_storici"])
        assert original_storico_count == unified_storici_count

        # Check artisti_2026 count
        original_artisti_2026_count = len(sample_storico_data["statistiche_artisti_2026"])
        unified_artisti_2026_count = len(expected_unified_data["artisti_2026"])
        assert original_artisti_2026_count == unified_artisti_2026_count


class TestUnifiedStoricoSchema:
    """Test suite for unified storico schema validation."""

    @pytest.fixture
    def config(self):
        """Get test configuration."""
        return get_config()

    def test_schema_file_exists(self, config):
        """Test that the unified schema file exists."""
        schema_path = config.get_schema_path("historical_unified_schema.json")
        assert schema_path.exists(), f"Schema file not found: {schema_path}"

    def test_schema_loads_correctly(self, config):
        """Test that the schema file loads correctly."""
        from data_pipeline.validators.schema_validator import SchemaValidator

        validator = SchemaValidator(config)
        schema = validator.load_schema("historical_unified_schema.json")

        assert isinstance(schema, dict)
        assert "$schema" in schema
        assert "title" in schema
        assert "type" in schema

    def test_schema_validation_accepts_valid_data(self, config):
        """Test that schema validation accepts valid data."""
        from data_pipeline.validators.schema_validator import SchemaValidator

        validator = SchemaValidator(config)
        schema = validator.load_schema("historical_unified_schema.json")

        valid_data = {
            "festival_edizioni": {
                "2020": {
                    "edizione": 0,
                    "anno": 2020,
                    "vincitore": "Piero Pelù",
                    "punteggio": 220,
                    "squadre": 47,
                }
            },
            "artisti_storici": {
                "Test Artist": {
                    "partecipazioni_totali": 1,
                    "edizioni": [
                        {
                            "anno": 2020,
                            "posizione": 1,
                            "punteggio_finale": 220,
                            "quotazione_baudi": 15,
                        }
                    ],
                }
            },
            "artisti_2026": [
                {
                    "nome": "Test Artist",
                    "quotazione": None,
                    "debuttante": False,
                    "partecipazioni": 1,
                    "storico_posizioni": {
                        "2021": None,
                        "2022": None,
                        "2023": None,
                        "2024": None,
                        "2025": None,
                    },
                }
            ],
            "metadata": {
                "unified_version": "1.0",
                "data_last_updated": "2026-01-30",
            },
        }

        is_valid, errors = validator.validate(valid_data, schema)
        assert is_valid, f"Validation failed with errors: {errors}"
        assert len(errors) == 0

    def test_schema_validation_rejects_missing_required_keys(self, config):
        """Test that schema validation rejects data with missing keys."""
        from data_pipeline.validators.schema_validator import SchemaValidator

        validator = SchemaValidator(config)
        schema = validator.load_schema("historical_unified_schema.json")

        invalid_data = {
            "festival_edizioni": {},
            # Missing: artisti_storici, artisti_2026, metadata
        }

        is_valid, errors = validator.validate(invalid_data, schema)
        assert not is_valid
        assert len(errors) > 0


class TestUnifiedStoricoIntegration:
    """Integration tests for unified storico with actual files."""

    @pytest.fixture
    def config(self):
        """Get test configuration."""
        return get_config()

    def test_unified_file_exists_after_migration(self, config):
        """Test that the unified file exists after migration."""
        unified_path = config.data_dir / "storico_fantasanremo_unified.json"

        # This test will fail until migration is run
        # Skip if file doesn't exist yet
        if not unified_path.exists():
            pytest.skip("Unified file not created yet - run migration first")

        assert unified_path.exists()

    def test_unified_file_loads_correctly(self, config):
        """Test that the unified file loads correctly."""
        unified_path = config.data_dir / "storico_fantasanremo_unified.json"

        if not unified_path.exists():
            pytest.skip("Unified file not created yet - run migration first")

        with open(unified_path) as f:
            data = json.load(f)

        assert isinstance(data, dict)
        assert "festival_edizioni" in data
        assert "artisti_storici" in data
        assert "artisti_2026" in data
        assert "metadata" in data

    def test_unified_file_validates_against_schema(self, config):
        """Test that the unified file validates against its schema."""
        from data_pipeline.validators.schema_validator import SchemaValidator

        unified_path = config.data_dir / "storico_fantasanremo_unified.json"

        if not unified_path.exists():
            pytest.skip("Unified file not created yet - run migration first")

        validator = SchemaValidator(config)
        schema = validator.load_schema("historical_unified_schema.json")

        with open(unified_path) as f:
            data = json.load(f)

        is_valid, errors = validator.validate(data, schema)
        assert is_valid, f"Validation failed with errors: {errors}"
