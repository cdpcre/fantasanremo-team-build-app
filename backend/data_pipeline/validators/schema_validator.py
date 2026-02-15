"""
Schema Validator Module

Valida i dati contro gli schemi JSON.
"""

import json
from typing import Any

from ..config import get_config, get_logger


class SchemaValidator:
    """
    Validatore di schemi JSON per i dati FantaSanremo.
    """

    def __init__(self, config=None):
        """
        Inizializza lo schema validator.

        Args:
            config: Configurazione pipeline (usa default se None)
        """
        self.config = config or get_config()
        self.logger = get_logger(f"{self.__class__.__name__}")
        self._schemas: dict[str, dict] = {}

    def load_schema(self, schema_name: str) -> dict[str, Any]:
        """
        Carica uno schema dal file.

        Args:
            schema_name: Nome file schema (es. "artist_schema.json")

        Returns:
            Dict con schema JSON
        """
        if schema_name in self._schemas:
            return self._schemas[schema_name]

        schema_path = self.config.get_schema_path(schema_name)

        if not schema_path.exists():
            self.logger.warning(f"Schema file not found: {schema_path}")
            # Return default schema
            return self._get_default_schema(schema_name)

        try:
            with open(schema_path) as f:
                schema = json.load(f)
                self._schemas[schema_name] = schema
                return schema
        except Exception as e:
            self.logger.error(f"Failed to load schema {schema_name}: {e}")
            return self._get_default_schema(schema_name)

    def validate(self, data: Any, schema: dict[str, Any]) -> tuple[bool, list[str]]:
        """
        Valida i dati contro uno schema.

        Args:
            data: Dati da validare
            schema: Schema JSON per validazione

        Returns:
            Tuple (is_valid: bool, errors: List[str])
        """
        try:
            import jsonschema

            jsonschema.validate(instance=data, schema=schema)
            return True, []

        except ImportError:
            self.logger.warning("jsonschema not installed, using basic validation")
            return self._basic_validate(data, schema)

        except jsonschema.ValidationError as e:
            return False, [f"Schema validation error: {e.message} at path {list(e.path)}"]

        except Exception as e:
            return False, [f"Validation error: {e}"]

    def validate_artisti_2026(self, data: Any) -> tuple[bool, list[str]]:
        """
        Valida dati artisti 2026.

        Args:
            data: Dati artisti (dict o lista)

        Returns:
            Tuple (is_valid: bool, errors: List[str])
        """
        schema = self.load_schema("artist_schema.json")
        return self.validate(data, schema)

    def validate_storico(self, data: Any) -> tuple[bool, list[str]]:
        """
        Valida dati storici.

        Args:
            data: Dati storici

        Returns:
            Tuple (is_valid: bool, errors: List[str])
        """
        schema = self.load_schema("historical_schema.json")
        return self.validate(data, schema)

    def validate_biografico(self, data: Any) -> tuple[bool, list[str]]:
        """
        Valida dati biografici.

        Args:
            data: Dati biografici

        Returns:
            Tuple (is_valid: bool, errors: List[str])
        """
        schema = self.load_schema("biographical_schema.json")
        return self.validate(data, schema)

    def validate_caratteristiche(self, data: Any) -> tuple[bool, list[str]]:
        """
        Valida dati caratteristiche.

        Args:
            data: Dati caratteristiche

        Returns:
            Tuple (is_valid: bool, errors: List[str])
        """
        schema = self.load_schema("characteristics_schema.json")
        return self.validate(data, schema)

    def validate_storico_unified(self, data: Any) -> tuple[bool, list[str]]:
        """
        Valida dati storici unificati.

        Args:
            data: Dati storici unificati

        Returns:
            Tuple (is_valid: bool, errors: List[str])
        """
        schema = self.load_schema("historical_unified_schema.json")
        return self.validate(data, schema)

    def _basic_validate(self, data: Any, schema: dict) -> tuple[bool, list[str]]:
        """
        Validazione base senza jsonschema.

        Args:
            data: Dati da validare
            schema: Schema con regole base

        Returns:
            Tuple (is_valid: bool, errors: List[str])
        """
        errors = []

        # Check type
        expected_type = schema.get("type")
        if expected_type == "object":
            if not isinstance(data, dict):
                errors.append(f"Expected object, got {type(data).__name__}")
                return False, errors
        elif expected_type == "array":
            if not isinstance(data, list):
                errors.append(f"Expected array, got {type(data).__name__}")
                return False, errors

        # Check required fields for objects
        if expected_type == "object":
            required = schema.get("required", [])
            for field in required:
                if field not in data:
                    errors.append(f"Missing required field: {field}")

        # Check array items
        if expected_type == "array" and isinstance(data, list):
            item_schema = schema.get("items")
            if item_schema and data:
                # Validate first item as sample
                is_valid, item_errors = self._basic_validate(data[0], item_schema)
                if not is_valid:
                    errors.extend([f"Item 0: {e}" for e in item_errors])

        # Check properties for objects
        if expected_type == "object" and isinstance(data, dict):
            properties = schema.get("properties", {})
            for prop, prop_schema in properties.items():
                if prop in data:
                    is_valid, prop_errors = self._basic_validate(data[prop], prop_schema)
                    if not is_valid:
                        errors.extend([f"{prop}: {e}" for e in prop_errors])

        return len(errors) == 0, errors

    def _get_default_schema(self, schema_name: str) -> dict:
        """Restituisce schema default se file non trovato."""
        defaults = {
            "artist_schema.json": {
                "type": "object",
                "required": ["edizione", "artisti"],
                "properties": {
                    "edizione": {"type": "integer"},
                    "artisti": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["id", "nome", "quotazione"],
                            "properties": {
                                "id": {"type": "integer", "minimum": 1},
                                "nome": {"type": "string", "minLength": 1},
                                "quotazione": {"type": "integer", "minimum": 13, "maximum": 17},
                            },
                        },
                    },
                },
            },
            "historical_schema.json": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["artista", "posizioni"],
                    "properties": {"artista": {"type": "string"}, "posizioni": {"type": "object"}},
                },
            },
            "biographical_schema.json": {
                "type": "object",
                "required": ["artisti_2026_biografico"],
                "properties": {
                    "artisti_2026_biografico": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["nome"],
                            "properties": {
                                "nome": {"type": "string"},
                                "genere_musicale": {"type": "string"},
                                "anno_nascita": {"type": "integer"},
                                "prima_partecipazione": {"type": "integer"},
                            },
                        },
                    }
                },
            },
            "characteristics_schema.json": {
                "type": "object",
                "required": ["caratteristiche_artisti_2026"],
                "properties": {
                    "caratteristiche_artisti_2026": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["nome"],
                            "properties": {
                                "nome": {"type": "string"},
                                "viralita_social": {
                                    "type": "integer",
                                    "minimum": 1,
                                    "maximum": 100,
                                },
                                "social_followers_total": {"type": "integer", "minimum": 0},
                                "social_followers_by_platform": {"type": "object"},
                                "social_followers_last_updated": {"type": "string"},
                                "storia_bonus_ottenuti": {"type": "integer", "minimum": 0},
                                "ad_personam_bonus_count": {"type": "integer", "minimum": 0},
                                "ad_personam_bonus_points": {"type": "integer", "minimum": 0},
                            },
                        },
                    }
                },
            },
        }

        return defaults.get(schema_name, {})


# Convenience function
def validate_data(data: Any, data_type: str, config=None) -> tuple[bool, list[str]]:
    """
    Funzione convenienza per validare dati.

    Args:
        data: Dati da validare
        data_type: Tipo dati ("artisti", "storico", "biografico", "caratteristiche",
            "storico_unified")
        config: Configurazione opzionale

    Returns:
        Tuple (is_valid: bool, errors: List[str])
    """
    validator = SchemaValidator(config)

    validators = {
        "artisti": validator.validate_artisti_2026,
        "storico": validator.validate_storico,
        "biografico": validator.validate_biografico,
        "caratteristiche": validator.validate_caratteristiche,
        "storico_unified": validator.validate_storico_unified,
    }

    validator_func = validators.get(data_type)
    if not validator_func:
        return False, [f"Unknown data type: {data_type}"]

    return validator_func(data)
