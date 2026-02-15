"""
Fantasanremo Data Pipeline - Configuration Module

Gestione centralizzata di percorsi, configurazioni, e regole di validazione.
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DataPipelineConfig:
    """Configurazione centralizzata per la data pipeline."""

    # Environment
    environment: str = field(default_factory=lambda: os.getenv("FS_ENV", "dev"))

    # Project paths
    project_root: Path = field(default_factory=lambda: Path(__file__).parent.parent.parent)
    data_dir: Path = field(default=None)
    backend_dir: Path = field(default=None)
    ml_dir: Path = field(default=None)
    models_dir: Path = field(default=None)

    # Data source URLs
    wikipedia_url: str = "https://it.wikipedia.org/wiki/Festival_di_Sanremo"
    fantasanremo_url: str = "https://www.fantasanremo.it"

    # File paths
    artisti_2026_path: Path = field(default=None)
    storico_unified_path: Path = field(default=None)
    regolamento_2026_path: Path = field(default=None)

    # Validation rules
    min_quotazione: int = 13
    max_quotazione: int = 17
    budget_team: int = 100
    team_size: int = 7
    titulari_count: int = 5
    riserve_count: int = 2

    # ML configuration
    training_years: list[int] = field(default_factory=lambda: [2020, 2021, 2022, 2024])
    validation_years: list[int] = field(default_factory=lambda: [2023, 2025])
    prediction_year: int = 2026

    # Data quality thresholds
    min_quality_score: int = 80
    min_required_field_coverage: float = 0.95
    min_biografico_coverage: float = 0.85
    min_caratteristiche_coverage: float = 0.80
    min_crossfile_match_rate: float = 0.90

    # Cache configuration
    cache_dir: Path = field(default=None)
    cache_ttl_hours: int = 24

    # Logging configuration
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Retry configuration
    max_retries: int = 3
    retry_base_delay: float = 1.0  # seconds
    retry_max_delay: float = 60.0  # seconds

    def __post_init__(self):
        """Initialize paths based on project root."""
        if self.data_dir is None:
            self.data_dir = self.project_root / "data"
        if self.backend_dir is None:
            self.backend_dir = self.project_root / "backend"
        if self.ml_dir is None:
            self.ml_dir = self.backend_dir / "ml"
        if self.models_dir is None:
            self.models_dir = self.ml_dir / "models"
        if self.cache_dir is None:
            self.cache_dir = self.backend_dir / ".cache"

        # Data file paths
        # Canonical enriched artists file (merged data)
        self.artisti_2026_path = self.data_dir / "artisti_2026_enriched.json"
        self.storico_unified_path = self.data_dir / "storico_fantasanremo_unified.json"
        self.regolamento_2026_path = self.data_dir / "regolamento_2026.json"

    def get_schema_path(self, schema_name: str) -> Path:
        """Get path to a JSON schema file."""
        schema_dir = self.backend_dir / "data_pipeline" / "schemas"
        return schema_dir / schema_name

    def ensure_directories(self):
        """Create necessary directories if they don't exist."""
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)


# Global config instance
_config: DataPipelineConfig = None


def get_config() -> DataPipelineConfig:
    """
    Get the global configuration instance.

    Returns:
        DataPipelineConfig: The configuration instance
    """
    global _config
    if _config is None:
        _config = DataPipelineConfig()
        _config.ensure_directories()
        _setup_logging(_config)
    return _config


def set_config(config: DataPipelineConfig):
    """Set a custom configuration instance."""
    global _config
    _config = config
    _config.ensure_directories()
    _setup_logging(_config)


def _setup_logging(config: DataPipelineConfig):
    """Setup logging based on configuration."""
    log_level = getattr(logging, config.log_level.upper(), logging.INFO)

    logging.basicConfig(
        level=log_level,
        format=config.log_format,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(config.backend_dir / "data_pipeline.log"),
        ],
    )


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance."""
    return logging.getLogger(name)


# Environment-specific configurations
def get_production_config() -> DataPipelineConfig:
    """Get production configuration."""
    config = DataPipelineConfig()
    config.environment = "prod"
    config.log_level = "WARNING"
    config.cache_ttl_hours = 6
    return config


def get_development_config() -> DataPipelineConfig:
    """Get development configuration."""
    return DataPipelineConfig()  # Default is dev


def get_test_config() -> DataPipelineConfig:
    """Get test configuration."""
    config = DataPipelineConfig()
    config.environment = "test"
    config.log_level = "DEBUG"
    config.cache_ttl_hours = 1
    return config


def load_config_from_env() -> DataPipelineConfig:
    """Load configuration from environment variables."""
    env = os.getenv("FS_ENV", "dev")

    if env == "prod":
        return get_production_config()
    elif env == "test":
        return get_test_config()
    else:
        return get_development_config()
