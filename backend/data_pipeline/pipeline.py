"""
Fantasanremo Data Pipeline Orchestration Module

Core orchestration per la data pipeline completa.
"""

import json
from typing import Any

from backend.utils.enriched_artists import (
    build_biografico_from_enriched,
    build_caratteristiche_from_enriched,
)

from .config import get_config, get_logger
from .sources.wikipedia_source import WikipediaDataSource
from .validators import BusinessRulesValidator, DataQualityChecker, SchemaValidator


class DataPipeline:
    """
    Orchestrazione completa della data pipeline.

    Steps:
    1. Fetch da tutte le sorgenti
    2. Validate schemas
    3. Apply business rules
    4. Run data quality checks
    5. Transform e clean
    6. Generate ML features
    7. Load to database
    8. Generate quality report
    """

    def __init__(self, config=None):
        """
        Inizializza la pipeline.

        Args:
            config: Configurazione pipeline
        """
        self.config = config or get_config()
        self.logger = get_logger("data_pipeline")

        # Components
        self.wikipedia_source = WikipediaDataSource(self.config)
        self.schema_validator = SchemaValidator(self.config)
        self.business_validator = BusinessRulesValidator(self.config)
        self.quality_checker = DataQualityChecker(self.config)

        # Data storage
        self.data_sources = {}
        self.validation_results = {}
        self.quality_report = {}

    def run_pipeline(
        self, steps: list[str] | None = None, dry_run: bool = False, verbose: bool = False
    ) -> dict[str, Any]:
        """
        Esegue la pipeline completa o steps specifici.

        Args:
            steps: Lista steps da eseguire (None = all)
            dry_run: Se True, non salva nulla
            verbose: Se True, log dettagliato

        Returns:
            Dict con risultati pipeline
        """
        if verbose:
            self.logger.setLevel("DEBUG")

        results = {"steps_completed": [], "steps_failed": [], "dry_run": dry_run}

        # Define available steps
        available_steps = ["fetch", "validate", "transform", "load", "generate_report"]

        # Default to all steps if not specified
        if steps is None:
            steps = available_steps

        # Validate steps
        invalid_steps = [s for s in steps if s not in available_steps]
        if invalid_steps:
            self.logger.warning(f"Invalid steps: {invalid_steps}")
            steps = [s for s in steps if s in available_steps]

        self.logger.info(f"Running pipeline steps: {', '.join(steps)}")
        self.logger.info(f"Dry run: {dry_run}")

        try:
            # Step 1: Fetch
            if "fetch" in steps:
                self.logger.info("Step 1: Fetching data from sources...")
                self.data_sources = self.fetch_all_sources()
                results["steps_completed"].append("fetch")
                self.logger.info(f"  Fetched {len(self.data_sources)} data sources")

            # Step 2: Validate
            if "validate" in steps:
                self.logger.info("Step 2: Validating data...")
                self.validation_results = self.validate_all_data()
                results["steps_completed"].append("validate")
                results["validation"] = self.validation_results

                # Check if validation passed
                all_valid = all(r.get("all_valid", True) for r in self.validation_results.values())
                if not all_valid:
                    self.logger.warning("  Validation completed with issues")

            # Step 3: Transform
            if "transform" in steps:
                self.logger.info("Step 3: Transforming data...")
                transformed_data = self.transform_data()
                results["steps_completed"].append("transform")
                results["transformed"] = transformed_data

            # Step 4: Load
            if "load" in steps and not dry_run:
                self.logger.info("Step 4: Loading data to database...")
                load_results = self.load_to_database()
                results["steps_completed"].append("load")
                results["load"] = load_results
            elif "load" in steps and dry_run:
                self.logger.info("Step 4: Load skipped (dry run)")

            # Step 5: Generate report
            if "generate_report" in steps:
                self.logger.info("Step 5: Generating quality report...")
                self.quality_report = self.generate_quality_report()
                results["steps_completed"].append("generate_report")
                results["quality_report"] = self.quality_report

            self.logger.info("Pipeline completed successfully!")

        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            results["steps_failed"].append(str(e))
            import traceback

            traceback.print_exc()

        return results

    def fetch_all_sources(self) -> dict[str, Any]:
        """Fetch da tutte le sorgenti."""
        sources = {}

        # Load local JSON files
        data_dir = self.config.data_dir

        # Load unified storico (required)
        unified_path = data_dir / "storico_fantasanremo_unified.json"
        if unified_path.exists():
            with open(unified_path) as f:
                sources["storico_unified"] = json.load(f)
            self.logger.info("  Loaded storico_unified from storico_fantasanremo_unified.json")
        else:
            raise FileNotFoundError("Missing required data file: storico_fantasanremo_unified.json")

        # Load enriched artists (single source for roster + bio + caratteristiche)
        artisti_path = self.config.artisti_2026_path
        if artisti_path.exists():
            with open(artisti_path) as f:
                sources["artisti_2026"] = json.load(f)
            self.logger.info(f"  Loaded artisti_2026 from {artisti_path.name}")

            # Build derived biografico/caratteristiche for downstream validators
            sources["biografico"] = build_biografico_from_enriched(sources["artisti_2026"])
            sources["caratteristiche"] = build_caratteristiche_from_enriched(
                sources["artisti_2026"]
            )
        else:
            self.logger.warning(f"  File not found: {artisti_path.name}")

        # Optionally fetch from Wikipedia
        # wikipedia_data = self.wikipedia_source.fetch_with_retry(year=2026)
        # sources["wikipedia_2026"] = wikipedia_data

        # Load voti stampa
        voti_stampa_path = data_dir / "voti_stampa.json"
        if voti_stampa_path.exists():
            with open(voti_stampa_path) as f:
                sources["voti_stampa"] = json.load(f)
            self.logger.info(f"  Loaded voti_stampa from {voti_stampa_path.name}")

        return sources

    def validate_all_data(self) -> dict[str, Any]:
        """Valida tutti i dati."""
        validation_results = {}

        # Schema validation
        schema_results = {}
        if "artisti_2026" in self.data_sources:
            is_valid, errors = self.schema_validator.validate_artisti_2026(
                self.data_sources["artisti_2026"]
            )
            schema_results["artisti_2026"] = {"valid": is_valid, "errors": errors}

        # Validate unified storico
        if "storico_unified" in self.data_sources:
            is_valid, errors = self.schema_validator.validate_storico_unified(
                self.data_sources["storico_unified"]
            )
            schema_results["storico_unified"] = {"valid": is_valid, "errors": errors}
        else:
            schema_results["storico_unified"] = {
                "valid": False,
                "errors": ["Missing storico_fantasanremo_unified.json"],
            }

        validation_results["schema"] = schema_results

        # Business rules validation
        business_results = {}
        if "artisti_2026" in self.data_sources:
            is_valid, errors = self.business_validator.validate_artisti_2026(
                self.data_sources["artisti_2026"]
            )
            business_results["artisti_2026"] = {"valid": is_valid, "errors": errors}

        validation_results["business"] = business_results

        # Cross-file consistency
        if all(
            key in self.data_sources
            for key in ["artisti_2026", "biografico", "caratteristiche", "storico_unified"]
        ):
            is_valid, errors = self.business_validator.validate_cross_file_consistency(
                self.data_sources["artisti_2026"],
                self.data_sources.get("biografico", {}),
                self.data_sources.get("caratteristiche", {}),
                self.data_sources.get("storico_unified", {}),
            )
            validation_results["cross_file"] = {"valid": is_valid, "errors": errors}

        # Data quality
        quality_report = self.quality_checker.check_quality(self.data_sources)
        validation_results["quality"] = quality_report

        return validation_results

    def transform_data(self) -> dict[str, int]:
        """Trasforma e pulisci i dati."""
        # For now, just return counts
        # In production, would apply transformations
        return {
            source: len(data.get("artisti", data)) if isinstance(data, dict) else len(data)
            for source, data in self.data_sources.items()
        }

    def load_to_database(self) -> dict[str, int]:
        """Carica dati nel database."""
        from backend.populate_db import DatabasePopulator

        populator = DatabasePopulator(self.config)
        result = populator.populate_database(
            validate=False,  # Already validated
            generate_predictions=False,  # Generate separately
            force_refresh=False,
        )

        return result.get("stats", {}).get("details", {})

    def generate_quality_report(self) -> dict[str, Any]:
        """Genera report qualità completo."""
        quality = self.validation_results.get("quality", {})
        acceptable = self.quality_checker.is_quality_acceptable(
            quality, threshold=self.config.min_quality_score
        )
        return {
            "data_sources_count": len(self.data_sources),
            "validation_summary": self._summarize_validation(),
            "quality_score": self._calculate_quality_score(),
            "acceptable": acceptable,
            "thresholds": {
                "min_quality_score": self.config.min_quality_score,
                "min_required_field_coverage": self.config.min_required_field_coverage,
                "min_biografico_coverage": self.config.min_biografico_coverage,
                "min_caratteristiche_coverage": self.config.min_caratteristiche_coverage,
                "min_crossfile_match_rate": self.config.min_crossfile_match_rate,
            },
        }

    def _summarize_validation(self) -> dict[str, int]:
        """Summarizza risultati validazione."""
        summary = {"total_validated": 0, "total_errors": 0, "total_warnings": 0}

        for category, results in self.validation_results.items():
            if category == "quality":
                continue

            if isinstance(results, dict):
                for source, result in results.items():
                    if isinstance(result, dict):
                        summary["total_validated"] += 1
                        summary["total_errors"] += len(result.get("errors", []))

        return summary

    def _calculate_quality_score(self) -> int:
        """Calcola punteggio qualità globale."""
        if "quality" in self.validation_results:
            return self.validation_results["quality"].get("overall_score", 0)
        return 50


def run_pipeline(
    steps: list[str] | None = None, dry_run: bool = False, verbose: bool = False, config=None
) -> dict[str, Any]:
    """
    Funzione convenienza per eseguire la pipeline.

    Args:
        steps: Lista steps da eseguire
        dry_run: Se True, non salva nulla
        verbose: Se True, log dettagliato
        config: Configurazione opzionale

    Returns:
        Dict con risultati
    """
    pipeline = DataPipeline(config)
    return pipeline.run_pipeline(steps=steps, dry_run=dry_run, verbose=verbose)
