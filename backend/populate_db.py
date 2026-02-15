"""
Enhanced Database Population Module

Popola il database usando la nuova data pipeline con validazione.
"""

import json
import os
import sys
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# New pipeline imports
from data_pipeline.config import get_config, get_logger
from data_pipeline.storage.database_store import DatabaseStore
from data_pipeline.validators import BusinessRulesValidator, DataQualityChecker, SchemaValidator
from database import Base, SessionLocal, engine
from ml.predict import generate_predictions_simple
from models import Artista, CaratteristicheArtista, EdizioneFantaSanremo, Predizione2026

from backend.utils.enriched_artists import (
    build_biografico_from_enriched,
    build_caratteristiche_from_enriched,
)
from backend.utils.name_normalization import index_by_normalized_name, normalize_artist_name


class DatabasePopulator:
    """
    Popola il database usando la nuova pipeline.
    """

    def __init__(self, config=None):
        """
        Inizializza il popolator.

        Args:
            config: Configurazione pipeline
        """
        self.config = config or get_config()
        self.logger = get_logger("database_populator")

        # Initialize database store
        self.db = SessionLocal()
        self.store = DatabaseStore(self.db, self.config)

        # Initialize validators
        self.schema_validator = SchemaValidator(self.config)
        self.business_validator = BusinessRulesValidator(self.config)
        self.quality_checker = DataQualityChecker(self.config)

        # Statistics
        self.stats = {"inserted": 0, "updated": 0, "skipped": 0, "errors": 0, "details": {}}

    def populate_database(
        self, validate: bool = True, generate_predictions: bool = True, force_refresh: bool = False
    ) -> dict:
        """
        Popola il database con tutti i dati.

        Args:
            validate: Se True, valida i dati prima dell'inserimento
            generate_predictions: Se True, genera predizioni ML
            force_refresh: Se True, sovrascrive dati esistenti

        Returns:
            Dict con statistiche popolamento
        """
        self.logger.info("=" * 60)
        self.logger.info("Starting Database Population")
        self.logger.info("=" * 60)

        try:
            # Create tables
            Base.metadata.create_all(bind=engine)

            # Check if database already populated
            existing = self.db.query(Artista).first()
            if existing and not force_refresh:
                self.logger.info("Database already populated. Use force_refresh=True to refresh.")
                return {"status": "already_populated"}

            if force_refresh and existing:
                self.logger.info("Force refresh enabled - clearing existing data...")
                self._clear_database()

            # Step 1: Load data sources
            self.logger.info("Step 1: Loading data sources...")
            data_sources = self._load_data_sources()
            self.stats["details"]["data_sources"] = len(data_sources)

            # Step 2: Validate data
            if validate:
                self.logger.info("Step 2: Validating data...")
                validation_results = self._validate_data(data_sources)
                self.stats["details"]["validation"] = validation_results

                if not validation_results.get("all_valid", True):
                    self.logger.warning("Data validation failed - proceeding with caution")

            # Step 3: Insert artists
            self.logger.info("Step 3: Inserting artists...")
            artist_stats = self._insert_artists(
                data_sources.get("artisti_2026", {}),
                biografico_data=data_sources.get("biografico", {}),
            )
            self.stats["details"]["artists"] = artist_stats

            # Step 4: Insert historical data
            self.logger.info("Step 4: Inserting historical data...")
            storico_list = self._build_storico_list(data_sources)
            storico_stats = self._insert_historical_data(storico_list)
            self.stats["details"]["historical"] = storico_stats

            # Step 5: Insert characteristics
            self.logger.info("Step 5: Inserting artist characteristics...")
            car_stats = self._insert_characteristics(data_sources.get("caratteristiche", {}))
            self.stats["details"]["characteristics"] = car_stats

            # Step 6: Generate ML predictions
            if generate_predictions:
                self.logger.info("Step 6: Generating ML predictions...")
                pred_stats = self._generate_predictions()
                self.stats["details"]["predictions"] = pred_stats

            # Final commit
            self.db.commit()

            # Generate quality report
            self.logger.info("Step 7: Generating quality report...")
            quality_report = self._generate_quality_report()

            self.logger.info("=" * 60)
            self.logger.info("Database populated successfully!")
            self.logger.info("=" * 60)

            return {"status": "success", "stats": self.stats, "quality_report": quality_report}

        except Exception as e:
            self.db.rollback()
            self.logger.error(f"Error populating database: {e}")
            self.stats["errors"] += 1
            raise

        finally:
            self.db.close()

    def _load_data_sources(self) -> dict:
        """Carica tutte le sorgenti dati JSON."""
        sources = {}
        data_dir = self.config.data_dir

        # Load enriched artisti 2026 (single source)
        artisti_path = self.config.artisti_2026_path
        if artisti_path.exists():
            with open(artisti_path) as f:
                import json

                sources["artisti_2026"] = json.load(f)
            self.logger.info(
                f"Loaded artisti_2026: {len(sources['artisti_2026'].get('artisti', []))} artists"
            )
            sources["biografico"] = build_biografico_from_enriched(sources["artisti_2026"])
            sources["caratteristiche"] = build_caratteristiche_from_enriched(
                sources["artisti_2026"]
            )

        # Load unified storico (required)
        unified_storico_path = data_dir / "storico_fantasanremo_unified.json"
        if unified_storico_path.exists():
            with open(unified_storico_path) as f:
                sources["storico_unified"] = json.load(f)
            self.logger.info("Loaded storico_unified")
        else:
            raise FileNotFoundError("Missing required data file: storico_fantasanremo_unified.json")

        if "biografico" in sources:
            self.logger.info(
                f"Loaded biografico: "
                f"{len(sources['biografico'].get('artisti_2026_biografico', []))} entries"
            )
        if "caratteristiche" in sources:
            self.logger.info(
                f"Loaded caratteristiche: "
                f"{len(sources['caratteristiche'].get('caratteristiche_artisti_2026', []))} entries"
            )

        return sources

    def _validate_data(self, data_sources: dict) -> dict:
        """Valida tutti i dati."""
        results = {"schema_validation": {}, "business_validation": {}, "all_valid": True}

        # Validate artisti 2026
        if "artisti_2026" in data_sources:
            is_valid, errors = self.schema_validator.validate_artisti_2026(
                data_sources["artisti_2026"]
            )
            results["schema_validation"]["artisti_2026"] = {"valid": is_valid, "errors": errors}

        # Validate unified storico
        if "storico_unified" in data_sources:
            is_valid, errors = self.schema_validator.validate_storico_unified(
                data_sources["storico_unified"]
            )
            results["schema_validation"]["storico_unified"] = {"valid": is_valid, "errors": errors}
        else:
            results["schema_validation"]["storico_unified"] = {
                "valid": False,
                "errors": ["Missing storico_fantasanremo_unified.json"],
            }

        # Business rules
        if "artisti_2026" in data_sources:
            is_valid, errors = self.business_validator.validate_artisti_2026(
                data_sources["artisti_2026"]
            )
            results["business_validation"]["artisti_2026"] = {"valid": is_valid, "errors": errors}

        results["all_valid"] = all(
            r.get("valid", True) for r in results["schema_validation"].values()
        )

        return results

    def _insert_artists(self, artisti_data: dict, biografico_data: dict | None = None) -> dict:
        """Insert artisti nel database."""
        artisti = artisti_data.get("artisti", [])

        # Build biografico map
        bio_map = {}
        if biografico_data and "artisti_2026_biografico" in biografico_data:
            bio_map = index_by_normalized_name(
                biografico_data["artisti_2026_biografico"], name_field="nome"
            )

        inserted = 0
        updated = 0

        for idx, artista in enumerate(artisti):
            nome = artista.get("nome")
            bio = bio_map.get(normalize_artist_name(nome), {})

            storico_list = artista.get("storico_fantasanremo", [])
            debuttante = artista.get("debuttante_2026")
            if debuttante is None:
                debuttante = len(storico_list) == 0

            record = {
                "id": artista.get("id") or idx + 1,
                "nome": nome,
                "quotazione_2026": artista.get("quotazione"),
                "genere_musicale": bio.get("genere_musicale"),
                "anno_nascita": bio.get("anno_nascita"),
                "prima_partecipazione": bio.get("prima_partecipazione"),
                "debuttante_2026": debuttante,
                "image_url": artista.get("image_url"),
            }

            success, message = self.store.upsert_artist(record, Artista)

            if "Inserted" in message:
                inserted += 1
            elif "Updated" in message:
                updated += 1

        self.logger.info(f"Artists: {inserted} inserted, {updated} updated")

        return {"inserted": inserted, "updated": updated}

    def _insert_historical_data(self, storico_data: list) -> dict:
        """Insert dati storici nel database."""
        if not storico_data:
            return {"inserted": 0, "updated": 0}

        # Build artist name to ID map
        artisti_map = {normalize_artist_name(a.nome): a.id for a in self.db.query(Artista).all()}

        inserted = 0
        updated = 0

        for entry in storico_data:
            artista_nome = entry.get("artista")
            artista_key = normalize_artist_name(artista_nome)
            if artista_key not in artisti_map:
                continue

            artista_id = artisti_map[artista_key]
            punteggi_map = entry.get("punteggi", {}) or {}

            for anno, posizione in entry.get("posizioni", {}).items():
                if posizione == "NP":
                    continue

                score = punteggi_map.get(anno)
                if score is None:
                    score = punteggi_map.get(str(anno))

                edizione_data = {
                    "anno": int(anno),
                    "punteggio_finale": score,
                    "posizione": int(posizione),
                    "quotazione_baudi": entry.get("quotazioni", {}).get(anno),
                }

                success, message = self.store.upsert_edizione(
                    edizione_data, EdizioneFantaSanremo, artista_id
                )

                if "Inserted" in message:
                    inserted += 1
                elif "Updated" in message:
                    updated += 1

        self.logger.info(f"Historical: {inserted} inserted, {updated} updated")

        return {"inserted": inserted, "updated": updated}

    def _insert_characteristics(self, caratteristiche_data: dict) -> dict:
        """Insert caratteristiche artisti nel database."""
        if not caratteristiche_data:
            return {"inserted": 0}

        artisti_map = {normalize_artist_name(a.nome): a.id for a in self.db.query(Artista).all()}
        caratteristiche_map = index_by_normalized_name(
            caratteristiche_data.get("caratteristiche_artisti_2026", []), name_field="nome"
        )

        inserted = 0

        for nome_key, caratteristiche in caratteristiche_map.items():
            if nome_key not in artisti_map:
                continue

            artista_id = artisti_map[nome_key]

            # Check if exists
            existing = (
                self.db.query(CaratteristicheArtista)
                .filter(CaratteristicheArtista.artista_id == artista_id)
                .first()
            )

            if existing:
                # Update
                existing.viralita_social = caratteristiche.get("viralita_social")
                existing.social_followers_total = caratteristiche.get("social_followers_total")
                existing.social_followers_by_platform = json.dumps(
                    caratteristiche.get("social_followers_by_platform", {}), ensure_ascii=False
                )
                existing.social_followers_last_updated = caratteristiche.get(
                    "social_followers_last_updated"
                )
                existing.storia_bonus_ottenuti = caratteristiche.get("storia_bonus_ottenuti")
                existing.ad_personam_bonus_count = caratteristiche.get("ad_personam_bonus_count")
                existing.ad_personam_bonus_points = caratteristiche.get("ad_personam_bonus_points")
            else:
                # Insert
                car = CaratteristicheArtista(
                    artista_id=artista_id,
                    viralita_social=caratteristiche.get("viralita_social"),
                    social_followers_total=caratteristiche.get("social_followers_total"),
                    social_followers_by_platform=json.dumps(
                        caratteristiche.get("social_followers_by_platform", {}), ensure_ascii=False
                    ),
                    social_followers_last_updated=caratteristiche.get(
                        "social_followers_last_updated"
                    ),
                    storia_bonus_ottenuti=caratteristiche.get("storia_bonus_ottenuti"),
                    ad_personam_bonus_count=caratteristiche.get("ad_personam_bonus_count"),
                    ad_personam_bonus_points=caratteristiche.get("ad_personam_bonus_points"),
                )
                self.db.add(car)
                inserted += 1

        self.logger.info(f"Characteristics: {inserted} inserted")

        return {"inserted": inserted}

    def _generate_predictions(self) -> dict:
        """Genera predizioni ML."""
        artisti_list = [
            {"id": a.id, "nome": a.nome, "quotazione_2026": a.quotazione_2026}
            for a in self.db.query(Artista).all()
        ]

        storico_list = [
            {"artista_id": e.artista_id, "anno": e.anno, "posizione": e.posizione}
            for e in self.db.query(EdizioneFantaSanremo).all()
        ]

        predictions = generate_predictions_simple(artisti_list, storico_list)

        for pred in predictions:
            # Check if exists
            existing = (
                self.db.query(Predizione2026)
                .filter(Predizione2026.artista_id == pred["artista_id"])
                .first()
            )

            if existing:
                existing.punteggio_predetto = pred["punteggio_predetto"]
                existing.confidence = pred.get("confidence", 0.5)
                existing.livello_performer = pred["livello_performer"]
            else:
                new_pred = Predizione2026(
                    artista_id=pred["artista_id"],
                    punteggio_predetto=pred["punteggio_predetto"],
                    confidence=pred.get("confidence", 0.5),
                    livello_performer=pred["livello_performer"],
                )
                self.db.add(new_pred)

        self.logger.info(f"Predictions: {len(predictions)} generated")

        return {"generated": len(predictions)}

    def _generate_quality_report(self) -> dict:
        """Genera report qualità dati."""
        artist_count = self.db.query(Artista).count()
        storico_count = self.db.query(EdizioneFantaSanremo).count()
        car_count = self.db.query(CaratteristicheArtista).count()
        pred_count = self.db.query(Predizione2026).count()

        return {
            "artists": artist_count,
            "historical_records": storico_count,
            "characteristics": car_count,
            "predictions": pred_count,
            "timestamp": datetime.now().isoformat(),
        }

    def _clear_database(self):
        """Pulisce il database."""
        # Drop and recreate tables to align schema changes
        Base.metadata.drop_all(bind=engine)
        Base.metadata.create_all(bind=engine)
        self.db.commit()
        self.logger.info("Database cleared and schema rebuilt")

    def _build_storico_list(self, data_sources: dict) -> list[dict]:
        """
        Normalize storico data into a list of {artista, posizioni, punteggi, quotazioni}.
        """
        if "storico_unified" in data_sources:
            unified = data_sources.get("storico_unified", {})
            artisti_storici = unified.get("artisti_storici", {})
            if artisti_storici:
                out = []
                for artista_nome, data in artisti_storici.items():
                    posizioni = {}
                    quotazioni = {}
                    punteggi = {}
                    for ed in data.get("edizioni", []):
                        anno = ed.get("anno")
                        posizione = ed.get("posizione")
                        if anno and posizione is not None:
                            posizioni[str(anno)] = posizione
                        punteggio = ed.get("punteggio_finale")
                        if anno and punteggio is not None:
                            punteggi[str(anno)] = punteggio
                        quot = ed.get("quotazione_baudi")
                        if anno and quot is not None:
                            quotazioni[str(anno)] = quot
                    out.append(
                        {
                            "artista": artista_nome,
                            "posizioni": posizioni,
                            "punteggi": punteggi,
                            "quotazioni": quotazioni,
                        }
                    )
                return out

            # Fallback: use artisti_2026 in unified (limited to recent years)
            artisti_2026 = unified.get("artisti_2026", [])
            if artisti_2026:
                out = []
                for entry in artisti_2026:
                    posizioni = entry.get("storico_posizioni", {})
                    out.append(
                        {
                            "artista": entry.get("nome"),
                            "posizioni": posizioni,
                            "punteggi": {},
                            "quotazioni": {},
                        }
                    )
                return out

        storico = data_sources.get("storico")
        if isinstance(storico, list):
            return storico
        if isinstance(storico, dict):
            return storico.get("storico_fantasanremo", [])

        storico_completo = data_sources.get("storico_completo", {})
        if isinstance(storico_completo, dict):
            out = []
            for entry in storico_completo.get("storico_fantasanremo_completo", []):
                artista_nome = entry.get("artista")
                posizioni = {}
                punteggi = {}
                quotazioni = {}
                for ed in entry.get("edizioni", []):
                    anno = ed.get("anno")
                    posizione = ed.get("posizione")
                    if anno and posizione is not None:
                        posizioni[str(anno)] = posizione
                    punteggio = ed.get("punteggio_finale")
                    if anno and punteggio is not None:
                        punteggi[str(anno)] = punteggio
                    quot = ed.get("quotazione_baudi")
                    if anno and quot is not None:
                        quotazioni[str(anno)] = quot
                out.append(
                    {
                        "artista": artista_nome,
                        "posizioni": posizioni,
                        "punteggi": punteggi,
                        "quotazioni": quotazioni,
                    }
                )
            return out

        return []


def populate_database(
    validate: bool = True,
    generate_predictions: bool = True,
    force_refresh: bool = False,
    config=None,
) -> dict:
    """
    Funzione convenienza per popolare il database.

    Args:
        validate: Se True, valida i dati
        generate_predictions: Se True, genera predizioni
        force_refresh: Se True, sovrascrive dati esistenti
        config: Configurazione opzionale

    Returns:
        Dict con risultati
    """
    populator = DatabasePopulator(config)
    return populator.populate_database(
        validate=validate, generate_predictions=generate_predictions, force_refresh=force_refresh
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Populate Fantasanremo database")
    parser.add_argument("--force-refresh", action="store_true", help="Force refresh existing data")
    parser.add_argument("--no-validate", action="store_true", help="Skip data validation")
    parser.add_argument(
        "--no-predictions", action="store_true", help="Skip ML predictions generation"
    )
    args = parser.parse_args()

    result = populate_database(
        validate=not args.no_validate,
        generate_predictions=not args.no_predictions,
        force_refresh=args.force_refresh,
    )

    if result["status"] == "success":
        print("\n✓ Database populated successfully!")
        print(f"  Artists: {result['stats']['details']['artists']['inserted']}")
        print(f"  Historical: {result['stats']['details']['historical']['inserted']}")
        print(f"  Characteristics: {result['stats']['details']['characteristics']['inserted']}")
        print(f"  Predictions: {result['stats']['details']['predictions']['generated']}")
    elif result["status"] == "already_populated":
        print("\nℹ Database already populated")
    else:
        print(f"\n✗ Error: {result}")
