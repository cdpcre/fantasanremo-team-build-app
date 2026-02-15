"""
Database Store Module

Gestisce le operazioni di salvataggio e caricamento dal database.
"""

from typing import Any

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ..config import get_config, get_logger


class DatabaseStore:
    """
    Gestisce le operazioni database con upsert e transaction management.
    """

    def __init__(self, session: Session, config=None):
        """
        Inizializza il database store.

        Args:
            session: SQLAlchemy session
            config: Configurazione pipeline (usa default se None)
        """
        self.session = session
        self.config = config or get_config()
        self.logger = get_logger(f"{self.__class__.__name__}")

        # Statistics
        self.stats = {"inserted": 0, "updated": 0, "skipped": 0, "errors": 0}

    def upsert_artist(
        self,
        artista_data: dict[str, Any],
        model_class: Any,
        id_field: str = "id",
        unique_field: str = "nome",
    ) -> tuple[bool, str]:
        """
        Esegue un upsert (insert o update) per un artista.

        Args:
            artista_data: Dict con dati artista
            model_class: Modello SQLAlchemy
            id_field: Campo ID
            unique_field: Campo unico per check esistenza

        Returns:
            Tuple (success: bool, message: str)
        """
        try:
            # Check if exists
            unique_value = artista_data.get(unique_field)
            if not unique_value:
                return False, f"Missing unique field: {unique_field}"

            existing = (
                self.session.query(model_class)
                .filter(getattr(model_class, unique_field) == unique_value)
                .first()
            )

            if existing:
                # Update existing record
                for key, value in artista_data.items():
                    if hasattr(existing, key):
                        setattr(existing, key, value)

                self.stats["updated"] += 1
                return True, f"Updated {unique_value}"

            else:
                # Insert new record
                new_record = model_class(**artista_data)
                self.session.add(new_record)
                self.session.flush()  # Get ID without committing

                self.stats["inserted"] += 1
                return True, f"Inserted {unique_value}"

        except IntegrityError as e:
            self.session.rollback()
            self.stats["errors"] += 1
            self.logger.error(f"Integrity error for {artista_data.get(unique_field)}: {e}")
            return False, f"Integrity error: {e}"

        except Exception as e:
            self.session.rollback()
            self.stats["errors"] += 1
            self.logger.error(f"Error upserting artist: {e}")
            return False, f"Error: {e}"

    def bulk_upsert(
        self,
        records: list[dict[str, Any]],
        model_class: Any,
        unique_field: str = "nome",
        batch_size: int = 100,
    ) -> dict[str, int]:
        """
        Esegue upsert in bulk per una lista di record.

        Args:
            records: Lista di dict con dati
            model_class: Modello SQLAlchemy
            unique_field: Campo unico
            batch_size: Dimensione batch per commit

        Returns:
            Dict con statistiche
        """
        self.logger.info(f"Starting bulk upsert of {len(records)} records")

        for i, record in enumerate(records):
            success, message = self.upsert_artist(record, model_class, unique_field=unique_field)

            if not success:
                self.logger.warning(f"Failed to upsert record {i}: {message}")

            # Commit periodically
            if (i + 1) % batch_size == 0:
                try:
                    self.session.commit()
                    self.logger.info(f"Committed batch at {i + 1} records")
                except Exception as e:
                    self.session.rollback()
                    self.logger.error(f"Batch commit failed: {e}")

        # Final commit
        try:
            self.session.commit()
            self.logger.info("Final commit successful")
        except Exception as e:
            self.session.rollback()
            self.logger.error(f"Final commit failed: {e}")

        return self.get_stats()

    def upsert_edizione(
        self, edizione_data: dict[str, Any], model_class: Any, artista_id: int
    ) -> tuple[bool, str]:
        """
        Upsert per edizione FantaSanremo.

        Args:
            edizione_data: Dict con dati edizione
            model_class: Modello SQLAlchemy
            artista_id: ID artista

        Returns:
            Tuple (success: bool, message: str)
        """
        try:
            # Check if exists
            existing = (
                self.session.query(model_class)
                .filter(
                    model_class.artista_id == artista_id,
                    model_class.anno == edizione_data.get("anno"),
                )
                .first()
            )

            if existing:
                # Update
                for key, value in edizione_data.items():
                    if hasattr(existing, key):
                        setattr(existing, key, value)

                self.stats["updated"] += 1
                return True, f"Updated edition {edizione_data.get('anno')}"

            else:
                # Insert
                edizione_data["artista_id"] = artista_id
                new_record = model_class(**edizione_data)
                self.session.add(new_record)
                self.session.flush()

                self.stats["inserted"] += 1
                return True, f"Inserted edition {edizione_data.get('anno')}"

        except Exception as e:
            self.session.rollback()
            self.stats["errors"] += 1
            self.logger.error(f"Error upserting edition: {e}")
            return False, f"Error: {e}"

    def get_artist_by_name(self, nome: str, model_class: Any) -> Any | None:
        """
        Recupera artista per nome.

        Args:
            nome: Nome artista
            model_class: Modello SQLAlchemy

        Returns:
            Istanza modello o None
        """
        try:
            return self.session.query(model_class).filter(model_class.nome == nome).first()
        except Exception as e:
            self.logger.error(f"Error fetching artist {nome}: {e}")
            return None

    def get_artist_by_id(self, artista_id: int, model_class: Any) -> Any | None:
        """
        Recupera artista per ID.

        Args:
            artista_id: ID artista
            model_class: Modello SQLAlchemy

        Returns:
            Istanza modello o None
        """
        try:
            return self.session.query(model_class).filter(model_class.id == artista_id).first()
        except Exception as e:
            self.logger.error(f"Error fetching artist ID {artista_id}: {e}")
            return None

    def execute_query(self, query: Any) -> list[Any]:
        """
        Esegue una query SQLAlchemy.

        Args:
            query: Query SQLAlchemy

        Returns:
            Lista di risultati
        """
        try:
            return query.all()
        except Exception as e:
            self.logger.error(f"Error executing query: {e}")
            return []

    def get_stats(self) -> dict[str, int]:
        """Restituisce le statistiche delle operazioni."""
        return self.stats.copy()

    def reset_stats(self):
        """Resetta le statistiche."""
        self.stats = {"inserted": 0, "updated": 0, "skipped": 0, "errors": 0}

    def commit(self) -> bool:
        """
        Esegue commit della transazione.

        Returns:
            True se successo, False altrimenti
        """
        try:
            self.session.commit()
            return True
        except Exception as e:
            self.session.rollback()
            self.logger.error(f"Commit failed: {e}")
            return False

    def rollback(self):
        """Esegue rollback della transazione."""
        self.session.rollback()
        self.logger.info("Transaction rolled back")
