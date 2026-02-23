"""
Base Data Source Abstract Class

Definisce l'interfaccia per tutte le sorgenti dati della pipeline.
"""

import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

from ..config import get_config, get_logger


class BaseDataSource(ABC):
    """
    Classe base astratta per le sorgenti dati.

    Tutte le sorgenti dati devono implementare questi metodi:
    - fetch(): Recupera i dati dalla sorgente
    - validate(): Valida i dati recuperati
    - transform(): Trasforma i dati nel formato atteso
    """

    def __init__(self, name: str, config=None):
        """
        Inizializza la sorgente dati.

        Args:
            name: Nome identificativo della sorgente
            config: Configurazione della pipeline (usa default se None)
        """
        self.name = name
        self.config = config or get_config()
        self.logger = get_logger(f"{self.__class__.__name__}")
        self._metadata: dict[str, Any] = {}

    @abstractmethod
    def fetch(self, **kwargs) -> Any:
        """
        Recupera i dati dalla sorgente.

        Args:
            **kwargs: Parametri specifici della sorgente

        Returns:
            Dati recuperati dalla sorgente
        """
        pass

    def validate(self, data: Any) -> bool:
        """
        Valida i dati recuperati.

        Args:
            data: Dati da validare

        Returns:
            True se i dati sono validi, False altrimenti
        """
        if data is None:
            self.logger.warning(f"Data validation failed for {self.name}: data is None")
            return False

        self._metadata["validated_at"] = datetime.now().isoformat()
        return True

    @abstractmethod
    def transform(self, data: Any) -> dict[str, Any]:
        """
        Trasforma i dati nel formato atteso.

        Args:
            data: Dati da trasformare

        Returns:
            Dict con i dati trasformati
        """
        pass

    def fetch_with_retry(self, max_retries: int | None = None, **kwargs) -> Any:
        """
        Fetch con retry logic e exponential backoff.

        Args:
            max_retries: Numero massimo di tentativi (usa config se None)
            **kwargs: Parametri passati a fetch()

        Returns:
            Dati recuperati o None se tutti i tentativi falliscono
        """
        max_retries = max_retries or self.config.max_retries
        base_delay = self.config.retry_base_delay
        max_delay = self.config.retry_max_delay

        last_exception = None

        for attempt in range(max_retries + 1):
            try:
                self.logger.info(
                    f"Fetching from {self.name} (attempt {attempt + 1}/{max_retries + 1})"
                )
                data = self.fetch(**kwargs)
                self._metadata["fetch_attempts"] = attempt + 1
                self._metadata["fetched_at"] = datetime.now().isoformat()
                return data

            except Exception as e:
                last_exception = e
                self.logger.warning(f"Attempt {attempt + 1} failed for {self.name}: {e}")

                if attempt < max_retries:
                    # Exponential backoff
                    delay = min(base_delay * (2**attempt), max_delay)
                    self.logger.info(f"Retrying in {delay:.1f} seconds...")
                    time.sleep(delay)

        # All retries exhausted
        self.logger.error(f"All fetch attempts failed for {self.name}")
        if last_exception:
            raise last_exception
        return None

    def get_metadata(self) -> dict[str, Any]:
        """
        Restituisce i metadati dell'ultima operazione.

        Returns:
            Dict con i metadati
        """
        return self._metadata.copy()

    def clear_metadata(self):
        """Pulisce i metadati."""
        self._metadata = {}

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
