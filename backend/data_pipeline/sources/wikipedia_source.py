"""
Wikipedia Data Source

Recupera dati da Wikipedia per FantaSanremo.
"""

import hashlib
import json
from datetime import datetime, timedelta
from typing import Any

from .base import BaseDataSource


class WikipediaDataSource(BaseDataSource):
    """
    Sorgente dati Wikipedia per FantaSanremo.

    Recupera informazioni sulle edizioni di Sanremo e sui partecipanti
    utilizzando web scraping o API MCP.
    """

    def __init__(self, config=None):
        super().__init__("wikipedia", config)
        self.cache_dir = self.config.cache_dir / "wikipedia"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch(self, year: int | None = None, **kwargs) -> dict[str, Any]:
        """
        Recupera dati da Wikipedia.

        Args:
            year: Anno dell'edizione (opzionale)
            **kwargs: Altri parametri

        Returns:
            Dict con i dati recuperati
        """
        # Try cache first
        cache_key = self._get_cache_key(year)
        cached_data = self._load_from_cache(cache_key)

        if cached_data and not self._is_cache_expired(cached_data):
            self.logger.info(f"Using cached data for year {year}")
            return cached_data["data"]

        # Fetch fresh data
        data = self._fetch_from_wikipedia(year)

        # Save to cache
        if data:
            self._save_to_cache(cache_key, data)

        return data

    def validate(self, data: Any) -> bool:
        """Valida i dati Wikipedia."""
        if not super().validate(data):
            return False

        if not isinstance(data, dict):
            self.logger.warning("Wikipedia data is not a dict")
            return False

        # Check for required fields
        required_fields = ["edizione", "artisti"]
        for field in required_fields:
            if field not in data:
                self.logger.warning(f"Missing required field: {field}")
                return False

        return True

    def transform(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Trasforma i dati Wikipedia nel formato FantaSanremo.

        Args:
            data: Dati grezzi da Wikipedia

        Returns:
            Dict con dati trasformati
        """
        transformed = {
            "source": "wikipedia",
            "fetched_at": datetime.now().isoformat(),
            "edizione": data.get("edizione"),
            "artisti": [],
        }

        for artista in data.get("artisti", []):
            transformed_artist = {
                "nome": artista.get("nome"),
                "canzone": artista.get("canzone"),
                "genere": self._extract_genre(artista),
                "anno_partecipazione": data.get("edizione"),
            }
            transformed["artisti"].append(transformed_artist)

        return transformed

    def _fetch_from_wikipedia(self, year: int | None) -> dict[str, Any]:
        """
        Recupera dati da Wikipedia usando web-reader MCP o BeautifulSoup.

        Args:
            year: Anno dell'edizione

        Returns:
            Dict con i dati recuperati
        """
        # Use web-reader MCP if available, otherwise fallback to mock data
        # This is a placeholder - actual implementation would use the MCP tool

        url = self.config.wikipedia_url
        if year:
            url = f"{url}_{year}"

        self.logger.info(f"Fetching data from {url}")

        # Placeholder return - implement actual scraping
        return {
            "edizione": year or datetime.now().year,
            "artisti": [],
            "metadata": {"source_url": url, "scraped_at": datetime.now().isoformat()},
        }

    def _get_cache_key(self, year: int | None) -> str:
        """Genera chiave cache per i dati."""
        key = f"wikipedia_{year or 'latest'}"
        return hashlib.md5(key.encode()).hexdigest()

    def _load_from_cache(self, cache_key: str) -> dict | None:
        """Carica dati dalla cache."""
        cache_file = self.cache_dir / f"{cache_key}.json"

        if not cache_file.exists():
            return None

        try:
            with open(cache_file) as f:
                return json.load(f)
        except Exception as e:
            self.logger.warning(f"Failed to load cache: {e}")
            return None

    def _save_to_cache(self, cache_key: str, data: dict):
        """Salva dati nella cache."""
        cache_file = self.cache_dir / f"{cache_key}.json"

        cache_data = {"cached_at": datetime.now().isoformat(), "data": data}

        try:
            with open(cache_file, "w") as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Saved data to cache: {cache_file.name}")
        except Exception as e:
            self.logger.warning(f"Failed to save cache: {e}")

    def _is_cache_expired(self, cached_data: dict) -> bool:
        """Verifica se la cache è scaduta."""
        cached_at_str = cached_data.get("cached_at")
        if not cached_at_str:
            return True

        try:
            cached_at = datetime.fromisoformat(cached_at_str)
            expiry = cached_at + timedelta(hours=self.config.cache_ttl_hours)
            return datetime.now() > expiry
        except Exception:
            return True

    def _extract_genre(self, artista: dict) -> str:
        """Estrae il genere musicale da dati artista."""
        # Try various fields that might contain genre info
        genre_fields = ["genere", "genre", "stile", "style"]

        for field in genre_fields:
            if field in artista and artista[field]:
                return artista[field]

        # Try to infer from description/bio
        description = artista.get("descrizione", "")
        description_lower = description.lower()

        genre_keywords = {
            "pop": ["pop", "cantautore"],
            "rap": ["rap", "hip hop", "trap", "urban"],
            "rock": ["rock", "alternative", "indie"],
            "dance": ["dance", "elettronica", "edm"],
        }

        for genre, keywords in genre_keywords.items():
            if any(kw in description_lower for kw in keywords):
                return genre

        return "Pop"  # Default

    def clear_cache(self, older_than_hours: int | None = None):
        """
        Pulisce la cache.

        Args:
            older_than_hours: Rimuovi solo file più vecchi di X ore (None = all)
        """
        cutoff_time = None
        if older_than_hours:
            cutoff_time = datetime.now() - timedelta(hours=older_than_hours)

        removed = 0
        for cache_file in self.cache_dir.glob("*.json"):
            if cutoff_time:
                # Check file modification time
                mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)
                if mtime > cutoff_time:
                    continue

            cache_file.unlink()
            removed += 1

        self.logger.info(f"Cleared {removed} cache files")


class WikipediaHistoricalDataSource(WikipediaDataSource):
    """
    Sorgente dati Wikipedia per dati storici.

    Recupera dati storici delle edizioni passate.
    """

    def __init__(self, config=None):
        super().__init__(config)
        self.name = "wikipedia_historical"

    def fetch_years(self, years: list[int]) -> list[dict[str, Any]]:
        """
        Recupera dati per più anni.

        Args:
            years: Lista di anni da recuperare

        Returns:
            Lista di dict con i dati per ogni anno
        """
        results = []

        for year in years:
            try:
                data = self.fetch_with_retry(year=year)
                if data:
                    results.append(data)
            except Exception as e:
                self.logger.error(f"Failed to fetch year {year}: {e}")

        return results

    def transform_to_fantasanremo_format(self, data: list[dict]) -> dict[str, Any]:
        """
        Trasforma dati storici in formato FantaSanremo.

        Args:
            data: Lista di dati grezzi per anno

        Returns:
            Dict con struttura storico_fantasanremo.json
        """
        storico = []

        for year_data in data:
            year = year_data.get("edizione")
            artisti = year_data.get("artisti", [])

            for artista in artisti:
                nome = artista.get("nome")
                if not nome:
                    continue

                # Find or create artist entry
                existing = next((a for a in storico if a["artista"] == nome), None)

                if not existing:
                    entry = {"artista": nome, "posizioni": {}, "quotazioni": {}}
                    storico.append(entry)
                    existing = entry

                # Add position for this year
                # Note: Actual position would come from Wikipedia table
                existing["posizioni"][str(year)] = artista.get("posizione", "NP")

        return {
            "storico_fantasanremo": storico,
            "source": "wikipedia_historical",
            "generated_at": datetime.now().isoformat(),
        }
