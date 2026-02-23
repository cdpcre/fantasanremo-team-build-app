"""
Business Rules Validator Module

Valida i dati contro le regole business di FantaSanremo.
"""

from backend.utils.name_normalization import index_by_normalized_name, normalize_artist_name

from ..config import get_config, get_logger


class BusinessRulesValidator:
    """
    Validatore di regole business per FantaSanremo.
    """

    def __init__(self, config=None):
        """
        Inizializza il validatore.

        Args:
            config: Configurazione pipeline (usa default se None)
        """
        self.config = config or get_config()
        self.logger = get_logger(f"{self.__class__.__name__}")

    def validate_artisti_2026(self, artisti_data: dict) -> tuple[bool, list[str]]:
        """
        Valida artisti 2026 contro regole business.

        Args:
            artisti_data: Dict con dati artisti

        Returns:
            Tuple (is_valid: bool, errors: List[str])
        """
        errors = []

        if not isinstance(artisti_data, dict) or "artisti" not in artisti_data:
            errors.append("Missing 'artisti' key in artisti_2026 data")
            return False, errors

        artisti = artisti_data["artisti"]
        if not isinstance(artisti, list):
            errors.append("'artisti' must be a list")
            return False, errors

        # Check unique IDs
        ids = [a.get("id") for a in artisti if "id" in a]
        if len(ids) != len(set(ids)):
            errors.append("Duplicate artist IDs found")

        # Check unique names
        nomi = [a.get("nome") for a in artisti if "nome" in a]
        if len(nomi) != len(set(nomi)):
            errors.append("Duplicate artist names found")

        # Validate quotations
        for artista in artisti:
            quotazione = artista.get("quotazione")
            if quotazione is not None:
                if not isinstance(quotazione, int):
                    errors.append(f"Invalid quotation type for {artista.get('nome')}: {quotazione}")
                elif (
                    quotazione < self.config.min_quotazione
                    or quotazione > self.config.max_quotazione
                ):
                    errors.append(
                        f"Invalid quotation for {artista.get('nome')}: {quotazione} "
                        f"(must be {self.config.min_quotazione}-{self.config.max_quotazione})"
                    )

        # Check team summability (should be possible to form 100 baudi teams)
        total = sum(a.get("quotazione", 0) for a in artisti if isinstance(a.get("quotazione"), int))
        if total < len(artisti) * self.config.min_quotazione:
            errors.append("Total quotations too low to form valid teams")

        return len(errors) == 0, errors

    def validate_storico(self, storico_data: list) -> tuple[bool, list[str]]:
        """
        Valida dati storici contro regole business.

        Args:
            storico_data: Lista di dict storico

        Returns:
            Tuple (is_valid: bool, errors: List[str])
        """
        errors = []

        if not isinstance(storico_data, list):
            errors.append("Storico data must be a list")
            return False, errors

        for entry in storico_data:
            if not isinstance(entry, dict):
                errors.append(f"Invalid entry type: {type(entry)}")
                continue

            # Check required fields
            if "artista" not in entry:
                errors.append(f"Missing 'artista' in entry: {entry}")
                continue

            if "posizioni" not in entry:
                errors.append(f"Missing 'posizioni' for artist: {entry.get('artista')}")
                continue

            # Validate positions
            posizioni = entry["posizioni"]
            if not isinstance(posizioni, dict):
                errors.append(f"Invalid posizioni type for {entry.get('artista')}")
                continue

            for anno, pos in posizioni.items():
                # Validate year format
                try:
                    anno_int = int(anno)
                    if anno_int < 2020 or anno_int > 2026:
                        errors.append(f"Invalid year {anno} for {entry.get('artista')}")
                except ValueError:
                    errors.append(f"Invalid year format {anno} for {entry.get('artista')}")

                # Validate position value
                if pos != "NP":
                    try:
                        pos_int = int(pos)
                        if pos_int < 1 or pos_int > 30:
                            errors.append(
                                f"Invalid position {pos} for {entry.get('artista')} in {anno}"
                            )
                    except ValueError:
                        errors.append(
                            f"Invalid position value {pos} for {entry.get('artista')} in {anno}"
                        )

        return len(errors) == 0, errors

    def validate_biografico(self, biografico_data: dict) -> tuple[bool, list[str]]:
        """
        Valida dati biografici.

        Args:
            biografico_data: Dict con dati biografici

        Returns:
            Tuple (is_valid: bool, errors: List[str])
        """
        errors = []

        if (
            not isinstance(biografico_data, dict)
            or "artisti_2026_biografico" not in biografico_data
        ):
            errors.append("Missing 'artisti_2026_biografico' key")
            return False, errors

        artisti = biografico_data["artisti_2026_biografico"]
        if not isinstance(artisti, list):
            errors.append("'artisti_2026_biografico' must be a list")
            return False, errors

        for artista in artisti:
            if not isinstance(artista, dict):
                continue

            nome = artista.get("nome")
            if not nome:
                errors.append("Missing artist name in biografico data")
                continue

            # Validate birth year if present
            anno_nascita = artista.get("anno_nascita")
            if anno_nascita is not None:
                if not isinstance(anno_nascita, int):
                    errors.append(f"Invalid birth year type for {nome}")
                elif anno_nascita < 1940 or anno_nascita > 2010:
                    errors.append(f"Suspicious birth year for {nome}: {anno_nascita}")

            # Validate genre if present
            genere = artista.get("genere_musicale")
            if genere and not isinstance(genere, str):
                errors.append(f"Invalid genre type for {nome}")

        return len(errors) == 0, errors

    def validate_caratteristiche(self, caratteristiche_data: dict) -> tuple[bool, list[str]]:
        """
        Valida dati caratteristiche.

        Args:
            caratteristiche_data: Dict con caratteristiche

        Returns:
            Tuple (is_valid: bool, errors: List[str])
        """
        errors = []

        if (
            not isinstance(caratteristiche_data, dict)
            or "caratteristiche_artisti_2026" not in caratteristiche_data
        ):
            errors.append("Missing 'caratteristiche_artisti_2026' key")
            return False, errors

        artisti = caratteristiche_data["caratteristiche_artisti_2026"]
        if not isinstance(artisti, list):
            errors.append("'caratteristiche_artisti_2026' must be a list")
            return False, errors

        valid_ranges = {
            "viralita_social": (1, 100),
            "storia_bonus_ottenuti": (0, 2500),
            "ad_personam_bonus_count": (0, 50),
            "ad_personam_bonus_points": (0, 500),
            "social_followers_total": (0, 1_000_000_000),
        }

        for artista in artisti:
            if not isinstance(artista, dict):
                continue

            nome = artista.get("nome")
            if not nome:
                errors.append("Missing artist name in caratteristiche data")
                continue

            for field, (min_val, max_val) in valid_ranges.items():
                value = artista.get(field)
                if value is not None:
                    if not isinstance(value, int):
                        errors.append(f"Invalid {field} type for {nome}")
                    elif value < min_val or value > max_val:
                        errors.append(f"{field} out of range for {nome}: {value}")

        return len(errors) == 0, errors

    def validate_cross_file_consistency(
        self, artisti_2026: dict, biografico: dict, caratteristiche: dict, storico: list
    ) -> tuple[bool, list[str]]:
        """
        Valida coerenza tra file diversi.

        Args:
            artisti_2026: Dati artisti 2026
            biografico: Dati biografici
            caratteristiche: Dati caratteristiche
            storico: Dati storici

        Returns:
            Tuple (is_valid: bool, errors: List[str])
        """
        errors = []
        warnings = []

        # Extract artist names from each source
        artisti_map = {}
        if isinstance(artisti_2026, dict) and "artisti" in artisti_2026:
            artisti_map = index_by_normalized_name(artisti_2026["artisti"], name_field="nome")
        artisti_names = set(artisti_map.keys())

        biografico_map = {}
        if isinstance(biografico, dict) and "artisti_2026_biografico" in biografico:
            biografico_map = index_by_normalized_name(
                biografico["artisti_2026_biografico"], name_field="nome"
            )
        biografico_names = set(biografico_map.keys())

        caratteristiche_map = {}
        if isinstance(caratteristiche, dict) and "caratteristiche_artisti_2026" in caratteristiche:
            caratteristiche_map = index_by_normalized_name(
                caratteristiche["caratteristiche_artisti_2026"], name_field="nome"
            )
        caratteristiche_names = set(caratteristiche_map.keys())

        storico_map = {}
        if isinstance(storico, list):
            storico_map = index_by_normalized_name(storico, name_field="artista")
        elif isinstance(storico, dict):
            # Unified historical structure: { artisti_storici: { "<name>": {...} } }
            storico_map = {
                normalize_artist_name(name): payload
                for name, payload in storico.get("artisti_storici", {}).items()
                if name
            }
        storico_names = set(storico_map.keys())

        # Compute artists that should have historical references.
        expected_history_names = set()
        for artista in artisti_2026.get("artisti", []):
            nome_key = normalize_artist_name(artista.get("nome"))
            if not nome_key:
                continue
            storico_locale = artista.get("storico_fantasanremo", [])
            if isinstance(storico_locale, list) and storico_locale:
                expected_history_names.add(nome_key)
                continue
            prima = artista.get("prima_partecipazione")
            if isinstance(prima, int) and prima < 2026:
                expected_history_names.add(nome_key)
                continue
            if artista.get("debuttante_2026") is False:
                expected_history_names.add(nome_key)

        # Check debuttante flag consistency
        if artisti_names and storico_names:
            debuttanti_in_file = set()
            for artista in artisti_2026.get("artisti", []):
                if artista.get("debuttante_2026", False):
                    debuttanti_in_file.add(normalize_artist_name(artista.get("nome")))

            # Artists in storico shouldn't be marked as debuttanti
            conflicting = debuttanti_in_file & storico_names
            if conflicting:
                warnings.extend(
                    [f"Artist marked as debuttante but has history: {name}" for name in conflicting]
                )

        # Check that all 2026 artists have biografico data
        if artisti_names and biografico_names:
            missing_biografico = artisti_names - biografico_names
            if missing_biografico:
                warnings.extend(
                    [
                        "Missing biografico data for: "
                        f"{artisti_map.get(name, {}).get('nome', name)}"
                        for name in sorted(missing_biografico)
                    ]
                )

        # Historical coverage checks should target only artists expected to have history.
        if expected_history_names and storico_names:
            missing_history = expected_history_names - storico_names
            if missing_history:
                warnings.extend(
                    [
                        "Missing storico_unified data for: "
                        f"{artisti_map.get(name, {}).get('nome', name)}"
                        for name in sorted(missing_history)
                    ]
                )

        # Check for name inconsistencies (slight variations)
        all_names = artisti_names | biografico_names | caratteristiche_names | storico_names
        for name in all_names:
            similar = self._find_similar_names(name, all_names)
            if len(similar) > 1:
                warnings.append(f"Possible name variation: {name} vs {similar}")

        # Check biografico debuttante flag consistency
        if biografico_names and storico_names:
            for entry in biografico.get("artisti_2026_biografico", []):
                nome = entry.get("nome")
                prima_part = entry.get("prima_partecipazione")
                if prima_part == 2026 and normalize_artist_name(nome) in storico_names:
                    warnings.append(
                        f"Artist {nome} has prima_partecipazione=2026 but exists in storico"
                    )

        # Log warnings as info, not errors
        for warning in warnings:
            self.logger.warning(f"Cross-file consistency warning: {warning}")

        return len(errors) == 0, errors

    def validate_team_composition(self, team: dict) -> tuple[bool, list[str]]:
        """
        Valida composizione team FantaSanremo.

        Args:
            team: Dict con team data

        Returns:
            Tuple (is_valid: bool, errors: List[str])
        """
        errors = []

        titolari = team.get("titolari", [])
        riserve = team.get("riserve", [])
        capitano = team.get("capitano")

        # Check team size
        if len(titolari) != self.config.titolari_count:
            errors.append(f"Must have {self.config.titolari_count} titolari, got {len(titolari)}")

        if len(riserve) != self.config.riserve_count:
            errors.append(f"Must have {self.config.riserve_count} riserve, got {len(riserve)}")

        # Check captain in titolari
        if capitano and capitano not in titolari:
            errors.append(f"Capitano {capitano} not in titolari")

        # Check unique artists
        all_artists = titolari + riserve
        if len(all_artists) != len(set(all_artists)):
            errors.append("Duplicate artists in team")

        # Check budget
        # Would need quotation data to validate

        return len(errors) == 0, errors

    def _find_similar_names(self, name: str, names: set[str], threshold: float = 0.8) -> list[str]:
        """Trova nomi simili (per variazioni)."""
        similar = []
        name_lower = name.lower()

        for other in names:
            if other == name:
                continue
            other_lower = other.lower()

            # Simple similarity check
            if name_lower in other_lower or other_lower in name_lower:
                similar.append(other)

        return similar
