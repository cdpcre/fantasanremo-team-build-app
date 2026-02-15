"""
Data Quality Checker Module

Verifica la qualità dei dati (completeness, accuracy, consistency).
"""

import statistics
from collections import Counter
from typing import Any

from backend.utils.name_normalization import normalize_artist_name

from ..config import get_config, get_logger


class DataQualityChecker:
    """
    Verificatore qualità dati per FantaSanremo.
    """

    def __init__(self, config=None):
        """
        Inizializza il quality checker.

        Args:
            config: Configurazione pipeline (usa default se None)
        """
        self.config = config or get_config()
        self.logger = get_logger(f"{self.__class__.__name__}")

    def check_quality(self, data_sources: dict[str, Any]) -> dict[str, Any]:
        """
        Esegue check qualità su tutti i data sources.

        Args:
            data_sources: Dict con nome -> dati

        Returns:
            Dict con report qualità
        """
        report = {
            "overall_score": 0,
            "sources": {},
            "issues": [],
            "warnings": [],
            "cross_file": {},
        }

        scores = []

        for source_name, data in data_sources.items():
            if source_name == "artisti_2026":
                source_report = self.check_artisti_quality(data)
            elif source_name == "storico":
                source_report = self.check_storico_quality(data)
            elif source_name == "storico_unified":
                source_report = self.check_storico_unified_quality(data)
            elif source_name == "biografico":
                source_report = self.check_biografico_quality(data)
            elif source_name == "caratteristiche":
                source_report = self.check_caratteristiche_quality(data)
            elif source_name == "voti_stampa":
                source_report = self.check_voti_stampa_quality(data)
            else:
                source_report = {"score": 50, "issues": ["Unknown source type"]}

            report["sources"][source_name] = source_report
            scores.append(source_report["score"])
            report["issues"].extend(source_report.get("issues", []))
            report["warnings"].extend(source_report.get("warnings", []))

        # Cross-file match rates
        cross_report = self._check_cross_file_matches(data_sources)
        report["cross_file"] = cross_report
        if cross_report.get("issues"):
            report["issues"].extend(cross_report["issues"])
        if cross_report.get("warnings"):
            report["warnings"].extend(cross_report["warnings"])

        # Calculate overall score
        if scores:
            overall = int(statistics.mean(scores))
            min_match = cross_report.get("min_match_rate")
            if min_match is not None and min_match < self.config.min_crossfile_match_rate:
                penalty = int((self.config.min_crossfile_match_rate - min_match) * 100)
                overall = max(0, overall - penalty)
            report["overall_score"] = overall

        return report

    def check_artisti_quality(self, data: dict) -> dict[str, Any]:
        """
        Verifica qualità dati artisti.

        Args:
            data: Dict artisti_2026

        Returns:
            Dict con report qualità
        """
        issues = []
        warnings = []
        score = 100

        if not isinstance(data, dict) or "artisti" not in data:
            return {"score": 0, "issues": ["Invalid data structure"], "warnings": []}

        artisti = data["artisti"]
        total = len(artisti)

        # Completeness
        required_fields = ["id", "nome", "quotazione"]
        missing_counts = Counter()

        for artista in artisti:
            for field in required_fields:
                if field not in artista or artista[field] is None:
                    missing_counts[field] += 1

        for field, count in missing_counts.items():
            if count > 0:
                pct = (count / total) * 100
                issues.append(f"{count} artists ({pct:.1f}%) missing {field}")
                score -= 10

        # Check for empty names
        empty_names = sum(1 for a in artisti if not a.get("nome"))
        if empty_names > 0:
            issues.append(f"{empty_names} artists with empty names")
            score -= 20

        # Check quotations distribution
        quotations = [a.get("quotazione") for a in artisti if isinstance(a.get("quotazione"), int)]
        if quotations:
            if len(quotations) < total:
                warnings.append(f"{total - len(quotations)} artists missing valid quotations")

            # Check distribution balance
            quote_counts = Counter(quotations)
            if len(quote_counts) < 3:  # Should have variety
                warnings.append("Limited quotation variety")

        completeness = 100 - (sum(missing_counts.values()) / total * 100 if total > 0 else 0)
        if completeness / 100 < self.config.min_required_field_coverage:
            issues.append(
                f"Core field coverage below threshold: {completeness:.1f}% "
                f"(min {self.config.min_required_field_coverage * 100:.0f}%)"
            )
            score -= 10

        return {
            "score": max(0, min(100, score)),
            "issues": issues,
            "warnings": warnings,
            "total_artists": total,
            "completeness": completeness,
        }

    def check_storico_quality(self, data: list) -> dict[str, Any]:
        """
        Verifica qualità dati storici.

        Args:
            data: Lista storico

        Returns:
            Dict con report qualità
        """
        issues = []
        warnings = []
        score = 100

        if not isinstance(data, list):
            return {"score": 0, "issues": ["Invalid data structure"], "warnings": []}

        total = len(data)

        # Completeness
        missing_artista = sum(1 for entry in data if not entry.get("artista"))
        if missing_artista > 0:
            issues.append(f"{missing_artista} entries missing artist name")
            score -= 20

        missing_posizioni = sum(1 for entry in data if not entry.get("posizioni"))
        if missing_posizioni > 0:
            issues.append(f"{missing_posizioni} entries missing positions")
            score -= 15

        # Check position values
        invalid_positions = 0
        np_count = 0

        for entry in data:
            posizioni = entry.get("posizioni", {})
            if isinstance(posizioni, dict):
                for anno, pos in posizioni.items():
                    if pos == "NP":
                        np_count += 1
                    elif isinstance(pos, int):
                        if pos < 1 or pos > 30:
                            invalid_positions += 1
                    else:
                        invalid_positions += 1

        if invalid_positions > 0:
            issues.append(f"{invalid_positions} invalid position values")
            score -= 10

        # Check year coverage
        years = set()
        for entry in data:
            posizioni = entry.get("posizioni", {})
            if isinstance(posizioni, dict):
                years.update(posizioni.keys())

        if years:
            min(int(y) for y in years if y.isdigit())
            max_year = max(int(y) for y in years if y.isdigit())
            expected_years = set(range(2020, max_year + 1))
            missing_years = expected_years - set(int(y) for y in years if y.isdigit())

            if missing_years:
                warnings.append(f"Missing data for years: {sorted(missing_years)}")

        return {
            "score": max(0, min(100, score)),
            "issues": issues,
            "warnings": warnings,
            "total_entries": total,
            "np_entries": np_count,
            "years_covered": len(years),
        }

    def check_storico_unified_quality(self, data: dict) -> dict[str, Any]:
        """
        Verifica qualità dati storici unificati.
        """
        issues = []
        warnings = []
        score = 100

        if not isinstance(data, dict) or "artisti_storici" not in data:
            return {"score": 0, "issues": ["Invalid unified storico structure"], "warnings": []}

        artisti_storici = data.get("artisti_storici", {})
        if not artisti_storici:
            return {"score": 0, "issues": ["Empty artisti_storici"], "warnings": []}

        total_entries = 0
        invalid_positions = 0
        years = set()

        for _, artista_data in artisti_storici.items():
            for ed in artista_data.get("edizioni", []):
                total_entries += 1
                anno = ed.get("anno")
                pos = ed.get("posizione")
                if anno:
                    years.add(str(anno))
                if pos is None or not isinstance(pos, int) or pos < 1 or pos > 30:
                    invalid_positions += 1

        if invalid_positions > 0:
            issues.append(f"{invalid_positions} invalid position values in unified storico")
            score -= 15

        if years:
            max_year = max(int(y) for y in years if str(y).isdigit())
            expected_years = set(range(2020, max_year + 1))
            missing_years = expected_years - set(int(y) for y in years if str(y).isdigit())
            if missing_years:
                warnings.append(f"Missing data for years: {sorted(missing_years)}")

        return {
            "score": max(0, min(100, score)),
            "issues": issues,
            "warnings": warnings,
            "total_entries": total_entries,
            "years_covered": len(years),
        }

    def check_biografico_quality(self, data: dict) -> dict[str, Any]:
        """
        Verifica qualità dati biografici.

        Args:
            data: Dict biografico

        Returns:
            Dict con report qualità
        """
        issues = []
        warnings = []
        score = 100

        if not isinstance(data, dict) or "artisti_2026_biografico" not in data:
            return {"score": 0, "issues": ["Invalid data structure"], "warnings": []}

        artisti = data["artisti_2026_biografico"]
        total = len(artisti)

        # Completeness
        missing_nome = sum(1 for a in artisti if not a.get("nome"))
        if missing_nome > 0:
            issues.append(f"{missing_nome} entries missing names")
            score -= 20

        missing_genere = sum(1 for a in artisti if not a.get("genere_musicale"))
        if missing_genere > 0:
            warnings.append(f"{missing_genere} artists missing genre")

        missing_anno_nascita = sum(1 for a in artisti if not a.get("anno_nascita"))
        if missing_anno_nascita > 0:
            warnings.append(f"{missing_anno_nascita} artists missing birth year")

        # Check data consistency
        suspicious_years = 0
        for artista in artisti:
            anno_nascita = artista.get("anno_nascita")
            if anno_nascita and (anno_nascita < 1940 or anno_nascita > 2010):
                suspicious_years += 1

        if suspicious_years > 0:
            warnings.append(f"{suspicious_years} artists with suspicious birth years")

        # Genre distribution
        generi = [a.get("genere_musicale") for a in artisti if a.get("genere_musicale")]
        if generi:
            genre_counts = Counter(generi)
            if len(genre_counts) < 3:
                warnings.append("Limited genre diversity")

        completeness = 100 - (missing_nome / total * 100 if total > 0 else 0)
        if completeness / 100 < self.config.min_biografico_coverage:
            warnings.append(
                f"Biografico coverage below target: {completeness:.1f}% "
                f"(min {self.config.min_biografico_coverage * 100:.0f}%)"
            )

        return {
            "score": max(0, min(100, score)),
            "issues": issues,
            "warnings": warnings,
            "total_entries": total,
            "completeness_score": completeness,
        }

    def check_caratteristiche_quality(self, data: dict) -> dict[str, Any]:
        """
        Verifica qualità dati caratteristiche.

        Args:
            data: Dict caratteristiche

        Returns:
            Dict con report qualità
        """
        issues = []
        warnings = []
        score = 100

        if not isinstance(data, dict) or "caratteristiche_artisti_2026" not in data:
            return {"score": 0, "issues": ["Invalid data structure"], "warnings": []}

        artisti = data["caratteristiche_artisti_2026"]
        total = len(artisti)

        # Completeness
        numeric_fields = [
            "viralita_social",
            "social_followers_total",
            "storia_bonus_ottenuti",
            "ad_personam_bonus_count",
            "ad_personam_bonus_points",
        ]
        missing_counts = Counter()

        for artista in artisti:
            for field in numeric_fields:
                if field not in artista or artista[field] is None:
                    missing_counts[field] += 1

        for field, count in missing_counts.items():
            if count > 0:
                pct = (count / total) * 100
                if pct > 50:
                    issues.append(f"{count} artists ({pct:.1f}%) missing {field}")
                    score -= 10
                else:
                    warnings.append(f"{count} artists ({pct:.1f}%) missing {field}")

        # Check value ranges
        out_of_range = 0
        for artista in artisti:
            value = artista.get("viralita_social")
            if value is not None and (value < 1 or value > 100):
                out_of_range += 1

            bonus = artista.get("storia_bonus_ottenuti")
            if bonus is not None and bonus < 0:
                out_of_range += 1

            count = artista.get("ad_personam_bonus_count")
            if count is not None and count < 0:
                out_of_range += 1

            points = artista.get("ad_personam_bonus_points")
            if points is not None and points < 0:
                out_of_range += 1

            followers = artista.get("social_followers_total")
            if followers is not None and followers < 0:
                out_of_range += 1

        if out_of_range > 0:
            issues.append(f"{out_of_range} values out of valid range")
            score -= 15

        # Coverage
        coverage = (total - missing_counts["viralita_social"]) / total * 100 if total > 0 else 0
        if coverage < 80:
            warnings.append(f"Low characteristics coverage: {coverage:.1f}%")
        if coverage / 100 < self.config.min_caratteristiche_coverage:
            warnings.append(
                f"Characteristics coverage below target: {coverage:.1f}% "
                f"(min {self.config.min_caratteristiche_coverage * 100:.0f}%)"
            )

        return {
            "score": max(0, min(100, score)),
            "issues": issues,
            "warnings": warnings,
            "total_entries": total,
            "coverage_pct": coverage,
        }

    def check_voti_stampa_quality(self, data: dict) -> dict[str, Any]:
        """
        Verifica qualità dati voti stampa.
        """
        issues: list[str] = []
        warnings: list[str] = []
        score = 100

        if not isinstance(data, dict) or "edizioni" not in data:
            return {"score": 0, "issues": ["Invalid voti_stampa structure"], "warnings": []}

        edizioni = data.get("edizioni", {})
        if not isinstance(edizioni, dict) or not edizioni:
            return {"score": 0, "issues": ["Empty voti_stampa.edizioni"], "warnings": []}

        total_votes = 0
        years_present: list[int] = []
        invalid_votes = 0

        for year, payload in edizioni.items():
            try:
                year_int = int(year)
                years_present.append(year_int)
            except (TypeError, ValueError):
                warnings.append(f"Invalid year key in voti_stampa: {year}")
                continue

            votes = payload.get("voti", []) if isinstance(payload, dict) else []
            if not isinstance(votes, list):
                issues.append(f"Invalid voti list for year {year}")
                score -= 10
                continue

            for vote in votes:
                total_votes += 1
                value = vote.get("voto") if isinstance(vote, dict) else None
                if not isinstance(value, (int, float)):
                    invalid_votes += 1
                    continue
                if value < 0 or value > 10:
                    invalid_votes += 1

        if invalid_votes > 0:
            issues.append(f"{invalid_votes} invalid vote values in voti_stampa")
            score -= 10

        if years_present:
            expected = set(range(min(years_present), max(years_present) + 1))
            missing = expected - set(years_present)
            if missing:
                warnings.append(f"Missing voti_stampa years: {sorted(missing)}")

        return {
            "score": max(0, min(100, score)),
            "issues": issues,
            "warnings": warnings,
            "years_covered": sorted(years_present),
            "total_votes": total_votes,
        }

    def _check_cross_file_matches(self, data_sources: dict[str, Any]) -> dict[str, Any]:
        """Check name match rates across data sources."""
        issues: list[str] = []
        warnings: list[str] = []

        artisti = data_sources.get("artisti_2026", {}).get("artisti", [])
        artist_keys = {normalize_artist_name(a.get("nome")) for a in artisti if a.get("nome")}
        if not artist_keys:
            return {
                "match_rates": {},
                "min_match_rate": None,
                "issues": ["No artist names found in artisti_2026"],
                "warnings": [],
            }

        def _keys(items: list[dict], name_field: str) -> set[str]:
            return {normalize_artist_name(i.get(name_field)) for i in items if i.get(name_field)}

        def _historical_expected_keys(items: list[dict]) -> set[str]:
            expected: set[str] = set()
            for item in items:
                if not isinstance(item, dict):
                    continue
                name = normalize_artist_name(item.get("nome"))
                if not name:
                    continue
                storico = item.get("storico_fantasanremo", [])
                has_history = isinstance(storico, list) and len(storico) > 0
                prima = item.get("prima_partecipazione")
                debuttante = item.get("debuttante_2026")
                if has_history:
                    expected.add(name)
                    continue
                if isinstance(prima, int) and prima < 2026:
                    expected.add(name)
                    continue
                if debuttante is False:
                    expected.add(name)
            return expected

        match_rates: dict[str, float] = {}

        biografico = data_sources.get("biografico", {}).get("artisti_2026_biografico", [])
        if biografico:
            biokeys = _keys(biografico, "nome")
            match_rates["biografico"] = len(artist_keys & biokeys) / len(artist_keys)

        caratteristiche = data_sources.get("caratteristiche", {}).get(
            "caratteristiche_artisti_2026", []
        )
        if caratteristiche:
            ckeys = _keys(caratteristiche, "nome")
            match_rates["caratteristiche"] = len(artist_keys & ckeys) / len(artist_keys)

        storico = data_sources.get("storico", [])
        if isinstance(storico, list) and storico:
            skeys = _keys(storico, "artista")
            match_rates["storico"] = len(artist_keys & skeys) / len(artist_keys)
        else:
            storico_unified = data_sources.get("storico_unified", {})
            if isinstance(storico_unified, dict) and storico_unified.get("artisti_storici"):
                skeys = {
                    normalize_artist_name(name)
                    for name in storico_unified.get("artisti_storici", {}).keys()
                }
                expected_hist_keys = _historical_expected_keys(artisti)
                if expected_hist_keys:
                    match_rates["storico_unified"] = len(expected_hist_keys & skeys) / len(
                        expected_hist_keys
                    )
                else:
                    match_rates["storico_unified"] = len(artist_keys & skeys) / len(artist_keys)

        min_match_rate = min(match_rates.values()) if match_rates else None
        if min_match_rate is not None and min_match_rate < self.config.min_crossfile_match_rate:
            warnings.append(
                f"Cross-file match rate below target: {min_match_rate * 100:.1f}% "
                f"(min {self.config.min_crossfile_match_rate * 100:.0f}%)"
            )

        return {
            "match_rates": match_rates,
            "min_match_rate": min_match_rate,
            "issues": issues,
            "warnings": warnings,
        }

    def calculate_quality_score(self, quality_report: dict) -> int:
        """
        Calcola score qualità globale.

        Args:
            quality_report: Report da check_quality()

        Returns:
            Int score 0-100
        """
        return quality_report.get("overall_score", 0)

    def is_quality_acceptable(self, quality_report: dict, threshold: int | None = None) -> bool:
        """
        Verifica se qualità è accettabile.

        Args:
            quality_report: Report qualità
            threshold: Soglia minima (default 70)

        Returns:
            True se accettabile
        """
        if threshold is None:
            threshold = self.config.min_quality_score
        score = self.calculate_quality_score(quality_report)
        return score >= threshold

    def get_quality_summary(self, quality_report: dict) -> str:
        """
        Genera summary leggibile del report qualità.

        Args:
            quality_report: Report qualità

        Returns:
            String summary
        """
        score = quality_report.get("overall_score", 0)
        issues = quality_report.get("issues", [])
        warnings = quality_report.get("warnings", [])

        status = (
            "✓ Excellent"
            if score >= 90
            else "✓ Good"
            if score >= 70
            else "⚠ Needs attention"
            if score >= 50
            else "✗ Poor"
        )

        lines = [f"Data Quality Score: {score}/100 ({status})", ""]

        if issues:
            lines.append(f"Issues ({len(issues)}):")
            for issue in issues[:5]:  # Limit to first 5
                lines.append(f"  - {issue}")
            if len(issues) > 5:
                lines.append(f"  ... and {len(issues) - 5} more")
            lines.append("")

        if warnings:
            lines.append(f"Warnings ({len(warnings)}):")
            for warning in warnings[:5]:
                lines.append(f"  - {warning}")
            if len(warnings) > 5:
                lines.append(f"  ... and {len(warnings) - 5} more")

        return "\n".join(lines)
