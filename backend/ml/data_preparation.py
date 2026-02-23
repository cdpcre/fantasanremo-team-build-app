"""
ML Data Preparation Module

Prepara i dati per il ML con split temporale per prevenire data leakage.
Training: 2020-2022, 2024 | Validation 1: 2023 | Validation 2: 2025
"""

import sys
from pathlib import Path

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import json
from typing import Any

import numpy as np
import pandas as pd

from backend.data_pipeline.config import get_config, get_logger
from backend.utils.enriched_artists import (
    build_biografico_from_enriched,
    build_caratteristiche_from_enriched,
)
from backend.utils.name_normalization import normalize_artist_name


class MLDataPreparation:
    """
    Prepara i dati per il ML con split temporale.

    Strategy:
    - Training set: 2020-2022, 2024 (anni non in validazione)
    - Validation set 1: 2023 (check metà periodo)
    - Validation set 2: 2025 (check performance recente)
    """

    def __init__(self, config=None):
        """
        Inizializza il preparatore dati.

        Args:
            config: Configurazione pipeline (usa default se None)
        """
        self.config = config or get_config()
        self.logger = get_logger(f"{self.__class__.__name__}")

        # Year splits from config
        self.training_years = self.config.training_years
        self.validation_years = self.config.validation_years
        self.prediction_year = self.config.prediction_year

        self.logger.info(f"Training years: {self.training_years}")
        self.logger.info(f"Validation years: {self.validation_years}")

    def load_all_sources(self) -> dict[str, Any]:
        """
        Carica tutte le sorgenti dati JSON.

        Returns:
            Dict con tutti i dati caricati
        """
        config = get_config()
        data_dir = config.data_dir

        sources = {}

        # Load enriched artisti 2026 (single source)
        artisti_path = config.artisti_2026_path
        if artisti_path.exists():
            with open(artisti_path) as f:
                sources["artisti_2026"] = json.load(f)
            self.logger.info(
                f"Loaded artisti_2026: {len(sources['artisti_2026'].get('artisti', []))} artists"
            )

            sources["biografico"] = build_biografico_from_enriched(sources["artisti_2026"])
            sources["caratteristiche"] = build_caratteristiche_from_enriched(
                sources["artisti_2026"]
            )

        # Load storico unified (required)
        storico_unified_path = data_dir / "storico_fantasanremo_unified.json"
        if storico_unified_path.exists():
            with open(storico_unified_path) as f:
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

        # Load regolamento 2026
        regolamento_path = data_dir / "regolamento_2026.json"
        if regolamento_path.exists():
            with open(regolamento_path) as f:
                sources["regolamento"] = json.load(f)
            self.logger.info("Loaded regolamento_2026")

        # Load final classifications (for real scores)
        classifiche_path = data_dir / "classifiche_finali.json"
        if classifiche_path.exists():
            with open(classifiche_path) as f:
                sources["classifiche"] = json.load(f)
            self.logger.info("Loaded classifiche_finali")

        # Load voti stampa (new source)
        voti_stampa_path = data_dir / "voti_stampa.json"
        if voti_stampa_path.exists():
            with open(voti_stampa_path) as f:
                sources["voti_stampa"] = json.load(f)
            self.logger.info("Loaded voti_stampa")

        return sources

    def calculate_real_scores(
        self, storico_completo: dict, sources: dict | None = None
    ) -> pd.DataFrame:
        """
        Calcola i punteggi reali dalle classifiche finali (formato unified).

        Uses real punteggio_finale from artisti_storici where available,
        and calibrated estimates from position for all other artists.

        Args:
            storico_completo: Dati storici completi in formato unified
            sources: Full sources dict (for calibrated score estimation)

        Returns:
            DataFrame con colonne: artista_nome, anno, punteggio_reale, punteggio_source
        """
        scores = []
        seen_keys: set[tuple[str, int]] = set()

        # 1. Real scores from artisti_storici (highest quality)
        artisti_storici = storico_completo.get("artisti_storici", {})
        for artista_nome, data in artisti_storici.items():
            for edizione in data.get("edizioni", []):
                anno = edizione.get("anno")
                punteggio = edizione.get("punteggio_finale")
                posizione = edizione.get("posizione")

                source = "real"
                if punteggio is None and posizione:
                    punteggio = self._estimate_score_from_position(posizione, anno, sources)
                    source = "estimated_from_storico_position"

                if anno and punteggio is not None:
                    scores.append(
                        {
                            "artista_nome": artista_nome,
                            "anno": anno,
                            "punteggio_reale": punteggio,
                            "posizione": posizione,
                            "punteggio_source": source,
                        }
                    )
                    seen_keys.add((normalize_artist_name(artista_nome), anno))

        # 2. Estimated scores from classifiche_finali (expanded data)
        if sources:
            classifiche = sources.get("classifiche", {})
            edizioni = classifiche.get("edizioni", {})
            for year_str, year_data in edizioni.items():
                try:
                    anno = int(year_str)
                except ValueError:
                    continue

                for entry in year_data.get("classifica_completa", []):
                    artista_nome = entry.get("artista")
                    posizione = entry.get("posizione")
                    if not artista_nome or posizione is None:
                        continue

                    key = (normalize_artist_name(artista_nome), anno)
                    if key in seen_keys:
                        continue

                    punteggio = self._estimate_score_from_position(posizione, anno, sources)
                    scores.append(
                        {
                            "artista_nome": artista_nome,
                            "anno": anno,
                            "punteggio_reale": punteggio,
                            "posizione": posizione,
                            "punteggio_source": "estimated_from_classifiche",
                        }
                    )
                    seen_keys.add(key)

        df = pd.DataFrame(scores)
        if not df.empty:
            self.logger.info(f"Calculated {len(df)} real scores from historical data")

        return df

    def _estimate_score_from_position(
        self, posizione: int, anno: int | None = None, sources: dict | None = None
    ) -> float:
        """
        Stima punteggio FantaSanremo dalla posizione usando calibrazione reale.

        Usa i record con sia posizione che punteggio_finale per fittare una
        regressione lineare posizione→punteggio, con fattore di scala per anno
        basato sui punteggi vincitori in festival_edizioni.
        """
        calibration = self._get_score_calibration(sources)
        base_intercept = calibration["intercept"]
        base_slope = calibration["slope"]

        # Base prediction from calibration
        estimated = base_intercept + base_slope * posizione

        # Scale by year if info available
        if anno and sources:
            edizioni = sources.get("storico_unified", {}).get("festival_edizioni", {})
            year_data = edizioni.get(str(anno), {})
            winner_score = year_data.get("punteggio")
            if winner_score:
                # Scale relative to average winner score in calibration data
                avg_winner = calibration.get("avg_winner_score", 400)
                if avg_winner > 0:
                    scale = winner_score / avg_winner
                    estimated *= scale

        return max(50, estimated)

    def _get_score_calibration(self, sources: dict | None = None) -> dict:
        """
        Fit a linear regression from known (posizione, punteggio_finale) pairs.

        Returns dict with 'slope', 'intercept', 'avg_winner_score'.
        """
        if not sources:
            return {"slope": -10.0, "intercept": 400.0, "avg_winner_score": 400.0}

        artisti_storici = sources.get("storico_unified", {}).get("artisti_storici", {})
        positions = []
        scores = []
        winner_scores = []

        for data in artisti_storici.values():
            for ed in data.get("edizioni", []):
                pos = ed.get("posizione")
                punteggio = ed.get("punteggio_finale")
                if pos is not None and punteggio is not None:
                    positions.append(pos)
                    scores.append(punteggio)
                    if pos == 1:
                        winner_scores.append(punteggio)

        if len(positions) < 3:
            return {"slope": -10.0, "intercept": 400.0, "avg_winner_score": 400.0}

        # Simple linear regression: score = intercept + slope * position
        positions_arr = np.array(positions, dtype=float)
        scores_arr = np.array(scores, dtype=float)
        slope, intercept = np.polyfit(positions_arr, scores_arr, 1)

        # Average winner score from festival_edizioni
        edizioni = sources.get("storico_unified", {}).get("festival_edizioni", {})
        all_winner_scores = [e.get("punteggio") for e in edizioni.values() if e.get("punteggio")]
        avg_winner = float(np.mean(all_winner_scores)) if all_winner_scores else 400.0

        return {
            "slope": float(slope),
            "intercept": float(intercept),
            "avg_winner_score": avg_winner,
        }

    def create_training_dataset(
        self, sources: dict[str, Any]
    ) -> tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
    ]:
        """
        Crea dataset di training con split temporale.

        Args:
            sources: Dict con tutte le sorgenti dati

        Returns:
            Tuple of (X_train, y_train, X_val_2023, y_val_2023, X_val_2025, y_val_2025, X_2026)
        """
        # Build historical dataframe (artist-year rows)
        historical_df = self._build_historical_dataframe(sources)

        if historical_df.empty:
            self.logger.warning("No historical data available")
            return self._create_empty_datasets()

        # Calculate scores - use unified format if available, otherwise legacy
        if "storico_unified" in sources:
            scores_df = self.calculate_real_scores(
                sources.get("storico_unified", {}), sources=sources
            )
        else:
            self.logger.warning("No storico_unified data available for scores calculation")
            scores_df = pd.DataFrame()

        # Merge scores
        if not scores_df.empty:
            historical_df = historical_df.merge(
                scores_df[["artista_nome", "anno", "punteggio_reale", "punteggio_source"]],
                on=["artista_nome", "anno"],
                how="left",
            )

        # Add time-aware historical features (no future leakage)
        historical_df = self._add_time_aware_features(historical_df)

        # Create splits by year (drop rows without target)
        train_df = historical_df[historical_df["anno"].isin(self.training_years)].copy()
        val_2023_df = historical_df[historical_df["anno"] == 2023].copy()
        val_2025_df = historical_df[historical_df["anno"] == 2025].copy()

        train_df = train_df[train_df["punteggio_reale"].notna()]
        val_2023_df = val_2023_df[val_2023_df["punteggio_reale"].notna()]
        val_2025_df = val_2025_df[val_2025_df["punteggio_reale"].notna()]

        self.logger.info(f"Training samples: {len(train_df)}")
        self.logger.info(f"Validation 2023 samples: {len(val_2023_df)}")
        self.logger.info(f"Validation 2025 samples: {len(val_2025_df)}")

        # Generate features for each split
        feature_columns = self._get_feature_columns()

        X_train = train_df[feature_columns] if not train_df.empty else pd.DataFrame()
        y_train = train_df["punteggio_reale"] if not train_df.empty else pd.Series()

        X_val_2023 = val_2023_df[feature_columns] if not val_2023_df.empty else pd.DataFrame()
        y_val_2023 = val_2023_df["punteggio_reale"] if not val_2023_df.empty else pd.Series()

        X_val_2025 = val_2025_df[feature_columns] if not val_2025_df.empty else pd.DataFrame()
        y_val_2025 = val_2025_df["punteggio_reale"] if not val_2025_df.empty else pd.Series()

        # Create 2026 dataset (features based on history only)
        X_2026 = self._create_2026_dataset(sources, historical_df)

        return X_train, y_train, X_val_2023, y_val_2023, X_val_2025, y_val_2025, X_2026

    def _build_historical_dataframe(self, sources: dict[str, Any]) -> pd.DataFrame:
        """
        Costruisce DataFrame storico da tutte le sorgenti.

        Includes:
        1. Artists from 2026 roster with their historical positions
        2. ALL artists from classifiche_finali.json (expanded training data)

        Args:
            sources: Dict con sorgenti dati

        Returns:
            DataFrame con colonne base per ML
        """
        records = []
        seen_keys: set[tuple[str, int]] = set()  # (normalized_name, year) to avoid duplicates

        # Map artist name -> id from 2026 roster for stable joins
        artisti_list = sources.get("artisti_2026", {}).get("artisti", [])
        artist_id_map = {
            normalize_artist_name(a.get("nome")): a.get("id") for a in artisti_list if a.get("nome")
        }

        # 1. Load 2026 roster artists with their storico_posizioni
        if "storico_unified" in sources:
            storico_dict = sources.get("storico_unified", {})
            for entry in storico_dict.get("artisti_2026", []):
                artista_nome = entry.get("nome")
                if not artista_nome:
                    continue

                artista_id = artist_id_map.get(normalize_artist_name(artista_nome))
                if artista_id is None:
                    self.logger.warning(f"Missing artista_id for {artista_nome}")
                    continue

                debuttante = entry.get("debuttante", False)
                partecipazioni = entry.get("partecipazioni", 0)
                storico_posizioni = entry.get("storico_posizioni", {})

                for year_key in ["2020", "2021", "2022", "2023", "2024", "2025"]:
                    pos = storico_posizioni.get(year_key)
                    if pos is not None:
                        voto_stampa = self._get_voto_stampa(sources, artista_nome, int(year_key))
                        records.append(
                            {
                                "artista_id": artista_id,
                                "artista_nome": artista_nome,
                                "anno": int(year_key),
                                "posizione": pos,
                                "debuttante": debuttante,
                                "partecipazioni": partecipazioni,
                                "voto_stampa": voto_stampa,
                            }
                        )
                        seen_keys.add((normalize_artist_name(artista_nome), int(year_key)))

        # 2. Expand with ALL artists from classifiche_finali.json
        classifiche = sources.get("classifiche", {})
        edizioni = classifiche.get("edizioni", {})
        # Generate synthetic IDs for non-2026 artists (start from a high number)
        next_synth_id = 10000

        for year_str, year_data in edizioni.items():
            try:
                anno = int(year_str)
            except ValueError:
                continue

            classifica = year_data.get("classifica_completa", [])
            for entry in classifica:
                artista_nome = entry.get("artista")
                posizione = entry.get("posizione")
                if not artista_nome or posizione is None:
                    continue

                norm_name = normalize_artist_name(artista_nome)
                key = (norm_name, anno)
                if key in seen_keys:
                    continue  # Already added from roster data

                # Get or create artista_id
                artista_id = artist_id_map.get(norm_name)
                if artista_id is None:
                    # Synthetic ID for non-2026 artists
                    artista_id = next_synth_id
                    artist_id_map[norm_name] = artista_id
                    next_synth_id += 1

                # Get voti stampa
                voto_stampa = self._get_voto_stampa(sources, artista_nome, anno)

                # Estimate punteggio from position using calibration
                records.append(
                    {
                        "artista_id": artista_id,
                        "artista_nome": artista_nome,
                        "anno": anno,
                        "posizione": posizione,
                        "debuttante": False,  # Unknown, default
                        "partecipazioni": 0,  # Unknown for non-roster
                        "voto_stampa": voto_stampa,
                    }
                )
                seen_keys.add(key)

        df = pd.DataFrame(records)
        if not df.empty:
            self.logger.info(
                f"Built historical dataframe: {len(df)} records, "
                f"{df['artista_id'].nunique()} unique artists"
            )

        return df

    def _add_time_aware_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggiunge feature storiche time-aware (solo anni precedenti).

        Args:
            df: DataFrame base con colonne artista_id, anno, posizione

        Returns:
            DataFrame con feature aggiuntive
        """
        if df.empty:
            return df

        df = df.sort_values(["artista_id", "anno"]).copy()
        feature_rows = []

        for artista_id, group in df.groupby("artista_id", sort=False):
            history_positions: list[int] = []
            history_years: list[int] = []

            for _, row in group.iterrows():
                current_year = int(row["anno"])
                features = self._compute_history_features(
                    history_positions, history_years, current_year
                )
                feature_rows.append(
                    {
                        "artista_id": artista_id,
                        "anno": current_year,
                        **features,
                    }
                )

                # Update history with current year after computing features
                if pd.notna(row["posizione"]):
                    history_positions.append(int(row["posizione"]))
                    history_years.append(current_year)

        features_df = pd.DataFrame(feature_rows)
        df = df.merge(features_df, on=["artista_id", "anno"], how="left")

        return df

    def _compute_history_features(
        self, history_positions: list[int], history_years: list[int], current_year: int
    ) -> dict[str, Any]:
        """Compute features using only past participations."""
        if not history_positions:
            return {
                "participations": 0,
                "avg_position": 0.0,
                "position_variance": 0.0,
                "position_trend": 0.0,
                "best_position": 0.0,
                "recent_avg": 0.0,
                "consistency_score": 0.0,
                "momentum_score": 0.0,
                "peak_performance": 0.0,
                "longevity_bonus": 0.0,
                "top10_finishes": 0,
                "top5_finishes": 0,
                "median_position": 0.0,
                "volatility_index": 0.0,
                "years_since_last": 0,
                "is_debuttante": 1,
                "is_recent": int(current_year >= 2024),
            }

        inverted_positions = [31 - p for p in history_positions]
        participations = len(inverted_positions)
        avg_position = float(np.mean(inverted_positions))
        position_variance = float(np.var(inverted_positions)) if participations > 1 else 0.0
        best_position = float(np.max(inverted_positions))

        # Recent average (last two participations)
        recent_vals = inverted_positions[-2:] if participations >= 2 else inverted_positions
        recent_avg = float(np.mean(recent_vals))
        position_trend = float(recent_avg - avg_position)

        # Consistency score: 1 - coefficient of variation
        std_dev = float(np.std(inverted_positions)) if participations > 1 else 0.0
        if avg_position > 0 and std_dev > 0:
            consistency_score = max(0.0, 1.0 - (std_dev / avg_position))
        else:
            consistency_score = 0.5 if participations == 1 else 0.0

        # Momentum score: exponential decay average (recent years weighted more)
        weights = np.exp(-np.arange(participations) * 0.3)
        weights = weights / weights.sum()
        momentum_score = float(np.sum(np.array(inverted_positions) * weights))

        peak_performance = float(np.max(inverted_positions))
        longevity_bonus = float(participations * 5)
        top10_finishes = int(np.sum(np.array(inverted_positions) >= 21))
        top5_finishes = int(np.sum(np.array(inverted_positions) >= 26))
        median_position = float(np.median(inverted_positions))
        volatility_index = float(std_dev)

        years_since_last = int(current_year - history_years[-1]) if history_years else 0

        return {
            "participations": participations,
            "avg_position": avg_position,
            "position_variance": position_variance,
            "position_trend": position_trend,
            "best_position": best_position,
            "recent_avg": recent_avg,
            "consistency_score": float(consistency_score),
            "momentum_score": momentum_score,
            "peak_performance": peak_performance,
            "longevity_bonus": longevity_bonus,
            "top10_finishes": top10_finishes,
            "top5_finishes": top5_finishes,
            "median_position": median_position,
            "volatility_index": volatility_index,
            "years_since_last": years_since_last,
            "is_debuttante": 0,
            "is_recent": int(current_year >= 2024),
        }

    def _create_2026_dataset(
        self, sources: dict[str, Any], historical_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Crea dataset 2026 per predizioni.

        Args:
            sources: Dict con sorgenti dati

        Returns:
            DataFrame artisti 2026 con feature
        """
        artisti = sources.get("artisti_2026", {}).get("artisti", [])
        bio_map = {
            normalize_artist_name(b.get("nome")): b
            for b in sources.get("biografico", {}).get("artisti_2026_biografico", [])
            if b.get("nome")
        }

        # Build history map from historical_df
        history_map: dict[int, tuple[list[int], list[int]]] = {}
        if historical_df is not None and not historical_df.empty:
            hist_sorted = historical_df.sort_values(["artista_id", "anno"])
            for artista_id, group in hist_sorted.groupby("artista_id", sort=False):
                positions = [int(p) for p in group["posizione"].tolist() if pd.notna(p)]
                years = [int(y) for y in group["anno"].tolist()]
                if pd.notna(artista_id):
                    history_map[int(artista_id)] = (positions, years)

        records = []
        for artista in artisti:
            record = {
                "artista_id": artista.get("id"),
                "artista_nome": artista.get("nome"),
                "quotazione_2026": artista.get("quotazione"),
                "anno": self.prediction_year,
            }

            artista_id = record["artista_id"]
            if artista_id:
                positions, years = history_map.get(int(artista_id), ([], []))
            else:
                positions, years = ([], [])
            record.update(self._compute_history_features(positions, years, self.prediction_year))

            # Add voti stampa (2026 or imputed)
            record["voto_stampa"] = self._get_voto_stampa(
                sources, artista.get("nome"), self.prediction_year
            )

            # Add biographical info
            bio = bio_map.get(normalize_artist_name(artista.get("nome")))
            if bio:
                record["genere_musicale"] = bio.get("genere_musicale")
                record["anno_nascita"] = bio.get("anno_nascita")
                record["prima_partecipazione"] = bio.get("prima_partecipazione")

            records.append(record)

        df = pd.DataFrame(records)

        return df

    def _get_voto_stampa(
        self, sources: dict[str, Any], artist_name: str, target_year: int
    ) -> float | None:
        """
        Recupera il voto della sala stampa per un artista e un anno specifico.
        Se manca per anno target, imputa con media anni passati (STRICT CAUSAL).

        Args:
            sources: Dict con sorgenti dati
            artist_name: Nome artista
            target_year: Anno target

        Returns:
            Voto (float) o None
        """
        voti_data = sources.get("voti_stampa", {}).get("edizioni", {})
        norm_name = normalize_artist_name(artist_name)

        # 1. Try direct match
        target_year_str = str(target_year)
        if target_year_str in voti_data:
            for entry in voti_data[target_year_str].get("voti", []):
                if normalize_artist_name(entry.get("artista")) == norm_name:
                    return float(entry.get("voto"))

        # 2. Imputation (Strict Causal: only years < target_year)
        past_votes = []
        for year_str, data in voti_data.items():
            try:
                year = int(year_str)
            except ValueError:
                continue

            if year < target_year:
                for entry in data.get("voti", []):
                    if normalize_artist_name(entry.get("artista")) == norm_name:
                        past_votes.append(float(entry.get("voto")))
                        break  # Found for this year

        if past_votes:
            return float(np.mean(past_votes))

        return None

    def _get_feature_columns(self) -> list[str]:
        """Restituisce colonne feature per ML."""
        return [
            "participations",
            "avg_position",
            "position_variance",
            "position_trend",
            "best_position",
            "recent_avg",
            "consistency_score",
            "momentum_score",
            "peak_performance",
            "longevity_bonus",
            "top10_finishes",
            "top5_finishes",
            "median_position",
            "volatility_index",
            "years_since_last",
            "is_debuttante",
            "is_recent",
            "voto_stampa",
        ]

    def _create_empty_datasets(self) -> tuple:
        """Crea dataset vuoti se nessun dato disponibile."""
        empty_df = pd.DataFrame()
        empty_series = pd.Series()
        return (empty_df, empty_series, empty_df, empty_series, empty_df, empty_series, empty_df)

    def get_split_summary(self) -> dict[str, Any]:
        """
        Restituisce summary degli split temporali.

        Returns:
            Dict con info sugli split
        """
        return {
            "training_years": self.training_years,
            "validation_years": self.validation_years,
            "prediction_year": self.prediction_year,
            "strategy": "Temporal split to prevent data leakage",
            "training_description": "2020-2022, 2024: Years not in validation",
            "validation_2023": "Mid-period check",
            "validation_2025": "Recent performance check",
        }
