"""
Feature Builder

Builds a unified, time-aware feature set with as_of_year support.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from backend.data_pipeline.config import get_config, get_logger

from .biographical_features import create_biographical_features
from .caratteristiche_features import create_caratteristiche_features
from .categorization import categorize_all_artists, get_archetype_features
from .data_preparation import MLDataPreparation
from .genre_features import create_genre_features
from .interaction_features import create_interaction_features
from .regulatory_features import create_regulatory_features


class FeatureBuilder:
    """Build unified, time-aware features for training and prediction."""

    def __init__(self, config=None):
        self.config = config or get_config()
        self.logger = get_logger("feature_builder")
        self.preparator = MLDataPreparation(self.config)

    def load_sources(self) -> dict[str, Any]:
        """Load all data sources using the shared preparator."""
        return self.preparator.load_all_sources()

    def build_sources_from_inputs(
        self,
        artisti_data: list[dict] | None = None,
        storico_data: list[dict] | dict | None = None,
        biografico_data: dict | None = None,
        caratteristiche_data: dict | None = None,
        regolamento_data: dict | None = None,
    ) -> dict[str, Any]:
        """Build a sources dict from in-memory inputs."""
        sources: dict[str, Any] = {}

        if artisti_data is not None:
            sources["artisti_2026"] = {"artisti": artisti_data}
        if storico_data is not None:
            sources["storico_unified"] = storico_data
        if biografico_data is not None:
            sources["biografico"] = biografico_data
        if caratteristiche_data is not None:
            sources["caratteristiche"] = caratteristiche_data
        if regolamento_data is not None:
            sources["regolamento"] = regolamento_data

        return sources

    def build_historical_dataset(self, sources: dict[str, Any]) -> pd.DataFrame:
        """Build historical artist-year base dataset with targets."""
        base_df = self.preparator._build_historical_dataframe(sources)
        if base_df.empty:
            return base_df

        # Calculate scores from unified storico (with calibration from sources)
        scores_df = self.preparator.calculate_real_scores(
            sources.get("storico_unified", {}), sources=sources
        )

        if not scores_df.empty:
            base_df = base_df.merge(
                scores_df[["artista_nome", "anno", "punteggio_reale", "punteggio_source"]],
                on=["artista_nome", "anno"],
                how="left",
            )
        else:
            base_df["punteggio_reale"] = np.nan
            base_df["punteggio_source"] = "missing"

        return base_df

    def build_training_frame(self, sources: dict[str, Any]) -> pd.DataFrame:
        """Build training dataset with time-aware features and targets."""
        df = self.build_historical_dataset(sources)
        if df.empty:
            return df

        # Add time-aware history features
        df = self.preparator._add_time_aware_features(df)

        # Add per-year static features (bio/genre/regulatory/etc.)
        df = self._attach_static_features_by_year(df, sources)

        # Fill missing values
        df = self._fill_missing_features(df)

        return df

    def build_prediction_frame(self, sources: dict[str, Any], as_of_year: int) -> pd.DataFrame:
        """Build feature frame for a given prediction year."""
        artisti = sources.get("artisti_2026", {}).get("artisti", [])
        if not artisti:
            return pd.DataFrame()

        # Build history map from historical data
        historical_df = self.preparator._build_historical_dataframe(sources)
        history_map: dict[int, tuple[list[int], list[int]]] = {}
        if historical_df is not None and not historical_df.empty:
            hist_filtered = historical_df[historical_df["anno"] < as_of_year].sort_values(
                ["artista_id", "anno"]
            )
            for artista_id, group in hist_filtered.groupby("artista_id", sort=False):
                if pd.isna(artista_id):
                    continue
                positions = [int(p) for p in group["posizione"].tolist() if pd.notna(p)]
                years = [int(y) for y in group["anno"].tolist()]
                history_map[int(artista_id)] = (positions, years)

        records: list[dict[str, Any]] = []
        for artista in artisti:
            artista_id = artista.get("id")
            if artista_id is None:
                continue
            positions, years = history_map.get(int(artista_id), ([], []))
            features = self.preparator._compute_history_features(positions, years, as_of_year)

            record = {
                "artista_id": artista_id,
                "artista_nome": artista.get("nome"),
                "quotazione_2026": artista.get("quotazione"),
                "anno": as_of_year,
                "voto_stampa": self.preparator._get_voto_stampa(
                    sources, artista.get("nome"), as_of_year
                ),
            }
            record.update(features)
            records.append(record)

        df = pd.DataFrame(records)

        # Attach static features for the prediction year
        df = self._attach_static_features(df, sources, as_of_year)

        # Fill missing values
        df = self._fill_missing_features(df)

        return df

    def get_feature_columns(self, df: pd.DataFrame) -> list[str]:
        """Return feature columns excluding identifiers/targets/metadata."""
        if df.empty:
            return []

        non_feature_cols = {
            "artista_id",
            "artista_nome",
            "anno",
            "punteggio_reale",
            "punteggio_source",
            "posizione",
            "debuttante",
            "partecipazioni",
            "quotazione_2026",
            "genre",
            "primary_archetype",
        }

        return [c for c in df.columns if c not in non_feature_cols]

    def split_by_years(
        self, df: pd.DataFrame, training_years: list[int], validation_years: list[int]
    ) -> dict[str, pd.DataFrame]:
        """Split dataset into train/validation by year."""
        if df.empty:
            return {
                "train": pd.DataFrame(),
                "val": {},
            }

        train_df = df[df["anno"].isin(training_years)].copy()
        train_df = train_df[train_df["punteggio_reale"].notna()]

        val_frames = {}
        for year in validation_years:
            val_df = df[df["anno"] == year].copy()
            val_df = val_df[val_df["punteggio_reale"].notna()]
            val_frames[year] = val_df

        return {
            "train": train_df,
            "val": val_frames,
        }

    def _attach_static_features_by_year(
        self, df: pd.DataFrame, sources: dict[str, Any]
    ) -> pd.DataFrame:
        """Attach static features for each year in the dataset."""
        if df.empty:
            return df

        frames = []
        for year in sorted(df["anno"].unique()):
            subset = df[df["anno"] == year].copy()
            subset = self._attach_static_features(subset, sources, int(year))
            frames.append(subset)

        return pd.concat(frames, ignore_index=True) if frames else df

    def _attach_static_features(
        self, df: pd.DataFrame, sources: dict[str, Any], as_of_year: int
    ) -> pd.DataFrame:
        """Attach per-artist static features for a specific year."""
        if df.empty:
            return df

        artisti_data = sources.get("artisti_2026", {}).get("artisti", [])
        biografico_data = sources.get("biografico")
        caratteristiche_data = sources.get("caratteristiche")
        regolamento_data = sources.get("regolamento")
        storico_stats = self._build_storico_posizioni_list(sources)

        # Ensure all features are keyed by artista_id and avoid duplicate name columns
        feature_frames = []

        if biografico_data:
            bio_df = create_biographical_features(
                artisti_data, biografico_data, storico_stats, as_of_year=as_of_year
            )
            bio_df = bio_df.drop(columns=["artista_nome"], errors="ignore")
            feature_frames.append(bio_df)

        if caratteristiche_data:
            car_df = create_caratteristiche_features(artisti_data, caratteristiche_data)
            car_df = car_df.drop(columns=["artista_nome"], errors="ignore")
            feature_frames.append(car_df)

        if biografico_data:
            genre_df = create_genre_features(
                artisti_data,
                biografico_data,
                storico_stats,
                as_of_year=as_of_year,
            )
            genre_df = genre_df.drop(columns=["artista_nome"], errors="ignore")
            feature_frames.append(genre_df)

        reg_df = create_regulatory_features(
            artisti_data,
            biografico_data,
            caratteristiche_data,
            regolamento_data,
            as_of_year=as_of_year,
        )
        reg_df = reg_df.drop(columns=["artista_nome"], errors="ignore")
        feature_frames.append(reg_df)

        cat_df = categorize_all_artists(
            artisti_data,
            biografico_data,
            caratteristiche_data,
            storico_stats,
            as_of_year=as_of_year,
        )
        arch_df = get_archetype_features(cat_df)
        feature_frames.append(arch_df)

        # Merge all static features
        features_df = None
        for frame in feature_frames:
            if frame is None or frame.empty:
                continue
            if features_df is None:
                features_df = frame
            else:
                features_df = features_df.merge(frame, on="artista_id", how="left")

        if features_df is not None and not features_df.empty:
            df = df.merge(features_df, on="artista_id", how="left")

        # Create interaction features
        df = create_interaction_features(df)

        return df

    def _build_storico_posizioni_list(self, sources: dict[str, Any]) -> list[dict] | None:
        """Build a list of {artista, posizioni} from unified storico sources."""
        artisti_storici = sources.get("storico_unified", {}).get("artisti_storici", {})
        if not artisti_storici:
            return None

        out = []
        for artista_nome, data in artisti_storici.items():
            posizioni = {}
            for ed in data.get("edizioni", []):
                anno = ed.get("anno")
                posizione = ed.get("posizione")
                if anno and posizione:
                    posizioni[str(anno)] = posizione
            out.append({"artista": artista_nome, "posizioni": posizioni})
        return out

    def _fill_missing_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing values in features DataFrame."""
        df = features_df.copy()

        # Binary columns - fill with 0
        binary_cols = [c for c in df.columns if c.startswith(("is_", "has_", "genre_", "gen_"))]
        for col in binary_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0).astype(int)

        # Numeric columns - fill with median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col not in binary_cols and col not in ["artista_id"]:
                median_val = df[col].median()
                if pd.isna(median_val):
                    median_val = 0
                df[col] = df[col].fillna(median_val)

        return df
