#!/usr/bin/env python3
"""
CSV Export Script for Fantasanremo Database

Exports all database tables to CSV files with joined artist names and ML metadata.
"""

import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

# Database path
DB_PATH = Path(__file__).resolve().parent.parent / "db" / "fantasanremo.db"

# ML metadata paths
ML_MODELS_DIR = Path(__file__).resolve().parent.parent / "backend" / "ml" / "models"
ENSEMBLE_META_PATH = ML_MODELS_DIR / "ensemble_meta.json"
PREDICTIONS_JSON_PATH = ML_MODELS_DIR / "predictions_2026.json"

# Feature descriptions mapping
FEATURE_DESCRIPTIONS = {
    # Historical features
    "voto_stampa": "Press vote score (0-100)",
    "participations": "Number of Sanremo participations",
    "avg_position": "Average finishing position",
    "position_variance": "Variance in finishing positions (consistency)",
    "top10_finishes": "Number of top 10 finishes",
    "top5_finishes": "Number of top 5 finishes",
    "years_since_last": "Years since last participation",
    # Biographical features
    "artist_age": "Artist age in 2026",
    "career_length": "Years since first participation",
    "is_debuttante_y": "Is debutante in 2026 (1=yes, 0=no)",
    "gen_z": "Born 1997-2012 (Gen Z cohort)",
    "millennial": "Born 1981-1996 (Millennial cohort)",
    "gen_x": "Born 1965-1980 (Gen X cohort)",
    # Social & characteristics
    "viral_potential": "Viral potential score (1-100)",
    "social_followers_total": "Total social media followers",
    "has_bonus_history": "Has received bonus points (1=yes, 0=no)",
    "bonus_count": "Count of bonus points received",
    # Genre features
    "genre_mainstream_pop": "Mainstream pop genre (1=yes, 0=no)",
    "genre_rap_urban": "Rap/urban genre (1=yes, 0=no)",
    # Archetype features
    "POP_MAINSTREAM": "Pop Mainstream archetype (1=yes, 0=no)",
    "DEBUTTANTE_POTENTIAL": "Debuttante Potential archetype (1=yes, 0=no)",
}

# Feature category mapping
FEATURE_CATEGORIES = {
    # Historical
    "voto_stampa": "Historical",
    "participations": "Historical",
    "avg_position": "Historical",
    "position_variance": "Historical",
    "top10_finishes": "Historical",
    "top5_finishes": "Historical",
    "years_since_last": "Historical",
    # Biographical
    "artist_age": "Biographical",
    "career_length": "Biographical",
    "is_debuttante_y": "Biographical",
    "gen_z": "Biographical",
    "millennial": "Biographical",
    "gen_x": "Biographical",
    # Social & characteristics
    "viral_potential": "Characteristics",
    "social_followers_total": "Characteristics",
    "has_bonus_history": "Characteristics",
    "bonus_count": "Characteristics",
    # Genre
    "genre_mainstream_pop": "Genre",
    "genre_rap_urban": "Genre",
    # Archetypes
    "POP_MAINSTREAM": "Archetype",
    "DEBUTTANTE_POTENTIAL": "Archetype",
}


def get_db_connection():
    """Get a connection to the SQLite database."""
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database not found at {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # Access columns by name
    return conn


def export_table_to_csv(conn, table_name: str, output_path: Path, join_artist: bool = False) -> int:
    """
    Export a database table to CSV.

    Args:
        conn: SQLite connection
        table_name: Name of the table to export
        output_path: Path to output CSV file
        join_artist: If True, join with artisti table to get artist name

    Returns:
        Number of rows exported
    """
    if join_artist:
        # Build JOIN query with artisti
        query = f"""
        SELECT t.*, a.nome as artista_nome
        FROM {table_name} t
        LEFT JOIN artisti a ON t.artista_id = a.id
        """
    else:
        query = f"SELECT * FROM {table_name}"

    df = pd.read_sql_query(query, conn)
    df.to_csv(output_path, index=False, encoding="utf-8")

    return len(df)


def export_aggregated_predictions(conn, output_path: Path) -> int:
    """
    Export artists with predictions in a single view.

    Args:
        conn: SQLite connection
        output_path: Path to output CSV file

    Returns:
        Number of rows exported
    """
    query = """
    SELECT
        a.id,
        a.nome,
        a.quotazione_2026,
        a.genere_musicale,
        a.anno_nascita,
        a.prima_partecipazione,
        a.debuttante_2026,
        p.punteggio_predetto,
        p.confidence,
        p.livello_performer
    FROM artisti a
    LEFT JOIN predizioni_2026 p ON a.id = p.artista_id
    ORDER BY a.id
    """

    df = pd.read_sql_query(query, conn)
    df.to_csv(output_path, index=False, encoding="utf-8")

    return len(df)


def export_ml_metadata(output_dir: Path) -> dict[str, int]:
    """
    Export ML metadata to CSV files.

    Args:
        output_dir: Directory to write CSV files

    Returns:
        Dict with file names and row counts
    """
    results = {}

    # Load ensemble metadata
    if ENSEMBLE_META_PATH.exists():
        with open(ENSEMBLE_META_PATH) as f:
            meta = json.load(f)

        # Export selected features
        features = meta.get("selected_features", [])
        if features:
            feature_data = []
            for feature in features:
                feature_data.append(
                    {
                        "feature_name": feature,
                        "description": FEATURE_DESCRIPTIONS.get(feature, ""),
                        "category": FEATURE_CATEGORIES.get(feature, "Unknown"),
                    }
                )

            df = pd.DataFrame(feature_data)
            features_path = output_dir / "ml_features.csv"
            df.to_csv(features_path, index=False, encoding="utf-8")
            results["ml_features.csv"] = len(features)

        # Export ensemble weights
        weights = meta.get("ensemble_weights", {})
        if weights:
            weight_data = [{"model": k, "weight": v} for k, v in weights.items()]
            df = pd.DataFrame(weight_data)
            weights_path = output_dir / "ml_ensemble_weights.csv"
            df.to_csv(weights_path, index=False, encoding="utf-8")
            results["ml_ensemble_weights.csv"] = len(weights)

        # Export year statistics
        year_stats = meta.get("year_stats", {})
        if year_stats:
            stat_data = [
                {"year": k, "mean": v.get("mean"), "std": v.get("std")}
                for k, v in year_stats.items()
            ]
            df = pd.DataFrame(stat_data)
            stats_path = output_dir / "ml_year_stats.csv"
            df.to_csv(stats_path, index=False, encoding="utf-8")
            results["ml_year_stats.csv"] = len(year_stats)

    return results


def verify_database(conn) -> dict[str, int]:
    """
    Verify database record counts.

    Args:
        conn: SQLite connection

    Returns:
        Dict with table names and record counts
    """
    cursor = conn.cursor()

    stats = {
        "artisti": cursor.execute("SELECT COUNT(*) FROM artisti").fetchone()[0],
        "edizioni_fantasanremo": cursor.execute(
            "SELECT COUNT(*) FROM edizioni_fantasanremo"
        ).fetchone()[0],
        "caratteristiche_artisti": cursor.execute(
            "SELECT COUNT(*) FROM caratteristiche_artisti"
        ).fetchone()[0],
        "predizioni_2026": cursor.execute("SELECT COUNT(*) FROM predizioni_2026").fetchone()[0],
    }
    return stats


def compare_with_source_files(db_stats: dict[str, int]) -> dict[str, dict]:
    """
    Compare database counts with JSON source files.

    Args:
        db_stats: Database statistics

    Returns:
        Dict with comparison results
    """
    results = {}

    # Compare predictions with JSON
    if PREDICTIONS_JSON_PATH.exists():
        with open(PREDICTIONS_JSON_PATH) as f:
            predictions_json = json.load(f)
        json_count = len(predictions_json)
        db_count = db_stats.get("predizioni_2026", 0)
        results["predictions"] = {
            "json": json_count,
            "database": db_count,
            "match": json_count == db_count,
        }

    # Compare ensemble metadata
    if ENSEMBLE_META_PATH.exists():
        with open(ENSEMBLE_META_PATH) as f:
            meta = json.load(f)
        feature_count = len(meta.get("selected_features", []))
        results["ml_features"] = {"json": feature_count}

    return results


def main():
    parser = argparse.ArgumentParser(description="Export Fantasanremo database to CSV files")
    parser.add_argument(
        "--table",
        action="append",
        choices=["artisti", "edizioni", "caratteristiche", "predictions", "aggregated", "ml"],
        help="Export specific table (can be specified multiple times)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="exports",
        help="Output directory for CSV files (default: exports/)",
    )
    parser.add_argument(
        "--verify-db",
        action="store_true",
        help="Verify database record counts and compare with source files",
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get database connection
    conn = get_db_connection()

    try:
        print("=" * 60)
        print("Fantasanremo CSV Export")
        print("=" * 60)
        print(f"Database: {DB_PATH}")

        # Verification mode
        if args.verify_db:
            print("\n[Database Verification]")
            db_stats = verify_database(conn)
            for table, count in db_stats.items():
                print(f"  {table}: {count} records")

            print("\n[Source File Comparison]")
            comparison = compare_with_source_files(db_stats)
            for name, data in comparison.items():
                if "match" in data:
                    status = "✓" if data["match"] else "✗"
                    print(f"  {name}: JSON={data['json']}, DB={data['database']} {status}")
                else:
                    print(f"  {name}: JSON={data['json']}")

            return

        # Determine which tables to export
        tables_to_export = args.table
        if not tables_to_export:
            tables_to_export = [
                "artisti",
                "edizioni",
                "caratteristiche",
                "predictions",
                "aggregated",
                "ml",
            ]

        export_counts = {}

        # Export artisti
        if "artisti" in tables_to_export:
            print("\n[Exporting artisti...]")
            path = output_dir / "artisti.csv"
            count = export_table_to_csv(conn, "artisti", path, join_artist=False)
            export_counts["artisti.csv"] = count
            print(f"  Exported {count} artists to {path}")

        # Export edizioni_fantasanremo
        if "edizioni" in tables_to_export:
            print("\n[Exporting edizioni_fantasanremo...]")
            path = output_dir / "edizioni_fantasanremo.csv"
            count = export_table_to_csv(conn, "edizioni_fantasanremo", path, join_artist=True)
            export_counts["edizioni_fantasanremo.csv"] = count
            print(f"  Exported {count} historical records to {path}")

        # Export caratteristiche_artisti
        if "caratteristiche" in tables_to_export:
            print("\n[Exporting caratteristiche_artisti...]")
            path = output_dir / "caratteristiche_artisti.csv"
            count = export_table_to_csv(conn, "caratteristiche_artisti", path, join_artist=True)
            export_counts["caratteristiche_artisti.csv"] = count
            print(f"  Exported {count} characteristic records to {path}")

        # Export predizioni_2026
        if "predictions" in tables_to_export:
            print("\n[Exporting predizioni_2026...]")
            path = output_dir / "predizioni_2026.csv"
            count = export_table_to_csv(conn, "predizioni_2026", path, join_artist=True)
            export_counts["predizioni_2026.csv"] = count
            print(f"  Exported {count} predictions to {path}")

        # Export aggregated predictions
        if "aggregated" in tables_to_export:
            print("\n[Exporting predizioni_aggregated...]")
            path = output_dir / "predizioni_aggregated.csv"
            count = export_aggregated_predictions(conn, path)
            export_counts["predizioni_aggregated.csv"] = count
            print(f"  Exported {count} aggregated records to {path}")

        # Export ML metadata
        if "ml" in tables_to_export:
            print("\n[Exporting ML metadata...]")
            ml_counts = export_ml_metadata(output_dir)
            export_counts.update(ml_counts)
            for name, count in ml_counts.items():
                print(f"  Exported {count} records to {name}")

        print("\n" + "=" * 60)
        print(f"Export complete! Files written to {output_dir}/")
        print("=" * 60)
        print("\nSummary:")
        for filename, count in export_counts.items():
            print(f"  {filename}: {count} rows")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
