#!/usr/bin/env python3
"""
Migration Script: Unificazione File JSON Storico Fantasanremo

Questo script unifica storico_fantasanremo.json e storico_fantasanremo_completo.json
in un unico file storico_fantasanremo_unified.json eliminando ridondanze.

Usage:
    uv run python scripts/migrate_unified_storico.py [--dry-run] [--no-backup]

Rollback:
    cp data/backups/storico_fantasanremo.json.*.bak data/storico_fantasanremo.json
    cp data/backups/storico_fantasanremo_completo.json.*.bak data/storico_fantasanremo_completo.json
"""

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any


def setup_paths() -> dict[str, Path]:
    """Setup all required paths."""
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"
    backup_dir = data_dir / "backups"

    return {
        "project_root": project_root,
        "data_dir": data_dir,
        "backup_dir": backup_dir,
        "storico_path": data_dir / "storico_fantasanremo.json",
        "storico_completo_path": data_dir / "storico_fantasanremo_completo.json",
        "unified_path": data_dir / "storico_fantasanremo_unified.json",
    }


def create_backup(paths: dict[str, Path], timestamp: str) -> dict[str, Path]:
    """Create backups of original files."""
    print("📦 Creating backups...")

    backup_files = {}

    for file_key in ["storico_path", "storico_completo_path"]:
        source = paths[file_key]
        if source.exists():
            backup_name = f"{source.name}.{timestamp}.bak"
            backup_path = paths["backup_dir"] / backup_name
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, backup_path)
            backup_files[file_key] = backup_path
            print(f"   ✅ Backed up: {backup_path}")
        else:
            print(f"   ⚠️  File not found: {source}")

    return backup_files


def load_json_data(paths: dict[str, Path]) -> dict[str, Any]:
    """Load JSON data from both files."""
    print("📖 Loading JSON files...")

    data = {}

    # Load storico_fantasanremo.json
    if paths["storico_path"].exists():
        with open(paths["storico_path"]) as f:
            data["storico"] = json.load(f)
        print("   ✅ Loaded storico_fantasanremo.json")
    else:
        raise FileNotFoundError("storico_fantasanremo.json not found")

    # Load storico_fantasanremo_completo.json
    if paths["storico_completo_path"].exists():
        with open(paths["storico_completo_path"]) as f:
            data["storico_completo"] = json.load(f)
        print("   ✅ Loaded storico_fantasanremo_completo.json")
    else:
        raise FileNotFoundError("storico_fantasanremo_completo.json not found")

    return data


def create_unified_structure(data: dict[str, Any]) -> dict[str, Any]:
    """Create the unified structure from both data sources."""
    print("🔧 Creating unified structure...")

    storico = data["storico"]
    storico_completo = data["storico_completo"]

    # 1. Extract festival_edizioni (from albo_oro in storico)
    festival_edizioni = {}
    for entry in storico.get("albo_oro", []):
        anno = entry["anno"]
        festival_edizioni[str(anno)] = {
            "edizione": entry["edizione"],
            "anno": entry["anno"],
            "vincitore": entry["vincitore"],
            "punteggio": entry["punteggio"],
            "squadre": entry["squadre"],
        }
    print(f"   ✅ festival_edizioni: {len(festival_edizioni)} editions")

    # 2. Extract artisti_storici (from storico_fantasanremo_completo)
    artisti_storici = {}
    for entry in storico_completo.get("storico_fantasanremo_completo", []):
        artista_nome = entry["artista"]
        artisti_storici[artista_nome] = {
            "partecipazioni_totali": len(entry.get("edizioni", [])),
            "edizioni": entry.get("edizioni", []),
        }
    print(f"   ✅ artisti_storici: {len(artisti_storici)} artists")

    # 3. Extract artisti_2026 (from statistiche_artisti_2026 in storico)
    artisti_2026 = []
    for entry in storico.get("statistiche_artisti_2026", []):
        artist_data = {
            "nome": entry["nome"],
            "quotazione": None,  # Will be filled from artisti_2026_enriched.json if needed
            "debuttante": entry.get("debuttante", False),
            "partecipazioni": entry.get("partecipazioni", 0),
        }

        # Convert year columns to storico_posizioni
        storico_posizioni = {}
        for year in ["2021", "2022", "2023", "2024", "2025"]:
            pos = entry.get(year)
            storico_posizioni[year] = pos

        artist_data["storico_posizioni"] = storico_posizioni

        # Add note if present
        if "note" in entry:
            artist_data["note"] = entry["note"]

        artisti_2026.append(artist_data)

    print(f"   ✅ artisti_2026: {len(artisti_2026)} artists")

    # 4. Create metadata
    metadata = {
        "unified_version": "1.0",
        "data_last_updated": datetime.now().strftime("%Y-%m-%d"),
        "source_files": ["storico_fantasanremo.json", "storico_fantasanremo_completo.json"],
        "festival_years": list(festival_edizioni.keys()),
        "total_artists_2026": len(artisti_2026),
        "total_historical_artists": len(artisti_storici),
    }

    return {
        "festival_edizioni": festival_edizioni,
        "artisti_storici": artisti_storici,
        "artisti_2026": artisti_2026,
        "metadata": metadata,
    }


def validate_unified_data(unified: dict[str, Any]) -> tuple[bool, list[str]]:
    """Validate the unified data structure."""
    print("🔍 Validating unified data...")

    errors = []

    # Check required top-level keys
    required_keys = ["festival_edizioni", "artisti_storici", "artisti_2026", "metadata"]
    for key in required_keys:
        if key not in unified:
            errors.append(f"Missing required key: {key}")

    # Validate festival_edizioni
    if "festival_edizioni" in unified:
        for anno, edizione in unified["festival_edizioni"].items():
            required_fields = ["edizione", "anno", "vincitore", "punteggio", "squadre"]
            for field in required_fields:
                if field not in edizione:
                    errors.append(f"festival_edizioni[{anno}]: missing {field}")

    # Validate artisti_storici
    if "artisti_storici" in unified:
        for artista, data in unified["artisti_storici"].items():
            if "edizioni" not in data:
                errors.append(f"artisti_storici[{artista}]: missing edizioni")
            if "partecipazioni_totali" not in data:
                errors.append(f"artisti_storici[{artista}]: missing partecipazioni_totali")

    # Validate artisti_2026
    if "artisti_2026" in unified:
        for i, artista in enumerate(unified["artisti_2026"]):
            if "nome" not in artista:
                errors.append(f"artisti_2026[{i}]: missing nome")
            if "storico_posizioni" not in artista:
                errors.append(f"artisti_2026[{i}]: missing storico_posizioni")

    # Validate metadata
    if "metadata" in unified:
        metadata = unified["metadata"]
        required_meta = ["unified_version", "data_last_updated"]
        for field in required_meta:
            if field not in metadata:
                errors.append(f"metadata: missing {field}")

    if errors:
        print(f"   ❌ Validation failed with {len(errors)} errors")
        for error in errors[:10]:  # Show first 10 errors
            print(f"      - {error}")
        if len(errors) > 10:
            print(f"      ... and {len(errors) - 10} more errors")
        return False, errors

    print("   ✅ Validation passed")
    return True, []


def verify_data_integrity(
    unified: dict[str, Any], original: dict[str, Any]
) -> tuple[bool, list[str]]:
    """Verify that no data was lost during migration."""
    print("🔍 Verifying data integrity...")

    issues = []

    # Check festival_edizioni count
    original_albo_count = len(original["storico"].get("albo_oro", []))
    unified_edizioni_count = len(unified.get("festival_edizioni", {}))
    if original_albo_count != unified_edizioni_count:
        issues.append(
            f"festival_edizioni count mismatch: original={original_albo_count}, "
            f"unified={unified_edizioni_count}"
        )

    # Check artisti_storici count
    original_storico_count = len(
        original["storico_completo"].get("storico_fantasanremo_completo", [])
    )
    unified_storici_count = len(unified.get("artisti_storici", {}))
    if original_storico_count != unified_storici_count:
        issues.append(
            f"artisti_storici count mismatch: original={original_storico_count}, "
            f"unified={unified_storici_count}"
        )

    # Check artisti_2026 count
    original_artisti_2026_count = len(
        original["storico"].get("statistiche_artisti_2026", [])
    )
    unified_artisti_2026_count = len(unified.get("artisti_2026", []))
    if original_artisti_2026_count != unified_artisti_2026_count:
        issues.append(
            f"artisti_2026 count mismatch: original={original_artisti_2026_count}, "
            f"unified={unified_artisti_2026_count}"
        )

    # Check that all artists in artisti_2026 have their historical positions
    for artista in unified.get("artisti_2026", []):
        nome = artista.get("nome")
        storico_posizioni = artista.get("storico_posizioni", {})
        if not storico_posizioni:
            issues.append(f"artisti_2026[{nome}]: empty storico_posizioni")

    if issues:
        print(f"   ❌ Data integrity check failed with {len(issues)} issues")
        for issue in issues:
            print(f"      - {issue}")
        return False, issues

    print("   ✅ Data integrity verified - no data loss detected")
    return True, []


def save_unified_file(unified: dict[str, Any], paths: dict[str, Path]) -> None:
    """Save the unified file."""
    print("💾 Saving unified file...")

    output_path = paths["unified_path"]
    with open(output_path, "w") as f:
        json.dump(unified, f, indent=2, ensure_ascii=False)

    print(f"   ✅ Saved: {output_path}")


def generate_summary(
    unified: dict[str, Any],
    validation_result: tuple[bool, list[str]],
    integrity_result: tuple[bool, list[str]],
) -> None:
    """Generate and print migration summary."""
    print("\n" + "=" * 60)
    print("📋 MIGRATION SUMMARY")
    print("=" * 60)

    metadata = unified.get("metadata", {})
    print(f"\nUnified Version: {metadata.get('unified_version', 'N/A')}")
    print(f"Date: {metadata.get('data_last_updated', 'N/A')}")

    print("\n📊 Data Counts:")
    print(f"   Festival Editions: {len(unified.get('festival_edizioni', {}))}")
    print(f"   Historical Artists: {len(unified.get('artisti_storici', {}))}")
    print(f"   2026 Artists: {len(unified.get('artisti_2026', []))}")

    print(f"\n✅ Validation: {'PASSED' if validation_result[0] else 'FAILED'}")
    print(f"✅ Integrity: {'VERIFIED' if integrity_result[0] else 'FAILED'}")

    if not validation_result[0] or not integrity_result[0]:
        print("\n⚠️  Migration completed with errors. Please review the issues above.")
    else:
        print("\n✅ Migration completed successfully!")

    print("=" * 60)


def main():
    """Main migration function."""
    parser = argparse.ArgumentParser(
        description="Unifica i file JSON storico Fantasanremo"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Esegui la migrazione senza scrivere file",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Non creare backup dei file originali",
    )

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("🔄 UNIFICAZIONE FILE JSON STORICO FANTASANREMO")
    print("=" * 60 + "\n")

    try:
        # Setup paths
        paths = setup_paths()

        # Create timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create backups (unless --no-backup)
        if not args.no_backup:
            create_backup(paths, timestamp)
        else:
            print("⚠️  Skipping backup (--no-backup flag)")

        # Load data
        original_data = load_json_data(paths)

        # Create unified structure
        unified_data = create_unified_structure(original_data)

        # Validate unified data
        is_valid, validation_errors = validate_unified_data(unified_data)

        # Verify data integrity
        integrity_ok, integrity_issues = verify_data_integrity(unified_data, original_data)

        # Save file (unless --dry-run)
        if not args.dry_run and is_valid and integrity_ok:
            save_unified_file(unified_data, paths)
        elif args.dry_run:
            print("\n🔍 DRY RUN - No files written")
        else:
            print("\n❌ Migration aborted due to validation/integrity errors")
            return 1

        # Generate summary
        generate_summary(
            unified_data,
            (is_valid, validation_errors),
            (integrity_ok, integrity_issues),
        )

        # Return appropriate exit code
        if is_valid and integrity_ok:
            return 0
        else:
            return 1

    except Exception as e:
        print(f"\n❌ Migration failed with error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
