#!/usr/bin/env python3
"""
Script to verify integrity of voti_stampa.json.
Checks if it covers all expected years (2020-2026).
"""

import json
import sys
from pathlib import Path

def main():
    project_root = Path(__file__).parent.parent
    data_file = project_root / "data" / "voti_stampa.json"
    
    if not data_file.exists():
        print(f"❌ Error: {data_file} not found")
        sys.exit(1)
        
    try:
        with open(data_file) as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ Error loading JSON: {e}")
        sys.exit(1)
        
    edizioni = data.get("edizioni", {})
    expected_years = [str(y) for y in range(2020, 2027)]
    missing_years = []
    
    print(f"Checking {data_file}...")
    
    for year in expected_years:
        if year not in edizioni:
            missing_years.append(year)
            print(f"  Warning: Missing data for year {year}")
        else:
            entries = edizioni[year].get("voti", [])
            print(f"  ✅ {year}: {len(entries)} entries")
            
    if missing_years:
        print(f"\n⚠️  Missing years: {', '.join(missing_years)}")
        # We don't exit with error as we might not have all previous years, 
        # but user asked to 'recover' if missing. 
        # For now we just report.
        sys.exit(1)
    else:
        print("\n✅ voti_stampa.json covers all expected years (2020-2026)")
        sys.exit(0)

if __name__ == "__main__":
    main()
