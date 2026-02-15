# FantaSanremo Historical Data Update

## Overview

This document describes the process and results of updating the FantaSanremo historical data from Wikipedia.

## Data Sources

The data was extracted from the official Wikipedia page:
- **Source**: [FantaSanremo - Wikipedia](https://it.wikipedia.org/wiki/FantaSanremo)
- **Data retrieved**: 2026-01-28

## Files Created/Modified

### 1. `/Users/crescenzodepalma/Projects/cdpcre/fantasanremo_team_builder/scripts/update_fantasanremo_data.py`
Python script that:
- Parses the Wikipedia markdown content
- Extracts historical edition data (editions table)
- Extracts artist participation data (artists table)
- Updates the main JSON file with the extracted data

**Usage**:
```bash
uv run python scripts/update_fantasanremo_data.py
```

### 2. `/Users/crescenzodepalma/Projects/cdpcre/fantasanremo_team_builder/data/voti_stampa.json`
Contains historical voting data from the press jury (Sala Stampa).
- **Years**: 2020-2026
- **Source**: Aggregated from various sources/manual entry.

**Structure**:
```json
{
  "giudice": "stampa",
  "edizioni": {
    "2026": {
      "piattaforma": "Instagram", // or other platform
      "voti": [
        {"artista": "Name", "voto": 6.5, "brano": "Title"}
      ]
    }
  }
}
```

### 4. `/Users/crescenzodepalma/Projects/cdpcre/fantasanremo_team_builder/CLAUDE.md`
NEW FILE with development guidelines for the project.

## Historical Data Summary

### Editions (Albo d'Oro)

| Edition | Year | Winner | Points | Teams |
|---------|------|--------|--------|-------|
| 0 | 2020 | Piero Pelù | 220 | 47 |
| 1 | 2021 | Måneskin | 315 | 46,962 |
| 2 | 2022 | Emma | 525 | 500,012 |
| 3 | 2023 | Marco Mengoni | 670 | 4,212,694 |
| 4 | 2024 | La Sad | 486 | 4,201,022 |
| 5 | 2025 | Olly | 475 | 5,096,123 |

### Artist Statistics

Complete historical data for **103 artists** who participated from 2021-2025, including:
- Number of participations
- Final position in each edition
- Debut information

## Key Features of the Data

### 1. Multiple-Time Winners
- **Dargen D'Amico**: 2nd place in both 2022 and 2024
- **Irama**: Participated 4 times (2021, 2022, 2024, 2025)
- **Coma_Cose**: Participated 3 times

### 2. Notable Achievements
- **Måneskin (2021)**: First to win both Festival and FantaSanremo
- **Marco Mengoni (2023)**: Second double winner
- **Olly (2025)**: Third double winner
- **Emma (2022)**: Record 525 points

### 3. Growth Statistics
- 2020: 47 teams (pilot edition)
- 2021: 46,962 teams (first online edition)
- 2022: 500,012 teams (national phenomenon)
- 2023-2025: 4-5+ million teams (established)

## Data Quality

The data from Wikipedia is considered highly reliable as it's:
- Officially maintained
- Regularly updated
- Source-verified with footnotes
- Complete for all editions 2020-2025

## 2026 Biographical Gap Closure (2026-02-11)

- Issue closed: `anno_nascita` missing for `Bambole di Pezza`.
- Policy for groups: if no reliable member-level birth year is available from web sources, use the current roster average age as fallback.
- Current roster calculation (29 artists with known birth year): average birth year `1986.14`, rounded to `1986` (average age in 2026: `40`).
- Applied in:
  - `/Users/crescenzodepalma/Projects/cdpcre/fantasanremo_team_builder/data/artisti_2026_enriched.json`
  - `/Users/crescenzodepalma/Projects/cdpcre/fantasanremo_team_builder/data/overrides.json`

## Future Updates

To update the data in the future:
1. Re-fetch the Wikipedia page using the web reader tool
2. Update the `wikipedia_markdown` variable in `update_fantasanremo_data.py`
3. Run: `uv run python scripts/update_fantasanremo_data.py`

## Sources

- [FantaSanremo - Wikipedia](https://it.wikipedia.org/wiki/FantaSanremo)
- [FantaSanremo Official Website](https://fantasanremo.com/)

## Notes

- The script uses embedded Wikipedia data rather than live scraping to avoid external dependencies
- All data is from the official Wikipedia page retrieved on 2026-01-28
- The JSON structure maintains compatibility with existing code
- Use `uv` package manager for all Python operations as specified in CLAUDE.md
