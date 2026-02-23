# Artist Categorization System

## Overview

Artists are categorized into 7 archetypes based on their characteristics, genre, age, and performance history. This categorization helps identify patterns and inform predictions.

## The 7 Archetypes

### 1. VIRAL_PHENOMENON

**Definition:** High social media viral artists

**Criteria:**
- viralità >= 80
- età < 35

**Description:** Young artists with exceptional social media presence. These artists generate buzz online and can surprise with strong performance.

**Examples:** Fedez, Elettra Lamborghini, J-Ax

### 2. VETERAN_PERFORMER

**Definition:** Experienced artists with multiple Sanremo participations

**Criteria:**
- partecipazioni >= 3
- età > 40

**Description:** Seasoned Sanremo veterans who know how to work the stage. Their experience gives them consistency but may lack the surprise element.

**Examples:** Arisa, Francesco Renga, Raf

### 3. INDIE_DARLING

**Definition:** Indie or Rock artists with lower social media presence

**Criteria:**
- Genere in [Indie, Rock]
- viralità < 50

**Description:** Alternative artists who prioritize music over social media. Their fanbase is loyal but smaller, leading to unpredictable FantaSanremo performance.

**Examples:** Fulminacci, Dargen D'Amico, Ditonellapiaga

### 4. RAP_TRAP_STAR

**Definition:** Young Rap/Hip-hop/Urban artists

**Criteria:**
- Genere in [Rap, Hip-hop, Urban, Trap]
- età < 30

**Description:** Young urban artists who bring contemporary sounds to Sanremo. Strong with younger audiences but may struggle with traditional voters.

**Examples:** Luche, Nayt, Samurai Jay, Chiello

### 5. POP_MAINSTREAM

**Definition:** Established Pop artists with high quotations

**Criteria:**
- Genere in [Pop, Pop/Rock, Dance]
- quotazione >= 15

**Description:** Mainstream pop artists with high market value. Safe picks with predictable (usually good) performance.

**Examples:** Tommaso Paradiso, Ermal Meta, Malika Ayane

### 6. LEGENDARY_STATUS

**Definition:** Legendary artists with long careers

**Criteria:**
- età > 60
- prima_partecipazione < 1990

**Description:** Italian music legends. Their status guarantees attention but performance can vary wildly based on song quality.

**Examples:** Patty Pravo (1950s career), Sal Da Vinci

### 7. DEBUTTANTE_POTENTIAL

**Definition:** New artists with high quotation suggesting industry confidence

**Criteria:**
- partecipazioni == 0
- quotazione >= 14

**Description:** First-time Sanremo artists given high quotations, suggesting industry expects strong performance. High risk, high reward category.

**Examples:** Bambole di Pezza, Eddie Brock, Leo Gassmann

## Categorization Logic

### Algorithm

```python
def categorize_artist(artist_data, biografico, caratteristiche, storico):
    # Extract attributes
    genere = biografico.get("genere_musicale", "Pop")
    anno_nascita = biografico.get("anno_nascita")
    eta = 2026 - anno_nascita if anno_nascita else 35
    partecipazioni = count_participations(storico)
    viralita = caratteristiche.get("viralita_social", 50)
    quotazione = artist_data.get("quotazione", 15)

    # Check each archetype in priority order
    if _is_viral_phenomenon(viralita, eta):
        return "VIRAL_PHENOMENON"
    elif _is_veteran_performer(partecipazioni, eta):
        return "VETERAN_PERFORMER"
    # ... etc

    return "UNCATEGORIZED"
```

### Priority Order

Archetypes are checked in order:
1. VIRAL_PHENOMENON (specific traits)
2. VETERAN_PERFORMER (experience)
3. INDIE_DARLING (genre-based)
4. RAP_TRAP_STAR (genre + age)
5. POP_MAINSTREAM (genre + quotation)
6. LEGENDARY_STATUS (age + career)
7. DEBUTTANTE_POTENTIAL (new artists)

## Usage

### Command Line

```bash
# Get categorization
uv run python -c "
from backend.ml.categorization import categorize_all_artists
from backend.data_pipeline.config import get_config
import json

config = get_config()
with open('data/artisti_2026_enriched.json') as f:
    data = json.load(f)

df = categorize_all_artists(data['artisti'])
print(df[['nome', 'primary_archetype']])
"
```

### Python API

```python
from backend.ml.categorization import categorize_all_artists, get_archetype_summary

# Categorize all artists
categorization_df = categorize_all_artists(
    artisti_data=artisti_2026,
    biografico_data=biografico,
    caratteristiche_data=caratteristiche,
    storico_data=storico
)

# Get summary
summary = get_archetype_summary(categorization_df)
print(summary["primary_distribution"])
```

## 2026 Artist Categorization

### By Archetype

| Archetype | Count | Artists |
|-----------|-------|---------|
| VIRAL_PHENOMENON | ~3 | Fedez & Marco Masini, Elettra Lamborghini, J-Ax |
| VETERAN_PERFORMER | ~4 | Arisa, Francesco Renga, Raf, Patty Pravo |
| INDIE_DARLING | ~5 | Dargen D'Amico, Fulminacci, Ditonellapiaga, etc. |
| RAP_TRAP_STAR | ~6 | Luche, Nayt, Samurai Jay, Chiello, etc. |
| POP_MAINSTREAM | ~6 | Tommaso Paradiso, Ermal Meta, Malika Ayane, etc. |
| LEGENDARY_STATUS | ~2 | Patty Pravo, Sal Da Vinci |
| DEBUTTANTE_POTENTIAL | ~8 | Bambole di Pezza, Eddie Brock, etc. |

## Implementation

**File:** `backend/ml/categorization.py`

**Key Functions:**
- `categorize_artist()` - Categorize single artist
- `categorize_all_artists()` - Categorize all artists
- `get_archetype_features()` - Get one-hot encoded features
- `get_archetype_summary()` - Get summary statistics

## Feature Integration

Archetype features are integrated into ML models as one-hot encoded columns:

```python
# Example: For "VIRAL_PHENOMENON"
{
    "VIRAL_PHENOMENON": 1,
    "VETERAN_PERFORMER": 0,
    "INDIE_DARLING": 0,
    "RAP_TRAP_STAR": 0,
    "POP_MAINSTREAM": 0,
    "LEGENDARY_STATUS": 0,
    "DEBUTTANTE_POTENTIAL": 0
}
```

## References

- Implementation: `backend/ml/categorization.py`
- Feature Integration: `backend/ml/features_enhanced.py`
- Feature Groups: `backend/ml/features_enhanced.py:get_feature_groups()`
