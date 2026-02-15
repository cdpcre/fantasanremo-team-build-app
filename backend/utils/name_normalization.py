from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable

_FEAT_PAT = re.compile(r"\b(feat\.?|ft\.?|featuring)\b", re.IGNORECASE)


def normalize_artist_name(name: str | None) -> str:
    """
    Normalize artist names for robust cross-file matching.

    - lowercases
    - strips accents
    - removes punctuation
    - removes feat/ft/featuring tokens
    - collapses whitespace
    """
    if not name:
        return ""

    text = name.strip().lower()
    text = _FEAT_PAT.sub(" ", text)
    text = text.replace("&", " and ").replace("+", " and ")

    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))

    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_artist_key(name: str | None) -> str:
    """Alias for normalize_artist_name, used for clarity."""
    return normalize_artist_name(name)


def index_by_normalized_name(items: Iterable[dict], name_field: str = "nome") -> dict[str, dict]:
    """
    Build a normalized-name -> item map. Keeps first occurrence on collision.
    """
    index: dict[str, dict] = {}
    for item in items:
        key = normalize_artist_name(item.get(name_field))
        if not key:
            continue
        if key not in index:
            index[key] = item
    return index
