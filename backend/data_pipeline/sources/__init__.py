"""
Data Pipeline Sources Module

Modulo per l'acquisizione dati da diverse sorgenti.
"""

from .base import BaseDataSource
from .wikipedia_source import WikipediaDataSource

__all__ = ["BaseDataSource", "WikipediaDataSource"]
