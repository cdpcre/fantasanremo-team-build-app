"""
Pytest configuration and shared fixtures for FantaSanremo backend tests.

This module provides:
- Database fixtures for testing
- Mock data fixtures
- Test client configuration
- Common test utilities
"""

import os
import tempfile
from collections.abc import Generator

# Configure test database URL before importing app/database modules
test_db_file = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
test_db_file.close()
TEST_DATABASE_URL = f"sqlite:///{test_db_file.name}"
os.environ.setdefault("DATABASE_URL", TEST_DATABASE_URL)
os.environ.setdefault("FS_AUTO_CREATE_DB", "false")

import pytest
from database import Base, get_db
from fastapi.testclient import TestClient
from main import app
from models import Artista, CaratteristicheArtista, EdizioneFantaSanremo, Predizione2026
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

# Create test engine
test_engine = create_engine(TEST_DATABASE_URL, connect_args={"check_same_thread": False})

TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)


@pytest.fixture(scope="function")
def db_session() -> Generator[Session, None, None]:
    """
    Create a new database session for each test function.

    The database is created fresh for each test, ensuring isolation.
    All changes are rolled back after the test completes.
    """
    # Create tables
    Base.metadata.create_all(bind=test_engine)

    # Create session
    session = TestSessionLocal()

    try:
        yield session
    finally:
        session.rollback()
        session.close()
        # Drop tables after test
        Base.metadata.drop_all(bind=test_engine)


@pytest.fixture(scope="function")
def client(db_session: Session) -> TestClient:
    """
    Create a test client with a mocked database session.

    Overrides the get_db dependency to use the test database.
    """

    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    test_client = TestClient(app)

    yield test_client

    # Clean up dependency override
    app.dependency_overrides.clear()


@pytest.fixture(scope="function")
def client_with_sample_data(
    db_session: Session,
    sample_artisti: list[Artista],
    sample_storico: list[EdizioneFantaSanremo],
    sample_predizioni: list[Predizione2026],
) -> TestClient:
    """
    Test client with all sample data loaded.
    Use this fixture when you need the client to have access to all test data.

    This fixture ensures all sample data is loaded before the client is created.
    """

    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    test_client = TestClient(app)

    yield test_client

    # Clean up dependency override
    app.dependency_overrides.clear()


@pytest.fixture
def sample_artisti(db_session: Session) -> list[Artista]:
    """
    Create sample artists in the test database.

    Returns a list of 10 artists with varying quotations and characteristics.
    """
    artisti_data = [
        {
            "nome": "Maneskin",
            "quotazione_2026": 17,
            "genere_musicale": "Rock",
            "anno_nascita": 2016,
            "prima_partecipazione": 2021,
            "debuttante_2026": False,
        },
        {
            "nome": "Annalisa",
            "quotazione_2026": 16,
            "genere_musicale": "Pop",
            "anno_nascita": 1985,
            "prima_partecipazione": 2018,
            "debuttante_2026": False,
        },
        {
            "nome": "Marco Mengoni",
            "quotazione_2026": 16,
            "genere_musicale": "Pop",
            "anno_nascita": 1988,
            "prima_partecipazione": 2013,
            "debuttante_2026": False,
        },
        {
            "nome": "Loredana Bertè",
            "quotazione_2026": 15,
            "genere_musicale": "Pop",
            "anno_nascita": 1950,
            "prima_partecipazione": 1980,
            "debuttante_2026": False,
        },
        {
            "nome": "Francesco Renga",
            "quotazione_2026": 15,
            "genere_musicale": "Pop",
            "anno_nascita": 1968,
            "prima_partecipazione": 2001,
            "debuttante_2026": False,
        },
        {
            "nome": "Giorgia",
            "quotazione_2026": 15,
            "genere_musicale": "Pop",
            "anno_nascita": 1971,
            "prima_partecipazione": 1995,
            "debuttante_2026": False,
        },
        {
            "nome": "Ultimo",
            "quotazione_2026": 14,
            "genere_musicale": "Pop",
            "anno_nascita": 1996,
            "prima_partecipazione": 2018,
            "debuttante_2026": False,
        },
        {
            "nome": "Irama",
            "quotazione_2026": 14,
            "genere_musicale": "Pop",
            "anno_nascita": 1995,
            "prima_partecipazione": 2020,
            "debuttante_2026": False,
        },
        {
            "nome": "Rosa Chemical",
            "quotazione_2026": 13,
            "genere_musicale": "Pop",
            "anno_nascita": 1994,
            "prima_partecipazione": 2023,
            "debuttante_2026": False,
        },
        {
            "nome": "Nuovo Artista",
            "quotazione_2026": 13,
            "genere_musicale": "Indie",
            "anno_nascita": 2000,
            "prima_partecipazione": None,
            "debuttante_2026": True,
        },
    ]

    artisti = []
    for data in artisti_data:
        artista = Artista(**data)
        db_session.add(artista)
        artisti.append(artista)

    db_session.commit()
    for artista in artisti:
        db_session.refresh(artista)

    return artisti


@pytest.fixture
def sample_storico(
    db_session: Session, sample_artisti: list[Artista]
) -> list[EdizioneFantaSanremo]:
    """
    Create sample historical FantaSanremo editions.

    Returns a list of historical editions for the sample artists.
    """
    storico_data = [
        # Maneskin - won in 2021
        {
            "artista_id": sample_artisti[0].id,
            "anno": 2021,
            "punteggio_finale": 315,
            "posizione": 1,
            "quotazione_baudi": 15,
        },
        # Mengoni - won in 2023
        {
            "artista_id": sample_artisti[2].id,
            "anno": 2023,
            "punteggio_finale": 670,
            "posizione": 1,
            "quotazione_baudi": 17,
        },
        # Annalisa - second place 2024
        {
            "artista_id": sample_artisti[1].id,
            "anno": 2024,
            "punteggio_finale": 450,
            "posizione": 2,
            "quotazione_baudi": 16,
        },
        # Loredana - various positions
        {
            "artista_id": sample_artisti[3].id,
            "anno": 2020,
            "punteggio_finale": 180,
            "posizione": 15,
            "quotazione_baudi": 14,
        },
        # Ultimo - mid-tier
        {
            "artista_id": sample_artisti[6].id,
            "anno": 2019,
            "punteggio_finale": 250,
            "posizione": 8,
            "quotazione_baudi": 14,
        },
        # Rosa - low tier
        {
            "artista_id": sample_artisti[8].id,
            "anno": 2023,
            "punteggio_finale": 90,
            "posizione": 25,
            "quotazione_baudi": 13,
        },
    ]

    storico = []
    for data in storico_data:
        edizione = EdizioneFantaSanremo(**data)
        db_session.add(edizione)
        storico.append(edizione)

    db_session.commit()
    return storico


@pytest.fixture
def sample_caratteristiche(
    db_session: Session, sample_artisti: list[Artista]
) -> list[CaratteristicheArtista]:
    """
    Create sample artist characteristics.

    Returns characteristics for some of the sample artists.
    """
    caratteristiche_data = [
        {
            "artista_id": sample_artisti[0].id,  # Maneskin - high charisma
            "viralita_social": 99,
            "social_followers_total": 5_000_000,
            "storia_bonus_ottenuti": 8,
            "ad_personam_bonus_count": 2,
            "ad_personam_bonus_points": 20,
        },
        {
            "artista_id": sample_artisti[2].id,  # Mengoni - high performer
            "viralita_social": 85,
            "social_followers_total": 3_200_000,
            "storia_bonus_ottenuti": 12,
            "ad_personam_bonus_count": 1,
            "ad_personam_bonus_points": 10,
        },
        {
            "artista_id": sample_artisti[6].id,  # Ultimo - mid tier
            "viralita_social": 80,
            "social_followers_total": 1_100_000,
            "storia_bonus_ottenuti": 4,
            "ad_personam_bonus_count": 0,
            "ad_personam_bonus_points": 0,
        },
    ]

    caratteristiche = []
    for data in caratteristiche_data:
        caratteristica = CaratteristicheArtista(**data)
        db_session.add(caratteristica)
        caratteristiche.append(caratteristica)

    db_session.commit()
    return caratteristiche


@pytest.fixture
def sample_predizioni(db_session: Session, sample_artisti: list[Artista]) -> list[Predizione2026]:
    """
    Create sample predictions for 2026.

    Returns predictions for all sample artists.
    """
    predictions = [
        (0, 580, 0.92, "HIGH"),  # Maneskin - high prediction
        (1, 520, 0.88, "HIGH"),  # Annalisa
        (2, 600, 0.95, "HIGH"),  # Mengoni
        (3, 380, 0.75, "MEDIUM"),  # Loredana
        (4, 350, 0.70, "MEDIUM"),  # Renga
        (5, 400, 0.78, "MEDIUM"),  # Giorgia
        (6, 280, 0.65, "MEDIUM"),  # Ultimo
        (7, 260, 0.60, "MEDIUM"),  # Irama
        (8, 150, 0.50, "LOW"),  # Rosa
        (9, 200, 0.55, "MEDIUM"),  # Nuovo Artista
    ]

    predizioni = []
    for idx, (artista_idx, score, confidence, level) in enumerate(predictions):
        predizione = Predizione2026(
            artista_id=sample_artisti[artista_idx].id,
            punteggio_predetto=score,
            confidence=confidence,
            livello_performer=level,
        )
        db_session.add(predizione)
        predizioni.append(predizione)

    db_session.commit()
    return predizioni


@pytest.fixture
def valid_team_data(sample_artisti: list[Artista]) -> dict:
    """
    Provide valid team data for testing.

    Returns a dictionary with artist IDs and captain ID that form a valid team.
    Budget: 15+14+13+13+14+14+13 = 96 baudi (under 100 limit)
    """
    return {
        "artisti_ids": [
            sample_artisti[3].id,  # 15 baudi
            sample_artisti[6].id,  # 14 baudi
            sample_artisti[8].id,  # 13 baudi
            sample_artisti[9].id,  # 13 baudi (debuttante)
            sample_artisti[7].id,  # 14 baudi
            sample_artisti[6].id + 10,  # Will be made invalid in tests
            sample_artisti[8].id + 10,  # Will be made invalid in tests
        ],
        "capitano_id": sample_artisti[3].id,
    }


@pytest.fixture
def budget_exceeded_team_data(sample_artisti: list[Artista]) -> dict:
    """
    Provide team data that exceeds the 100 baudi budget.

    Budget: 17+17+16+16+16+16+16 = 114 baudi (exceeds 100 limit)
    """
    return {
        "artisti_ids": [
            sample_artisti[0].id,  # 17 baudi
            sample_artisti[0].id,  # Duplicate (invalid)
            sample_artisti[1].id,  # 16 baudi
            sample_artisti[2].id,  # 16 baudi
            sample_artisti[3].id,  # 15 baudi
            sample_artisti[4].id,  # 15 baudi
            sample_artisti[5].id,  # 15 baudi
        ],
        "capitano_id": sample_artisti[0].id,
    }


@pytest.fixture
def simulate_team_data(sample_artisti: list[Artista]) -> dict:
    """
    Provide valid team data for simulation (5 artists).

    Uses 5 artists for the simulate endpoint.
    """
    return {
        "artisti_ids": [
            sample_artisti[0].id,  # Maneskin
            sample_artisti[2].id,  # Mengoni
            sample_artisti[3].id,  # Loredana
            sample_artisti[6].id,  # Ultimo
            sample_artisti[8].id,  # Rosa
        ],
        "capitano_id": sample_artisti[0].id,
    }


# Cleanup function to delete the temporary database file
def pytest_configure(config):
    """Configure pytest with cleanup hooks."""
    pass


def pytest_unconfigure(config):
    """Clean up temporary database file after all tests."""
    try:
        if os.path.exists(test_db_file.name):
            os.unlink(test_db_file.name)
    except Exception:
        pass
