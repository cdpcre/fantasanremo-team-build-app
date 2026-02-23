"""
Database model tests for FantaSanremo backend.

This module tests:
- Model creation and validation
- Relationships between models
- Field constraints and defaults
- Cascade behaviors
- Query operations

Test categories:
- Model instantiation
- Field validation
- Relationship integrity
- Database constraints
- ORM queries
"""

import pytest
from models import Artista, CaratteristicheArtista, EdizioneFantaSanremo, Predizione2026
from sqlalchemy import inspect
from sqlalchemy.orm import Session


class TestArtistaModel:
    """Test cases for Artista model."""

    def test_create_artista_minimal(self, db_session: Session):
        """Test creating an artist with minimal required fields."""
        artista = Artista(nome="Test Artist", quotazione_2026=15)

        db_session.add(artista)
        db_session.commit()
        db_session.refresh(artista)

        assert artista.id is not None
        assert artista.nome == "Test Artist"
        assert artista.quotazione_2026 == 15
        assert artista.debuttante_2026 is True  # Default value

    def test_create_artista_all_fields(self, db_session: Session):
        """Test creating an artist with all fields populated."""
        artista = Artista(
            nome="Complete Artist",
            quotazione_2026=17,
            genere_musicale="Pop",
            anno_nascita=1990,
            prima_partecipazione=2020,
            debuttante_2026=False,
        )

        db_session.add(artista)
        db_session.commit()
        db_session.refresh(artista)

        assert artista.nome == "Complete Artist"
        assert artista.quotazione_2026 == 17
        assert artista.genere_musicale == "Pop"
        assert artista.anno_nascita == 1990
        assert artista.prima_partecipazione == 2020
        assert artista.debuttante_2026 is False

    def test_artista_unique_nome(self, db_session: Session):
        """Test that artist names must be unique."""
        artista1 = Artista(nome="Duplicate Name", quotazione_2026=15)

        db_session.add(artista1)
        db_session.commit()

        # Try to create another artist with same name
        artista2 = Artista(nome="Duplicate Name", quotazione_2026=16)

        db_session.add(artista2)

        with pytest.raises(Exception):  # IntegrityError
            db_session.commit()

    def test_artista_quotazione_range(self, db_session: Session):
        """Test that quotazione is within valid range (application level)."""
        # Valid quotations
        for quota in [13, 14, 15, 16, 17]:
            artista = Artista(nome=f"Artist {quota}", quotazione_2026=quota)
            db_session.add(artista)

        db_session.commit()

        # Verify all created
        artisti = db_session.query(Artista).all()
        assert len(artisti) == 5

    def test_artista_relationships_empty(self, db_session: Session):
        """Test that new artist has empty relationship lists."""
        artista = Artista(nome="Relationship Test", quotazione_2026=14)

        db_session.add(artista)
        db_session.commit()
        db_session.refresh(artista)

        assert len(artista.edizioni_fantasanremo) == 0
        assert artista.caratteristiche is None
        assert artista.predizione_2026 is None

    def test_artista_str_representation(self, db_session: Session):
        """Test artist string representation."""
        artista = Artista(nome="String Test", quotazione_2026=15)

        db_session.add(artista)
        db_session.commit()

        # Just verify it can be converted to string
        assert str(artista) is not None
        assert "String Test" in str(artista) or hasattr(artista, "nome")


class TestEdizioneFantaSanremoModel:
    """Test cases for EdizioneFantaSanremo model."""

    def test_create_edizione_complete(self, db_session: Session):
        """Test creating a complete historical edition."""
        artista = Artista(nome="Historical Artist", quotazione_2026=15)
        db_session.add(artista)
        db_session.commit()
        db_session.refresh(artista)

        edizione = EdizioneFantaSanremo(
            artista_id=artista.id, anno=2023, punteggio_finale=450, posizione=2, quotazione_baudi=16
        )

        db_session.add(edizione)
        db_session.commit()
        db_session.refresh(edizione)

        assert edizione.id is not None
        assert edizione.artista_id == artista.id
        assert edizione.anno == 2023
        assert edizione.punteggio_finale == 450
        assert edizione.posizione == 2
        assert edizione.quotazione_baudi == 16

    def test_create_edizione_minimal(self, db_session: Session):
        """Test creating an edition with minimal fields."""
        artista = Artista(nome="Minimal Artist", quotazione_2026=14)
        db_session.add(artista)
        db_session.commit()
        db_session.refresh(artista)

        edizione = EdizioneFantaSanremo(artista_id=artista.id, anno=2024)

        db_session.add(edizione)
        db_session.commit()
        db_session.refresh(edizione)

        assert edizione.anno == 2024
        assert edizione.punteggio_finale is None
        assert edizione.posizione is None
        assert edizione.quotazione_baudi is None

    def test_edizione_artista_relationship(self, db_session: Session):
        """Test the relationship between edition and artist."""
        artista = Artista(nome="Related Artist", quotazione_2026=16)
        db_session.add(artista)
        db_session.commit()
        db_session.refresh(artista)

        edizione = EdizioneFantaSanremo(
            artista_id=artista.id, anno=2023, punteggio_finale=500, posizione=1
        )

        db_session.add(edizione)
        db_session.commit()
        db_session.refresh(edizione)

        # Test forward relationship
        assert edizione.artista.id == artista.id
        assert edizione.artista.nome == "Related Artist"

        # Test reverse relationship
        assert len(artista.edizioni_fantasanremo) == 1
        assert artista.edizioni_fantasanremo[0].id == edizione.id

    def test_edizione_multiple_per_artist(self, db_session: Session):
        """Test that an artist can have multiple historical editions."""
        artista = Artista(nome="Multiple Artist", quotazione_2026=17)
        db_session.add(artista)
        db_session.commit()
        db_session.refresh(artista)

        # Create multiple editions
        for year, score, pos in [(2020, 300, 5), (2021, 400, 3), (2023, 500, 1)]:
            edizione = EdizioneFantaSanremo(
                artista_id=artista.id, anno=year, punteggio_finale=score, posizione=pos
            )
            db_session.add(edizione)

        db_session.commit()
        db_session.refresh(artista)

        assert len(artista.edizioni_fantasanremo) == 3

        # Verify ordering by year
        anni = [e.anno for e in artista.edizioni_fantasanremo]
        assert anni == [2020, 2021, 2023]


class TestCaratteristicheArtistaModel:
    """Test cases for CaratteristicheArtista model."""

    def test_create_caratteristiche_complete(self, db_session: Session):
        """Test creating complete artist characteristics."""
        artista = Artista(nome="Charismatic Artist", quotazione_2026=16)
        db_session.add(artista)
        db_session.commit()
        db_session.refresh(artista)

        caratteristiche = CaratteristicheArtista(
            artista_id=artista.id,
            viralita_social=85,
            social_followers_total=2500000,
            storia_bonus_ottenuti=10,
            ad_personam_bonus_count=2,
            ad_personam_bonus_points=20,
        )

        db_session.add(caratteristiche)
        db_session.commit()
        db_session.refresh(caratteristiche)

        assert caratteristiche.id is not None
        assert caratteristiche.artista_id == artista.id
        assert caratteristiche.viralita_social == 85
        assert caratteristiche.social_followers_total == 2500000
        assert caratteristiche.storia_bonus_ottenuti == 10
        assert caratteristiche.ad_personam_bonus_count == 2
        assert caratteristiche.ad_personam_bonus_points == 20

    def test_caratteristiche_minimal(self, db_session: Session):
        """Test creating characteristics with minimal values."""
        artista = Artista(nome="Minimal Char", quotazione_2026=14)
        db_session.add(artista)
        db_session.commit()
        db_session.refresh(artista)

        caratteristiche = CaratteristicheArtista(
            artista_id=artista.id,
            viralita_social=50,
            storia_bonus_ottenuti=0,
        )

        db_session.add(caratteristiche)
        db_session.commit()
        db_session.refresh(caratteristiche)

        assert caratteristiche.viralita_social == 50
        assert caratteristiche.storia_bonus_ottenuti == 0

    def test_caratteristiche_unique_per_artist(self, db_session: Session):
        """Test that characteristics are unique per artist (one-to-one)."""
        artista = Artista(nome="Unique Char Artist", quotazione_2026=15)
        db_session.add(artista)
        db_session.commit()
        db_session.refresh(artista)

        caratteristiche1 = CaratteristicheArtista(
            artista_id=artista.id,
            viralita_social=80,
            storia_bonus_ottenuti=5,
        )

        db_session.add(caratteristiche1)
        db_session.commit()

        # Try to create second characteristics for same artist
        caratteristiche2 = CaratteristicheArtista(
            artista_id=artista.id,
            viralita_social=90,
            storia_bonus_ottenuti=8,
        )

        db_session.add(caratteristiche2)

        with pytest.raises(Exception):  # IntegrityError
            db_session.commit()

    def test_caratteristiche_relationship(self, db_session: Session):
        """Test the relationship between characteristics and artist."""
        artista = Artista(nome="Char Relationship", quotazione_2026=17)
        db_session.add(artista)
        db_session.commit()
        db_session.refresh(artista)

        caratteristiche = CaratteristicheArtista(
            artista_id=artista.id,
            viralita_social=99,
            storia_bonus_ottenuti=12,
        )

        db_session.add(caratteristiche)
        db_session.commit()
        db_session.refresh(artista)

        # Test forward relationship
        assert artista.caratteristiche.id == caratteristiche.id
        assert artista.caratteristiche.viralita_social == 99

        # Test reverse relationship
        assert caratteristiche.artista.id == artista.id
        assert caratteristiche.artista.nome == "Char Relationship"


class TestPredizione2026Model:
    """Test cases for Predizione2026 model."""

    def test_create_predizione_complete(self, db_session: Session):
        """Test creating a complete prediction."""
        artista = Artista(nome="Predicted Artist", quotazione_2026=16)
        db_session.add(artista)
        db_session.commit()
        db_session.refresh(artista)

        predizione = Predizione2026(
            artista_id=artista.id,
            punteggio_predetto=450.5,
            confidence=0.85,
            livello_performer="HIGH",
        )

        db_session.add(predizione)
        db_session.commit()
        db_session.refresh(predizione)

        assert predizione.id is not None
        assert predizione.artista_id == artista.id
        assert predizione.punteggio_predetto == 450.5
        assert predizione.confidence == 0.85
        assert predizione.livello_performer == "HIGH"

    def test_create_predizione_minimal(self, db_session: Session):
        """Test creating prediction with minimal required fields."""
        artista = Artista(nome="Minimal Prediction", quotazione_2026=15)
        db_session.add(artista)
        db_session.commit()
        db_session.refresh(artista)

        predizione = Predizione2026(
            artista_id=artista.id, punteggio_predetto=300.0, livello_performer="MEDIUM"
        )

        db_session.add(predizione)
        db_session.commit()
        db_session.refresh(predizione)

        assert predizione.confidence is None
        assert predizione.livello_performer == "MEDIUM"

    def test_predizione_unique_per_artist(self, db_session: Session):
        """Test that predictions are unique per artist (one-to-one)."""
        artista = Artista(nome="Unique Prediction Artist", quotazione_2026=17)
        db_session.add(artista)
        db_session.commit()
        db_session.refresh(artista)

        predizione1 = Predizione2026(
            artista_id=artista.id, punteggio_predetto=500.0, livello_performer="HIGH"
        )

        db_session.add(predizione1)
        db_session.commit()

        # Try to create second prediction for same artist
        predizione2 = Predizione2026(
            artista_id=artista.id, punteggio_predetto=550.0, livello_performer="HIGH"
        )

        db_session.add(predizione2)

        with pytest.raises(Exception):  # IntegrityError
            db_session.commit()

    def test_predizione_relationship(self, db_session: Session):
        """Test the relationship between prediction and artist."""
        artista = Artista(nome="Prediction Relationship", quotazione_2026=16)
        db_session.add(artista)
        db_session.commit()
        db_session.refresh(artista)

        predizione = Predizione2026(
            artista_id=artista.id,
            punteggio_predetto=420.0,
            confidence=0.75,
            livello_performer="MEDIUM",
        )

        db_session.add(predizione)
        db_session.commit()
        db_session.refresh(artista)

        # Test forward relationship
        assert artista.predizione_2026.id == predizione.id
        assert artista.predizione_2026.punteggio_predetto == 420.0

        # Test reverse relationship
        assert predizione.artista.id == artista.id
        assert predizione.artista.nome == "Prediction Relationship"


class TestModelQueries:
    """Test cases for database queries and ORM operations."""

    def test_query_artist_by_quotazione_range(self, db_session: Session):
        """Test querying artists by quotation range."""
        # Create artists with different quotations
        for quota, nome in [
            (13, "Low"),
            (14, "Mid-Low"),
            (15, "Mid"),
            (16, "High"),
            (17, "Very High"),
        ]:
            artista = Artista(nome=f"Artist {nome}", quotazione_2026=quota)
            db_session.add(artista)

        db_session.commit()

        # Query high-value artists
        high_value = db_session.query(Artista).filter(Artista.quotazione_2026 >= 16).all()

        assert len(high_value) == 2

        # Query mid-range artists
        mid_range = db_session.query(Artista).filter(Artista.quotazione_2026.between(14, 16)).all()

        assert len(mid_range) == 3

    def test_query_debuttanti(self, db_session: Session):
        """Test querying debuttant artists."""
        # Create debuttanti and veterans
        debuttante = Artista(nome="New Artist", quotazione_2026=14, debuttante_2026=True)

        veteran = Artista(nome="Old Artist", quotazione_2026=15, debuttante_2026=False)

        db_session.add(debuttante)
        db_session.add(veteran)
        db_session.commit()

        # Query only debuttanti
        new_artists = db_session.query(Artista).filter(Artista.debuttante_2026).all()

        assert len(new_artists) == 1
        assert new_artists[0].nome == "New Artist"

    def test_query_join_artist_with_storico(self, db_session: Session):
        """Test querying artists with their historical data."""
        artista = Artista(nome="Join Test Artist", quotazione_2026=16)
        db_session.add(artista)
        db_session.commit()
        db_session.refresh(artista)

        edizione = EdizioneFantaSanremo(
            artista_id=artista.id, anno=2023, punteggio_finale=400, posizione=3
        )
        db_session.add(edizione)
        db_session.commit()

        # Join query
        result = (
            db_session.query(Artista, EdizioneFantaSanremo)
            .join(EdizioneFantaSanremo, Artista.id == EdizioneFantaSanremo.artista_id)
            .filter(EdizioneFantaSanremo.anno == 2023)
            .first()
        )

        assert result is not None
        artist_obj, edizione_obj = result
        assert artist_obj.nome == "Join Test Artist"
        assert edizione_obj.punteggio_finale == 400

    def test_query_top_performers(self, db_session: Session):
        """Test querying top performers by historical score."""
        # Create artist with multiple editions
        artista = Artista(nome="Top Performer", quotazione_2026=17)
        db_session.add(artista)
        db_session.commit()
        db_session.refresh(artista)

        for score, pos in [(500, 1), (450, 2), (400, 5)]:
            edizione = EdizioneFantaSanremo(
                artista_id=artista.id, anno=2020 + pos, punteggio_finale=score, posizione=pos
            )
            db_session.add(edizione)

        db_session.commit()

        # Query best position
        best_edition = (
            db_session.query(EdizioneFantaSanremo)
            .filter(EdizioneFantaSanremo.artista_id == artista.id)
            .order_by(EdizioneFantaSanremo.posizione.asc())
            .first()
        )

        assert best_edition.posizione == 1
        assert best_edition.punteggio_finale == 500

    def test_query_artists_with_caratteristiche(self, db_session: Session):
        """Test querying artists with high viralita_social."""
        # Create artists with different viralita levels
        for nome, viralita in [
            ("Low Viralita", 40),
            ("Medium Viralita", 70),
            ("High Viralita", 95),
        ]:
            artista = Artista(nome=nome, quotazione_2026=15)
            db_session.add(artista)
            db_session.commit()
            db_session.refresh(artista)

            caratteristiche = CaratteristicheArtista(
                artista_id=artista.id,
                viralita_social=viralita,
                storia_bonus_ottenuti=5,
            )
            db_session.add(caratteristiche)

        db_session.commit()

        # Query artists with high viralita
        high_viralita = (
            db_session.query(Artista)
            .join(CaratteristicheArtista)
            .filter(CaratteristicheArtista.viralita_social >= 80)
            .all()
        )

        assert len(high_viralita) == 1
        assert high_viralita[0].nome == "High Viralita"

    def test_cascade_delete_relationships(self, db_session: Session):
        """Test that deleting an artist requires handling related records first."""
        artista = Artista(nome="Delete Test", quotazione_2026=15)
        db_session.add(artista)
        db_session.commit()
        db_session.refresh(artista)

        # Add related records
        edizione = EdizioneFantaSanremo(artista_id=artista.id, anno=2023)
        db_session.add(edizione)

        caratteristiche = CaratteristicheArtista(
            artista_id=artista.id,
            viralita_social=75,
            storia_bonus_ottenuti=3,
        )
        db_session.add(caratteristiche)

        predizione = Predizione2026(
            artista_id=artista.id, punteggio_predetto=300.0, livello_performer="MEDIUM"
        )
        db_session.add(predizione)

        db_session.commit()

        artista_id = artista.id

        # First delete related records (as cascade is not configured)
        db_session.delete(edizione)
        db_session.delete(caratteristiche)
        db_session.delete(predizione)
        db_session.commit()

        # Now delete artist
        db_session.delete(artista)
        db_session.commit()

        # Verify all records are deleted
        assert db_session.query(Artista).filter_by(id=artista_id).first() is None
        assert (
            db_session.query(EdizioneFantaSanremo).filter_by(artista_id=artista_id).first() is None
        )
        assert (
            db_session.query(CaratteristicheArtista).filter_by(artista_id=artista_id).first()
            is None
        )
        assert db_session.query(Predizione2026).filter_by(artista_id=artista_id).first() is None

    def test_model_table_names(self, db_session: Session):
        """Test that model table names are correctly set."""
        assert Artista.__tablename__ == "artisti"
        assert EdizioneFantaSanremo.__tablename__ == "edizioni_fantasanremo"
        assert CaratteristicheArtista.__tablename__ == "caratteristiche_artisti"
        assert Predizione2026.__tablename__ == "predizioni_2026"

    def test_model_columns_exist(self, db_session: Session):
        """Test that expected columns exist in models."""
        inspector = inspect(db_session.bind)

        # Check Artista columns
        artisti_columns = [c["name"] for c in inspector.get_columns("artisti")]
        assert "id" in artisti_columns
        assert "nome" in artisti_columns
        assert "quotazione_2026" in artisti_columns
        assert "genere_musicale" in artisti_columns
        assert "anno_nascita" in artisti_columns
        assert "prima_partecipazione" in artisti_columns
        assert "debuttante_2026" in artisti_columns

        # Check EdizioneFantaSanremo columns
        edizioni_columns = [c["name"] for c in inspector.get_columns("edizioni_fantasanremo")]
        assert "id" in edizioni_columns
        assert "artista_id" in edizioni_columns
        assert "anno" in edizioni_columns
        assert "punteggio_finale" in edizioni_columns
        assert "posizione" in edizioni_columns
        assert "quotazione_baudi" in edizioni_columns

        # Check CaratteristicheArtista columns
        caratteristiche_columns = [
            c["name"] for c in inspector.get_columns("caratteristiche_artisti")
        ]
        assert "id" in caratteristiche_columns
        assert "artista_id" in caratteristiche_columns
        assert "viralita_social" in caratteristiche_columns
        assert "social_followers_total" in caratteristiche_columns
        assert "social_followers_by_platform" in caratteristiche_columns
        assert "social_followers_last_updated" in caratteristiche_columns
        assert "storia_bonus_ottenuti" in caratteristiche_columns
        assert "ad_personam_bonus_count" in caratteristiche_columns
        assert "ad_personam_bonus_points" in caratteristiche_columns


class TestModelEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_artista_with_future_year(self, db_session: Session):
        """Test creating artist with future participation year."""
        artista = Artista(nome="Future Artist", quotazione_2026=15, prima_partecipazione=2030)

        db_session.add(artista)
        db_session.commit()
        db_session.refresh(artista)

        assert artista.prima_partecipazione == 2030

    def test_artista_with_very_old_birth_year(self, db_session: Session):
        """Test creating artist with very old birth year."""
        artista = Artista(nome="Vintage Artist", quotazione_2026=14, anno_nascita=1940)

        db_session.add(artista)
        db_session.commit()

        assert artista.anno_nascita == 1940

    def test_edizione_without_position(self, db_session: Session):
        """Test creating edition without position (non-participant)."""
        artista = Artista(nome="NP Artist", quotazione_2026=13)
        db_session.add(artista)
        db_session.commit()
        db_session.refresh(artista)

        edizione = EdizioneFantaSanremo(
            artista_id=artista.id,
            anno=2024,
            posizione=None,  # Non-positioned
        )

        db_session.add(edizione)
        db_session.commit()

        assert edizione.posizione is None

    def test_caratteristiche_boundary_values(self, db_session: Session):
        """Test characteristics with boundary values."""
        artista = Artista(nome="Boundary Test", quotazione_2026=15)
        db_session.add(artista)
        db_session.commit()
        db_session.refresh(artista)

        caratteristiche = CaratteristicheArtista(
            artista_id=artista.id,
            viralita_social=1,  # Minimum
            storia_bonus_ottenuti=0,  # Minimum
            ad_personam_bonus_count=0,
            ad_personam_bonus_points=0,
        )

        db_session.add(caratteristiche)
        db_session.commit()

        assert caratteristiche.viralita_social == 1
        assert caratteristiche.storia_bonus_ottenuti == 0

    def test_predizione_confidence_range(self, db_session: Session):
        """Test prediction with boundary confidence values."""
        artista = Artista(nome="Confidence Test", quotazione_2026=16)
        db_session.add(artista)
        db_session.commit()
        db_session.refresh(artista)

        # Test with confidence = 0
        pred1 = Predizione2026(
            artista_id=artista.id, punteggio_predetto=300.0, confidence=0.0, livello_performer="LOW"
        )
        db_session.add(pred1)
        db_session.commit()

        # Test with confidence = 1 (after deleting first)
        db_session.delete(pred1)
        db_session.commit()

        pred2 = Predizione2026(
            artista_id=artista.id,
            punteggio_predetto=400.0,
            confidence=1.0,
            livello_performer="HIGH",
        )
        db_session.add(pred2)
        db_session.commit()

        assert pred2.confidence == 1.0
