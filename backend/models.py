from database import Base
from sqlalchemy import Boolean, Column, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship


class Artista(Base):
    __tablename__ = "artisti"

    id = Column(Integer, primary_key=True, index=True)
    nome = Column(String, unique=True, index=True, nullable=False)
    quotazione_2026 = Column(Integer, nullable=False)  # Baudi (13-17)
    genere_musicale = Column(String)
    anno_nascita = Column(Integer)
    prima_partecipazione = Column(Integer)  # Anno prima partecipazione a Sanremo
    debuttante_2026 = Column(Boolean, default=True)
    image_url = Column(String, nullable=True)  # URL foto artista per avatar

    # Relationships
    edizioni_fantasanremo = relationship("EdizioneFantaSanremo", back_populates="artista")
    caratteristiche = relationship(
        "CaratteristicheArtista", back_populates="artista", uselist=False
    )
    predizione_2026 = relationship("Predizione2026", back_populates="artista", uselist=False)


class EdizioneFantaSanremo(Base):
    __tablename__ = "edizioni_fantasanremo"

    id = Column(Integer, primary_key=True, index=True)
    artista_id = Column(Integer, ForeignKey("artisti.id"), nullable=False)
    anno = Column(Integer, nullable=False)
    punteggio_finale = Column(Integer)  # Punteggio FantaSanremo
    posizione = Column(Integer)  # Posizione classifica (NULL se NP)
    quotazione_baudi = Column(Integer)  # Quotazione dell'anno

    artista = relationship("Artista", back_populates="edizioni_fantasanremo")


class CaratteristicheArtista(Base):
    __tablename__ = "caratteristiche_artisti"

    id = Column(Integer, primary_key=True, index=True)
    artista_id = Column(Integer, ForeignKey("artisti.id"), unique=True, nullable=False)
    viralita_social = Column(Integer)  # 1-100 (derived from real followers)
    social_followers_total = Column(Integer)
    social_followers_by_platform = Column(Text)  # JSON string
    social_followers_last_updated = Column(String)
    storia_bonus_ottenuti = Column(Integer)  # Somma punti FantaSanremo storici (proxy bonus/malus)
    ad_personam_bonus_count = Column(Integer)
    ad_personam_bonus_points = Column(Integer)

    artista = relationship("Artista", back_populates="caratteristiche")


class Predizione2026(Base):
    __tablename__ = "predizioni_2026"

    id = Column(Integer, primary_key=True, index=True)
    artista_id = Column(Integer, ForeignKey("artisti.id"), unique=True, nullable=False)
    punteggio_predetto = Column(Float, nullable=False)
    confidence = Column(Float)  # 0-1
    livello_performer = Column(String)  # "HIGH", "MEDIUM", "LOW", "DEBUTTANTE"

    artista = relationship("Artista", back_populates="predizione_2026")
