from pydantic import BaseModel, ConfigDict, Field


class ArtistaBase(BaseModel):
    nome: str
    quotazione_2026: int = Field(ge=13, le=17)
    genere_musicale: str | None = None
    anno_nascita: int | None = None
    prima_partecipazione: int | None = None
    debuttante_2026: bool = True
    image_url: str | None = None


class Artista(ArtistaBase):
    id: int

    model_config = ConfigDict(from_attributes=True)


class EdizioneFantaSanremoBase(BaseModel):
    anno: int
    punteggio_finale: int | None = None
    posizione: int | None = None
    quotazione_baudi: int | None = None


class EdizioneFantaSanremo(EdizioneFantaSanremoBase):
    id: int
    artista_id: int

    model_config = ConfigDict(from_attributes=True)


class CaratteristicheArtistaBase(BaseModel):
    viralita_social: int | None = Field(default=None, ge=1, le=100)
    social_followers_total: int | None = Field(default=None, ge=0)
    social_followers_by_platform: str | None = None
    social_followers_last_updated: str | None = None
    storia_bonus_ottenuti: int | None = Field(default=None, ge=0)
    ad_personam_bonus_count: int | None = Field(default=None, ge=0)
    ad_personam_bonus_points: int | None = Field(default=None, ge=0)


class CaratteristicheArtista(CaratteristicheArtistaBase):
    id: int
    artista_id: int

    model_config = ConfigDict(from_attributes=True)


class Predizione2026Base(BaseModel):
    punteggio_predetto: float
    confidence: float | None = None
    livello_performer: str = Field(pattern="^(HIGH|MEDIUM|LOW|DEBUTTANTE)$")


class Predizione2026(Predizione2026Base):
    id: int
    artista_id: int

    model_config = ConfigDict(from_attributes=True)


class ArtistaWithPredizione(Artista):
    """Artista con predizione ML per la lista."""

    predizione_2026: Predizione2026 | None = None

    model_config = ConfigDict(from_attributes=True)


class ArtistaWithStorico(Artista):
    edizioni_fantasanremo: list[EdizioneFantaSanremo] = []
    caratteristiche: CaratteristicheArtista | None = None
    predizione_2026: Predizione2026 | None = None


class TeamValidateRequest(BaseModel):
    artisti_ids: list[int] = Field(min_length=7, max_length=7)
    capitano_id: int


class TeamValidateResponse(BaseModel):
    valid: bool
    message: str
    budget_totale: int
    budget_rimanente: int
    punteggio_simulato: float | None = None


class TeamSimulateRequest(BaseModel):
    artisti_ids: list[int] = Field(min_length=5, max_length=5)
    capitano_id: int


class TeamSimulateResponse(BaseModel):
    punteggio_totale: float
    punteggio_dettaglio: list[dict]
    punteggio_capitano: float
    punteggio_titolari: float
