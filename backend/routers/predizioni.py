import logging

from database import get_db
from fastapi import APIRouter, Depends, HTTPException
from models import Artista
from models import Predizione2026 as PredizioneModel
from schemas import Predizione2026
from sqlalchemy.orm import Session, joinedload

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("", response_model=list[Predizione2026])
def get_predizioni(db: Session = Depends(get_db)):
    """Get all 2026 predictions"""
    logger.info("Fetching all 2026 predictions")

    predizioni = (
        db.query(PredizioneModel)
        .options(joinedload(PredizioneModel.artista))
        .join(Artista)
        .order_by(PredizioneModel.punteggio_predetto.desc())
        .all()
    )

    logger.info(f"Retrieved {len(predizioni)} predictions")

    if predizioni:
        logger.debug(
            f"Highest prediction: {predizioni[0].punteggio_predetto} "
            f"(Artist ID: {predizioni[0].artista_id})"
        )
        logger.debug(
            f"Lowest prediction: {predizioni[-1].punteggio_predetto} "
            f"(Artist ID: {predizioni[-1].artista_id})"
        )

    return predizioni


@router.get("/{artista_id}", response_model=Predizione2026)
def get_predizione_artista(artista_id: int, db: Session = Depends(get_db)):
    """Get prediction for a specific artist"""
    logger.info(f"Fetching prediction for artist ID: {artista_id}")

    predizione = (
        db.query(PredizioneModel)
        .options(joinedload(PredizioneModel.artista))
        .filter(PredizioneModel.artista_id == artista_id)
        .first()
    )

    if not predizione:
        logger.warning(f"Prediction not found for artist ID: {artista_id}")
        raise HTTPException(status_code=404, detail="Predizione non trovata")

    logger.debug(
        f"Retrieved prediction for artist: {predizione.artista.nome} "
        f"(Score: {predizione.punteggio_predetto}, Level: {predizione.livello_performer})"
    )

    return predizione
