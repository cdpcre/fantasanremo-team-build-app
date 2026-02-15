import logging

from database import get_db
from fastapi import APIRouter, Depends, HTTPException
from models import Artista
from models import EdizioneFantaSanremo as EdizioneModel
from schemas import EdizioneFantaSanremo
from sqlalchemy.orm import Session

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("", response_model=list[EdizioneFantaSanremo])
def get_storico_aggregate(anno: int | None = None, db: Session = Depends(get_db)):
    """Get all historical FantaSanremo results, optionally filtered by year"""
    logger.info(f"Fetching historical data - year filter: {anno}")

    query = db.query(EdizioneModel)

    if anno is not None:
        query = query.filter(EdizioneModel.anno == anno)
        logger.debug(f"Applied year filter: {anno}")

    storico = query.order_by(EdizioneModel.anno, EdizioneModel.posizione).all()
    logger.info(f"Retrieved {len(storico)} historical records")

    return storico


@router.get("/artista/{artista_id}", response_model=list[EdizioneFantaSanremo])
def get_storico_artista(artista_id: int, db: Session = Depends(get_db)):
    """Get historical results for a specific artist"""
    logger.info(f"Fetching historical data for artist ID: {artista_id}")

    artista = db.query(Artista).filter(Artista.id == artista_id).first()

    if not artista:
        logger.warning(f"Artist not found for historical query - ID: {artista_id}")
        raise HTTPException(status_code=404, detail="Artista non trovato")

    storico = (
        db.query(EdizioneModel)
        .filter(EdizioneModel.artista_id == artista_id)
        .order_by(EdizioneModel.anno)
        .all()
    )

    logger.debug(
        f"Retrieved {len(storico)} historical records for artist: {artista.nome} (ID: {artista_id})"
    )

    return storico
