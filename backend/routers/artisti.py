import logging

from database import get_db
from fastapi import APIRouter, Depends, HTTPException, Query
from models import Artista as ArtistaModel
from schemas import ArtistaWithPredizione, ArtistaWithStorico
from sqlalchemy.orm import Session, joinedload

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("", response_model=list[ArtistaWithPredizione])
def get_artisti(
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(20, ge=1, le=100, description="Number of records to return (1-100)"),
    min_quotazione: int | None = Query(
        None, ge=13, le=17, description="Minimum artist price filter"
    ),
    max_quotazione: int | None = Query(
        None, ge=13, le=17, description="Maximum artist price filter"
    ),
    db: Session = Depends(get_db),
):
    """Get all artists with optional price filters and ML predictions"""
    logger.info(
        f"Fetching artists with filters - skip: {skip}, limit: {limit}, "
        f"min_quotazione: {min_quotazione}, max_quotazione: {max_quotazione}"
    )

    query = db.query(ArtistaModel).options(joinedload(ArtistaModel.predizione_2026))

    if min_quotazione is not None:
        query = query.filter(ArtistaModel.quotazione_2026 >= min_quotazione)
        logger.debug(f"Applied min_quotazione filter: {min_quotazione}")
    if max_quotazione is not None:
        query = query.filter(ArtistaModel.quotazione_2026 <= max_quotazione)
        logger.debug(f"Applied max_quotazione filter: {max_quotazione}")

    artisti = query.offset(skip).limit(limit).all()
    logger.info(f"Retrieved {len(artisti)} artists")

    return artisti


@router.get("/{artista_id}", response_model=ArtistaWithStorico)
def get_artista(artista_id: int, db: Session = Depends(get_db)):
    """Get artist details with history and predictions"""
    logger.info(f"Fetching artist details for ID: {artista_id}")

    artista = (
        db.query(ArtistaModel)
        .options(
            joinedload(ArtistaModel.edizioni_fantasanremo),
            joinedload(ArtistaModel.caratteristiche),
            joinedload(ArtistaModel.predizione_2026),
        )
        .filter(ArtistaModel.id == artista_id)
        .first()
    )

    if not artista:
        logger.warning(f"Artist not found with ID: {artista_id}")
        raise HTTPException(status_code=404, detail="Artista non trovato")

    logger.debug(f"Retrieved artist: {artista.nome} (ID: {artista_id})")
    return artista
