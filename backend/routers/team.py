import logging

from database import get_db
from fastapi import APIRouter, Depends, HTTPException
from models import Artista as ArtistaModel
from models import Predizione2026
from schemas import (
    TeamSimulateRequest,
    TeamSimulateResponse,
    TeamValidateRequest,
    TeamValidateResponse,
)
from sqlalchemy.orm import Session

router = APIRouter()
logger = logging.getLogger(__name__)

MAX_BUDGET = 100


def calculate_team_score(artisti_ids: list[int], capitano_id: int, db: Session) -> float:
    """Calculate simulated team score"""
    logger.debug(f"Calculating team score - artists: {artisti_ids}, captain: {capitano_id}")

    predizioni = db.query(Predizione2026).filter(Predizione2026.artista_id.in_(artisti_ids)).all()

    pred_map = {p.artista_id: p.punteggio_predetto for p in predizioni}

    punteggio_titolari = sum(pred_map.get(aid, 0) for aid in artisti_ids)
    punteggio_capitano = pred_map.get(capitano_id, 0)

    # Capitano bonus: 2x multiplier
    punteggio_totale = punteggio_titolari + punteggio_capitano

    logger.debug(
        f"Score calculation - titolari: {punteggio_titolari}, "
        f"capitano_bonus: {punteggio_capitano}, total: {punteggio_totale}"
    )

    return punteggio_totale


@router.post("/validate", response_model=TeamValidateResponse)
def validate_team(request: TeamValidateRequest, db: Session = Depends(get_db)):
    """Validate team (budget 100 baudi, 7 artists)"""
    logger.info(f"Validating team - artists: {request.artisti_ids}, captain: {request.capitano_id}")

    # Check unique artists
    if len(set(request.artisti_ids)) != 7:
        logger.warning(f"Team validation failed: not 7 unique artists - {request.artisti_ids}")
        return TeamValidateResponse(
            valid=False,
            message="La squadra deve contenere 7 artisti unici",
            budget_totale=0,
            budget_rimanente=MAX_BUDGET,
        )

    # Check capitano in team
    if request.capitano_id not in request.artisti_ids:
        logger.warning(
            f"Team validation failed: captain {request.capitano_id} "
            f"not in team {request.artisti_ids}"
        )
        return TeamValidateResponse(
            valid=False,
            message="Il capitano deve essere uno dei 7 artisti",
            budget_totale=0,
            budget_rimanente=MAX_BUDGET,
        )

    # Get artists and calculate budget
    artisti = db.query(ArtistaModel).filter(ArtistaModel.id.in_(request.artisti_ids)).all()

    if len(artisti) != 7:
        logger.error(f"Team validation failed: only {len(artisti)}/7 artists found in database")
        return TeamValidateResponse(
            valid=False,
            message="Alcuni artisti non sono stati trovati",
            budget_totale=0,
            budget_rimanente=MAX_BUDGET,
        )

    budget_totale = sum(a.quotazione_2026 for a in artisti)
    logger.debug(f"Team budget calculated: {budget_totale} baudi")

    if budget_totale > MAX_BUDGET:
        logger.warning(f"Team validation failed: budget exceeded - {budget_totale} > {MAX_BUDGET}")
        return TeamValidateResponse(
            valid=False,
            message=f"Budget superato! Totale: {budget_totale} Baudi (max: {MAX_BUDGET})",
            budget_totale=budget_totale,
            budget_rimanente=MAX_BUDGET - budget_totale,
        )

    # Calculate simulated score
    punteggio_simulato = calculate_team_score(request.artisti_ids, request.capitano_id, db)

    logger.info(
        f"Team validated successfully - budget: {budget_totale}/{MAX_BUDGET}, "
        f"simulated_score: {round(punteggio_simulato, 2)}"
    )

    return TeamValidateResponse(
        valid=True,
        message="Squadra valida!",
        budget_totale=budget_totale,
        budget_rimanente=MAX_BUDGET - budget_totale,
        punteggio_simulato=round(punteggio_simulato, 2),
    )


@router.post("/simulate", response_model=TeamSimulateResponse)
def simulate_team(request: TeamSimulateRequest, db: Session = Depends(get_db)):
    """Simulate team score based on predictions"""
    logger.info(f"Simulating team - artists: {request.artisti_ids}, captain: {request.capitano_id}")

    if request.capitano_id not in request.artisti_ids:
        logger.warning(
            f"Simulation failed: captain {request.capitano_id} not in team {request.artisti_ids}"
        )
        raise HTTPException(status_code=400, detail="Il capitano deve essere tra i titolari")

    predizioni = (
        db.query(Predizione2026).filter(Predizione2026.artista_id.in_(request.artisti_ids)).all()
    )

    if len(predizioni) != 5:
        logger.error(f"Simulation failed: missing predictions - expected 5, got {len(predizioni)}")
        raise HTTPException(status_code=404, detail="Mancano predizioni per alcuni artisti")

    pred_map = {p.artista_id: p.punteggio_predetto for p in predizioni}

    dettaglio = []
    for aid in request.artisti_ids:
        dettaglio.append(
            {
                "artista_id": aid,
                "punteggio": pred_map.get(aid, 0),
                "capitano": aid == request.capitano_id,
            }
        )

    punteggio_titolari = sum(d["punteggio"] for d in dettaglio)
    punteggio_capitano = pred_map.get(request.capitano_id, 0)
    punteggio_totale = punteggio_titolari + punteggio_capitano

    logger.info(
        f"Team simulation completed - total_score: {round(punteggio_totale, 2)}, "
        f"titolari: {round(punteggio_titolari, 2)}, captain_bonus: {round(punteggio_capitano, 2)}"
    )

    return TeamSimulateResponse(
        punteggio_totale=round(punteggio_totale, 2),
        punteggio_dettaglio=dettaglio,
        punteggio_capitano=round(punteggio_capitano, 2),
        punteggio_titolari=round(punteggio_titolari, 2),
    )
