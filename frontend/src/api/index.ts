import axios from 'axios'
import type { ArtistaWithPredizione, ArtistaWithStorico, EdizioneFantaSanremo, Predizione2026, TeamValidateRequest, TeamValidateResponse, TeamSimulateRequest, TeamSimulateResponse } from '@/types'

const rawApiBaseUrl = import.meta.env.VITE_API_URL?.trim() ?? ''
const API_BASE_URL = rawApiBaseUrl
  .replace(/\/+$/, '')
  .replace(/\/api$/, '')
const API_MODE = import.meta.env.VITE_API_MODE || 'remote'
const USE_LOCAL_DATA = API_MODE === 'local'
const MAX_BUDGET = 100

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
})

interface LocalSnapshot {
  generated_at: string
  source_db: string
  artisti: ArtistaWithStorico[]
}

let snapshotPromise: Promise<LocalSnapshot> | null = null

const getLocalSnapshot = async (): Promise<LocalSnapshot> => {
  if (!snapshotPromise) {
    snapshotPromise = fetch('/data/vercel_snapshot.json')
      .then(async response => {
        if (!response.ok) {
          throw new Error(`Static data not available (${response.status})`)
        }
        return response.json() as Promise<LocalSnapshot>
      })
  }
  return snapshotPromise
}

const getLocalPredizioni = (artisti: ArtistaWithStorico[]): Predizione2026[] => {
  return artisti
    .map(artista => artista.predizione_2026)
    .filter((predizione): predizione is Predizione2026 => !!predizione)
}

// Artisti API
export const getArtisti = async (params?: { min_quotazione?: number; max_quotazione?: number; limit?: number }): Promise<ArtistaWithPredizione[]> => {
  if (USE_LOCAL_DATA) {
    const snapshot = await getLocalSnapshot()
    const min = params?.min_quotazione
    const max = params?.max_quotazione
    const limit = params?.limit ?? 100

    let result = snapshot.artisti

    if (typeof min === 'number') {
      result = result.filter(artista => artista.quotazione_2026 >= min)
    }
    if (typeof max === 'number') {
      result = result.filter(artista => artista.quotazione_2026 <= max)
    }

    return result
      .slice(0, limit)
      .map(artista => ({
        id: artista.id,
        nome: artista.nome,
        quotazione_2026: artista.quotazione_2026,
        genere_musicale: artista.genere_musicale,
        anno_nascita: artista.anno_nascita,
        prima_partecipazione: artista.prima_partecipazione,
        debuttante_2026: artista.debuttante_2026,
        image_url: artista.image_url,
        predizione_2026: artista.predizione_2026
      }))
  }

  const response = await api.get('/api/artisti', {
    params: {
      limit: 100,
      ...params,
    },
  })
  return response.data
}

export const getArtista = async (id: number): Promise<ArtistaWithStorico> => {
  if (USE_LOCAL_DATA) {
    const snapshot = await getLocalSnapshot()
    const artista = snapshot.artisti.find(item => item.id === id)
    if (!artista) {
      throw new Error('Artista non trovato')
    }
    return artista
  }

  const response = await api.get(`/api/artisti/${id}`)
  return response.data
}

// Storico API
export const getStoricoAggregate = async (anno?: number): Promise<EdizioneFantaSanremo[]> => {
  if (USE_LOCAL_DATA) {
    const snapshot = await getLocalSnapshot()
    let storico = snapshot.artisti.flatMap(artista => artista.edizioni_fantasanremo || [])
    if (typeof anno === 'number') {
      storico = storico.filter(edizione => edizione.anno === anno)
    }
    return storico
  }

  const response = await api.get('/api/storico', { params: { anno } })
  return response.data
}

export const getStoricoArtista = async (artistaId: number): Promise<EdizioneFantaSanremo[]> => {
  if (USE_LOCAL_DATA) {
    const snapshot = await getLocalSnapshot()
    const artista = snapshot.artisti.find(item => item.id === artistaId)
    return artista?.edizioni_fantasanremo || []
  }

  const response = await api.get(`/api/storico/artista/${artistaId}`)
  return response.data
}

// Predizioni API
export const getPredizioni = async (): Promise<Predizione2026[]> => {
  if (USE_LOCAL_DATA) {
    const snapshot = await getLocalSnapshot()
    return getLocalPredizioni(snapshot.artisti)
  }

  const response = await api.get('/api/predizioni')
  return response.data
}

export const getPredizioneArtista = async (artistaId: number): Promise<Predizione2026> => {
  if (USE_LOCAL_DATA) {
    const snapshot = await getLocalSnapshot()
    const artista = snapshot.artisti.find(item => item.id === artistaId)
    if (!artista?.predizione_2026) {
      throw new Error('Predizione non trovata')
    }
    return artista.predizione_2026
  }

  const response = await api.get(`/api/predizioni/${artistaId}`)
  return response.data
}

// Team API
export const validateTeam = async (request: TeamValidateRequest): Promise<TeamValidateResponse> => {
  if (USE_LOCAL_DATA) {
    const snapshot = await getLocalSnapshot()
    const uniqueArtisti = new Set(request.artisti_ids)

    if (uniqueArtisti.size !== 7) {
      return {
        valid: false,
        message: 'La squadra deve contenere 7 artisti unici',
        budget_totale: 0,
        budget_rimanente: MAX_BUDGET
      }
    }

    if (!uniqueArtisti.has(request.capitano_id)) {
      return {
        valid: false,
        message: 'Il capitano deve essere uno dei 7 artisti',
        budget_totale: 0,
        budget_rimanente: MAX_BUDGET
      }
    }

    const selected = request.artisti_ids
      .map(id => snapshot.artisti.find(artista => artista.id === id))
      .filter((artista): artista is ArtistaWithStorico => !!artista)

    if (selected.length !== 7) {
      return {
        valid: false,
        message: 'Alcuni artisti non sono stati trovati',
        budget_totale: 0,
        budget_rimanente: MAX_BUDGET
      }
    }

    const budgetTotale = selected.reduce((total, artista) => total + artista.quotazione_2026, 0)

    if (budgetTotale > MAX_BUDGET) {
      return {
        valid: false,
        message: `Budget superato! Totale: ${budgetTotale} Baudi (max: ${MAX_BUDGET})`,
        budget_totale: budgetTotale,
        budget_rimanente: MAX_BUDGET - budgetTotale
      }
    }

    const punteggi = new Map<number, number>(
      selected.map(artista => [artista.id, artista.predizione_2026?.punteggio_predetto || 0])
    )
    const punteggioTitolari = request.artisti_ids.reduce((total, id) => total + (punteggi.get(id) || 0), 0)
    const punteggioCapitano = punteggi.get(request.capitano_id) || 0

    return {
      valid: true,
      message: 'Squadra valida!',
      budget_totale: budgetTotale,
      budget_rimanente: MAX_BUDGET - budgetTotale,
      punteggio_simulato: Math.round((punteggioTitolari + punteggioCapitano) * 100) / 100
    }
  }

  const response = await api.post('/api/team/validate', request)
  return response.data
}

export const simulateTeam = async (request: TeamSimulateRequest): Promise<TeamSimulateResponse> => {
  if (USE_LOCAL_DATA) {
    const snapshot = await getLocalSnapshot()

    if (!request.artisti_ids.includes(request.capitano_id)) {
      throw new Error('Il capitano deve essere tra i titolari')
    }

    const punteggioDettaglio = request.artisti_ids.map(artistaId => {
      const artista = snapshot.artisti.find(item => item.id === artistaId)
      return {
        artista_id: artistaId,
        punteggio: artista?.predizione_2026?.punteggio_predetto || 0,
        capitano: artistaId === request.capitano_id
      }
    })

    const punteggioTitolari = punteggioDettaglio.reduce((total, entry) => total + entry.punteggio, 0)
    const punteggioCapitano = punteggioDettaglio.find(entry => entry.capitano)?.punteggio || 0

    return {
      punteggio_totale: Math.round((punteggioTitolari + punteggioCapitano) * 100) / 100,
      punteggio_dettaglio: punteggioDettaglio,
      punteggio_capitano: Math.round(punteggioCapitano * 100) / 100,
      punteggio_titolari: Math.round(punteggioTitolari * 100) / 100
    }
  }

  const response = await api.post('/api/team/simulate', request)
  return response.data
}

export default api
