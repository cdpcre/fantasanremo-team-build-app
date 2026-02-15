export interface Artista {
  id: number
  nome: string
  quotazione_2026: number
  genere_musicale?: string
  anno_nascita?: number
  prima_partecipazione?: number
  debuttante_2026: boolean
  image_url?: string | null
}

export interface EdizioneFantaSanremo {
  id: number
  artista_id: number
  anno: number
  punteggio_finale?: number
  posizione?: number
  quotazione_baudi?: number
}

export interface Predizione2026 {
  id: number
  artista_id: number
  punteggio_predetto: number
  confidence?: number
  livello_performer: 'HIGH' | 'MEDIUM' | 'LOW' | 'DEBUTTANTE'
  interval_lower?: number
  interval_upper?: number
}

export interface ArtistaWithPredizione extends Artista {
  predizione_2026?: Predizione2026
}

export interface ArtistaWithStorico extends Artista {
  edizioni_fantasanremo: EdizioneFantaSanremo[]
  predizione_2026?: Predizione2026
}

export interface TeamValidateRequest {
  artisti_ids: number[]
  capitano_id: number
}

export interface TeamValidateResponse {
  valid: boolean
  message: string
  budget_totale: number
  budget_rimanente: number
  punteggio_simulato?: number
}

export interface TeamSimulateRequest {
  artisti_ids: number[]
  capitano_id: number
}

export interface TeamSimulateResponse {
  punteggio_totale: number
  punteggio_dettaglio: Array<{
    artista_id: number
    punteggio: number
    capitano: boolean
  }>
  punteggio_capitano: number
  punteggio_titolari: number
}
