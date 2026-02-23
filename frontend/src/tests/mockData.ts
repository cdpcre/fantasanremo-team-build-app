import type { Artista, Predizione2026, EdizioneFantaSanremo, ArtistaWithStorico } from '@/types'

export const mockArtisti: Artista[] = [
  {
    id: 1,
    nome: 'Annalisa',
    quotazione_2026: 17,
    genere_musicale: 'Pop',
    anno_nascita: 1985,
    prima_partecipazione: 2021,
    debuttante_2026: false,
  },
  {
    id: 2,
    nome: 'Achille Lauro',
    quotazione_2026: 16,
    genere_musicale: 'Rap',
    anno_nascita: 1990,
    prima_partecipazione: 2020,
    debuttante_2026: false,
  },
  {
    id: 3,
    nome: 'Francesca Michielin',
    quotazione_2026: 15,
    genere_musicale: 'Pop',
    anno_nascita: 1992,
    prima_partecipazione: 2016,
    debuttante_2026: false,
  },
  {
    id: 4,
    nome: 'Nuovo Artista',
    quotazione_2026: 10,
    genere_musicale: 'Indie',
    anno_nascita: 2000,
    debuttante_2026: true,
  },
  {
    id: 5,
    nome: 'Max Pezzali',
    quotazione_2026: 14,
    genere_musicale: 'Pop Rock',
    anno_nascita: 1967,
    prima_partecipazione: 2023,
    debuttante_2026: false,
  },
  {
    id: 6,
    nome: 'Giorgia',
    quotazione_2026: 18,
    genere_musicale: 'Pop',
    anno_nascita: 1971,
    prima_partecipazione: 1995,
    debuttante_2026: false,
  },
  {
    id: 7,
    nome: 'Lorenzo Fragola',
    quotazione_2026: 12,
    genere_musicale: 'Pop',
    anno_nascita: 1995,
    prima_partecipazione: 2015,
    debuttante_2026: false,
  },
]

export const mockPredizioni: Predizione2026[] = [
  {
    id: 1,
    artista_id: 1,
    punteggio_predetto: 85.5,
    confidence: 0.85,
    livello_performer: 'HIGH',
  },
  {
    id: 2,
    artista_id: 2,
    punteggio_predetto: 72.3,
    confidence: 0.75,
    livello_performer: 'MEDIUM',
  },
  {
    id: 3,
    artista_id: 3,
    punteggio_predetto: 65.8,
    confidence: 0.70,
    livello_performer: 'MEDIUM',
  },
  {
    id: 4,
    artista_id: 4,
    punteggio_predetto: 45.2,
    confidence: 0.60,
    livello_performer: 'LOW',
  },
  {
    id: 5,
    artista_id: 5,
    punteggio_predetto: 58.9,
    confidence: 0.65,
    livello_performer: 'LOW',
  },
  {
    id: 6,
    artista_id: 6,
    punteggio_predetto: 68.2,
    confidence: 0.72,
    livello_performer: 'MEDIUM',
  },
  {
    id: 7,
    artista_id: 7,
    punteggio_predetto: 55.4,
    confidence: 0.62,
    livello_performer: 'LOW',
  },
]

export const mockStorico: EdizioneFantaSanremo[] = [
  {
    id: 1,
    artista_id: 1,
    anno: 2025,
    punteggio_finale: 78,
    posizione: 3,
    quotazione_baudi: 15,
  },
  {
    id: 2,
    artista_id: 2,
    anno: 2024,
    punteggio_finale: 82,
    posizione: 2,
    quotazione_baudi: 14,
  },
]

export const mockArtistaWithZeroConfidence: ArtistaWithStorico = {
  id: 2,
  nome: 'Marco Masini & Fedez',
  quotazione_2026: 15,
  genere_musicale: 'Pop/Rap',
  anno_nascita: 1970,
  prima_partecipazione: 2026,
  debuttante_2026: false,
  image_url: null,
  predizione_2026: {
    id: 2,
    artista_id: 2,
    punteggio_predetto: 349.1,
    confidence: 0,
    livello_performer: 'MEDIUM',
    interval_lower: 300,
    interval_upper: 400
  },
  edizioni_fantasanremo: []
}

export const mockArtistiWithPred = mockArtisti.map((artista) => ({
  ...artista,
  predizione_2026: mockPredizioni.find((p) => p.artista_id === artista.id),
}))
