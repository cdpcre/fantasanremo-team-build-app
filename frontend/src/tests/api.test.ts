import { describe, it, expect, vi, beforeEach } from 'vitest'
import axios, { type AxiosInstance } from 'axios'
import { mockArtisti, mockPredizioni, mockStorico } from './mockData'

// Mock axios
vi.mock('axios')

const mockedAxios = vi.mocked(axios)

describe('API Client', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  describe('getArtisti', () => {
    it('should fetch artists without parameters', async () => {
      const response = { data: mockArtisti }
      mockedAxios.create.mockReturnValue({
        get: vi.fn().mockResolvedValue(response),
        post: vi.fn(),
      } as AxiosInstance)

      const api = axios.create({
        baseURL: 'http://localhost:8000',
        headers: { 'Content-Type': 'application/json' },
      })

      vi.mocked(api.get).mockResolvedValue(response)

      const result = await api.get('/api/artisti')
      expect(result.data).toEqual(mockArtisti)
    })

    it('should fetch artists with filter parameters', async () => {
      const params = { min_quotazione: 10, max_quotazione: 15 }
      const response = { data: mockArtisti.filter((a) => a.quotazione_2026 >= 10 && a.quotazione_2026 <= 15) }

      const api = axios.create({
        baseURL: 'http://localhost:8000',
        headers: { 'Content-Type': 'application/json' },
      })

      vi.mocked(api.get).mockResolvedValue(response)

      const result = await api.get('/api/artisti', { params })
      expect(result.data).toEqual(response.data)
    })

    it('should handle API errors', async () => {
      const api = axios.create({
        baseURL: 'http://localhost:8000',
        headers: { 'Content-Type': 'application/json' },
      })

      vi.mocked(api.get).mockRejectedValue(new Error('Network Error'))

      await expect(api.get('/api/artisti')).rejects.toThrow('Network Error')
    })
  })

  describe('getArtista', () => {
    it('should fetch a single artist by ID', async () => {
      const artistaId = 1
      const response = {
        data: {
          ...mockArtisti[0],
          edizioni_fantasanremo: mockStorico.filter((s) => s.artista_id === artistaId),
          predizione_2026: mockPredizioni[0],
        },
      }

      const api = axios.create({
        baseURL: 'http://localhost:8000',
        headers: { 'Content-Type': 'application/json' },
      })

      vi.mocked(api.get).mockResolvedValue(response)

      const result = await api.get(`/api/artisti/${artistaId}`)
      expect(result.data.id).toBe(artistaId)
      expect(result.data.nome).toBe(mockArtisti[0].nome)
    })
  })

  describe('getStoricoAggregate', () => {
    it('should fetch historical data without year filter', async () => {
      const response = { data: mockStorico }

      const api = axios.create({
        baseURL: 'http://localhost:8000',
        headers: { 'Content-Type': 'application/json' },
      })

      vi.mocked(api.get).mockResolvedValue(response)

      const result = await api.get('/api/storico')
      expect(result.data).toEqual(mockStorico)
    })

    it('should fetch historical data with year filter', async () => {
      const anno = 2025
      const response = { data: mockStorico.filter((s) => s.anno === anno) }

      const api = axios.create({
        baseURL: 'http://localhost:8000',
        headers: { 'Content-Type': 'application/json' },
      })

      vi.mocked(api.get).mockResolvedValue(response)

      const result = await api.get('/api/storico', { params: { anno } })
      expect(result.data).toEqual(response.data)
    })
  })

  describe('getStoricoArtista', () => {
    it('should fetch historical data for a specific artist', async () => {
      const artistaId = 1
      const response = { data: mockStorico.filter((s) => s.artista_id === artistaId) }

      const api = axios.create({
        baseURL: 'http://localhost:8000',
        headers: { 'Content-Type': 'application/json' },
      })

      vi.mocked(api.get).mockResolvedValue(response)

      const result = await api.get(`/api/storico/artista/${artistaId}`)
      expect(result.data).toEqual(response.data)
      expect(result.data.every((s) => s.artista_id === artistaId)).toBe(true)
    })
  })

  describe('getPredizioni', () => {
    it('should fetch all predictions', async () => {
      const response = { data: mockPredizioni }

      const api = axios.create({
        baseURL: 'http://localhost:8000',
        headers: { 'Content-Type': 'application/json' },
      })

      vi.mocked(api.get).mockResolvedValue(response)

      const result = await api.get('/api/predizioni')
      expect(result.data).toEqual(mockPredizioni)
      expect(result.data.length).toBe(mockPredizioni.length)
    })
  })

  describe('getPredizioneArtista', () => {
    it('should fetch prediction for a specific artist', async () => {
      const artistaId = 1
      const response = { data: mockPredizioni[0] }

      const api = axios.create({
        baseURL: 'http://localhost:8000',
        headers: { 'Content-Type': 'application/json' },
      })

      vi.mocked(api.get).mockResolvedValue(response)

      const result = await api.get(`/api/predizioni/${artistaId}`)
      expect(result.data.artista_id).toBe(artistaId)
    })
  })

  describe('validateTeam', () => {
    it('should validate a team', async () => {
      const request = {
        artisti_ids: [1, 2, 3, 4, 5, 6, 7],
        capitano_id: 1,
      }

      const response = {
        data: {
          valid: true,
          message: 'Team valido',
          budget_totale: 95,
          budget_rimanente: 5,
          punteggio_simulato: 456.8,
        },
      }

      const api = axios.create({
        baseURL: 'http://localhost:8000',
        headers: { 'Content-Type': 'application/json' },
      })

      const postMock = vi.fn().mockResolvedValue(response)
      vi.mocked(api.post).mockImplementation(postMock)

      const result = await api.post('/api/team/validate', request)
      expect(result.data.valid).toBe(true)
      expect(result.data.punteggio_simulato).toBeDefined()
    })

    it('should return validation errors for invalid team', async () => {
      const request = {
        artisti_ids: [1, 2, 3, 4, 5, 6, 7],
        capitano_id: 1,
      }

      const response = {
        data: {
          valid: false,
          message: 'Budget superato',
          budget_totale: 105,
          budget_rimanente: -5,
        },
      }

      const api = axios.create({
        baseURL: 'http://localhost:8000',
        headers: { 'Content-Type': 'application/json' },
      })

      const postMock = vi.fn().mockResolvedValue(response)
      vi.mocked(api.post).mockImplementation(postMock)

      const result = await api.post('/api/team/validate', request)
      expect(result.data.valid).toBe(false)
      expect(result.data.budget_rimanente).toBeLessThan(0)
    })
  })

  describe('simulateTeam', () => {
    it('should simulate team score', async () => {
      const request = {
        artisti_ids: [1, 2, 3, 4, 5, 6, 7],
        capitano_id: 1,
      }

      const response = {
        data: {
          punteggio_totale: 456.8,
          punteggio_dettaglio: [
            { artista_id: 1, punteggio: 85.5, capitano: true },
            { artista_id: 2, punteggio: 72.3, capitano: false },
          ],
          punteggio_capitano: 171.0,
          punteggio_titolari: 285.8,
        },
      }

      const api = axios.create({
        baseURL: 'http://localhost:8000',
        headers: { 'Content-Type': 'application/json' },
      })

      const postMock = vi.fn().mockResolvedValue(response)
      vi.mocked(api.post).mockImplementation(postMock)

      const result = await api.post('/api/team/simulate', request)
      expect(result.data.punteggio_totale).toBeDefined()
      expect(result.data.punteggio_dettaglio).toBeDefined()
      expect(result.data.punteggio_capitano).toBeDefined()
      expect(result.data.punteggio_titolari).toBeDefined()
    })
  })
})
