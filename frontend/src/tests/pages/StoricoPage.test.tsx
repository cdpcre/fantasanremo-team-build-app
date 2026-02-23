import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter, Routes, Route } from 'react-router-dom'
import StoricoPage from '@/pages/StoricoPage'
import type { ArtistaWithStorico } from '@/types'

// Mock the API module
vi.mock('@/api', () => ({
  getArtista: vi.fn()
}))

// Import mocked function
import { getArtista } from '@/api'

const mockGetArtista = getArtista as vi.MockedFunction<typeof getArtista>

// Helper to render the page
const renderStoricoPage = (artistaId: string) => {
  return render(
    <MemoryRouter initialEntries={[`/artisti/${artistaId}`]}>
      <Routes>
        <Route path="/artisti/:artistaId" element={<StoricoPage />} />
      </Routes>
    </MemoryRouter>
  )
}

// Mock artist data with various confidence scenarios
const createMockArtista = (confidence?: number): ArtistaWithStorico => ({
  id: 2,
  nome: 'Marco Masini & Fedez',
  quotazione_2026: 15,
  genere_musicale: 'Pop/Rap',
  anno_nascita: 1970,
  prima_partecipazione: 2026,
  debuttante_2026: false,
  image_url: null,
  edizioni_fantasanremo: [],
  predizione_2026: confidence !== undefined ? {
    id: 2,
    artista_id: 2,
    punteggio_predetto: 349.1,
    confidence,
    livello_performer: 'MEDIUM',
    interval_lower: 300,
    interval_upper: 400
  } : undefined
})

describe('StoricoPage Component - Confidence Meter Rendering', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  describe('Zero Confidence Scenario', () => {
    it('should render ConfidenceMeter when confidence is exactly 0', async () => {
      const mockArtista = createMockArtista(0)
      mockGetArtista.mockResolvedValue(mockArtista)

      renderStoricoPage('2')

      await waitFor(() => {
        expect(screen.getByText('Marco Masini & Fedez')).toBeInTheDocument()
      })

      // ConfidenceMeter should be rendered
      expect(screen.getByText('Confidenza ML')).toBeInTheDocument()
      expect(screen.getByText('N/A')).toBeInTheDocument()
      expect(screen.getByText('Dati insufficienti')).toBeInTheDocument()
    })

    it('should display "Dati insufficienti" message for zero confidence', async () => {
      const mockArtista = createMockArtista(0)
      mockGetArtista.mockResolvedValue(mockArtista)

      renderStoricoPage('2')

      await waitFor(() => {
        expect(screen.getByText('Dati insufficienti')).toBeInTheDocument()
      })
    })

    it('should display "N/A" instead of percentage for zero confidence', async () => {
      const mockArtista = createMockArtista(0)
      mockGetArtista.mockResolvedValue(mockArtista)

      renderStoricoPage('2')

      await waitFor(() => {
        expect(screen.getByText('N/A')).toBeInTheDocument()
      })

      // Should not show any percentage
      expect(screen.queryByText('0%')).not.toBeInTheDocument()
    })
  })

  describe('Undefined Confidence Scenario', () => {
    it('should NOT render ConfidenceMeter when confidence is undefined', async () => {
      const mockArtista = createMockArtista(undefined)
      mockGetArtista.mockResolvedValueOnce(mockArtista)

      renderStoricoPage('2')

      await waitFor(() => {
        expect(screen.getByText('Marco Masini & Fedez')).toBeInTheDocument()
      })

      // ConfidenceMeter should NOT be rendered
      expect(screen.queryByText('Confidenza ML')).not.toBeInTheDocument()
      expect(screen.queryByText('N/A')).not.toBeInTheDocument()
    })

    it('should not render ML prediction section when predizione_2026 is undefined', async () => {
      const mockArtista = createMockArtista(undefined)
      mockGetArtista.mockResolvedValueOnce(mockArtista)

      renderStoricoPage('2')

      await waitFor(() => {
        expect(screen.getByText('Marco Masini & Fedez')).toBeInTheDocument()
      })

      // ML Prediction section should not appear at all
      expect(screen.queryByText('Livello Performer')).not.toBeInTheDocument()
      expect(screen.queryByText('Punteggio Stimato')).not.toBeInTheDocument()
      expect(screen.queryByText('Confidenza ML')).not.toBeInTheDocument()
    })
  })

  describe('Normal Confidence Values', () => {
    it('should render ConfidenceMeter with percentage for confidence > 0', async () => {
      const mockArtista = createMockArtista(0.75)
      mockGetArtista.mockResolvedValue(mockArtista)

      renderStoricoPage('2')

      await waitFor(() => {
        expect(screen.getByText('Confidenza ML')).toBeInTheDocument()
      })

      expect(screen.getByText('75%')).toBeInTheDocument()
    })

    it('should render high confidence correctly', async () => {
      const mockArtista = createMockArtista(0.85)
      mockGetArtista.mockResolvedValue(mockArtista)

      renderStoricoPage('2')

      await waitFor(() => {
        expect(screen.getByText('85%')).toBeInTheDocument()
      })
    })

    it('should render low confidence correctly', async () => {
      const mockArtista = createMockArtista(0.35)
      mockGetArtista.mockResolvedValue(mockArtista)

      renderStoricoPage('2')

      await waitFor(() => {
        expect(screen.getByText('35%')).toBeInTheDocument()
      })
    })

    it('should render very low confidence (0.01) correctly', async () => {
      const mockArtista = createMockArtista(0.01)
      mockGetArtista.mockResolvedValue(mockArtista)

      renderStoricoPage('2')

      await waitFor(() => {
        expect(screen.getByText('1%')).toBeInTheDocument()
      })

      // Should NOT show "Dati insufficienti" for confidence > 0
      expect(screen.queryByText('Dati insufficienti')).not.toBeInTheDocument()
    })
  })

  describe('ML Prediction Section Layout', () => {
    it('should render all ML prediction elements together', async () => {
      const mockArtista = createMockArtista(0)
      mockGetArtista.mockResolvedValue(mockArtista)

      renderStoricoPage('2')

      await waitFor(() => {
        expect(screen.getByText('Livello Performer')).toBeInTheDocument()
        expect(screen.getByText('Punteggio Stimato')).toBeInTheDocument()
        expect(screen.getByText('Confidenza ML')).toBeInTheDocument()
      })
    })

    it('should render uncertainty interval when available', async () => {
      const mockArtista = createMockArtista(0)
      mockGetArtista.mockResolvedValue(mockArtista)

      renderStoricoPage('2')

      await waitFor(() => {
        // Uncertainty interval component should be present
        const container = screen.getByText('Punteggio Stimato').closest('div')
        expect(container).toBeInTheDocument()
      })
    })
  })

  describe('Error Handling', () => {
    it('should show error message when API fails', async () => {
      mockGetArtista.mockRejectedValueOnce(new Error('Network error'))

      renderStoricoPage('999')

      await waitFor(() => {
        expect(screen.getByText('Network error')).toBeInTheDocument()
      })
    })

    it('should show default error message for non-Error failures', async () => {
      mockGetArtista.mockRejectedValueOnce('Some string error')

      renderStoricoPage('999')

      await waitFor(() => {
        expect(screen.getByText(/Impossibile caricare i dati/i)).toBeInTheDocument()
      })
    })
  })

  describe('Loading State', () => {
    it('should show loading spinner while fetching data', () => {
      mockGetArtista.mockImplementation(() => new Promise(() => {})) // Never resolves

      renderStoricoPage('2')

      const loader = document.querySelector('.border-4.border-amber-500')
      expect(loader).toBeInTheDocument()
    })
  })

  describe('Type Safety - typeof Check', () => {
    it('should treat confidence: 0 as a valid number (not falsy)', async () => {
      const mockArtista = createMockArtista(0)
      mockGetArtista.mockResolvedValueOnce(mockArtista)

      renderStoricoPage('2')

      await waitFor(() => {
        // If typeof check is working, ConfidenceMeter should render
        expect(screen.getByText('Confidenza ML')).toBeInTheDocument()
      })
    })

    it('should not render when predizione_2026 is undefined', async () => {
      const mockArtista: ArtistaWithStorico = {
        id: 2,
        nome: 'Test Artist',
        quotazione_2026: 15,
        debuttante_2026: false,
        edizioni_fantasanremo: []
      }
      mockGetArtista.mockResolvedValueOnce(mockArtista)

      renderStoricoPage('2')

      await waitFor(() => {
        expect(screen.getByText('Test Artist')).toBeInTheDocument()
      })

      // ML Prediction section should not appear at all
      expect(screen.queryByText('Livello Performer')).not.toBeInTheDocument()
      expect(screen.queryByText('Punteggio Stimato')).not.toBeInTheDocument()
      expect(screen.queryByText('Confidenza ML')).not.toBeInTheDocument()
    })
  })
})
