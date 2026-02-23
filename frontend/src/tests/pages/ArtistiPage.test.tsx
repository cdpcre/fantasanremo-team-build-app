import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router-dom'
import ArtistiPage from '@/pages/ArtistiPage'
import * as api from '@/api'
import { mockArtistiWithPred } from '../mockData'

// Mock the API module
vi.mock('@/api')

describe('ArtistiPage Component', () => {
  const mockGetArtisti = vi.mocked(api.getArtisti)

  beforeEach(() => {
    vi.clearAllMocks()
  })

  const renderArtistiPage = () => {
    return render(
      <MemoryRouter>
        <ArtistiPage />
      </MemoryRouter>
    )
  }

  describe('Initial Render and Loading', () => {
    it('should show loading state initially', () => {
      mockGetArtisti.mockImplementation(() => new Promise(() => {})) // Never resolves

      renderArtistiPage()

      expect(screen.getByText('Caricamento...')).toBeInTheDocument()
    })

    it('should load and display artists', async () => {
      mockGetArtisti.mockResolvedValue(mockArtistiWithPred)

      renderArtistiPage()

      await waitFor(() => {
        expect(screen.getByText('🎤 Artisti 2026')).toBeInTheDocument()
      })

      expect(screen.getByText(`${mockArtistiWithPred.length} artisti`)).toBeInTheDocument()
    })

    it('should render artist cards with correct data', async () => {
      mockGetArtisti.mockResolvedValue(mockArtistiWithPred)

      renderArtistiPage()

      await waitFor(() => {
        expect(screen.getByText('Annalisa')).toBeInTheDocument()
      })

      expect(screen.getByText('Achille Lauro')).toBeInTheDocument()
      expect(screen.getByText('Francesca Michielin')).toBeInTheDocument()
    })

    it('should display artist quotations', async () => {
      mockGetArtisti.mockResolvedValue(mockArtistiWithPred)

      renderArtistiPage()

      await waitFor(() => {
        expect(screen.getByText('17')).toBeInTheDocument()
        expect(screen.getByText('16')).toBeInTheDocument()
      })
    })

    it('should display artist genres', async () => {
      mockGetArtisti.mockResolvedValue(mockArtistiWithPred)

      renderArtistiPage()

      await waitFor(() => {
        expect(screen.getAllByText('Pop').length).toBeGreaterThan(0)
        expect(screen.getByText('Rap')).toBeInTheDocument()
      })
    })
  })

  describe('Search Functionality', () => {
    it('should have search input field', async () => {
      mockGetArtisti.mockResolvedValue(mockArtistiWithPred)

      renderArtistiPage()

      await waitFor(() => {
        expect(screen.getByPlaceholderText('Cerca artista...')).toBeInTheDocument()
      })
    })

    it('should filter artists by search term', async () => {
      mockGetArtisti.mockResolvedValue(mockArtistiWithPred)

      renderArtistiPage()

      await waitFor(() => {
        expect(screen.getByText('Annalisa')).toBeInTheDocument()
      })

      const searchInput = screen.getByPlaceholderText('Cerca artista...')
      const user = userEvent.setup()

      await user.type(searchInput, 'Annalisa')

      await waitFor(() => {
        expect(screen.getByText('Annalisa')).toBeInTheDocument()
        expect(screen.queryByText('Achille Lauro')).not.toBeInTheDocument()
      })
    })

    it('should be case insensitive', async () => {
      mockGetArtisti.mockResolvedValue(mockArtistiWithPred)

      renderArtistiPage()

      await waitFor(() => {
        expect(screen.getByText('Annalisa')).toBeInTheDocument()
      })

      const searchInput = screen.getByPlaceholderText('Cerca artista...')
      const user = userEvent.setup()

      await user.type(searchInput, 'annalisa')

      await waitFor(() => {
        expect(screen.getByText('Annalisa')).toBeInTheDocument()
      })
    })

    it('should show no results message when search matches nothing', async () => {
      mockGetArtisti.mockResolvedValue(mockArtistiWithPred)

      renderArtistiPage()

      await waitFor(() => {
        expect(screen.getByText('Annalisa')).toBeInTheDocument()
      })

      const searchInput = screen.getByPlaceholderText('Cerca artista...')
      const user = userEvent.setup()

      await user.type(searchInput, 'NonExistentArtist')

      await waitFor(() => {
        expect(screen.getByText('Nessun artista trovato con i filtri selezionati')).toBeInTheDocument()
      })
    })
  })

  describe('Price Filter', () => {
    it('should have price filter dropdown', async () => {
      mockGetArtisti.mockResolvedValue(mockArtistiWithPred)

      renderArtistiPage()

      await waitFor(() => {
        expect(screen.getByText('Tutti i prezzi')).toBeInTheDocument()
      })
    })

    it('should filter by minimum price', async () => {
      mockGetArtisti.mockResolvedValue(mockArtistiWithPred)

      renderArtistiPage()

      await waitFor(() => {
        expect(screen.getByText('Annalisa')).toBeInTheDocument()
      })

      // Select 17 Baudi filter
      const select = screen.getByDisplayValue(/Tutti i prezzi/i)
      const user = userEvent.setup()

      await user.selectOptions(select, '17')

      await waitFor(() => {
        expect(mockGetArtisti).toHaveBeenCalledWith({ min_quotazione: 17 })
      })
    })

    it('should not call API on search change', async () => {
      mockGetArtisti.mockResolvedValue(mockArtistiWithPred)

      renderArtistiPage()

      await waitFor(() => {
        expect(screen.getByText('Annalisa')).toBeInTheDocument()
      })

      const callCount = mockGetArtisti.mock.calls.length

      const searchInput = screen.getByPlaceholderText('Cerca artista...')
      const user = userEvent.setup()

      await user.type(searchInput, 'test')

      // API should not be called again for search filter (it's client-side)
      expect(mockGetArtisti.mock.calls.length).toBe(callCount)
    })
  })

  describe('Prediction Badges', () => {
    it('should display HIGH level badge', async () => {
      mockGetArtisti.mockResolvedValue(mockArtistiWithPred)

      renderArtistiPage()

      await waitFor(() => {
        expect(screen.getAllByText('HIGH').length).toBeGreaterThan(0)
      })
    })

    it('should display MEDIUM level badge', async () => {
      mockGetArtisti.mockResolvedValue(mockArtistiWithPred)

      renderArtistiPage()

      await waitFor(() => {
        expect(screen.getAllByText('MED').length).toBeGreaterThan(0)
      })
    })

    it('should display LOW level badge', async () => {
      mockGetArtisti.mockResolvedValue(mockArtistiWithPred)

      renderArtistiPage()

      await waitFor(() => {
        expect(screen.getAllByText('LOW').length).toBeGreaterThan(0)
      })
    })

    it('should display debuttantebadge', async () => {
      mockGetArtisti.mockResolvedValue(mockArtistiWithPred)

      renderArtistiPage()

      await waitFor(() => {
        expect(screen.getByText('DEBUTTANTE')).toBeInTheDocument()
      })
    })
  })

  describe('Predicted Scores', () => {
    it('should display predicted scores', async () => {
      mockGetArtisti.mockResolvedValue(mockArtistiWithPred)

      renderArtistiPage()

      await waitFor(() => {
        expect(screen.getAllByText('Punteggio stimato').length).toBeGreaterThan(0)
      })
    })

    it('should render predicted score values', async () => {
      mockGetArtisti.mockResolvedValue(mockArtistiWithPred)

      renderArtistiPage()

      await waitFor(() => {
        expect(screen.getByText('86')).toBeInTheDocument() // 85.5 rounded
      })
    })
  })

  describe('Navigation', () => {
    it('should have links to artist detail pages', async () => {
      mockGetArtisti.mockResolvedValue(mockArtistiWithPred)

      renderArtistiPage()

      await waitFor(() => {
        expect(screen.getByText('Annalisa')).toBeInTheDocument()
      })

      const artistCard = screen.getByText('Annalisa').closest('a')
      expect(artistCard).toHaveAttribute('href', '/storico/1')
    })
  })

  describe('Artist Count Display', () => {
    it('should display total artist count', async () => {
      mockGetArtisti.mockResolvedValue(mockArtistiWithPred)

      renderArtistiPage()

      await waitFor(() => {
        expect(screen.getByText(`${mockArtistiWithPred.length} artisti`)).toBeInTheDocument()
      })
    })

    it('should update count when filtering', async () => {
      mockGetArtisti.mockResolvedValue(mockArtistiWithPred)

      renderArtistiPage()

      await waitFor(() => {
        expect(screen.getByText(`${mockArtistiWithPred.length} artisti`)).toBeInTheDocument()
      })

      const searchInput = screen.getByPlaceholderText('Cerca artista...')
      const user = userEvent.setup()

      await user.type(searchInput, 'Annalisa')

      await waitFor(() => {
        expect(screen.getByText('1 artisti')).toBeInTheDocument()
      })
    })
  })

  describe('Error Handling', () => {
    it('should handle API errors gracefully', async () => {
      mockGetArtisti.mockRejectedValue(new Error('API Error'))

      // Spy on console.error
      const consoleSpy = vi.spyOn(console, 'error').mockImplementation(() => {})

      renderArtistiPage()

      await waitFor(() => {
        expect(consoleSpy).toHaveBeenCalled()
      })

      expect(screen.queryByText('Caricamento...')).not.toBeInTheDocument()

      consoleSpy.mockRestore()
    })
  })

  describe('Empty State', () => {
    it('should show empty state when no artists match filters', async () => {
      mockGetArtisti.mockResolvedValue([])

      renderArtistiPage()

      await waitFor(() => {
        expect(screen.getByText('Nessun artista trovato con i filtri selezionati')).toBeInTheDocument()
      })
    })
  })
})
