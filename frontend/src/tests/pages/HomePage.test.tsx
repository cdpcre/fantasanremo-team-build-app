import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import HomePage from '@/pages/HomePage'

describe('HomePage Component', () => {
  const renderHomePage = () => {
    return render(
      <MemoryRouter>
        <HomePage />
      </MemoryRouter>
    )
  }

  describe('Hero Section', () => {
    it('should render welcome message', () => {
      renderHomePage()

      // The hero title is split across multiple elements
      const headingElements = screen.getAllByRole('heading', { level: 2 })
      const welcomeText = headingElements.map(el => el.textContent).join(' ')
      expect(welcomeText).toContain('FantaSanremo')
      expect(welcomeText).toContain('Team Builder 2026!')
    })

    it('should render description', () => {
      renderHomePage()

      expect(
        screen.getByText(/Costruisci la tua squadra ideale con i 30 Big artisti in gara/i)
      ).toBeInTheDocument()
    })

    it('should render CTA buttons', () => {
      renderHomePage()

      expect(screen.getByText('Crea la tua squadra')).toBeInTheDocument()
      expect(screen.getByText('Esplora gli artisti')).toBeInTheDocument()
    })

    it('should render Edizione badge', () => {
      renderHomePage()

      expect(screen.getByText('Edizione 2026')).toBeInTheDocument()
    })
  })

  describe('Stats Cards', () => {
    it('should render all stat cards', () => {
      renderHomePage()

      expect(screen.getByText('Artisti in gara')).toBeInTheDocument()
      expect(screen.getByText('Budget disponibile')).toBeInTheDocument()
      expect(screen.getByText('Artisti per squadra')).toBeInTheDocument()
      expect(screen.getByText('Date Festival')).toBeInTheDocument()
    })

    it('should display correct stat values', () => {
      renderHomePage()

      expect(screen.getByText('30')).toBeInTheDocument() // Artisti
      expect(screen.getByText('100')).toBeInTheDocument() // Budget
      expect(screen.getByText('7')).toBeInTheDocument() // Per squadra
      expect(screen.getByText('24-28 Feb')).toBeInTheDocument() // Date
    })

    it('should display stat subtitles', () => {
      renderHomePage()

      expect(screen.getByText('Big 2026')).toBeInTheDocument()
      expect(screen.getByText('Baudi')).toBeInTheDocument()
      expect(screen.getByText('5 titolari + 2 riserve')).toBeInTheDocument()
      expect(screen.getByText('2026')).toBeInTheDocument()
    })

    it('should have stat card icons (using lucide-react SVG)', () => {
      const { container } = renderHomePage()

      // Check for SVG icons from lucide-react
      const svgs = container.querySelectorAll('svg')
      expect(svgs.length).toBeGreaterThan(0)
    })
  })

  describe('Quick Links', () => {
    it('should render link cards', () => {
      renderHomePage()

      expect(screen.getByText('Lista Artisti')).toBeInTheDocument()
      expect(screen.getByText('Team Builder')).toBeInTheDocument()
    })

    it('should render info card', () => {
      renderHomePage()

      expect(screen.getByText('Regolamento 2026')).toBeInTheDocument()
      expect(
        screen.getByText(/Budget 100 Baudi, 7 artisti \(5 titolari \+ 2 riserve\)/i)
      ).toBeInTheDocument()
      expect(screen.getByText(/Il capitano vale doppio!/i)).toBeInTheDocument()
    })

    it('should have correct link descriptions', () => {
      renderHomePage()

      expect(
        screen.getByText('Visualizza tutti i 30 Big con quotazioni e predizioni ML')
      ).toBeInTheDocument()
      expect(screen.getByText('Crea la tua squadra e simula il punteggio con l\'algoritmo')).toBeInTheDocument()
    })

    it('should have working links', () => {
      renderHomePage()

      const artistiLink = screen.getByText('Lista Artisti').closest('a')
      const teamBuilderLink = screen.getByText('Team Builder').closest('a')

      expect(artistiLink).toHaveAttribute('href', '/artisti')
      expect(teamBuilderLink).toHaveAttribute('href', '/team-builder')
    })

    it('should render Accesso Rapido section', () => {
      renderHomePage()

      expect(screen.getByText('Accesso Rapido')).toBeInTheDocument()
    })
  })

  describe('Albo d\'Oro', () => {
    it('should render albo d\'oro section', () => {
      renderHomePage()

      expect(screen.getByText('Albo d\'Oro')).toBeInTheDocument()
      expect(screen.getByText('Storico vincitori del FantaSanremo')).toBeInTheDocument()
    })

    it('should render table headers', () => {
      renderHomePage()

      expect(screen.getByText('Edizione')).toBeInTheDocument()
      expect(screen.getByText('Anno')).toBeInTheDocument()
      expect(screen.getByText('Squadre')).toBeInTheDocument()
      expect(screen.getByText('Vincitore')).toBeInTheDocument()
      expect(screen.getByText('Punteggio')).toBeInTheDocument()
    })

    it('should render all winners', () => {
      renderHomePage()

      expect(screen.getByText('Olly')).toBeInTheDocument()
      expect(screen.getByText('La Sad')).toBeInTheDocument()
      expect(screen.getByText('Marco Mengoni')).toBeInTheDocument()
      expect(screen.getByText('Emma')).toBeInTheDocument()
      expect(screen.getByText('Måneskin')).toBeInTheDocument()
    })

    it('should render correct scores', () => {
      renderHomePage()

      expect(screen.getByText('475')).toBeInTheDocument()
      expect(screen.getByText('486')).toBeInTheDocument()
      expect(screen.getByText('670')).toBeInTheDocument()
      expect(screen.getByText('525')).toBeInTheDocument()
      expect(screen.getByText('315')).toBeInTheDocument()
    })

    it('should render edition numbers', () => {
      renderHomePage()

      expect(screen.getByText('5ª')).toBeInTheDocument()
      expect(screen.getByText('4ª')).toBeInTheDocument()
      expect(screen.getByText('3ª')).toBeInTheDocument()
      expect(screen.getByText('2ª')).toBeInTheDocument()
      expect(screen.getByText('1ª')).toBeInTheDocument()
    })

    it('should highlight special row with Record badge', () => {
      renderHomePage()

      expect(screen.getByText('Record')).toBeInTheDocument()
    })

    it('should render team participation numbers', () => {
      renderHomePage()

      expect(screen.getByText('5.09M')).toBeInTheDocument()
      expect(screen.getByText('4.20M')).toBeInTheDocument()
      expect(screen.getByText('4.21M')).toBeInTheDocument()
      expect(screen.getByText('500K')).toBeInTheDocument()
      expect(screen.getByText('47K')).toBeInTheDocument()
    })
  })

  describe('ML Feature Highlight Section', () => {
    it('should render ML feature section', () => {
      renderHomePage()

      expect(screen.getByText('Machine Learning')).toBeInTheDocument()
      expect(screen.getByText(/Predizioni basate su 50\+ feature/i)).toBeInTheDocument()
    })

    it('should render ML stats', () => {
      renderHomePage()

      expect(screen.getByText('50+')).toBeInTheDocument()
      expect(screen.getByText('108')).toBeInTheDocument()
      expect(screen.getByText('5')).toBeInTheDocument()
    })

    it('should render Feature label', () => {
      renderHomePage()

      expect(screen.getByText('Feature')).toBeInTheDocument()
      expect(screen.getByText('Samples')).toBeInTheDocument()
      expect(screen.getByText('Modelli')).toBeInTheDocument()
    })

    it('should have Scopri di più link', () => {
      renderHomePage()

      expect(screen.getByText('Scopri di più')).toBeInTheDocument()
    })
  })

  describe('Layout and Styling', () => {
    it('should have proper spacing between sections', () => {
      const { container } = renderHomePage()

      const sections = container.querySelectorAll('section')
      expect(sections.length).toBeGreaterThan(0)
    })

    it('should use grid layout for stats cards', () => {
      const { container } = renderHomePage()

      const sections = container.querySelectorAll('section')
      const statsSection = Array.from(sections).find((section) =>
        section.textContent?.includes('Artisti in gara')
      )
      expect(statsSection).toHaveClass('grid')
    })
  })

  describe('Accessibility', () => {
    it('should have proper heading hierarchy', () => {
      renderHomePage()

      // The hero heading text is split across multiple lines
      const h2Elements = screen.getAllByRole('heading', { level: 2 })
      expect(h2Elements.length).toBeGreaterThan(0)
      const headingText = h2Elements.map(el => el.textContent).join(' ')
      expect(headingText).toContain('FantaSanremo')
      expect(headingText).toContain('Team Builder 2026!')
    })

    it('should have icon elements present', () => {
      const { container } = renderHomePage()

      // Check for SVG icons from lucide-react
      const svgs = container.querySelectorAll('svg')
      expect(svgs.length).toBeGreaterThan(5) // Multiple icons throughout the page
    })
  })
})
