import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import { MemoryRouter, Routes, Route } from 'react-router-dom'
import Layout from '@/components/Layout'

describe('Layout Component', () => {
  const renderLayout = (initialEntries = ['/']) => {
    return render(
      <MemoryRouter initialEntries={initialEntries}>
        <Layout>
          <div>Test Content</div>
        </Layout>
        <Routes>
          <Route path="/" element={<div>Home Page</div>} />
          <Route path="/artisti" element={<div>Artisti Page</div>} />
          <Route path="/team-builder" element={<div>Team Builder Page</div>} />
        </Routes>
      </MemoryRouter>
    )
  }

  describe('Header', () => {
    it('should render the header with title', () => {
      renderLayout()

      expect(screen.getByText('FantaSanremo')).toBeInTheDocument()
      expect(screen.getByText('Team Builder 2026')).toBeInTheDocument()
    })

    it('should display festival dates', () => {
      renderLayout()

      expect(screen.getAllByText(/24-28 Febbraio 2026/i).length).toBeGreaterThanOrEqual(1)
    })

    it('should have logo with Sparkles icon', () => {
      const { container } = renderLayout()

      const sparklesIcon = container.querySelector('svg')
      expect(sparklesIcon).toBeInTheDocument()
    })
  })

  describe('Navigation', () => {
    it('should render all navigation links', () => {
      renderLayout()

      expect(screen.getByText('Home')).toBeInTheDocument()
      expect(screen.getByText('Artisti')).toBeInTheDocument()
      expect(screen.getByText('Team Builder')).toBeInTheDocument()
    })

    it('should highlight active route - Home', () => {
      renderLayout(['/'])

      // Find the Home link within main navigation
      const homeLinks = screen.getAllByText('Home')
      const homeLink = homeLinks.find((link) => link.closest('a')?.getAttribute('aria-current') === 'page')
      expect(homeLink?.closest('a')).toHaveAttribute('aria-current', 'page')
    })

    it('should highlight active route - Artisti', () => {
      renderLayout(['/artisti'])

      // Check for aria-current="page" attribute
      const artistiLinks = screen.getAllByText('Artisti')
      const artistiLink = artistiLinks.find((link) => link.closest('a')?.getAttribute('aria-current') === 'page')
      expect(artistiLink?.closest('a')).toHaveAttribute('aria-current', 'page')
    })

    it('should highlight active route - Team Builder', () => {
      renderLayout(['/team-builder'])

      // Check for aria-current="page" attribute
      const teamBuilderLinks = screen.getAllByText('Team Builder')
      const teamBuilderLink = teamBuilderLinks.find((link) => link.closest('a')?.getAttribute('aria-current') === 'page')
      expect(teamBuilderLink?.closest('a')).toHaveAttribute('aria-current', 'page')
    })

    it('should not highlight inactive routes', () => {
      // Find links in navigation that are NOT the current page
      const { container } = renderLayout(['/'])

      const artistiLinks = container.querySelectorAll('nav a[href="/artisti"]')
      const teamBuilderLinks = container.querySelectorAll('nav a[href="/team-builder"]')

      // Inactive links should not have aria-current="page"
      artistiLinks.forEach(link => {
        expect(link).not.toHaveAttribute('aria-current', 'page')
      })
      teamBuilderLinks.forEach(link => {
        expect(link).not.toHaveAttribute('aria-current', 'page')
      })
    })
  })

  describe('Main Content', () => {
    it('should render children content', () => {
      renderLayout()

      expect(screen.getByText('Test Content')).toBeInTheDocument()
    })
  })

  describe('Footer', () => {
    it('should render footer with budget info', () => {
      renderLayout()

      expect(screen.getByText(/Budget: 100 Baudi/i)).toBeInTheDocument()
      expect(screen.getByText(/Squadra: 7 Artisti/i)).toBeInTheDocument()
    })

    it('should render disclaimer', () => {
      renderLayout()

      expect(screen.getByText(/I dati e le predizioni sono a scopo simulativo/i)).toBeInTheDocument()
    })

    it('should render regulation section', () => {
      renderLayout()

      expect(screen.getByText('Regolamento')).toBeInTheDocument()
    })

    it('should render links section', () => {
      renderLayout()

      expect(screen.getByText('Collegamenti')).toBeInTheDocument()
    })
  })

  describe('Breadcrumb Navigation', () => {
    it('should not show breadcrumbs on home page', () => {
      renderLayout(['/'])

      const breadcrumb = screen.queryByRole('navigation', { name: 'Breadcrumb' })
      expect(breadcrumb).not.toBeInTheDocument()
    })

    it('should show breadcrumbs on inner pages', () => {
      renderLayout(['/artisti'])

      // Breadcrumbs should contain the page name
      const breadcrumb = screen.getByRole('navigation', { name: 'Breadcrumb' })
      expect(breadcrumb).toBeInTheDocument()
      expect(screen.getByText('Artisti')).toBeInTheDocument()
    })
  })

  describe('Accessibility', () => {
    it('should have navigation semantically structured', () => {
      const { container } = renderLayout()

      const navs = container.querySelectorAll('nav')
      expect(navs.length).toBeGreaterThan(0)
    })

    it('should have proper aria labels', () => {
      const { container } = renderLayout()

      const mainNav = container.querySelector('nav[aria-label="Main navigation"]')
      expect(mainNav).toBeInTheDocument()
    })

    it('should have mobile menu button with proper aria attributes', () => {
      const { container } = renderLayout()

      const menuButton = container.querySelector('button[aria-label="Apri menu"]')
      expect(menuButton).toBeInTheDocument()
    })
  })

  describe('Responsive Design', () => {
    it('should render mobile menu button', () => {
      const { container } = renderLayout()

      const menuButton = container.querySelector('button[aria-label*="menu"]')
      expect(menuButton).toBeInTheDocument()
    })
  })
})
