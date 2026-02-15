import { Link, useLocation } from 'react-router-dom'
import { useState, useEffect } from 'react'
import {
  Home,
  Mic,
  Trophy,
  Menu,
  X,
  Sparkles,
  Calendar,
  Shield,
  Github,
  Twitter,
  Info
} from 'lucide-react'
import ThemeToggle from './ui/ThemeToggle'

interface LayoutProps {
  children: React.ReactNode
}

// Breadcrumb configuration
const breadcrumbTitles: Record<string, string> = {
  '/': 'Home',
  '/artisti': 'Artisti',
  '/team-builder': 'Team Builder',
  '/storico': 'Storico'
}

export default function Layout({ children }: LayoutProps) {
  const location = useLocation()
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)
  const [scrolled, setScrolled] = useState(false)

  const isActive = (path: string) => location.pathname === path

  // Handle scroll effect for header
  useEffect(() => {
    const handleScroll = () => {
      setScrolled(window.scrollY > 10)
    }
    window.addEventListener('scroll', handleScroll)
    return () => window.removeEventListener('scroll', handleScroll)
  }, [])

  // Close mobile menu on route change
  useEffect(() => {
    setMobileMenuOpen(false)
  }, [location.pathname])

  // Generate breadcrumbs
  const generateBreadcrumbs = () => {
    const pathnames = location.pathname.split('/').filter(x => x)
    if (pathnames.length === 0) return null

    return (
      <nav className="flex items-center space-x-2 text-sm text-gray-400 mb-6" aria-label="Breadcrumb">
        <Link to="/" className="hover:text-amber-400 transition-colors">
          <Home className="w-4 h-4" />
        </Link>
        {pathnames.map((name, index) => {
          const routeTo = `/${pathnames.slice(0, index + 1).join('/')}`
          const isLast = index === pathnames.length - 1
          return (
            <div key={routeTo} className="flex items-center space-x-2">
              <span className="text-navy-600 dark:text-navy-400">/</span>
              {isLast ? (
                <span className="text-amber-400 font-medium">{breadcrumbTitles[routeTo] || name}</span>
              ) : (
                <Link to={routeTo} className="hover:text-amber-400 transition-colors">
                  {breadcrumbTitles[routeTo] || name}
                </Link>
              )}
            </div>
          )
        })}
      </nav>
    )
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 via-gray-100 to-gray-50 dark:from-navy-950 dark:via-navy-900 dark:to-navy-950 transition-colors duration-300" data-testid="layout">
      {/* Animated background pattern */}
      <div className="fixed inset-0 opacity-[0.02] dark:opacity-[0.03] pointer-events-none" aria-hidden="true">
        <svg className="w-full h-full">
          <pattern id="grid-pattern" width="40" height="40" patternUnits="userSpaceOnUse">
            <circle cx="2" cy="2" r="1" fill="currentColor" className="text-amber-400" />
          </pattern>
          <rect width="100%" height="100%" fill="url(#grid-pattern)" />
        </svg>
      </div>

      {/* Header */}
      <header
        className={`sticky top-0 z-50 transition-all duration-300 ${
          scrolled
            ? 'bg-white/95 dark:bg-navy-950/95 backdrop-blur-md border-b border-gray-200 dark:border-navy-700 shadow-lg shadow-gray-200/50 dark:shadow-navy-950/50'
            : 'bg-gradient-to-r from-white via-gray-50 to-white dark:from-navy-900 dark:via-navy-800 dark:to-navy-900 border-b border-gray-200 dark:border-navy-700'
        }`}
        data-testid="header"
      >
        <div className="max-w-7xl mx-auto px-6 sm:px-8 lg:px-12">
          <div className="flex items-center justify-between h-16 md:h-20">
            {/* Logo */}
            <Link
              to="/"
              className="flex items-center space-x-3 group"
              aria-label="FantaSanremo Team Builder Home"
            >
              <div className="relative">
                <div className="absolute inset-0 bg-amber-400/20 blur-lg rounded-full group-hover:bg-amber-400/30 transition-all duration-300" />
                <Sparkles className="w-8 h-8 text-amber-400 relative z-10" />
              </div>
              <div className="hidden sm:block">
                <h1 className="text-xl md:text-2xl font-bold text-amber-500 dark:text-amber-400 group-hover:text-amber-600 dark:group-hover:text-amber-500 transition-colors">
                  FantaSanremo
                </h1>
                <p className="text-xs text-gray-600 dark:text-gray-400">Team Builder 2026</p>
              </div>
            </Link>

            {/* Desktop Navigation */}
            <nav className="hidden md:flex items-center space-x-1" aria-label="Main navigation">
              <NavLink to="/" active={isActive('/')} icon={<Home className="w-4 h-4" />}>
                Home
              </NavLink>
              <NavLink to="/artisti" active={isActive('/artisti')} icon={<Mic className="w-4 h-4" />}>
                Artisti
              </NavLink>
              <NavLink to="/team-builder" active={isActive('/team-builder')} icon={<Trophy className="w-4 h-4" />}>
                Team Builder
              </NavLink>
            </nav>

            {/* Festival Info */}
            <div className="hidden lg:flex items-center space-x-2 text-sm text-gray-600 dark:text-gray-400 bg-gray-100 dark:bg-navy-800/50 px-4 py-2 rounded-full border border-gray-200 dark:border-navy-700">
              <Calendar className="w-4 h-4 text-amber-500 dark:text-amber-400" />
              <span>24-28 Febbraio 2026</span>
            </div>

            {/* Theme Toggle */}
            <div className="hidden md:block">
              <ThemeToggle />
            </div>

            {/* Mobile Menu Button */}
            <button
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
              className="md:hidden p-2 rounded-lg text-gray-600 dark:text-gray-400 hover:text-amber-500 dark:hover:text-amber-400 hover:bg-gray-100 dark:hover:bg-navy-700 transition-colors"
              aria-label={mobileMenuOpen ? 'Chiudi menu' : 'Apri menu'}
              aria-expanded={mobileMenuOpen}
            >
              {mobileMenuOpen ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
            </button>
          </div>
        </div>

        {/* Mobile Navigation */}
        <nav
          className={`md:hidden transition-all duration-300 overflow-hidden ${
            mobileMenuOpen
              ? 'max-h-64 opacity-100 border-t border-gray-200 dark:border-navy-700'
              : 'max-h-0 opacity-0'
          }`}
          aria-label="Mobile navigation"
        >
          <div className="px-4 py-4 space-y-2">
            <MobileNavLink to="/" active={isActive('/')} icon={<Home className="w-5 h-5" />}>
              Home
            </MobileNavLink>
            <MobileNavLink to="/artisti" active={isActive('/artisti')} icon={<Mic className="w-5 h-5" />}>
              Artisti
            </MobileNavLink>
            <MobileNavLink
              to="/team-builder"
              active={isActive('/team-builder')}
              icon={<Trophy className="w-5 h-5" />}
            >
              Team Builder
            </MobileNavLink>
            <div className="pt-2 border-t border-gray-200 dark:border-navy-700">
              <div className="flex items-center space-x-2 text-sm text-gray-600 dark:text-gray-400 px-4 py-2">
                <Calendar className="w-4 h-4 text-amber-500 dark:text-amber-400" />
                <span>24-28 Febbraio 2026</span>
              </div>
            </div>

            {/* Mobile Theme Toggle */}
            <div className="pt-2 border-t border-gray-200 dark:border-navy-700 px-4 py-2">
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-600 dark:text-gray-400">Tema</span>
                <ThemeToggle />
              </div>
            </div>
          </div>
        </nav>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-6 sm:px-8 lg:px-12 py-8 relative z-10 overflow-x-hidden">
        {generateBreadcrumbs()}
        {children}
      </main>

      {/* Footer */}
      <footer className="relative z-10 mt-16 border-t border-gray-200 dark:border-navy-800 bg-white/50 dark:bg-navy-900/50 backdrop-blur" data-testid="footer">
        <div className="max-w-7xl mx-auto px-6 sm:px-8 lg:px-12 py-12">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {/* Brand */}
            <div className="space-y-4">
              <div className="flex items-center space-x-2">
                <Sparkles className="w-6 h-6 text-amber-500 dark:text-amber-400" />
                <span className="font-bold text-amber-500 dark:text-amber-400">FantaSanremo 2026</span>
              </div>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                Costruisci la tua squadra ideale e vinci il torneo di FantaSanremo!
              </p>
            </div>

            {/* Game Info */}
            <div className="space-y-4">
              <h3 className="font-semibold text-gray-800 dark:text-gray-200 flex items-center space-x-2">
                <Info className="w-4 h-4 text-amber-500 dark:text-amber-400" />
                <span>Regolamento</span>
              </h3>
              <ul className="space-y-2 text-sm text-gray-600 dark:text-gray-400">
                <li>Budget: 100 Baudi</li>
                <li>Squadra: 7 Artisti (5 titolari + 2 riserve)</li>
                <li>Il capitano vale doppio!</li>
              </ul>
            </div>

            {/* Links */}
            <div className="space-y-4">
              <h3 className="font-semibold text-gray-800 dark:text-gray-200">Collegamenti</h3>
              <div className="flex space-x-4">
                <a
                  href="https://twitter.com/FantaSanremo"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-gray-600 dark:text-gray-400 hover:text-amber-500 dark:hover:text-amber-400 transition-colors"
                  aria-label="Twitter FantaSanremo"
                >
                  <Twitter className="w-5 h-5" />
                </a>
                <a
                  href="https://github.com"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-gray-600 dark:text-gray-400 hover:text-amber-500 dark:hover:text-amber-400 transition-colors"
                  aria-label="GitHub"
                >
                  <Github className="w-5 h-5" />
                </a>
              </div>
            </div>
          </div>

          {/* Bottom bar */}
          <div className="mt-8 pt-8 border-t border-gray-200 dark:border-navy-800">
            <div className="flex flex-col sm:flex-row justify-between items-center space-y-4 sm:space-y-0">
              <p className="text-sm text-gray-600 dark:text-gray-500">
                I dati e le predizioni sono a scopo simulativo
              </p>
              <p className="text-sm text-gray-600 dark:text-gray-500 flex items-center space-x-2">
                <Shield className="w-4 h-4" />
                <span>FantaSanremo Team Builder</span>
              </p>
            </div>
          </div>
        </div>
      </footer>
    </div>
  )
}

function NavLink({
  to,
  active,
  icon,
  children
}: {
  to: string
  active: boolean
  icon: React.ReactNode
  children: React.ReactNode
}) {
  return (
    <Link
      to={to}
      className={`relative flex items-center space-x-2 px-4 py-2 rounded-lg transition-all duration-200 ${
        active
          ? 'text-amber-600 dark:text-amber-400 font-semibold bg-amber-500/10 dark:bg-amber-400/10 border border-amber-500/30 dark:border-amber-400/30 shadow-lg shadow-amber-500/10 dark:shadow-amber-400/10'
          : 'text-gray-700 dark:text-gray-300 hover:text-amber-600 dark:hover:text-amber-400 hover:bg-gray-100 dark:hover:bg-navy-700'
      }`}
      aria-current={active ? 'page' : undefined}
    >
      {active && (
        <span className="absolute inset-0 rounded-lg bg-gradient-to-r from-amber-500/0 via-amber-500/10 dark:from-amber-400/0 dark:via-amber-400/10 to-amber-500/0 dark:to-amber-400/0 animate-pulse" />
      )}
      <span className="relative z-10 flex items-center space-x-2">
        {icon}
        <span>{children}</span>
      </span>
    </Link>
  )
}

function MobileNavLink({
  to,
  active,
  icon,
  children
}: {
  to: string
  active: boolean
  icon: React.ReactNode
  children: React.ReactNode
}) {
  return (
    <Link
      to={to}
      className={`flex items-center space-x-3 px-4 py-3 rounded-lg transition-all duration-200 ${
        active
          ? 'bg-amber-500/10 dark:bg-amber-400/10 border border-amber-500/30 dark:border-amber-400/30 text-amber-600 dark:text-amber-400 font-semibold'
          : 'text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-navy-700 hover:text-amber-600 dark:hover:text-amber-400'
      }`}
      aria-current={active ? 'page' : undefined}
    >
      {icon}
      <span>{children}</span>
      {active && (
        <span className="ml-auto">
          <span className="block w-2 h-2 rounded-full bg-amber-500 dark:bg-amber-400 animate-pulse" />
        </span>
      )}
    </Link>
  )
}
