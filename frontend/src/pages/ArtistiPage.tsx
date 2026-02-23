import { useCallback, useEffect, useState, useMemo } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import { getArtisti } from '@/api'
import type { ArtistaWithPredizione } from '@/types'
import { Card, Button, Input, Badge, EmptyState, PerformanceLevelBadge, UncertaintyInterval } from '@/components/ui'
import { Search, Filter, Music, TrendingUp, User, SlidersHorizontal } from 'lucide-react'
import Avatar from '@/components/ui/Avatar'
import { getInitials } from '@/utils/getInitials'

interface FilterState {
  search: string
  minPrice: number | undefined
  maxPrice: number | undefined
  genre: string | undefined
  level: string | undefined
  sortBy: 'name' | 'price' | 'score' | 'price-desc'
}

const sortOptions = [
  { value: 'name', label: 'Nome A-Z' },
  { value: 'price', label: 'Prezzo crescente' },
  { value: 'price-desc', label: 'Prezzo decrescente' },
  { value: 'score', label: 'Punteggio' }
]

const priceFilterOptions = [
  { value: '', label: 'Tutti i prezzi' },
  { value: '17', label: '17 Baudi' },
  { value: '16', label: '16+ Baudi' },
  { value: '15', label: '15+ Baudi' },
  { value: '14', label: '14+ Baudi' },
  { value: '13', label: '13+ Baudi' }
]

const levelOptions = [
  { value: '', label: 'Tutti i livelli' },
  { value: 'HIGH', label: '🔥 HIGH Performer' },
  { value: 'MEDIUM', label: '⚡ MEDIUM Performer' },
  { value: 'LOW', label: '🌱 LOW Performer' },
  { value: 'DEBUTTANTE', label: '🌟 DEBUTTANTE' }
]

export default function ArtistiPage() {
  const [artisti, setArtisti] = useState<ArtistaWithPredizione[]>([])
  const [loading, setLoading] = useState(true)
  const [filter, setFilter] = useState<FilterState>({
    search: '',
    minPrice: undefined,
    maxPrice: undefined,
    genre: undefined,
    level: undefined,
    sortBy: 'name'
  })
  const [showFilters, setShowFilters] = useState(false)

  const loadArtisti = useCallback(async () => {
    setLoading(true)
    try {
      const data = await getArtisti({
        min_quotazione: filter.minPrice,
        max_quotazione: filter.maxPrice
      })
      setArtisti(data)
    } catch (error) {
      console.error('Error loading artisti:', error)
    } finally {
      setLoading(false)
    }
  }, [filter.minPrice, filter.maxPrice])

  useEffect(() => {
    loadArtisti()
  }, [loadArtisti])

  // Extract unique genres
  const genres = useMemo(() => {
    const genreSet = new Set<string>()
    artisti.forEach(a => {
      if (a.genere_musicale) genreSet.add(a.genere_musicale)
    })
    return Array.from(genreSet).sort()
  }, [artisti])

  const genreOptions = useMemo(() => [
    { value: '', label: 'Tutti i generi' },
    ...genres.map(g => ({ value: g, label: g }))
  ], [genres])

  const filteredAndSortedArtisti = useMemo(() => {
    let result = artisti.filter(a => {
      const matchesSearch = a.nome.toLowerCase().includes(filter.search.toLowerCase())
      const matchesGenre = !filter.genre || a.genere_musicale === filter.genre
      const matchesLevel = !filter.level || a.predizione_2026?.livello_performer === filter.level
      return matchesSearch && matchesGenre && matchesLevel
    })

    // Sort
    result = result.sort((a, b) => {
      switch (filter.sortBy) {
        case 'name':
          return a.nome.localeCompare(b.nome)
        case 'price':
          return a.quotazione_2026 - b.quotazione_2026
        case 'price-desc':
          return b.quotazione_2026 - a.quotazione_2026
        case 'score':
          return (b.predizione_2026?.punteggio_predetto || 0) - (a.predizione_2026?.punteggio_predetto || 0)
        default:
          return 0
      }
    })

    return result
  }, [artisti, filter.search, filter.genre, filter.level, filter.sortBy])

  // Calculate stats
  const stats = useMemo(() => ({
    total: filteredAndSortedArtisti.length,
    debuttanti: filteredAndSortedArtisti.filter(a => a.debuttante_2026).length,
    highPerformers: filteredAndSortedArtisti.filter(a => a.predizione_2026?.livello_performer === 'HIGH').length,
    avgScore: filteredAndSortedArtisti.length > 0
      ? filteredAndSortedArtisti.reduce((sum, a) => sum + (a.predizione_2026?.punteggio_predetto || 0), 0) / filteredAndSortedArtisti.length
      : 0
  }), [filteredAndSortedArtisti])

  const clearFilters = () => {
    setFilter({
      search: '',
      minPrice: undefined,
      maxPrice: undefined,
      genre: undefined,
      level: undefined,
      sortBy: 'name'
    })
  }

  const hasActiveFilters = filter.search || filter.minPrice || filter.maxPrice || filter.genre || filter.level

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="space-y-6 overflow-x-hidden"
    >
      {/* Header */}
      <motion.div
        initial={{ y: -20, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        className="flex flex-col sm:flex-row sm:items-center justify-between gap-4"
      >
        <div>
          <h2 className="text-3xl font-bold text-amber-400 flex items-center gap-2">
            <Music className="w-8 h-8" />
            Artisti 2026
          </h2>
          <p className="text-gray-400 mt-1">Scopri tutti gli artisti in gara</p>
        </div>
        <div className="flex items-center gap-3">
          <Button
            variant={showFilters ? 'primary' : 'secondary'}
            onClick={() => setShowFilters(!showFilters)}
          >
            <SlidersHorizontal className="w-4 h-4" />
            Filtri
          </Button>
          <Badge variant="gold" size="md" className="text-sm">
            {stats.total} artisti
          </Badge>
        </div>
      </motion.div>

      {/* Stats Cards */}
      <motion.div
        initial={{ y: -10, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ delay: 0.1 }}
        className="grid grid-cols-2 lg:grid-cols-4 gap-3"
      >
        <StatCard
          icon={<User className="w-5 h-5" />}
          label="Totale"
          value={stats.total}
          color="gold"
        />
        <StatCard
          icon={<TrendingUp className="w-5 h-5" />}
          label="Debuttanti"
          value={stats.debuttanti}
          color="cyan"
        />
        <StatCard
          icon={<TrendingUp className="w-5 h-5" />}
          label="High Perf."
          value={stats.highPerformers}
          color="green"
        />
        <StatCard
          icon={<TrendingUp className="w-5 h-5" />}
          label="Media Score"
          value={Math.round(stats.avgScore)}
          color="purple"
        />
      </motion.div>

      {/* Search and Sort Bar */}
      <Card className="p-4">
        <div className="flex flex-col sm:flex-row gap-3">
          <div className="flex-1">
            <Input
              placeholder="Cerca artista per nome..."
              value={filter.search}
              onChange={(e) => setFilter({ ...filter, search: e.target.value })}
              icon={<Search className="w-5 h-5 text-gray-400" />}
            />
          </div>
          <div className="sm:w-48">
            <select
              value={filter.sortBy}
              onChange={(e) => setFilter({ ...filter, sortBy: e.target.value as FilterState['sortBy'] })}
              className="w-full px-4 py-2.5 bg-white dark:bg-navy-900 border border-gray-300 dark:border-navy-700 rounded-lg text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-amber-500/50 focus:border-amber-500 transition-all duration-200 cursor-pointer appearance-none"
            >
              {sortOptions.map(opt => (
                <option key={opt.value} value={opt.value}>{opt.label}</option>
              ))}
            </select>
          </div>
        </div>
      </Card>

      {/* Advanced Filters */}
      <AnimatePresence>
        {showFilters && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.3 }}
          >
            <Card className="p-5 space-y-4">
              <div className="flex items-center justify-between">
                <h3 className="text-lg font-semibold text-amber-400 flex items-center gap-2">
                  <Filter className="w-5 h-5" />
                  Filtri avanzati
                </h3>
                {hasActiveFilters && (
                  <Button variant="ghost" size="sm" onClick={clearFilters}>
                    Cancella filtri
                  </Button>
                )}
              </div>
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
                <div className="space-y-1">
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">Prezzo minimo</label>
                  <select
                    value={filter.minPrice?.toString() ?? ''}
                    onChange={(e) => setFilter({ ...filter, minPrice: e.target.value ? parseInt(e.target.value) : undefined })}
                    className="w-full px-4 py-2.5 bg-white dark:bg-navy-900 border border-gray-300 dark:border-navy-700 rounded-lg text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-amber-500/50 focus:border-amber-500 transition-all duration-200 cursor-pointer appearance-none"
                  >
                    {priceFilterOptions.map(opt => (
                      <option key={opt.value} value={opt.value}>{opt.label}</option>
                    ))}
                  </select>
                </div>
                <div className="space-y-1">
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">Prezzo massimo</label>
                  <select
                    value={filter.maxPrice?.toString() ?? ''}
                    onChange={(e) => setFilter({ ...filter, maxPrice: e.target.value ? parseInt(e.target.value) : undefined })}
                    className="w-full px-4 py-2.5 bg-white dark:bg-navy-900 border border-gray-300 dark:border-navy-700 rounded-lg text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-amber-500/50 focus:border-amber-500 transition-all duration-200 cursor-pointer appearance-none"
                  >
                    {priceFilterOptions.map(opt => (
                      <option key={opt.value} value={opt.value}>{opt.label}</option>
                    ))}
                  </select>
                </div>
                <div className="space-y-1">
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">Genere musicale</label>
                  <select
                    value={filter.genre ?? ''}
                    onChange={(e) => setFilter({ ...filter, genre: e.target.value || undefined })}
                    className="w-full px-4 py-2.5 bg-white dark:bg-navy-900 border border-gray-300 dark:border-navy-700 rounded-lg text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-amber-500/50 focus:border-amber-500 transition-all duration-200 cursor-pointer appearance-none"
                  >
                    {genreOptions.map(opt => (
                      <option key={opt.value} value={opt.value}>{opt.label}</option>
                    ))}
                  </select>
                </div>
                <div className="space-y-1">
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">Livello performer</label>
                  <select
                    value={filter.level ?? ''}
                    onChange={(e) => setFilter({ ...filter, level: e.target.value || undefined })}
                    className="w-full px-4 py-2.5 bg-white dark:bg-navy-900 border border-gray-300 dark:border-navy-700 rounded-lg text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-amber-500/50 focus:border-amber-500 transition-all duration-200 cursor-pointer appearance-none"
                  >
                    {levelOptions.map(opt => (
                      <option key={opt.value} value={opt.value}>{opt.label}</option>
                    ))}
                  </select>
                </div>
              </div>

              {/* Price Range Slider */}
              <div className="pt-4 border-t border-gray-200 dark:border-navy-700">
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-4">Range prezzo</label>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
                  <div>
                    <div className="flex justify-between text-sm mb-2">
                      <span className="text-gray-400">Prezzo minimo</span>
                      <span className="font-bold text-amber-400">{filter.minPrice || 13} baudi</span>
                    </div>
                    <input
                      type="range"
                      min="13"
                      max="17"
                      value={filter.minPrice || 13}
                      onChange={(e) => setFilter({ ...filter, minPrice: parseInt(e.target.value) })}
                      className="w-full h-2 bg-gray-200 dark:bg-navy-900 rounded-lg appearance-none cursor-pointer accent-amber-500"
                    />
                    <div className="flex justify-between text-xs text-gray-500 mt-1">
                      <span>13</span>
                      <span>17</span>
                    </div>
                  </div>
                  <div>
                    <div className="flex justify-between text-sm mb-2">
                      <span className="text-gray-400">Prezzo massimo</span>
                      <span className="font-bold text-amber-400">{filter.maxPrice || 17} baudi</span>
                    </div>
                    <input
                      type="range"
                      min="13"
                      max="17"
                      value={filter.maxPrice || 17}
                      onChange={(e) => setFilter({ ...filter, maxPrice: parseInt(e.target.value) })}
                      className="w-full h-2 bg-gray-200 dark:bg-navy-900 rounded-lg appearance-none cursor-pointer accent-amber-500"
                    />
                    <div className="flex justify-between text-xs text-gray-500 mt-1">
                      <span>13</span>
                      <span>17</span>
                    </div>
                  </div>
                </div>
              </div>
            </Card>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Active Filters Display */}
      {hasActiveFilters && (
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          className="flex flex-wrap gap-2"
        >
          {filter.search && (
            <Badge variant="default" size="sm" className="flex items-center gap-1">
              Search: "{filter.search}"
              <button onClick={() => setFilter({ ...filter, search: '' })} className="ml-1 hover:text-white">×</button>
            </Badge>
          )}
          {filter.minPrice && (
            <Badge variant="default" size="sm" className="flex items-center gap-1">
              Min: {filter.minPrice}
              <button onClick={() => setFilter({ ...filter, minPrice: undefined })} className="ml-1 hover:text-white">×</button>
            </Badge>
          )}
          {filter.maxPrice && (
            <Badge variant="default" size="sm" className="flex items-center gap-1">
              Max: {filter.maxPrice}
              <button onClick={() => setFilter({ ...filter, maxPrice: undefined })} className="ml-1 hover:text-white">×</button>
            </Badge>
          )}
          {filter.genre && (
            <Badge variant="default" size="sm" className="flex items-center gap-1">
              {filter.genre}
              <button onClick={() => setFilter({ ...filter, genre: undefined })} className="ml-1 hover:text-white">×</button>
            </Badge>
          )}
          {filter.level && (
            <Badge variant="default" size="sm" className="flex items-center gap-1">
              {filter.level}
              <button onClick={() => setFilter({ ...filter, level: undefined })} className="ml-1 hover:text-white">×</button>
            </Badge>
          )}
        </motion.div>
      )}

      {/* Artisti Grid */}
      {loading ? (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4"
        >
          {[...Array(6)].map((_, i) => (
            <LoadingArtistCard key={i} />
          ))}
        </motion.div>
      ) : filteredAndSortedArtisti.length > 0 ? (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4"
        >
          <AnimatePresence>
            {filteredAndSortedArtisti.map((artista, index) => (
              <motion.div
                key={artista.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, scale: 0.8 }}
                transition={{ delay: index * 0.03 }}
              >
                <Link to={`/storico/${artista.id}`} className="min-w-0">
                  <ArtistaCard artista={artista} />
                </Link>
              </motion.div>
            ))}
          </AnimatePresence>
        </motion.div>
      ) : (
        <EmptyState
          icon={<Music className="text-6xl text-gray-600" />}
          title="Nessun artista trovato"
          description="Prova a cambiare i filtri di ricerca"
          action={{
            label: 'Cancella filtri',
            onClick: clearFilters
          }}
        />
      )}
    </motion.div>
  )
}

function StatCard({ icon, label, value, color }: { icon: React.ReactNode, label: string, value: number, color: 'gold' | 'cyan' | 'green' | 'purple' }) {
  const colorStyles = {
    gold: 'text-amber-400 bg-amber-500/10 border-amber-500/20',
    cyan: 'text-cyan-400 bg-cyan-500/10 border-cyan-500/20',
    green: 'text-green-400 bg-green-500/10 border-green-500/20',
    purple: 'text-purple-400 bg-purple-500/10 border-purple-500/20'
  }

  return (
    <motion.div
      whileHover={{ y: -2 }}
      className={`p-4 rounded-xl border ${colorStyles[color]}`}
    >
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          {icon}
          <span className="text-sm text-gray-400">{label}</span>
        </div>
        <span className="text-2xl font-bold">{value}</span>
      </div>
    </motion.div>
  )
}

function ArtistaCard({ artista }: { artista: ArtistaWithPredizione }) {
  const navigate = useNavigate()
  return (
    <Card variant="interactive" hover={true} className="h-full">
      <div className="px-4 pt-6">
        <div className="flex items-start gap-3 mb-4">
          <Avatar
            initials={getInitials(artista.nome)}
            src={artista.image_url || undefined}
            size="lg"
            className="flex-shrink-0"
            alt={artista.nome}
          />
          <div className="flex-1 min-w-0">
            <h3 className="text-lg font-bold text-amber-600 dark:text-amber-400 truncate">
              {artista.nome}
            </h3>
            {artista.genere_musicale && (
              <p className="text-sm text-gray-500 dark:text-gray-400 flex items-center gap-1 mt-1">
                <Music className="w-3 h-3" />
                {artista.genere_musicale}
              </p>
            )}
          </div>
          <div className="text-right flex-shrink-0 pr-2">
            <div className="text-3xl font-bold text-amber-600 dark:text-amber-400">{artista.quotazione_2026}</div>
            <div className="text-xs text-gray-500 dark:text-gray-400">baudi</div>
          </div>
        </div>
      </div>

      <div className="px-4 flex flex-wrap items-center gap-2 mb-4">
        {artista.predizione_2026 && (
          <PerformanceLevelBadge level={artista.predizione_2026.livello_performer} size="sm" />
        )}
        {artista.debuttante_2026 && (
          <Badge variant="info" size="sm">DEBUT</Badge>
        )}
      </div>

      {artista.predizione_2026 && (
        <div className="px-4 pt-4 border-t border-gray-200 dark:border-navy-700">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm text-gray-500 dark:text-gray-400">Punteggio stimato</span>
            <span className="font-mono font-bold text-amber-600 dark:text-amber-400 text-lg">
              {Math.round(artista.predizione_2026.punteggio_predetto)}
            </span>
          </div>
          {artista.predizione_2026.interval_lower && artista.predizione_2026.interval_upper && (
            <div className="flex items-center justify-between">
              <span className="text-xs text-gray-400 dark:text-gray-500">Intervallo</span>
              <UncertaintyInterval
                lower={artista.predizione_2026.interval_lower}
                upper={artista.predizione_2026.interval_upper}
                predicted={artista.predizione_2026.punteggio_predetto}
                size="sm"
              />
            </div>
          )}
        </div>
      )}

      {/* Quick Actions */}
      <div className="mt-4 pt-4 px-4 pb-6 border-t border-gray-200 dark:border-navy-700 flex gap-2">
        <Button variant="ghost" size="sm" className="flex-1">
          Dettagli
        </Button>
        <Button
          variant="primary"
          size="sm"
          className="flex-1"
          onClick={(e) => {
            e.preventDefault()
            navigate('/team-builder')
          }}
        >
          Aggiungi
        </Button>
      </div>
    </Card>
  )
}

function LoadingArtistCard() {
  return (
    <div className="bg-white dark:bg-navy-800 rounded-xl p-5 border border-gray-200 dark:border-navy-700 animate-pulse">
      <div className="flex items-start justify-between mb-4">
        <div className="flex-1 space-y-2">
          <div className="h-6 bg-gray-200 dark:bg-navy-700 rounded w-3/4"></div>
          <div className="h-4 bg-gray-200 dark:bg-navy-700 rounded w-1/2"></div>
        </div>
        <div className="text-right">
          <div className="h-8 bg-gray-200 dark:bg-navy-700 rounded w-12"></div>
          <div className="h-3 bg-gray-200 dark:bg-navy-700 rounded w-16 mt-1"></div>
        </div>
      </div>
      <div className="flex gap-2 mb-4">
        <div className="h-6 bg-gray-200 dark:bg-navy-700 rounded w-16"></div>
        <div className="h-6 bg-gray-200 dark:bg-navy-700 rounded w-12"></div>
      </div>
      <div className="pt-4 border-t border-gray-200 dark:border-navy-700">
        <div className="flex justify-between">
          <div className="h-4 bg-gray-200 dark:bg-navy-700 rounded w-24"></div>
          <div className="h-6 bg-gray-200 dark:bg-navy-700 rounded w-16"></div>
        </div>
      </div>
    </div>
  )
}
