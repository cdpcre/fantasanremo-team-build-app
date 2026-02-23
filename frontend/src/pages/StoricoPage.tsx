import { useCallback, useEffect, useState } from 'react'
import { useParams, Link, useNavigate } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import { ArrowLeft, Trophy, Calendar, Music, TrendingUp, Award, Star } from 'lucide-react'
import { getArtista } from '@/api'
import type { ArtistaWithStorico } from '@/types'
import {
  Card,
  StatCard,
  ConfidenceMeter,
  PerformanceChart,
  PerformanceLevelBadge,
  AnimatedCounter,
  Timeline,
  UncertaintyInterval
} from '@/components/ui'
import Avatar from '@/components/ui/Avatar'
import { getInitials } from '@/utils/getInitials'

export default function StoricoPage() {
  const { artistaId } = useParams<{ artistaId: string }>()
  const navigate = useNavigate()
  const [artista, setArtista] = useState<ArtistaWithStorico | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  // Define callback BEFORE useEffect that uses it
  const loadArtista = useCallback(async (id: number) => {
    setLoading(true)
    setError(null)
    try {
      const data = await getArtista(id)
      setArtista(data)
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : 'Impossibile caricare i dati dell\'artista'
      console.error('Error loading artista:', error)
      setError(errorMsg)
      // Delay redirect slightly so user sees error
      setTimeout(() => navigate('/artisti'), 2000)
    } finally {
      setLoading(false)
    }
  }, [navigate])

  useEffect(() => {
    if (artistaId) {
      loadArtista(parseInt(artistaId))
    }
  }, [artistaId, loadArtista])

  // Calculate stats
  const getStats = useCallback(() => {
    if (!artista) return null

    const completedEditions = artista.edizioni_fantasanremo.filter(e => e.posizione !== null)
    const avgPosition = completedEditions.length > 0
      ? completedEditions.reduce((sum, e) => sum + (e.posizione || 0), 0) / completedEditions.length
      : null
    const bestPosition = completedEditions.length > 0
      ? Math.min(...completedEditions.map(e => e.posizione || 0))
      : null
    const totalPoints = completedEditions.reduce((sum, e) => sum + (e.punteggio_finale || 0), 0)

    return {
      totalEditions: artista.edizioni_fantasanremo.length,
      completedEditions: completedEditions.length,
      avgPosition: avgPosition ? Math.round(avgPosition) : null,
      bestPosition,
      totalPoints
    }
  }, [artista])

  const stats = getStats()

  // Display error message if present
  if (error) {
    return (
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="space-y-6"
      >
        <div className="bg-red-500/20 border border-red-500 text-red-600 dark:text-red-400 rounded-lg p-4 mb-4">
          {error}
        </div>
        {loading && (
          <div className="flex items-center justify-center py-12">
            <motion.div
              animate={{ rotate: 360 }}
              transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
              className="w-12 h-12 border-4 border-amber-500 border-t-transparent rounded-full"
            />
          </div>
        )}
      </motion.div>
    )
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center py-12">
        <motion.div
          animate={{ rotate: 360 }}
          transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
          className="w-12 h-12 border-4 border-amber-500 border-t-transparent rounded-full"
        />
      </div>
    )
  }

  if (!artista) {
    return (
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-center py-12 bg-white dark:bg-navy-800 rounded-xl border border-gray-200 dark:border-navy-700"
      >
        <Music className="w-16 h-16 text-gray-400 mx-auto mb-4" />
        <p className="text-gray-600 dark:text-gray-400 text-lg">Artista non trovato</p>
      </motion.div>
    )
  }

  const chartData = artista.edizioni_fantasanremo
    .filter(e => e.posizione !== null)
    .map(e => ({
      anno: e.anno,
      posizione: e.posizione || 0,
      punteggio: e.punteggio_finale || 0
    }))

  const timelineData = artista.edizioni_fantasanremo.map(e => ({
    anno: e.anno,
    posizione: e.posizione,
    punteggio: e.punteggio_finale,
    quotazione: e.quotazione_baudi
  }))

  return (
    <div className="space-y-6 w-full max-w-full overflow-x-auto">
      {/* Back button & Print toggle */}
      <motion.div
        initial={{ opacity: 0, x: -20 }}
        animate={{ opacity: 1, x: 0 }}
        className="flex items-center justify-between"
      >
        <Link
          to="/artisti"
          className="inline-flex items-center gap-2 text-amber-500 hover:text-amber-400 dark:text-amber-400 dark:hover:text-amber-300 transition-colors font-medium"
        >
          <ArrowLeft className="w-5 h-5" />
          Torna alla lista artisti
        </Link>
        <button
          onClick={() => window.print()}
          className="px-4 py-2 bg-white dark:bg-navy-800 border border-gray-200 dark:border-navy-700 rounded-lg hover:bg-gray-50 dark:hover:bg-navy-700 transition-colors text-sm font-medium text-gray-700 dark:text-gray-300"
        >
          🖨️ Stampa
        </button>
      </motion.div>

      {/* Header Card */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <Card hover={false} className="p-4 sm:p-6 bg-gradient-to-br from-amber-500/10 to-cyan-500/10 dark:from-amber-500/20 dark:to-cyan-500/20 border-amber-500/30 dark:border-amber-500/20 overflow-x-auto">
          <div className="flex flex-col md:flex-row md:items-start md:justify-between gap-6">
            <div className="flex-1">
              <div className="flex items-start gap-4">
                <motion.div
                  whileHover={{ rotate: 5, scale: 1.05 }}
                  className="w-20 h-20"
                >
                  <Avatar
                    initials={getInitials(artista.nome)}
                    src={artista.image_url || undefined}
                    size="xl"
                    className="w-20 h-20 text-2xl ring-4 ring-amber-500/30 shadow-lg shadow-amber-500/30"
                    alt={artista.nome}
                  />
                </motion.div>
                <div className="flex-1">
                  <h2 className="text-3xl md:text-4xl font-bold text-gray-900 dark:text-white mb-2">
                    {artista.nome}
                  </h2>
                  {artista.genere_musicale && (
                    <motion.div
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      transition={{ delay: 0.2 }}
                      className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-cyan-500/20 dark:bg-cyan-500/30 border border-cyan-500/30 text-cyan-600 dark:text-cyan-400 text-sm font-medium"
                    >
                      <Music className="w-4 h-4" />
                      {artista.genere_musicale}
                    </motion.div>
                  )}
                </div>
              </div>
            </div>

            <motion.div
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.3, type: 'spring' }}
              className="text-center md:text-right"
            >
              <div className="text-5xl md:text-6xl font-bold text-amber-500">
                <AnimatedCounter value={artista.quotazione_2026} />
              </div>
              <div className="text-sm text-gray-600 dark:text-gray-400 font-medium mt-1">
                baudi 2026
              </div>
            </motion.div>
          </div>

          {/* ML Prediction */}
          {artista.predizione_2026 && (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.4 }}
              className="mt-6 p-5 rounded-xl bg-white/80 dark:bg-navy-800/80 backdrop-blur border border-gray-200 dark:border-navy-700"
            >
              <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
                <div className="flex items-center gap-3">
                  <Star className="w-6 h-6 text-amber-500" />
                  <div>
                    <div className="text-sm text-gray-600 dark:text-gray-400">Livello Performer</div>
                    <PerformanceLevelBadge
                      level={artista.predizione_2026.livello_performer}
                      size="lg"
                    />
                  </div>
                </div>

                <div className="flex items-center gap-8">
                  <div>
                    <div className="text-sm text-gray-600 dark:text-gray-400 mb-1">Punteggio Stimato</div>
                    <div className="text-3xl font-bold text-amber-500">
                      <AnimatedCounter
                        value={artista.predizione_2026.punteggio_predetto}
                        format={(v) => v.toFixed(0)}
                      />
                    </div>
                    {artista.predizione_2026.interval_lower && artista.predizione_2026.interval_upper && (
                      <div className="mt-2">
                        <UncertaintyInterval
                          lower={artista.predizione_2026.interval_lower}
                          upper={artista.predizione_2026.interval_upper}
                          predicted={artista.predizione_2026.punteggio_predetto}
                          size="sm"
                        />
                      </div>
                    )}
                  </div>

                  {typeof artista.predizione_2026.confidence === 'number' && (
                    <div className="w-48">
                      <ConfidenceMeter
                        confidence={artista.predizione_2026.confidence}
                        label="Confidenza ML"
                        size="md"
                      />
                    </div>
                  )}
                </div>
              </div>
            </motion.div>
          )}
        </Card>
      </motion.div>

      {/* Stats Grid */}
      {stats && stats.completedEditions > 0 && (
        <div className="grid grid-cols-2 sm:grid-cols-2 lg:grid-cols-4 gap-3 sm:gap-4 w-full">
          <StatCard
            label="Partecipazioni"
            value={stats.completedEditions}
            icon={Calendar}
            color="gold"
            delay={0.1}
          />
          <StatCard
            label="Posizione Media"
            value={`#${stats.avgPosition || '-'}`}
            icon={TrendingUp}
            color="cyan"
            delay={0.2}
          />
          <StatCard
            label="Miglior Risultato"
            value={`#${stats.bestPosition || '-'}`}
            icon={Trophy}
            color="green"
            delay={0.3}
          />
          <StatCard
            label="Punti Totali"
            value={stats.totalPoints}
            icon={Award}
            color="purple"
            delay={0.4}
          />
        </div>
      )}

      {/* Performance Chart */}
      {chartData.length > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5 }}
          className="w-full min-w-0"
        >
          <Card className="hover:shadow-amber-500/20 overflow-visible">
            <div className="flex items-center gap-3 mb-6">
              <div className="p-2 rounded-lg bg-gradient-to-br from-amber-400/20 to-amber-600/20 dark:from-amber-400/30 dark:to-amber-600/30">
                <TrendingUp className="w-5 h-5 text-amber-500" />
              </div>
              <h3 className="text-xl font-bold text-gray-900 dark:text-white">
                Storico Performance
              </h3>
            </div>
            <PerformanceChart data={chartData} />
          </Card>
        </motion.div>
      )}

      {/* Timeline */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.6 }}
        className="w-full overflow-x-auto"
      >
        <Card className="p-4 sm:p-6 min-w-0">
          <div className="flex items-center gap-3 mb-6">
            <div className="p-2 rounded-lg bg-gradient-to-br from-cyan-400/20 to-cyan-600/20 dark:from-cyan-400/30 dark:to-cyan-600/30">
              <Calendar className="w-5 h-5 text-cyan-500" />
            </div>
            <h3 className="text-xl font-bold text-gray-900 dark:text-white">
              Timeline Partecipazioni
            </h3>
          </div>
          <Timeline items={timelineData} />
        </Card>
      </motion.div>

      {/* Additional Info Cards */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.7 }}
        className="grid grid-cols-1 md:grid-cols-3 gap-4"
      >
        {artista.anno_nascita && (
          <Card delay={0}>
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-lg bg-purple-500/20 dark:bg-purple-500/30">
                <Calendar className="w-5 h-5 text-purple-500" />
              </div>
              <div>
                <div className="text-sm text-gray-600 dark:text-gray-400">Anno di nascita</div>
                <div className="text-xl font-bold text-gray-900 dark:text-white">{artista.anno_nascita}</div>
              </div>
            </div>
          </Card>
        )}

        {artista.prima_partecipazione && (
          <Card delay={0.1}>
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-lg bg-amber-500/20 dark:bg-amber-500/30">
                <Trophy className="w-5 h-5 text-amber-500" />
              </div>
              <div>
                <div className="text-sm text-gray-600 dark:text-gray-400">Prima partecipazione</div>
                <div className="text-xl font-bold text-gray-900 dark:text-white">{artista.prima_partecipazione}</div>
              </div>
            </div>
          </Card>
        )}

        <Card delay={0.2}>
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-lg bg-green-500/20 dark:bg-green-500/30">
              <Star className="w-5 h-5 text-green-500" />
            </div>
            <div>
              <div className="text-sm text-gray-600 dark:text-gray-400">Debuttante 2026</div>
              <div className="text-xl font-bold text-gray-900 dark:text-white">
                {artista.debuttante_2026 ? 'SÌ' : 'NO'}
              </div>
            </div>
          </div>
        </Card>
      </motion.div>

      {/* Historical Comparison Table */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.8 }}
      >
        <Card>
          <div className="flex items-center gap-3 mb-6">
            <div className="p-2 rounded-lg bg-gradient-to-br from-purple-400/20 to-purple-600/20 dark:from-purple-400/30 dark:to-purple-600/30">
              <Trophy className="w-5 h-5 text-purple-500" />
            </div>
            <h3 className="text-xl font-bold text-gray-900 dark:text-white">
              Dettaglio Partecipazioni
            </h3>
          </div>

          <div className="overflow-x-auto -mx-2 sm:-mx-4 sm:mx-0 px-2 sm:px-4 sm:px-0">
            <table className="w-full min-w-[500px]">
              <thead>
                <tr className="border-b border-gray-200 dark:border-navy-700">
                  <th className="text-left py-3 px-4 text-sm font-semibold text-gray-700 dark:text-gray-300">
                    Anno
                  </th>
                  <th className="text-left py-3 px-4 text-sm font-semibold text-gray-700 dark:text-gray-300">
                    Posizione
                  </th>
                  <th className="text-left py-3 px-4 text-sm font-semibold text-gray-700 dark:text-gray-300">
                    Punteggio
                  </th>
                  <th className="text-left py-3 px-4 text-sm font-semibold text-gray-700 dark:text-gray-300">
                    Quotazione
                  </th>
                </tr>
              </thead>
              <tbody>
                <AnimatePresence>
                  {artista.edizioni_fantasanremo.length === 0 ? (
                    <motion.tr
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      exit={{ opacity: 0 }}
                    >
                      <td colSpan={4} className="text-center py-8 text-gray-500 dark:text-gray-400">
                        Nessuna partecipazione registrata
                      </td>
                    </motion.tr>
                  ) : (
                    artista.edizioni_fantasanremo.map((edizione, index) => (
                      <motion.tr
                        key={edizione.id}
                        initial={{ opacity: 0, x: -20 }}
                        animate={{ opacity: 1, x: 0 }}
                        exit={{ opacity: 0, x: 20 }}
                        transition={{ delay: index * 0.05 }}
                        className="border-b border-gray-100 dark:border-navy-700/50 hover:bg-gray-50 dark:hover:bg-navy-700/30 transition-colors"
                      >
                        <td className="py-3 px-4 font-semibold text-gray-900 dark:text-white">
                          {edizione.anno}
                        </td>
                        <td className="py-3 px-4">
                          {edizione.posizione ? (
                            <span className="inline-flex items-center gap-1 px-2 py-1 rounded-md bg-amber-500/20 text-amber-600 dark:text-amber-400 font-mono font-bold border border-amber-500/30">
                              #{edizione.posizione}
                              {edizione.posizione <= 3 && <Trophy className="w-3 h-3" />}
                            </span>
                          ) : (
                            <span className="text-gray-400 dark:text-gray-500">NP</span>
                          )}
                        </td>
                        <td className="py-3 px-4 font-mono text-gray-700 dark:text-gray-300">
                          {edizione.punteggio_finale || '-'}
                        </td>
                        <td className="py-3 px-4">
                          {edizione.quotazione_baudi ? (
                            <span className="text-cyan-600 dark:text-cyan-400 font-medium">
                              {edizione.quotazione_baudi} baudi
                            </span>
                          ) : (
                            <span className="text-gray-400 dark:text-gray-500">-</span>
                          )}
                        </td>
                      </motion.tr>
                    ))
                  )}
                </AnimatePresence>
              </tbody>
            </table>
          </div>
        </Card>
      </motion.div>
    </div>
  )
}
