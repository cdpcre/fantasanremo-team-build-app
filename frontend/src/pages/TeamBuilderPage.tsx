import { useEffect, useState, useMemo } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { getArtisti, validateTeam } from '@/api'
import type { ArtistaWithPredizione, TeamValidateRequest } from '@/types'
import { Card, Button, Badge, ProgressBar, PerformanceLevelBadge, EmptyState, AnimatedCounter } from '@/components/ui'
import { Users, Star, Crown, Trash2, Trophy, TrendingUp, Zap, Shield, Plus, Minus } from 'lucide-react'
import Avatar from '@/components/ui/Avatar'
import { getInitials } from '@/utils/getInitials'

interface SelectedArtista extends ArtistaWithPredizione {
  role: 'titolare' | 'riserva'
}

interface ValidationState {
  valid: boolean
  message: string
  budget_totale: number
  budget_rimanente: number
  punteggio_simulato?: number
}

export default function TeamBuilderPage() {
  const [artisti, setArtisti] = useState<ArtistaWithPredizione[]>([])
  const [selected, setSelected] = useState<Map<number, SelectedArtista>>(new Map())
  const [capitanoId, setCapitanoId] = useState<number | null>(null)
  const [validation, setValidation] = useState<ValidationState | null>(null)
  const [loading, setLoading] = useState(false)
  const [validating, setValidating] = useState(false)

  useEffect(() => {
    loadArtisti()
  }, [])

  const loadArtisti = async () => {
    setLoading(true)
    try {
      const data = await getArtisti()
      setArtisti(data)
    } catch (error) {
      console.error('Error loading artisti:', error)
    } finally {
      setLoading(false)
    }
  }

  const toggleArtista = (artista: ArtistaWithPredizione) => {
    const newSelected = new Map(selected)

    if (newSelected.has(artista.id)) {
      newSelected.delete(artista.id)
      if (capitanoId === artista.id) {
        setCapitanoId(null)
      }
    } else {
      const titolari = Array.from(newSelected.values()).filter(a => a.role === 'titolare')
      const riserve = Array.from(newSelected.values()).filter(a => a.role === 'riserva')

      if (titolari.length < 5) {
        newSelected.set(artista.id, { ...artista, role: 'titolare' })
      } else if (riserve.length < 2) {
        newSelected.set(artista.id, { ...artista, role: 'riserva' })
      } else {
        return
      }
    }

    setSelected(newSelected)
    setValidation(null)
  }

  const removeArtista = (artistaId: number) => {
    const newSelected = new Map(selected)
    newSelected.delete(artistaId)
    if (capitanoId === artistaId) {
      setCapitanoId(null)
    }
    setSelected(newSelected)
    setValidation(null)
  }

  const setCapitano = (artistaId: number) => {
    if (selected.get(artistaId)?.role === 'titolare') {
      setCapitanoId(artistaId)
    }
  }

  const handleValidate = async () => {
    const artistiIds = Array.from(selected.keys())

    if (artistiIds.length !== 7) {
      setValidation({
        valid: false,
        message: 'Seleziona 7 artisti (5 titolari + 2 riserve)',
        budget_totale: Array.from(selected.values()).reduce((sum, a) => sum + a.quotazione_2026, 0),
        budget_rimanente: 0
      })
      return
    }

    if (!capitanoId) {
      setValidation({
        valid: false,
        message: 'Seleziona un capitano tra i titolari',
        budget_totale: Array.from(selected.values()).reduce((sum, a) => sum + a.quotazione_2026, 0),
        budget_rimanente: 100 - Array.from(selected.values()).reduce((sum, a) => sum + a.quotazione_2026, 0)
      })
      return
    }

    setValidating(true)
    try {
      const request: TeamValidateRequest = {
        artisti_ids: artistiIds,
        capitano_id: capitanoId
      }
      const result = await validateTeam(request)
      setValidation(result)
    } catch (error) {
      console.error('Error validating team:', error)
    } finally {
      setValidating(false)
    }
  }

  const clearTeam = () => {
    setSelected(new Map())
    setCapitanoId(null)
    setValidation(null)
  }

  const selectedList = Array.from(selected.values())
  const titolari = selectedList.filter(a => a.role === 'titolare')
  const riserve = selectedList.filter(a => a.role === 'riserva')
  const budgetUsed = selectedList.reduce((sum, a) => sum + a.quotazione_2026, 0)
  const budgetRemaining = 100 - budgetUsed

  const teamStats = useMemo(() => ({
    avgScore: selectedList.length > 0
      ? selectedList.reduce((sum, a) => sum + (a.predizione_2026?.punteggio_predetto || 0), 0) / selectedList.length
      : 0,
    highPerformers: selectedList.filter(a => a.predizione_2026?.livello_performer === 'HIGH').length,
    debuttanti: selectedList.filter(a => a.debuttante_2026).length,
    captainBonus: capitanoId ? selected.get(capitanoId)?.predizione_2026?.punteggio_predetto || 0 : 0
  }), [selectedList, capitanoId, selected])

  const availableArtisti = artisti.filter(a => !selected.has(a.id))
  const canAdd = (artista: ArtistaWithPredizione) => {
    return budgetUsed + artista.quotazione_2026 <= 100
  }

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="space-y-6 overflow-x-hidden"
      data-testid="team-builder-page"
    >
      {/* Header */}
      <motion.div
        initial={{ y: -20, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        className="flex flex-col sm:flex-row sm:items-center justify-between gap-4"
      >
        <div>
          <h2 className="text-3xl font-bold text-amber-400 flex items-center gap-2">
            <Users className="w-8 h-8" />
            Team Builder
          </h2>
          <p className="text-gray-400 mt-1">Crea la tua squadra vincente</p>
        </div>
        {selectedList.length > 0 && (
          <Button
            variant="secondary"
            onClick={clearTeam}
            data-testid="clear-team-button"
          >
            <Trash2 className="w-4 h-4" />
            Cancella squadra
          </Button>
        )}
      </motion.div>

      {/* Budget Indicator */}
      <motion.div
        initial={{ y: -10, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ delay: 0.1 }}
      >
        <Card className="p-5">
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center gap-2">
              <Trophy className="w-5 h-5 text-amber-400" />
              <span className="font-semibold text-gray-200">Budget</span>
            </div>
            <span className={`font-mono font-bold text-lg ${budgetRemaining < 0 ? 'text-red-400' : 'text-amber-400'}`} data-testid="budget-used">
              {budgetUsed} / 100 Baudi
            </span>
          </div>
          <ProgressBar
            value={budgetUsed}
            max={100}
            size="lg"
            showLabel={false}
            animated={true}
          />
          <div className="flex items-center justify-between mt-3 text-sm">
            <span className="text-gray-400">Rimanenti:</span>
            <span className={`font-bold ${budgetRemaining < 0 ? 'text-red-400' : budgetRemaining < 20 ? 'text-yellow-400' : 'text-amber-400'}`} data-testid="budget-remaining">
              {budgetRemaining} Baudi
            </span>
          </div>
        </Card>
      </motion.div>

      {/* Team Stats */}
      {selectedList.length > 0 && (
        <motion.div
          initial={{ y: -10, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ delay: 0.15 }}
          className="grid grid-cols-2 lg:grid-cols-4 gap-3"
        >
          <TeamStatCard
            icon={<TrendingUp className="w-4 h-4" />}
            label="Media Score"
            value={teamStats.avgScore.toFixed(0)}
            color="gold"
          />
          <TeamStatCard
            icon={<Zap className="w-4 h-4" />}
            label="High Perf."
            value={teamStats.highPerformers}
            color="purple"
          />
          <TeamStatCard
            icon={<Shield className="w-4 h-4" />}
            label="Debuttanti"
            value={teamStats.debuttanti}
            color="cyan"
          />
          <TeamStatCard
            icon={<Star className="w-4 h-4" />}
            label="Capitano"
            value={teamStats.captainBonus > 0 ? '+' + teamStats.captainBonus.toFixed(0) : '-'}
            color="green"
          />
        </motion.div>
      )}

      {/* Selected Team */}
      <AnimatePresence>
        {selectedList.length > 0 && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="space-y-4"
            data-testid="selected-team"
          >
            <Card className="p-5">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-xl font-bold text-amber-400 flex items-center gap-2">
                  <Users className="w-5 h-5" />
                  La tua squadra ({selectedList.length}/7)
                </h3>
                <div className="flex gap-2">
                  <Badge variant={selectedList.length === 7 ? 'success' : 'warning'} size="sm">
                    {selectedList.length === 7 ? 'Completa' : 'In costruzione'}
                  </Badge>
                </div>
              </div>

              {/* Titolari */}
              {titolari.length > 0 && (
                <div className="mb-4" data-testid="titolari-section">
                  <h4 className="text-sm text-gray-400 mb-3 flex items-center gap-2">
                    <Shield className="w-4 h-4" />
                    TITOLARI ({titolari.length}/5)
                  </h4>
                  <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-3">
                    {titolari.map((artista, index) => (
                      <motion.div
                        key={artista.id}
                        initial={{ opacity: 0, scale: 0.9 }}
                        animate={{ opacity: 1, scale: 1 }}
                        exit={{ opacity: 0, scale: 0.9 }}
                        transition={{ delay: index * 0.05 }}
                      >
                        <SelectedArtistaCard
                          artista={artista}
                          isCapitano={capitanoId === artista.id}
                          isCaptainEligible={true}
                          onSetCapitano={() => setCapitano(artista.id)}
                          onRemove={() => removeArtista(artista.id)}
                          testId={`titolare-${artista.id}`}
                        />
                      </motion.div>
                    ))}
                    {titolari.length < 5 && [...Array(5 - titolari.length)].map((_, i) => (
                      <EmptySlot key={`empty-t-${i}`} label="Titolare" />
                    ))}
                  </div>
                </div>
              )}

              {/* Riserve */}
              {riserve.length > 0 && (
                <div data-testid="riserve-section">
                  <h4 className="text-sm text-gray-400 mb-3 flex items-center gap-2">
                    <Shield className="w-4 h-4" />
                    RISERVE ({riserve.length}/2)
                  </h4>
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                    {riserve.map((artista, index) => (
                      <motion.div
                        key={artista.id}
                        initial={{ opacity: 0, scale: 0.9 }}
                        animate={{ opacity: 1, scale: 1 }}
                        exit={{ opacity: 0, scale: 0.9 }}
                        transition={{ delay: index * 0.05 }}
                      >
                        <SelectedArtistaCard
                          artista={artista}
                          isCapitano={false}
                          isCaptainEligible={false}
                          onSetCapitano={() => {}}
                          onRemove={() => removeArtista(artista.id)}
                          testId={`riserva-${artista.id}`}
                        />
                      </motion.div>
                    ))}
                    {riserve.length < 2 && [...Array(2 - riserve.length)].map((_, i) => (
                      <EmptySlot key={`empty-r-${i}`} label="Riserva" />
                    ))}
                  </div>
                </div>
              )}

              {/* Validate Button */}
              {selectedList.length === 7 && (
                <motion.div
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="mt-4"
                >
                  {!capitanoId ? (
                    <div className="p-4 bg-yellow-500/10 border border-yellow-500/30 rounded-lg text-yellow-400 text-sm text-center">
                      <Star className="w-4 h-4 inline mr-2" />
                      Seleziona un capitano tra i titolari per simulare il punteggio
                    </div>
                  ) : (
                    <Button
                      variant="primary"
                      onClick={handleValidate}
                      loading={validating}
                      className="w-full"
                      size="lg"
                      data-testid="validate-button"
                    >
                      {validating ? 'Calcolo in corso...' : 'Simula Punteggio'}
                    </Button>
                  )}
                </motion.div>
              )}
            </Card>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Validation Result */}
      <AnimatePresence>
        {validation && (
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.95 }}
          >
            <Card className={`p-6 border-2 ${validation.valid ? 'border-green-500' : 'border-red-500'}`}>
              <div className="flex items-start gap-4">
                <div className={`p-3 rounded-full ${validation.valid ? 'bg-green-500/20' : 'bg-red-500/20'}`}>
                  {validation.valid ? (
                    <Trophy className="w-6 h-6 text-green-400" />
                  ) : (
                    <Shield className="w-6 h-6 text-red-400" />
                  )}
                </div>
                <div className="flex-1">
                  <h3 className={`font-bold text-xl mb-2 ${validation.valid ? 'text-green-400' : 'text-red-400'}`}>
                    {validation.valid ? 'Squadra Valida!' : 'Squadra Non Valida'}
                  </h3>
                  <p className="text-gray-300 mb-4">{validation.message}</p>
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <span className="text-gray-400 text-sm">Budget:</span>{' '}
                      <span className="text-amber-400 font-mono font-bold">{validation.budget_totale}/100</span>
                    </div>
                    {validation.punteggio_simulato && (
                      <div>
                        <span className="text-gray-400 text-sm">Punteggio stimato:</span>{' '}
                        <span className="text-amber-400 font-mono font-bold text-xl">
                          <AnimatedCounter value={validation.punteggio_simulato} />
                        </span>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </Card>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Available Artists */}
      <Card className="p-5" data-testid="available-artists">
        <h3 className="text-xl font-bold text-amber-400 mb-4 flex items-center gap-2">
          <Users className="w-5 h-5" />
          Artisti disponibili ({availableArtisti.length})
        </h3>
        {loading ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
            {[...Array(6)].map((_, i) => (
              <div key={i} className="h-40 bg-gray-200 dark:bg-navy-700 animate-pulse rounded-lg" />
            ))}
          </div>
        ) : availableArtisti.length > 0 ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
            {availableArtisti.map((artista) => (
              <motion.div
                key={artista.id}
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                whileHover={{ scale: 1.01 }}
                whileTap={{ scale: 0.98 }}
              >
                <AvailableArtistaCard
                  artista={artista}
                  canAdd={canAdd(artista)}
                  onAdd={() => toggleArtista(artista)}
                  testId={`available-artista-${artista.id}`}
                />
              </motion.div>
            ))}
          </div>
        ) : (
          <EmptyState
            icon={<Users className="text-6xl text-gray-600" />}
            title="Tutti gli artisti sono stati selezionati"
            description="Hai aggiunto tutti gli artisti disponibili alla tua squadra"
          />
        )}
      </Card>
    </motion.div>
  )
}

function TeamStatCard({ icon, label, value, color }: { icon: React.ReactNode, label: string, value: string | number, color: 'gold' | 'cyan' | 'green' | 'purple' }) {
  const colorStyles = {
    gold: 'text-amber-400 bg-amber-500/10',
    cyan: 'text-cyan-400 bg-cyan-500/10',
    green: 'text-green-400 bg-green-500/10',
    purple: 'text-purple-400 bg-purple-500/10'
  }

  return (
    <motion.div
      whileHover={{ y: -2 }}
      className={`p-3 rounded-xl border border-gray-300 dark:border-navy-700 ${colorStyles[color]}`}
    >
      <div className="flex items-center gap-2 mb-1">
        {icon}
        <span className="text-xs text-gray-400">{label}</span>
      </div>
      <span className="text-xl font-bold">{value}</span>
    </motion.div>
  )
}

interface SelectedArtistaCardProps {
  artista: SelectedArtista
  isCapitano: boolean
  isCaptainEligible: boolean
  onSetCapitano: () => void
  onRemove: () => void
  testId?: string
}

function SelectedArtistaCard({
  artista,
  isCapitano,
  isCaptainEligible,
  onSetCapitano,
  onRemove,
  testId
}: SelectedArtistaCardProps) {
  return (
    <motion.div
      className={`p-3 rounded-lg border-2 transition-all ${
        isCapitano
          ? 'border-amber-500 bg-amber-500/10 shadow-lg shadow-amber-500/20'
          : 'border-gray-300 dark:border-navy-700 bg-white dark:bg-navy-900 hover:border-gray-400 dark:hover:border-navy-600'
      }`}
      data-testid={testId}
    >
      <div className="flex items-start gap-2 mb-2">
        <Avatar
          initials={getInitials(artista.nome)}
          src={artista.image_url || undefined}
          size="sm"
          className="flex-shrink-0"
          alt={artista.nome}
        />
        <div className="flex-1 min-w-0">
          <div className="text-xs text-gray-400 dark:text-gray-400 text-gray-500 mb-1">{artista.quotazione_2026} baudi</div>
          <div className="font-semibold text-sm truncate text-amber-600 dark:text-amber-400">{artista.nome}</div>
        </div>
        <button
          onClick={(e) => { e.stopPropagation(); onRemove(); }}
          className="p-1 hover:bg-red-500/20 rounded transition-colors"
        >
          <Minus className="w-4 h-4 text-red-400" />
        </button>
      </div>
      <div className="flex items-center gap-2 mb-2">
        <PerformanceLevelBadge level={artista.predizione_2026?.livello_performer || 'MEDIUM'} size="sm" />
        {artista.debuttante_2026 && (
          <Badge variant="info" size="sm">DEBUT</Badge>
        )}
      </div>
      {isCaptainEligible && (
        <motion.button
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          onClick={onSetCapitano}
          className={`w-full text-xs font-bold py-1.5 rounded transition-colors ${
            isCapitano
              ? 'bg-amber-500 text-navy-950'
              : 'bg-gray-200 dark:bg-navy-700 text-gray-700 dark:text-gray-300 hover:bg-gray-300 dark:hover:bg-navy-600'
          }`}
        >
          {isCapitano ? (
            <span className="flex items-center justify-center gap-1">
              <Crown className="w-3 h-3" /> CAPITANO
            </span>
          ) : (
            <span className="flex items-center justify-center gap-1">
              <Star className="w-3 h-3" /> Imposta C
            </span>
          )}
        </motion.button>
      )}
    </motion.div>
  )
}

interface AvailableArtistaCardProps {
  artista: ArtistaWithPredizione
  canAdd: boolean
  onAdd: () => void
  testId?: string
}

function AvailableArtistaCard({
  artista,
  canAdd,
  onAdd,
  testId
}: AvailableArtistaCardProps) {
  return (
    <div
      className={`p-4 rounded-lg border-2 cursor-pointer transition-all ${
        !canAdd
          ? 'border-red-500/50 bg-red-500/5 opacity-50'
          : 'border-gray-300 dark:border-navy-700 bg-white dark:bg-navy-900 hover:border-amber-500 hover:bg-gray-50 dark:hover:bg-navy-800'
      }`}
      onClick={canAdd ? onAdd : undefined}
      data-testid={testId}
    >
      <div className="flex justify-between items-start mb-3 gap-3">
        <Avatar
          initials={getInitials(artista.nome)}
          src={artista.image_url || undefined}
          size="md"
          className="flex-shrink-0"
          alt={artista.nome}
        />
        <div className="flex-1 min-w-0">
          <div className="font-semibold text-amber-600 dark:text-amber-400 truncate">{artista.nome}</div>
          {artista.genere_musicale && (
            <p className="text-sm text-gray-500 dark:text-gray-400 truncate">{artista.genere_musicale}</p>
          )}
        </div>
        <div className="text-right flex-shrink-0">
          <div className="text-xl font-bold text-amber-600 dark:text-amber-400">{artista.quotazione_2026}</div>
          <div className="text-xs text-gray-500 dark:text-gray-400">baudi</div>
        </div>
      </div>
      <div className="flex items-center justify-between pt-3 border-t border-gray-200 dark:border-navy-700">
        <div className="flex items-center gap-2">
          <PerformanceLevelBadge level={artista.predizione_2026?.livello_performer || 'MEDIUM'} size="sm" />
          {artista.debuttante_2026 && (
            <Badge variant="info" size="sm">DEBUT</Badge>
          )}
        </div>
        {artista.predizione_2026 && (
          <div className="text-right">
            <div className="text-xs text-gray-500 dark:text-gray-400">Score</div>
            <div className="font-mono font-bold text-amber-600 dark:text-amber-400 text-sm">
              {artista.predizione_2026.punteggio_predetto.toFixed(0)}
            </div>
          </div>
        )}
      </div>
      {canAdd && (
        <div className="mt-3 flex items-center justify-center text-amber-400 text-sm font-semibold">
          <Plus className="w-4 h-4 mr-1" /> Aggiungi
        </div>
      )}
    </div>
  )
}

function EmptySlot({ label }: { label: string }) {
  return (
    <motion.div
      whileHover={{ scale: 1.02 }}
      className="p-3 rounded-lg border-2 border-dashed border-gray-300 dark:border-navy-700 bg-gray-100 dark:bg-navy-900/50 flex items-center justify-center min-h-[100px]"
    >
      <div className="text-center text-gray-500">
        <Plus className="w-6 h-6 mx-auto mb-1 opacity-50" />
        <div className="text-xs">{label}</div>
      </div>
    </motion.div>
  )
}
