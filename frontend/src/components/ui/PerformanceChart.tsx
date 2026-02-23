import { useEffect, useMemo, useState } from 'react'
import {
  BarChart,
  Bar,
  ComposedChart,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
  Scatter
} from 'recharts'
import { motion } from 'framer-motion'

interface PerformanceChartProps {
  data: Array<{
    anno: number
    posizione: number
    punteggio: number
  }>
}

export default function PerformanceChart({ data }: PerformanceChartProps) {
  const [isMobile, setIsMobile] = useState(() =>
    typeof window !== 'undefined' ? window.innerWidth < 640 : false
  )

  useEffect(() => {
    if (typeof window === 'undefined') {
      return
    }

    const handleResize = () => {
      setIsMobile(window.innerWidth < 640)
    }

    window.addEventListener('resize', handleResize)
    return () => {
      window.removeEventListener('resize', handleResize)
    }
  }, [])

  const chartColors = useMemo(
    () => ({
      posizione: '#ffb800',
      punteggio: '#06b6d4',
      grid: 'rgba(26, 47, 74, 0.3)',
      text: '#9ca3af'
    }),
    []
  )

  const maxPosition = useMemo(() => {
    const positions = data.map((entry) => entry.posizione).filter((value) => value > 0)
    return positions.length > 0 ? Math.max(...positions) : 1
  }, [data])

  // Use a stable ranking ceiling so good placements stay visually high even
  // when an artist has few historical entries.
  const rankingCeiling = useMemo(() => Math.max(30, maxPosition), [maxPosition])

  const rankingPlotData = useMemo(
    () =>
      data.map((entry) => ({
        ...entry,
        rankingValue: rankingCeiling + 1 - entry.posizione
      })),
    [data, rankingCeiling]
  )

  const maxScore = useMemo(() => {
    const scores = data.map((entry) => entry.punteggio).filter((value) => value > 0)
    return scores.length > 0 ? Math.max(...scores) : 100
  }, [data])

  const xAxisTick = useMemo(
    () => ({ fill: chartColors.text, fontSize: isMobile ? 10 : 12 }),
    [chartColors.text, isMobile]
  )

  const yAxisTick = useMemo(
    () => ({ fill: chartColors.text, fontSize: isMobile ? 10 : 12 }),
    [chartColors.text, isMobile]
  )

  const chartMargin = useMemo(
    () => ({ top: 8, right: isMobile ? 8 : 20, left: 0, bottom: isMobile ? 2 : 8 }),
    [isMobile]
  )

  const xAxisInterval = isMobile && data.length > 4 ? 'preserveStartEnd' : 0
  const positionChartHeight = isMobile ? 190 : 230
  const scoreChartHeight = isMobile ? 190 : 220
  const scoreDomainMax = Math.max(100, Math.ceil(maxScore * 1.1))

  const PositionTooltip = ({
    active,
    payload
  }: {
    active?: boolean
    payload?: Array<{
      name: string
      value: number
      color: string
      payload: { anno: number }
    }>
  }) => {
    if (active && payload && payload.length) {
      const row = payload[0]?.payload as { anno: number; posizione: number } | undefined
      if (!row) return null
      return (
        <div className="bg-white dark:bg-navy-800 border border-gray-200 dark:border-navy-700 rounded-lg p-3 shadow-xl">
          <p className="font-semibold text-gray-900 dark:text-white mb-2">
            Edizione {row.anno}
          </p>
          <p className="text-sm" style={{ color: chartColors.posizione }}>
            Posizione: #{row.posizione}
          </p>
        </div>
      )
    }
    return null
  }

  const ScoreTooltip = ({
    active,
    payload
  }: {
    active?: boolean
    payload?: Array<{
      name: string
      value: number
      color: string
      payload: { anno: number }
    }>
  }) => {
    if (active && payload && payload.length) {
      const row = payload[0]?.payload as { anno: number; punteggio: number } | undefined
      if (!row) return null
      return (
        <div className="bg-white dark:bg-navy-800 border border-gray-200 dark:border-navy-700 rounded-lg p-3 shadow-xl">
          <p className="font-semibold text-gray-900 dark:text-white mb-2">
            Edizione {row.anno}
          </p>
          <p className="text-sm" style={{ color: chartColors.punteggio }}>
            Punteggio: {row.punteggio}
          </p>
        </div>
      )
    }
    return null
  }

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.5 }}
      className="w-full min-w-0"
    >
      <div className="space-y-5 sm:space-y-6">
        <div>
          <div className="flex items-center gap-2 mb-2">
            <div className="h-3 w-3 rounded-full" style={{ backgroundColor: chartColors.posizione }} />
            <span className="text-xs sm:text-sm font-medium text-gray-700 dark:text-gray-300">
              Posizione in classifica (alto = meglio)
            </span>
          </div>
          <ResponsiveContainer width="100%" height={positionChartHeight} minWidth={0}>
            <ComposedChart data={rankingPlotData} syncId="artist-history" margin={chartMargin}>
              <CartesianGrid strokeDasharray="3 3" stroke={chartColors.grid} />
              <XAxis
                dataKey="anno"
                stroke={chartColors.text}
                tick={xAxisTick}
                interval={xAxisInterval}
                minTickGap={isMobile ? 20 : 8}
              />
              <YAxis
                stroke={chartColors.text}
                tick={yAxisTick}
                domain={[1, rankingCeiling]}
                tickFormatter={(value) => `#${rankingCeiling + 1 - Number(value)}`}
                allowDecimals={false}
                width={isMobile ? 34 : 42}
              />
              <Tooltip content={<PositionTooltip />} />
              <Bar
                dataKey="rankingValue"
                name="Posizione"
                fill="rgba(255, 184, 0, 0.25)"
                barSize={isMobile ? 10 : 12}
                radius={[10, 10, 0, 0]}
              />
              <Scatter
                dataKey="rankingValue"
                name="Posizione"
                fill={chartColors.posizione}
                stroke="#f8fafc"
                strokeWidth={2}
              />
            </ComposedChart>
          </ResponsiveContainer>
        </div>

        <div>
          <div className="flex items-center gap-2 mb-2">
            <div className="h-3 w-3 rounded-full" style={{ backgroundColor: chartColors.punteggio }} />
            <span className="text-xs sm:text-sm font-medium text-gray-700 dark:text-gray-300">
              Punteggio Fantasanremo
            </span>
          </div>
          <ResponsiveContainer width="100%" height={scoreChartHeight} minWidth={0}>
            <BarChart data={data} syncId="artist-history" margin={chartMargin}>
              <CartesianGrid strokeDasharray="3 3" stroke={chartColors.grid} />
              <XAxis
                dataKey="anno"
                stroke={chartColors.text}
                tick={xAxisTick}
                interval={xAxisInterval}
                minTickGap={isMobile ? 20 : 8}
              />
              <YAxis
                stroke={chartColors.text}
                tick={yAxisTick}
                domain={[0, scoreDomainMax]}
                allowDecimals={false}
                width={isMobile ? 34 : 42}
              />
              <Tooltip content={<ScoreTooltip />} />
              <Bar
                dataKey="punteggio"
                name="Punteggio"
                fill={chartColors.punteggio}
                radius={[4, 4, 0, 0]}
                maxBarSize={isMobile ? 26 : 44}
              >
                {data.map((_entry, index) => (
                  <Cell key={`score-cell-${index}`} fill={chartColors.punteggio} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </motion.div>
  )
}
