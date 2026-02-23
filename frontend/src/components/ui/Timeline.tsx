import { motion } from 'framer-motion'
import { Trophy, Music, Calendar } from 'lucide-react'

interface TimelineItem {
  anno: number
  posizione?: number
  punteggio?: number
  quotazione?: number
}

interface TimelineProps {
  items: TimelineItem[]
}

export default function Timeline({ items }: TimelineProps) {
  return (
    <div className="relative">
      {/* Timeline line */}
      <div className="absolute left-4 top-0 bottom-0 w-0.5 bg-gradient-to-b from-amber-500 via-cyan-500 to-purple-500" />

      <div className="space-y-4">
        {items.map((item, index) => (
          <motion.div
            key={item.anno}
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.4, delay: index * 0.1 }}
            className="relative pl-12"
          >
            {/* Timeline dot */}
            <motion.div
              whileHover={{ scale: 1.2 }}
              className="absolute left-[2px] w-8 h-8 rounded-full bg-navy-800 dark:bg-navy-700 border-2 border-amber-500 flex items-center justify-center z-10"
            >
              {item.posizione && item.posizione <= 3 ? (
                <Trophy className="w-4 h-4 text-amber-500" />
              ) : (
                <Music className="w-4 h-4 text-amber-500" />
              )}
            </motion.div>

            {/* Content */}
            <div className="bg-white dark:bg-navy-800 rounded-lg border border-gray-200 dark:border-navy-700 p-4 shadow-md hover:shadow-lg transition-shadow">
              <div className="flex items-center justify-between mb-2">
                <span className="text-lg font-bold text-gray-900 dark:text-white">
                  {item.anno}
                </span>
                {item.posizione && (
                  <span
                    className={`px-2 py-1 rounded-md text-sm font-semibold ${
                      item.posizione <= 3
                        ? 'bg-amber-500/20 text-amber-500 border border-amber-500'
                        : 'bg-navy-500/20 text-gray-600 dark:text-gray-400 border border-navy-500'
                    }`}
                  >
                    #{item.posizione}
                  </span>
                )}
              </div>

              <div className="flex gap-4 text-sm text-gray-600 dark:text-gray-400">
                {item.punteggio && (
                  <div className="flex items-center gap-1">
                    <Calendar className="w-4 h-4" />
                    <span>{item.punteggio} pt</span>
                  </div>
                )}
                {item.quotazione && (
                  <div className="flex items-center gap-1">
                    <span className="text-amber-500">{item.quotazione} baudi</span>
                  </div>
                )}
              </div>
            </div>
          </motion.div>
        ))}
      </div>
    </div>
  )
}
