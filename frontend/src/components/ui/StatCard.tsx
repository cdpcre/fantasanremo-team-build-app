import { motion } from 'framer-motion'
import { LucideIcon } from 'lucide-react'

interface StatCardProps {
  label: string
  value: string | number
  icon?: LucideIcon
  color?: 'gold' | 'cyan' | 'green' | 'red' | 'purple'
  delay?: number
  trend?: {
    value: number
    label: string
  }
}

const colorStyles = {
  gold: 'text-amber-500',
  cyan: 'text-cyan-500',
  green: 'text-green-500',
  red: 'text-red-500',
  purple: 'text-purple-500',
}

const bgStyles = {
  gold: 'bg-amber-500/10 dark:bg-amber-500/20',
  cyan: 'bg-cyan-500/10 dark:bg-cyan-500/20',
  green: 'bg-green-500/10 dark:bg-green-500/20',
  red: 'bg-red-500/10 dark:bg-red-500/20',
  purple: 'bg-purple-500/10 dark:bg-purple-500/20',
}

export default function StatCard({ label, value, icon: Icon, color = 'gold', delay = 0, trend }: StatCardProps) {
  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.4, delay }}
      whileHover={{ scale: 1.02 }}
      className="bg-white dark:bg-navy-800 rounded-xl border border-gray-200 dark:border-navy-700 p-6 shadow-lg hover:shadow-xl transition-all duration-300"
    >
      <div className="flex items-start justify-between">
        <div className="flex-1">
          <p className="text-sm text-gray-600 dark:text-gray-400 mb-1">{label}</p>
          <p className={`text-3xl font-bold ${colorStyles[color]}`}>{value}</p>
          {trend && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: delay + 0.2 }}
              className={`mt-2 text-sm ${trend.value >= 0 ? 'text-green-500' : 'text-red-500'}`}
            >
              {trend.value >= 0 ? '↑' : '↓'} {Math.abs(trend.value)}% {trend.label}
            </motion.div>
          )}
        </div>
        {Icon && (
          <div className={`p-3 rounded-lg ${bgStyles[color]}`}>
            <Icon className={`w-6 h-6 ${colorStyles[color]}`} />
          </div>
        )}
      </div>
    </motion.div>
  )
}
