import { motion } from 'framer-motion'

type PerformanceLevel = 'HIGH' | 'MEDIUM' | 'LOW' | 'DEBUTTANTE'

interface PerformanceLevelBadgeProps {
  level: PerformanceLevel
  size?: 'sm' | 'md' | 'lg'
}

const levelConfig = {
  HIGH: {
    label: 'Alto',
    bgColor: 'bg-green-500/20 dark:bg-green-500/30',
    borderColor: 'border-green-500',
    textColor: 'text-green-500',
    icon: '🔥',
  },
  MEDIUM: {
    label: 'Medio',
    bgColor: 'bg-yellow-500/20 dark:bg-yellow-500/30',
    borderColor: 'border-yellow-500',
    textColor: 'text-yellow-500',
    icon: '⚡',
  },
  LOW: {
    label: 'Basso',
    bgColor: 'bg-red-500/20 dark:bg-red-500/30',
    borderColor: 'border-red-500',
    textColor: 'text-red-500',
    icon: '💧',
  },
  DEBUTTANTE: {
    label: 'Debuttante',
    bgColor: 'bg-purple-500/20 dark:bg-purple-500/30',
    borderColor: 'border-purple-500',
    textColor: 'text-purple-500',
    icon: '🌟',
  },
}

const sizeStyles = {
  sm: { padding: 'px-2 py-1', text: 'text-xs', icon: 'text-sm' },
  md: { padding: 'px-3 py-1.5', text: 'text-sm', icon: 'text-base' },
  lg: { padding: 'px-4 py-2', text: 'text-base', icon: 'text-lg' },
}

export default function PerformanceLevelBadge({ level, size = 'md' }: PerformanceLevelBadgeProps) {
  const config = levelConfig[level]
  const style = sizeStyles[size]

  return (
    <motion.div
      initial={{ scale: 0.8, opacity: 0 }}
      animate={{ scale: 1, opacity: 1 }}
      transition={{ duration: 0.3 }}
      whileHover={{ scale: 1.05 }}
      className={`inline-flex items-center gap-2 ${style.padding} ${style.text} rounded-lg border ${config.bgColor} ${config.borderColor} ${config.textColor} font-semibold`}
    >
      <span className={style.icon}>{config.icon}</span>
      <span>{config.label}</span>
    </motion.div>
  )
}
