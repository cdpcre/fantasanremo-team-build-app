import { motion } from 'framer-motion'

export interface ProgressBarProps {
  value: number
  max?: number
  size?: 'sm' | 'md' | 'lg'
  color?: 'gold' | 'green' | 'yellow' | 'red' | 'blue'
  showLabel?: boolean
  label?: string
  animated?: boolean
}

export default function ProgressBar({
  value,
  max = 100,
  size = 'md',
  color = 'gold',
  showLabel = false,
  label,
  animated = true
}: ProgressBarProps) {
  const percentage = Math.min((value / max) * 100, 100)

  const sizeClasses = {
    sm: 'h-2',
    md: 'h-3',
    lg: 'h-4'
  }

  const colorClasses = {
    gold: 'bg-gradient-to-r from-amber-600 to-amber-400',
    green: 'bg-gradient-to-r from-green-600 to-green-400',
    yellow: 'bg-gradient-to-r from-yellow-600 to-yellow-400',
    red: 'bg-gradient-to-r from-red-600 to-red-400',
    blue: 'bg-gradient-to-r from-blue-600 to-blue-400'
  }

  const actualColor = color

  return (
    <div className="space-y-1">
      {(label || showLabel) && (
        <div className="flex items-center justify-between text-sm">
          {label && <span className="text-gray-300">{label}</span>}
          {showLabel && (
            <span className={`font-bold ${value > max ? 'text-red-400' : 'text-amber-400'}`}>
              {value} / {max}
            </span>
          )}
        </div>
      )}
      <div className={`w-full bg-navy-900 rounded-full overflow-hidden ${sizeClasses[size]}`}>
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${percentage}%` }}
          transition={{
            type: 'spring',
            stiffness: 300,
            damping: 20,
            delay: animated ? 0.1 : 0
          }}
          className={`h-full ${colorClasses[actualColor]} relative overflow-hidden`}
        >
          <motion.div
            className="absolute inset-0 bg-white/20"
            animate={animated ? {
              x: ['-100%', '100%']
            } : {}}
            transition={{
              duration: 1.5,
              repeat: Infinity,
              ease: 'linear',
              repeatDelay: 0.5
            }}
          />
        </motion.div>
      </div>
    </div>
  )
}
