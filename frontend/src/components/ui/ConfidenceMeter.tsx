import { motion } from 'framer-motion'

interface ConfidenceMeterProps {
  confidence: number
  label?: string
  size?: 'sm' | 'md' | 'lg'
}

const sizeStyles = {
  sm: { height: 8, text: 'text-xs' },
  md: { height: 12, text: 'text-sm' },
  lg: { height: 16, text: 'text-base' },
}

export default function ConfidenceMeter({ confidence, label, size = 'md' }: ConfidenceMeterProps) {
  // Handle edge case: confidence = 0 means no data available
  if (confidence === 0) {
    const { height } = sizeStyles[size]
    return (
      <div className="space-y-2">
        {label && (
          <div className="flex justify-between items-center">
            <span className="text-gray-600 dark:text-gray-400 text-sm">{label}</span>
            <span className="text-sm font-semibold text-gray-400 dark:text-gray-500">
              N/A
            </span>
          </div>
        )}
        <div className="relative w-full bg-gray-200 dark:bg-navy-900 rounded-full overflow-hidden opacity-40"
             style={{ height: `${height}px` }}>
          <span className="absolute inset-0 flex items-center justify-center text-xs text-gray-400 dark:text-gray-500">
            Dati insufficienti
          </span>
        </div>
      </div>
    )
  }

  const percentage = Math.round(confidence * 100)
  const { height, text } = sizeStyles[size]

  const getColor = () => {
    if (percentage >= 70) return 'bg-green-500'
    if (percentage >= 40) return 'bg-yellow-500'
    return 'bg-red-500'
  }

  return (
    <div className="space-y-2">
      {label && (
        <div className="flex justify-between items-center">
          <span className="text-gray-600 dark:text-gray-400 text-sm">{label}</span>
          <span className={`font-semibold ${text} text-gray-900 dark:text-white`}>{percentage}%</span>
        </div>
      )}
      <div className="relative w-full bg-gray-200 dark:bg-navy-900 rounded-full overflow-hidden" style={{ height: `${height}px` }}>
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${percentage}%` }}
          transition={{ duration: 1, ease: 'easeOut' }}
          className={`h-full ${getColor()} rounded-full`}
        />
      </div>
    </div>
  )
}
