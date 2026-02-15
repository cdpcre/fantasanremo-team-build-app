import { motion } from 'framer-motion'

interface UncertaintyIntervalProps {
  lower: number
  upper: number
  predicted: number
  size?: 'sm' | 'md' | 'lg'
}

const sizeStyles = {
  sm: { barWidth: 'w-16', barHeight: 'h-1.5', text: 'text-xs' },
  md: { barWidth: 'w-24', barHeight: 'h-2', text: 'text-sm' },
  lg: { barWidth: 'w-32', barHeight: 'h-2.5', text: 'text-base' },
}

export default function UncertaintyInterval({ lower, upper, predicted, size = 'md' }: UncertaintyIntervalProps) {
  const width = upper - lower
  const leftPos = width > 0 ? ((predicted - lower) / width) * 100 : 50
  const style = sizeStyles[size]

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.3 }}
      className="flex items-center gap-2"
    >
      <span className={`text-muted-foreground ${style.text}`}>
        [{lower.toFixed(0)} - {upper.toFixed(0)}]
      </span>
      <div className={`${style.barWidth} ${style.barHeight} bg-gray-700 dark:bg-gray-600 rounded-full relative`}>
        <motion.div
          initial={{ left: '0%' }}
          animate={{ left: `${Math.max(0, Math.min(100, leftPos))}%` }}
          transition={{ duration: 0.5, ease: 'easeOut' }}
          className={`absolute ${style.barHeight} bg-yellow-500 rounded-full`}
          style={{ width: size === 'sm' ? '3px' : size === 'lg' ? '5px' : '4px' }}
        />
      </div>
    </motion.div>
  )
}
