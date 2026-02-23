import { motion } from 'framer-motion'
import { spinnerVariants } from '@/utils/animations'

interface LoadingSpinnerProps {
  size?: 'sm' | 'md' | 'lg'
  color?: 'gold' | 'cyan' | 'white'
}

const sizeStyles = {
  sm: 'w-6 h-6 border-2',
  md: 'w-12 h-12 border-4',
  lg: 'w-16 h-16 border-4',
}

const colorStyles = {
  gold: 'border-amber-500 border-t-transparent',
  cyan: 'border-cyan-500 border-t-transparent',
  white: 'border-white border-t-transparent',
}

export default function LoadingSpinner({
  size = 'md',
  color = 'gold'
}: LoadingSpinnerProps) {
  return (
    <motion.div
      variants={spinnerVariants}
      animate="rotate"
      className={`${sizeStyles[size]} ${colorStyles[color]} rounded-full`}
      role="status"
      aria-label="Caricamento"
    >
      <span className="sr-only">Caricamento in corso...</span>
    </motion.div>
  )
}

/**
 * Full page loading overlay
 */
interface LoadingOverlayProps {
  message?: string
}

export function LoadingOverlay({ message = 'Caricamento...' }: LoadingOverlayProps) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-white/80 dark:bg-navy-950/80 backdrop-blur-sm">
      <div className="text-center space-y-4">
        <LoadingSpinner size="lg" />
        <motion.p
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.2 }}
          className="text-gray-700 dark:text-gray-300"
        >
          {message}
        </motion.p>
      </div>
    </div>
  )
}
