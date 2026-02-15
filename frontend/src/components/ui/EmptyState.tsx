import { ReactNode } from 'react'
import { motion } from 'framer-motion'

export interface EmptyStateProps {
  icon?: ReactNode
  title: string
  description?: string
  action?: {
    label: string
    onClick: () => void
  }
}

export default function EmptyState({ icon, title, description, action }: EmptyStateProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="flex flex-col items-center justify-center py-16 px-4 text-center"
    >
      {icon && (
        <motion.div
          initial={{ scale: 0 }}
          animate={{ scale: 1 }}
          transition={{ delay: 0.2, type: 'spring', stiffness: 200 }}
          className="text-6xl mb-4"
        >
          {icon}
        </motion.div>
      )}
      <h3 className="text-xl font-bold text-gray-300 mb-2">{title}</h3>
      {description && (
        <p className="text-gray-400 mb-6 max-w-md">{description}</p>
      )}
      {action && (
        <motion.button
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.3 }}
          onClick={action.onClick}
          className="px-6 py-2 bg-amber-500 hover:bg-amber-600 text-navy-950 font-semibold rounded-lg transition-colors"
        >
          {action.label}
        </motion.button>
      )}
    </motion.div>
  )
}
