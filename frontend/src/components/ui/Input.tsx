import { forwardRef, ReactNode } from 'react'
import { motion, type HTMLMotionProps } from 'framer-motion'

export interface InputProps extends Omit<HTMLMotionProps<'input'>, 'size'> {
  label?: string
  error?: string
  icon?: ReactNode
  className?: string
}

const Input = forwardRef<HTMLInputElement, InputProps>(
  ({ className = '', label, error, icon, ...props }, ref) => {
    return (
      <div className="space-y-1">
        {label && (
          <label className="block text-sm font-medium text-gray-300">
            {label}
          </label>
        )}
        <div className="relative">
          {icon && (
            <div className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400">
              {icon}
            </div>
          )}
          <motion.input
            ref={ref}
            className={`w-full px-4 py-2.5 bg-white dark:bg-navy-900 border rounded-lg text-gray-900 dark:text-gray-100 placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-amber-500/50 focus:border-amber-500 transition-all duration-200 ${
              icon ? 'pl-10' : 'pl-4'
            } ${
              error ? 'border-red-500 focus:ring-red-500/50' : 'border-gray-300 dark:border-navy-700'
            } ${className}`}
            whileFocus={{ scale: 1.01 }}
            {...props}
          />
        </div>
        {error && (
          <motion.p
            initial={{ opacity: 0, y: -5 }}
            animate={{ opacity: 1, y: 0 }}
            className="text-sm text-red-400"
          >
            {error}
          </motion.p>
        )}
      </div>
    )
  }
)

Input.displayName = 'Input'

export default Input
