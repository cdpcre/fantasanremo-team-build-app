import { forwardRef, ReactNode } from 'react'
import { motion, HTMLMotionProps, TargetAndTransition, VariantLabels } from 'framer-motion'

export interface ButtonProps extends Omit<HTMLMotionProps<'button'>, 'whileHover' | 'whileTap'> {
  variant?: 'primary' | 'secondary' | 'ghost' | 'danger'
  size?: 'sm' | 'md' | 'lg'
  loading?: boolean
  whileHover?: TargetAndTransition | VariantLabels
  whileTap?: TargetAndTransition | VariantLabels
  children?: ReactNode
}

const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className = '', variant = 'primary', size = 'md', loading, disabled, children, whileHover, whileTap, ...props }, ref) => {
    const isDisabled = disabled || loading
    const hoverProps = isDisabled ? {} : { whileHover: whileHover ?? { scale: 1.02 } }
    const tapProps = isDisabled ? {} : { whileTap: whileTap ?? { scale: 0.98 } }

    const baseClasses = 'font-semibold rounded-lg transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed inline-flex items-center justify-center gap-2'
    const variantClasses = {
      primary: 'bg-amber-500 hover:bg-amber-600 text-navy-950 shadow-lg shadow-amber-500/20 hover:shadow-amber-500/40',
      secondary: 'bg-gray-200 dark:bg-navy-700 hover:bg-gray-300 dark:hover:bg-navy-600 text-gray-800 dark:text-gray-100 border border-gray-300 dark:border-navy-600',
      ghost: 'bg-transparent hover:bg-gray-200 dark:hover:bg-navy-700 text-gray-800 dark:text-gray-100 border border-transparent hover:border-gray-300 dark:hover:border-navy-600',
      danger: 'bg-red-500 hover:bg-red-600 text-white shadow-lg shadow-red-500/20'
    }

    const sizeClasses = {
      sm: 'px-3 py-1.5 text-sm',
      md: 'px-4 py-2 text-base',
      lg: 'px-6 py-3 text-lg'
    }

    const classes = `${baseClasses} ${variantClasses[variant]} ${sizeClasses[size]} ${className}`

    return (
      <motion.button
        ref={ref}
        className={classes}
        disabled={disabled || loading}
        {...hoverProps}
        {...tapProps}
        {...props}
      >
        {loading && (
          <svg className="animate-spin h-4 w-4" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
          </svg>
        )}
        {children}
      </motion.button>
    )
  }
)

Button.displayName = 'Button'

export default Button
