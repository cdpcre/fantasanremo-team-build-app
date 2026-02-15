import { HTMLAttributes, ReactNode } from 'react'
import { motion, HTMLMotionProps } from 'framer-motion'

export interface BadgeProps extends Omit<HTMLMotionProps<'span'>, keyof HTMLAttributes<HTMLSpanElement>> {
  variant?: 'default' | 'success' | 'warning' | 'danger' | 'info' | 'gold'
  size?: 'sm' | 'md'
  className?: string
  children?: ReactNode
}

const Badge = ({
  className = '',
  variant = 'default',
  size = 'sm',
  children,
  ...props
}: BadgeProps) => {
  const baseClasses = 'font-bold rounded inline-flex items-center gap-1'

  const variantClasses = {
    default: 'bg-gray-500/20 text-gray-400 border border-gray-500/30',
    success: 'bg-green-500/20 text-green-400 border border-green-500/30',
    warning: 'bg-yellow-500/20 text-yellow-400 border border-yellow-500/30',
    danger: 'bg-red-500/20 text-red-400 border border-red-500/30',
    info: 'bg-blue-500/20 text-blue-400 border border-blue-500/30',
    gold: 'bg-amber-500/20 text-amber-400 border border-amber-500/30'
  }

  const sizeClasses = {
    sm: 'px-2 py-1 text-xs',
    md: 'px-3 py-1.5 text-sm'
  }

  const classes = `${baseClasses} ${variantClasses[variant]} ${sizeClasses[size]} ${className}`

  return (
    <motion.span
      className={classes}
      initial={{ scale: 0.9, opacity: 0 }}
      animate={{ scale: 1, opacity: 1 }}
      transition={{ duration: 0.2 }}
      {...props}
    >
      {children}
    </motion.span>
  )
}

export default Badge
