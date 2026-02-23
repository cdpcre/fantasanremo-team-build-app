import { motion } from 'framer-motion'
import { cn } from '@/utils/cn'

/**
 * Loading spinner component for indicating loading states.
 * Fully accessible with ARIA live regions.
 *
 * @example
 * ```tsx
 * <Loading size="md" />
 * ```
 *
 * @example With text
 * ```tsx
 * <Loading text="Loading artists..." />
 * ```
 */
export interface LoadingProps {
  size?: 'xs' | 'sm' | 'md' | 'lg' | 'xl'
  color?: 'gold' | 'white' | 'navy'
  text?: string
  className?: string
  ariaLabel?: string
}

const sizeStyles = {
  xs: 'h-4 w-4 border-2',
  sm: 'h-6 w-6 border-2',
  md: 'h-8 w-8 border-3',
  lg: 'h-12 w-12 border-4',
  xl: 'h-16 w-16 border-4',
}

const colorStyles = {
  gold: 'border-amber-500 border-t-transparent',
  white: 'border-white border-t-transparent',
  navy: 'border-navy-600 border-t-transparent',
}

export default function Loading({
  size = 'md',
  color = 'gold',
  text,
  className = '',
  ariaLabel = 'Loading',
}: LoadingProps) {
  return (
    <div className={cn('flex flex-col items-center justify-center gap-3', className)}>
      <motion.div
        className={cn('rounded-full', sizeStyles[size], colorStyles[color])}
        animate={{ rotate: 360 }}
        transition={{
          duration: 1,
          repeat: Infinity,
          ease: 'linear',
        }}
        role="status"
        aria-label={ariaLabel}
      />
      {text && (
        <motion.p
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.2 }}
          className="text-sm text-gray-400"
        >
          {text}
        </motion.p>
      )}
    </div>
  )
}

/**
 * Loading dots component for alternative loading indicator
 */
export interface LoadingDotsProps {
  size?: 'sm' | 'md' | 'lg'
  color?: 'gold' | 'white' | 'navy'
  className?: string
}

const dotSizeStyles = {
  sm: 'h-1.5 w-1.5',
  md: 'h-2 w-2',
  lg: 'h-3 w-3',
}

const dotColorStyles = {
  gold: 'bg-amber-500',
  white: 'bg-white',
  navy: 'bg-navy-600',
}

export function LoadingDots({
  size = 'md',
  color = 'gold',
  className = '',
}: LoadingDotsProps) {
  return (
    <div className={cn('flex items-center gap-1', className)} role="status" aria-label="Loading">
      {[0, 1, 2].map((index) => (
        <motion.div
          key={index}
          className={cn('rounded-full', dotSizeStyles[size], dotColorStyles[color])}
          animate={{
            scale: [1, 1.2, 1],
            opacity: [0.5, 1, 0.5],
          }}
          transition={{
            duration: 0.8,
            repeat: Infinity,
            delay: index * 0.15,
            ease: 'easeInOut',
          }}
        />
      ))}
    </div>
  )
}

/**
 * Full page loading overlay
 */
export interface LoadingPageProps {
  text?: string
  logo?: React.ReactNode
}

export function LoadingPage({ text = 'Loading...', logo }: LoadingPageProps) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-navy-950">
      <motion.div
        initial={{ opacity: 0, scale: 0.8 }}
        animate={{ opacity: 1, scale: 1 }}
        className="flex flex-col items-center gap-6"
      >
        {logo && (
          <motion.div
            animate={{ rotate: 360 }}
            transition={{ duration: 3, repeat: Infinity, ease: 'linear' }}
          >
            {logo}
          </motion.div>
        )}
        <Loading size="lg" text={text} />
      </motion.div>
    </div>
  )
}

/**
 * Loading skeleton for content placeholders
 */
export interface LoadingSkeletonProps {
  count?: number
  className?: string
}

export function LoadingSkeleton({ count = 3, className = '' }: LoadingSkeletonProps) {
  return (
    <div className={cn('space-y-4', className)}>
      {Array.from({ length: count }).map((_, index) => (
        <motion.div
          key={index}
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: index * 0.1 }}
          className="bg-navy-800 rounded-lg p-4 space-y-3"
        >
          <div className="flex items-center gap-4">
            <div className="h-12 w-12 bg-navy-700 rounded-full animate-pulse" />
            <div className="flex-1 space-y-2">
              <div className="h-4 bg-navy-700 rounded w-3/4 animate-pulse" />
              <div className="h-3 bg-navy-700 rounded w-1/2 animate-pulse" />
            </div>
          </div>
        </motion.div>
      ))}
    </div>
  )
}
