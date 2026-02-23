import { ReactNode, useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { cn } from '@/utils/cn'

/**
 * Avatar component for displaying user or entity images.
 * Supports images, initials, and fallback icons.
 *
 * @example
 * ```tsx
 * <Avatar src="/avatar.jpg" alt="User name" size="md" />
 * ```
 *
 * @example With initials
 * ```tsx
 * <Avatar initials="JD" size="lg" />
 * ```
 *
 * @example With status indicator
 * ```tsx
 * <Avatar src="/avatar.jpg" status="online" />
 * ```
 */
export interface AvatarProps {
  src?: string
  alt?: string
  initials?: string
  fallback?: ReactNode
  size?: 'xs' | 'sm' | 'md' | 'lg' | 'xl'
  status?: 'online' | 'offline' | 'away' | 'busy'
  className?: string
  onClick?: () => void
}

const sizeStyles = {
  xs: 'h-6 w-6 text-xs',
  sm: 'h-8 w-8 text-sm',
  md: 'h-10 w-10 text-base',
  lg: 'h-12 w-12 text-lg',
  xl: 'h-16 w-16 text-xl',
}

const statusSizeStyles = {
  xs: 'h-1.5 w-1.5',
  sm: 'h-2 w-2',
  md: 'h-2.5 w-2.5',
  lg: 'h-3 w-3',
  xl: 'h-3.5 w-3.5',
}

const statusColors = {
  online: 'bg-green-500',
  offline: 'bg-gray-400',
  away: 'bg-yellow-500',
  busy: 'bg-red-500',
}

export default function Avatar({
  src,
  alt,
  initials,
  fallback,
  size = 'md',
  status,
  className = '',
  onClick,
}: AvatarProps) {
  const [imageError, setImageError] = useState(false)
  const [imageLoaded, setImageLoaded] = useState(false)

  // Reset image states when src changes
  useEffect(() => {
    if (src) {
      setImageError(false)
      setImageLoaded(false)
    }
  }, [src])

  const handleClick = () => {
    if (onClick) {
      onClick()
    }
  }

  const renderContent = () => {
    // Show image if available and not errored
    if (src && !imageError) {
      return (
        <>
          {/* Skeleton loader */}
          {!imageLoaded && (
            <div className="absolute inset-0 bg-navy-700 dark:bg-navy-700 bg-gray-200 rounded-full animate-pulse" />
          )}
          {/* Image - always rendered but visibility controlled by opacity */}
          <motion.img
            src={src}
            alt={alt || 'Avatar'}
            className="h-full w-full object-cover rounded-full"
            initial={{ opacity: 0 }}
            animate={{ opacity: imageLoaded ? 1 : 0 }}
            onError={() => setImageError(true)}
            onLoad={() => setImageLoaded(true)}
          />
        </>
      )
    }

    // Show initials if available
    if (initials) {
      return (
        <span className="font-semibold text-amber-600 dark:text-amber-400 flex items-center justify-center w-full h-full">
          {initials.slice(0, 2).toUpperCase()}
        </span>
      )
    }

    // Show fallback
    if (fallback) {
      return <>{fallback}</>
    }

    // Default fallback
    return (
      <svg
        className="h-2/3 w-2/3 text-gray-400"
        fill="currentColor"
        viewBox="0 0 24 24"
      >
        <path d="M24 20.993V24H0v-2.996A14.977 14.977 0 0112.004 15c4.904 0 9.26 2.354 11.996 5.993zM16.002 8.999a4 4 0 11-8 0 4 4 0 018 0z" />
      </svg>
    )
  }

  return (
    <div
      className={cn(
        'relative inline-flex items-center justify-center rounded-full bg-gray-100 dark:bg-navy-800 overflow-hidden',
        sizeStyles[size],
        onClick && 'cursor-pointer hover:ring-2 hover:ring-amber-500 transition-all',
        className
      )}
      onClick={handleClick}
      role="img"
      aria-label={alt || (initials && `Avatar for ${initials}`)}
    >
      {renderContent()}

      {/* Status indicator */}
      {status && (
        <span
          className={cn(
            'absolute bottom-0 right-0 rounded-full border-2 border-white dark:border-navy-950',
            statusColors[status],
            statusSizeStyles[size]
          )}
          aria-label={`Status: ${status}`}
        />
      )}
    </div>
  )
}

/**
 * Avatar group component for displaying multiple avatars
 */
export interface AvatarGroupProps {
  children: ReactNode
  max?: number
  total?: number
  className?: string
}

export function AvatarGroup({
  children,
  max = 3,
  total,
  className = '',
}: AvatarGroupProps) {
  const avatars = Array.isArray(children) ? children : [children]
  const visibleAvatars = avatars.slice(0, max)
  const remainingCount = (total ?? avatars.length) - max

  return (
    <div className={cn('flex -space-x-2', className)}>
      {visibleAvatars.map((avatar, index) => (
        <div key={index} className="ring-2 ring-white dark:ring-navy-950 rounded-full">
          {avatar}
        </div>
      ))}
      {remainingCount > 0 && (
        <div
          className={cn(
            'ring-2 ring-white dark:ring-navy-950 rounded-full bg-navy-700 dark:bg-navy-700 bg-gray-200 flex items-center justify-center text-xs font-medium text-gray-600 dark:text-gray-300',
            'h-8 w-8'
          )}
        >
          +{remainingCount}
        </div>
      )}
    </div>
  )
}
