import { motion } from 'framer-motion'

export interface SkeletonProps {
  className?: string
  variant?: 'text' | 'rectangular' | 'circular'
  width?: string | number
  height?: string | number
  animation?: 'pulse' | 'wave' | 'none'
}

export default function Skeleton({
  className = '',
  variant = 'rectangular',
  width,
  height,
  animation = 'pulse'
}: SkeletonProps) {
  const variantClasses = {
    text: 'rounded',
    rectangular: 'rounded-lg',
    circular: 'rounded-full'
  }

  const pulseAnimation = {
    opacity: [0.4, 0.8, 0.4],
    transition: { duration: 1.5, repeat: Infinity, ease: 'easeInOut' as const }
  }
  const waveAnimation = {
    x: ['-100%', '100%'],
    transition: { duration: 1.5, repeat: Infinity, ease: 'easeInOut' as const }
  }
  const selectedAnimation = animation === 'pulse'
    ? pulseAnimation
    : animation === 'wave'
      ? waveAnimation
      : {}

  return (
    <motion.div
      className={`bg-navy-700 ${variantClasses[variant]} ${className}`}
      style={{ width, height }}
      animate={selectedAnimation}
    />
  )
}

export const ArtistCardSkeleton = () => (
  <div className="bg-navy-800 rounded-xl p-5 border border-navy-700">
    <div className="flex justify-between items-start mb-3">
      <div className="flex-1 space-y-2">
        <Skeleton width="70%" height={20} variant="text" />
        <Skeleton width="50%" height={14} variant="text" />
      </div>
      <div className="text-right space-y-1">
        <Skeleton width={40} height={28} variant="rectangular" />
        <Skeleton width={50} height={12} variant="text" />
      </div>
    </div>
    <div className="flex items-center justify-between pt-3 border-t border-navy-700">
      <div className="flex gap-2">
        <Skeleton width={50} height={20} variant="rectangular" />
        <Skeleton width={70} height={20} variant="rectangular" />
      </div>
      <div className="text-right space-y-1">
        <Skeleton width={60} height={12} variant="text" />
        <Skeleton width={50} height={18} variant="text" />
      </div>
    </div>
  </div>
)
