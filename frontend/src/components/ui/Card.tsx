import { ReactNode, HTMLAttributes } from 'react'
import { motion, HTMLMotionProps } from 'framer-motion'
import { cn } from '@/utils/cn'

export interface CardProps extends Omit<HTMLMotionProps<'div'>, keyof HTMLAttributes<HTMLDivElement>> {
  children: ReactNode
  className?: string
  delay?: number
  hover?: boolean
  variant?: 'default' | 'elevated' | 'bordered' | 'interactive'
}

export default function Card({
  children,
  className = '',
  delay = 0,
  hover = true,
  variant = 'default',
  ...props
}: CardProps) {
  const variantStyles = {
    default: 'bg-white dark:bg-navy-800/50 backdrop-blur border border-gray-200 dark:border-navy-700 rounded-xl',
    elevated: 'bg-white dark:bg-navy-800/80 backdrop-blur border border-gray-200 dark:border-navy-700 rounded-xl shadow-lg',
    bordered: 'bg-white dark:bg-navy-800/30 backdrop-blur border-2 border-gray-300 dark:border-navy-600 rounded-xl',
    interactive: 'bg-white dark:bg-navy-800/50 backdrop-blur border border-gray-200 dark:border-navy-700 rounded-xl cursor-pointer transition-all duration-200 hover:border-amber-500/50 hover:shadow-glow-amber',
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, delay }}
      whileHover={hover ? { y: -2 } : {}}
      className={cn(variantStyles[variant], 'overflow-hidden', className)}
      {...props}
    >
      {children}
    </motion.div>
  )
}

/**
 * Card header component
 */
export interface CardHeaderProps extends HTMLAttributes<HTMLDivElement> {
  children: ReactNode
  className?: string
}

export function CardHeader({ children, className = '', ...props }: CardHeaderProps) {
  return (
    <div className={cn('p-6 pb-4', className)} {...props}>
      {children}
    </div>
  )
}

/**
 * Card body component
 */
export interface CardBodyProps extends HTMLAttributes<HTMLDivElement> {
  children: ReactNode
  className?: string
}

export function CardBody({ children, className = '', ...props }: CardBodyProps) {
  return (
    <div className={cn('px-6 py-4', className)} {...props}>
      {children}
    </div>
  )
}

/**
 * Card footer component
 */
export interface CardFooterProps extends HTMLAttributes<HTMLDivElement> {
  children: ReactNode
  className?: string
}

export function CardFooter({ children, className = '', ...props }: CardFooterProps) {
  return (
    <div className={cn('p-6 pt-4 border-t border-gray-200 dark:border-navy-700', className)} {...props}>
      {children}
    </div>
  )
}

/**
 * Card title component
 */
export interface CardTitleProps extends HTMLAttributes<HTMLHeadingElement> {
  children: ReactNode
  className?: string
}

export function CardTitle({ children, className = '', ...props }: CardTitleProps) {
  return (
    <h3 className={cn('text-lg font-semibold text-gray-100', className)} {...props}>
      {children}
    </h3>
  )
}

/**
 * Card description component
 */
export interface CardDescriptionProps extends HTMLAttributes<HTMLParagraphElement> {
  children: ReactNode
  className?: string
}

export function CardDescription({ children, className = '', ...props }: CardDescriptionProps) {
  return (
    <p className={cn('text-sm text-gray-400 mt-1', className)} {...props}>
      {children}
    </p>
  )
}
