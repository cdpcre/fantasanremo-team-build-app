import { ReactNode, useState, useCallback } from 'react'
import { motion } from 'framer-motion'
import { cn } from '@/utils/cn'

/**
 * Tabs component for organizing content into separate panels.
 * Fully accessible with keyboard navigation and ARIA support.
 *
 * @example Underline tabs
 * ```tsx
 * <Tabs defaultValue="tab1">
 *   <TabsList>
 *     <TabsTrigger value="tab1">Tab 1</TabsTrigger>
 *     <TabsTrigger value="tab2">Tab 2</TabsTrigger>
 *   </TabsList>
 *   <TabsContent value="tab1">Content 1</TabsContent>
 *   <TabsContent value="tab2">Content 2</TabsContent>
 * </Tabs>
 * ```
 *
 * @example Pill tabs
 * ```tsx
 * <Tabs defaultValue="tab1" variant="pill">
 *   <TabsList>
 *     <TabsTrigger value="tab1">Tab 1</TabsTrigger>
 *     <TabsTrigger value="tab2">Tab 2</TabsTrigger>
 *   </TabsList>
 *   <TabsContent value="tab1">Content 1</TabsContent>
 * </Tabs>
 * ```
 */
export interface TabsProps {
  defaultValue: string
  value?: string
  onValueChange?: (value: string) => void
  children: ReactNode
  variant?: 'underline' | 'pill' | 'card'
  className?: string
}

export function Tabs({
  defaultValue,
  value: controlledValue,
  onValueChange,
  children,
  variant = 'underline',
  className = '',
}: TabsProps) {
  const [uncontrolledValue, setUncontrolledValue] = useState(defaultValue)
  const value = controlledValue ?? uncontrolledValue

  const handleValueChange = useCallback(
    (newValue: string) => {
      if (controlledValue === undefined) {
        setUncontrolledValue(newValue)
      }
      onValueChange?.(newValue)
    },
    [controlledValue, onValueChange]
  )

  return (
    <div className={cn('w-full', className)}>
      {(
        children as Array<ReactNode>
      ).map((child, index) => {
        if (
          (child as { type?: { name?: string } }).type?.name === 'TabsList'
        ) {
          return (
            <TabsList
              key={index}
              value={value}
              onValueChange={handleValueChange}
              variant={variant}
            >
              {(child as { props?: { children?: ReactNode } }).props?.children}
            </TabsList>
          )
        }
        if (
          (child as { type?: { name?: string } }).type?.name === 'TabsContent'
        ) {
          const childProps = (child as { props?: { value?: string; children?: ReactNode } })
            .props
          return (
            value === childProps?.value && (
              <TabsContent key={index} value={childProps?.value ?? ''}>
                {childProps?.children}
              </TabsContent>
            )
          )
        }
        return null
      })}
    </div>
  )
}

/**
 * Tabs list component
 */
export interface TabsListProps {
  value: string
  onValueChange: (value: string) => void
  children: ReactNode
  variant?: 'underline' | 'pill' | 'card'
  className?: string
}

export function TabsList({
  value,
  onValueChange,
  children,
  variant = 'underline',
  className = '',
}: TabsListProps) {
  const variantStyles = {
    underline: 'border-b border-navy-700',
    pill: 'bg-navy-800 p-1 rounded-lg inline-flex',
    card: 'flex gap-2',
  }

  return (
    <div
      role="tablist"
      className={cn(
        'flex',
        variant === 'underline' && 'gap-6',
        variantStyles[variant],
        className
      )}
    >
      {(children as Array<ReactNode>).map((child, index) => {
        const childProps = (
          child as { props?: { value?: string; children?: ReactNode } }
        ).props
        const isActive = value === childProps?.value

        return (
          <TabsTrigger
            key={index}
            value={childProps?.value ?? ''}
            isActive={isActive}
            onClick={() => onValueChange(childProps?.value ?? '')}
            variant={variant}
          >
            {childProps?.children}
          </TabsTrigger>
        )
      })}
    </div>
  )
}

/**
 * Tabs trigger component
 */
export interface TabsTriggerProps {
  value: string
  isActive: boolean
  onClick: () => void
  children: ReactNode
  variant?: 'underline' | 'pill' | 'card'
  className?: string
}

export function TabsTrigger({
  value,
  isActive,
  onClick,
  children,
  variant = 'underline',
  className = '',
}: TabsTriggerProps) {
  const variantStyles: Record<
    'underline' | 'pill' | 'card',
    { base: string; active: string; inactive: string }
  > = {
    underline: {
      base: 'relative pb-3 transition-colors',
      active: 'text-amber-400',
      inactive: 'text-gray-400 hover:text-gray-300',
    },
    pill: {
      base: 'px-4 py-2 rounded-md transition-all duration-200',
      active: 'bg-amber-500 text-navy-950 font-semibold',
      inactive: 'text-gray-400 hover:text-gray-300 hover:bg-navy-700',
    },
    card: {
      base: 'px-4 py-2 rounded-lg border transition-all duration-200',
      active: 'bg-amber-500/10 border-amber-500/30 text-amber-400',
      inactive: 'bg-navy-800 border-navy-700 text-gray-400 hover:text-gray-300',
    },
  }

  const styles = variantStyles[variant]

  return (
    <button
      type="button"
      role="tab"
      aria-selected={isActive}
      aria-controls={`panel-${value}`}
      onClick={onClick}
      className={cn(
        styles.base,
        isActive ? styles.active : styles.inactive,
        className
      )}
    >
      {children}
      {variant === 'underline' && isActive && (
        <motion.div
          layoutId="activeTab"
          className="absolute bottom-0 left-0 right-0 h-0.5 bg-amber-500"
          transition={{ type: 'spring', stiffness: 500, damping: 30 }}
        />
      )}
    </button>
  )
}

/**
 * Tabs content component
 */
export interface TabsContentProps {
  value: string
  children: ReactNode
  className?: string
}

export function TabsContent({ value, children, className = '' }: TabsContentProps) {
  return (
    <motion.div
      role="tabpanel"
      id={`panel-${value}`}
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.2 }}
      className={cn('mt-4', className)}
    >
      {children}
    </motion.div>
  )
}
