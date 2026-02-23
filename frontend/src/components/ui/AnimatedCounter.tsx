import { motion, useSpring, useTransform } from 'framer-motion'
import { useEffect, useState } from 'react'

interface AnimatedCounterProps {
  value: number
  duration?: number
  format?: (value: number) => string
  className?: string
}

export default function AnimatedCounter({
  value,
  duration = 1.5,
  format = (v) => v.toFixed(0),
  className = '',
}: AnimatedCounterProps) {
  const [displayValue, setDisplayValue] = useState(0)

  const spring = useSpring(0, {
    bounce: 0,
    duration: duration * 1000,
  })

  const display = useTransform(spring, (latest) => {
    return format(latest)
  })

  useEffect(() => {
    spring.set(value)
  }, [spring, value])

  useEffect(() => {
    const unsubscribe = display.on('change', (v) => {
      setDisplayValue(v as unknown as number)
    })
    return unsubscribe
  }, [display])

  return (
    <motion.span
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className={className}
    >
      {displayValue as unknown as string}
    </motion.span>
  )
}
