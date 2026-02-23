import { Variants, Transition } from 'framer-motion'

/**
 * Animation utilities and variants for Framer Motion
 * Centralized animation configurations for consistent motion design
 */

// Common transition settings
export const transitions: Record<string, Transition> = {
  smooth: { type: 'spring', stiffness: 300, damping: 30 },
  bouncy: { type: 'spring', stiffness: 400, damping: 20 },
  gentle: { type: 'spring', stiffness: 200, damping: 40 },
  fast: { duration: 0.2 },
  normal: { duration: 0.3 },
  slow: { duration: 0.5 },
}

// Page transition variants
export const pageVariants: Variants = {
  initial: { opacity: 0, y: 20 },
  enter: { opacity: 1, y: 0 },
  exit: { opacity: 0, y: -20 },
}

export const pageTransition = transitions.normal

// Fade in variants
export const fadeInVariants: Variants = {
  hidden: { opacity: 0 },
  visible: { opacity: 1 },
}

// Slide up variants
export const slideUpVariants: Variants = {
  hidden: { opacity: 0, y: 60 },
  visible: { opacity: 1, y: 0 },
}

// Slide down variants
export const slideDownVariants: Variants = {
  hidden: { opacity: 0, y: -60 },
  visible: { opacity: 1, y: 0 },
}

// Scale variants
export const scaleVariants: Variants = {
  hidden: { opacity: 0, scale: 0.9 },
  visible: { opacity: 1, scale: 1 },
}

// Stagger container for list items
export const staggerContainer: Variants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: {
      staggerChildren: 0.1,
      delayChildren: 0.2,
    },
  },
}

// Stagger item for list children
export const staggerItem: Variants = {
  hidden: { opacity: 0, y: 20 },
  visible: {
    opacity: 1,
    y: 0,
    transition: transitions.smooth,
  },
}

// Card hover variants
export const cardHoverVariants: Variants = {
  rest: {
    scale: 1,
    boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)',
  },
  hover: {
    scale: 1.02,
    boxShadow: '0 20px 40px rgba(255, 184, 0, 0.15)',
    transition: transitions.smooth,
  },
}

// Button press variants
export const buttonTapVariants = {
  tap: { scale: 0.95 },
  hover: { scale: 1.05 },
}

// Modal variants
export const modalVariants: Variants = {
  hidden: {
    opacity: 0,
    scale: 0.95,
    y: 20,
  },
  visible: {
    opacity: 1,
    scale: 1,
    y: 0,
    transition: {
      type: 'spring',
      damping: 25,
      stiffness: 300,
    },
  },
  exit: {
    opacity: 0,
    scale: 0.95,
    y: 20,
    transition: { duration: 0.2 },
  },
}

// Modal overlay/backdrop
export const modalOverlayVariants: Variants = {
  hidden: { opacity: 0 },
  visible: { opacity: 1 },
  exit: { opacity: 0 },
}

// Toast notification variants
export const toastVariants: Variants = {
  hidden: {
    x: '100%',
    opacity: 0,
  },
  visible: {
    x: 0,
    opacity: 1,
    transition: {
      type: 'spring',
      damping: 25,
      stiffness: 300,
    },
  },
  exit: {
    x: '100%',
    opacity: 0,
    transition: { duration: 0.2 },
  },
}

// Skeleton loading variants
export const skeletonVariants: Variants = {
  start: {
    opacity: 0.6,
  },
  end: {
    opacity: 1,
    transition: {
      repeat: Infinity,
      repeatType: 'reverse' as const,
      duration: 0.8,
    },
  },
}

// Shimmer effect for skeletons
export const shimmerVariants: Variants = {
  start: {
    x: '-100%',
  },
  end: {
    x: '100%',
    transition: {
      repeat: Infinity,
      duration: 1.5,
      ease: 'linear',
    },
  },
}

// Loading spinner variants
export const spinnerVariants: Variants = {
  rotate: {
    rotate: 360,
    transition: {
      repeat: Infinity,
      duration: 1,
      ease: 'linear',
    },
  },
}

// Pulse animation
export const pulseVariants: Variants = {
  pulse: {
    scale: [1, 1.05, 1],
    opacity: [1, 0.7, 1],
    transition: {
      repeat: Infinity,
      duration: 2,
    },
  },
}

// Ripple effect for buttons
export const rippleVariants: Variants = {
  start: {
    scale: 0,
    opacity: 0.5,
  },
  end: {
    scale: 4,
    opacity: 0,
    transition: {
      duration: 0.6,
      ease: 'easeOut',
    },
  },
}

// Accordion variants
export const accordionVariants: Variants = {
  open: {
    height: 'auto',
    opacity: 1,
    transition: {
      type: 'spring',
      stiffness: 300,
      damping: 30,
    },
  },
  closed: {
    height: 0,
    opacity: 0,
    transition: {
      type: 'spring',
      stiffness: 300,
      damping: 30,
    },
  },
}

// Tab content variants
export const tabContentVariants: Variants = {
  hidden: {
    opacity: 0,
    x: -20,
  },
  visible: {
    opacity: 1,
    x: 0,
    transition: transitions.smooth,
  },
}

// List item variants with stagger
export const listItemVariants: Variants = {
  hidden: {
    opacity: 0,
    x: -20,
  },
  visible: {
    opacity: 1,
    x: 0,
    transition: transitions.smooth,
  },
}

// Hero section animations
export const heroVariants: Variants = {
  hidden: {},
  visible: {
    transition: {
      staggerChildren: 0.15,
      delayChildren: 0.3,
    },
  },
}

export const heroItemVariants: Variants = {
  hidden: { opacity: 0, y: 30 },
  visible: {
    opacity: 1,
    y: 0,
    transition: transitions.bouncy,
  },
}

// Stats card variants
export const statsVariants: Variants = {
  hidden: { opacity: 0, scale: 0.8 },
  visible: {
    opacity: 1,
    scale: 1,
    transition: transitions.smooth,
  },
}

// Progress bar variants
export const progressVariants = {
  start: { width: '0%' },
  complete: (percentage: number) => ({
    width: `${percentage}%`,
    transition: {
      type: 'spring',
      stiffness: 100,
      damping: 15,
      mass: 1,
    },
  }),
}

// Counter animation (for numbers)
export const counterVariants: Variants = {
  hidden: { opacity: 0, scale: 0.5 },
  visible: {
    opacity: 1,
    scale: 1,
    transition: {
      type: 'spring',
      stiffness: 200,
      damping: 20,
    },
  },
}

// Floating animation (for decorative elements)
export const floatVariants: Variants = {
  floating: {
    y: [-10, 10, -10],
    transition: {
      repeat: Infinity,
      duration: 3,
      ease: 'easeInOut',
    },
  },
}

// Glow pulse animation
export const glowPulseVariants: Variants = {
  pulse: {
    boxShadow: [
      '0 0 20px rgba(255, 184, 0, 0.3)',
      '0 0 40px rgba(255, 184, 0, 0.5)',
      '0 0 20px rgba(255, 184, 0, 0.3)',
    ],
    transition: {
      repeat: Infinity,
      duration: 2,
    },
  },
}

// Image gallery variants
export const galleryVariants: Variants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: {
      staggerChildren: 0.1,
    },
  },
}

export const galleryItemVariants: Variants = {
  hidden: {
    opacity: 0,
    scale: 0.8,
  },
  visible: {
    opacity: 1,
    scale: 1,
    transition: transitions.smooth,
  },
}

/**
 * Helper function to create staggered animation delays
 * @param index - Item index
 * @param baseDelay - Base delay in seconds
 * @returns Delay value for animation
 */
export function getStaggerDelay(index: number, baseDelay: number = 0.1): number {
  return index * baseDelay
}

/**
 * Helper function to create spring transition with custom parameters
 * @param stiffness - Spring stiffness
 * @param damping - Spring damping
 * @returns Transition object
 */
export function createSpringTransition(stiffness: number = 300, damping: number = 30): Transition {
  return { type: 'spring' as const, stiffness, damping }
}

/**
 * Preset animation combinations for common use cases
 */
export const animationPresets = {
  // Card entrance
  cardEntrance: {
    initial: { opacity: 0, y: 20 },
    animate: { opacity: 1, y: 0 },
    transition: transitions.smooth,
  },

  // Button hover
  buttonHover: {
    whileHover: { scale: 1.05 },
    whileTap: { scale: 0.95 },
    transition: transitions.fast,
  },

  // Modal entrance
  modalEntrance: {
    initial: 'hidden',
    animate: 'visible',
    exit: 'hidden',
    variants: modalVariants,
  },

  // List stagger
  listStagger: {
    variants: staggerContainer,
    initial: 'hidden',
    animate: 'visible',
  },

  // Fade in up
  fadeInUp: {
    initial: { opacity: 0, y: 20 },
    animate: { opacity: 1, y: 0 },
    transition: transitions.normal,
  },
}
