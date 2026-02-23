# Frontend Utilities

## Overview

The frontend includes several utility modules for common operations like animations, logging, and CSS class management.

---

## Table of Contents

- [Animations](#animations)
- [Logger](#logger)
- [Class Name Helper (cn)](#class-name-helper-cn)
- [Theme Context](#theme-context)

---

## Animations

**Location:** `/frontend/src/utils/animations.ts`

Centralized animation configurations using Framer Motion. Provides consistent motion design across the application.

### Features

- **Pre-built variants** for common animations (fade, slide, scale, modal, toast, etc.)
- **Transition presets** (smooth, bouncy, gentle, fast, normal, slow)
- **Stagger animations** for list items
- **Loading animations** (spinner, skeleton, shimmer)
- **Interactive animations** (button tap, card hover)

### Usage Examples

```tsx
import { motion } from 'framer-motion'
import {
  fadeInVariants,
  slideUpVariants,
  modalVariants,
  staggerContainer,
  transitions
} from '@/utils/animations'

// Fade in animation
<motion.div
  variants={fadeInVariants}
  initial="hidden"
  animate="visible"
>
  Content
</motion.div>

// Slide up with custom transition
<motion.div
  variants={slideUpVariants}
  initial="hidden"
  animate="visible"
  transition={transitions.smooth}
>
  Content
</motion.div>

// Modal with enter/exit animations
<Modal isOpen={show}>
  <motion.div
    variants={modalVariants}
    initial="hidden"
    animate="visible"
    exit="exit"
  >
    Modal content
  </motion.div>
</Modal>

// Staggered list
<motion.ul
  variants={staggerContainer}
  initial="hidden"
  animate="visible"
>
  {items.map((item, i) => (
    <motion.li key={i} variants={staggerItem}>
      {item}
    </motion.li>
  ))}
</motion.ul>
```

### Available Variants

| Variant | Description |
|---------|-------------|
| `pageVariants` | Page transition animations |
| `fadeInVariants` | Simple fade in/out |
| `slideUpVariants` | Slide up from bottom |
| `slideDownVariants` | Slide down from top |
| `scaleVariants` | Scale with fade |
| `staggerContainer` | Container for staggered children |
| `staggerItem` | Individual staggered item |
| `cardHoverVariants` | Card hover effects |
| `buttonTapVariants` | Button press/tap effects |
| `modalVariants` | Modal enter/exit |
| `modalOverlayVariants` | Modal backdrop |
| `toastVariants` | Toast slide in/out |
| `skeletonVariants` | Skeleton loading pulse |
| `shimmerVariants` | Shimmer effect |
| `spinnerVariants` | Loading spinner rotation |
| `pulseVariants` | Pulse animation |
| `accordionVariants` | Accordion open/close |
| `tabContentVariants` | Tab content transitions |
| `listItemVariants` | List item animations |
| `heroVariants` | Hero section stagger |
| `statsVariants` | Stats card animations |
| `progressVariants` | Progress bar animations |
| `counterVariants` | Counter number animations |
| `floatVariants` | Floating decoration animation |
| `glowPulseVariants` | Glow pulse effect |

### Available Transitions

```tsx
import { transitions } from '@/utils/animations'

// Use presets
transitions.smooth  // Spring: { type: 'spring', stiffness: 300, damping: 30 }
transitions.bouncy  // Spring: { type: 'spring', stiffness: 400, damping: 20 }
transitions.gentle  // Spring: { type: 'spring', stiffness: 200, damping: 40 }
transitions.fast    // Duration: 0.2s
transitions.normal  // Duration: 0.3s
transitions.slow    // Duration: 0.5s
```

### Helper Functions

```tsx
import { getStaggerDelay, createSpringTransition } from '@/utils/animations'

// Calculate stagger delay
const delay = getStaggerDelay(index, 0.1) // index * 0.1

// Create custom spring transition
const customSpring = createSpringTransition(400, 25)
```

### Animation Presets

```tsx
import { animationPresets } from '@/utils/animations'

// Card entrance
<motion.div {...animationPresets.cardEntrance} />

// Button with hover/tap
<motion.button {...animationPresets.buttonHover} />

// Modal entrance
<motion.div {...animationPresets.modalEntrance} />

// Staggered list
<motion.ul {...animationPresets.listStagger} />

// Fade in up
<motion.div {...animationPresets.fadeInUp} />
```

---

## Logger

**Location:** `/frontend/src/utils/logger.ts`

Simple logging utility with different log levels and context support.

### Features

- Multiple log levels (DEBUG, INFO, WARN, ERROR)
- Automatic timestamp formatting
- Context/metadata support
- Development-only debug logs
- Structured logging format

### Usage

```tsx
import { Logger } from '@/utils/logger'

// Info logging
Logger.info('User logged in', { userId: 123 })

// Warning
Logger.warn('API rate limit approaching', { requests: 95 })

// Error with context
Logger.error('Failed to load data', { endpoint: '/api/artisti' })

// Debug (only in development)
Logger.debug('Component state updated', { state: newState })
```

### Log Levels

| Level | Description | Production |
|-------|-------------|------------|
| `DEBUG` | Detailed debugging info | Hidden |
| `INFO` | General information | Shown |
| `WARN` | Warning messages | Shown |
| `ERROR` | Error messages | Shown |

### Output Format

```
[2026-02-08T10:30:45.123Z] [INFO] User logged in {"userId":123}
[2026-02-08T10:30:46.456Z] [ERROR] Failed to load data {"endpoint":"/api/artisti"}
```

---

## Class Name Helper (cn)

**Location:** `/frontend/src/utils/cn.ts`

Utility function for merging Tailwind CSS classes without conflicts. Combines `clsx` and `tailwind-merge`.

### Usage

```tsx
import { cn } from '@/utils/cn'

// Basic usage
className={cn('px-4 py-2', 'bg-blue-500')}

// Conditional classes
className={cn(
  'base-class',
  isActive && 'active-class',
  isError && 'error-class'
)}

// Override Tailwind classes (last one wins)
className={cn('px-4 py-2', 'px-6')} // Result: px-6 py-2

// With component props
function Button({ variant, size, className, ...props }) {
  return (
    <button
      className={cn(
        'base-button-styles',
        variant === 'primary' && 'bg-primary',
        variant === 'secondary' && 'bg-secondary',
        size === 'sm' && 'text-sm',
        size === 'lg' && 'text-lg',
        className // User classes override defaults
      )}
      {...props}
    />
  )
}
```

### Benefits

1. **Conflict Resolution:** Tailwind classes are properly merged (last class wins)
2. **Conditional Logic:** Easy conditional classes
3. **Composition:** Build complex class strings from multiple sources
4. **Type Safety:** Works with TypeScript

---

## Theme Context

**Location:** `/frontend/src/contexts/ThemeContext.tsx`

React context for managing light/dark theme with localStorage persistence and system preference detection.

### Features

- Light/dark theme toggle
- LocalStorage persistence
- System preference detection (prefers-color-scheme)
- Automatic theme application to document
- TypeScript support

### Usage

```tsx
import { ThemeProvider, useTheme } from '@/contexts/ThemeContext'

// 1. Wrap your app with ThemeProvider
function App() {
  return (
    <ThemeProvider>
      <YourApp />
    </ThemeProvider>
  )
}

// 2. Use the theme hook in components
function ThemeToggle() {
  const { theme, toggleTheme, setTheme } = useTheme()

  return (
    <div>
      <p>Current theme: {theme}</p>
      <button onClick={toggleTheme}>Toggle Theme</button>
      <button onClick={() => setTheme('light')}>Light</button>
      <button onClick={() => setTheme('dark')}>Dark</button>
    </div>
  )
}

// 3. Apply theme-specific styles
function ThemedComponent() {
  const { theme } = useTheme()

  return (
    <div className={cn(
      'base-styles',
      theme === 'dark' && 'dark:bg-gray-900 dark:text-white'
    )}>
      Content
    </div>
  )
}
```

### Context API

```tsx
interface ThemeContextType {
  theme: 'light' | 'dark'    // Current theme
  toggleTheme: () => void    // Toggle between light/dark
  setTheme: (theme: Theme) => void  // Set specific theme
}
```

### Implementation Details

- **Initialization:** Checks localStorage → system preference → defaults to 'light'
- **Persistence:** Saves theme to localStorage on change
- **DOM Update:** Adds/removes 'dark' class on `<html>` element
- **Type Safety:** Throws error if `useTheme()` is used outside `ThemeProvider`

### Tailwind Dark Mode Configuration

Ensure your `tailwind.config.js` is configured:

```js
module.exports = {
  darkMode: 'class', // Important! Use class-based dark mode
  // ... rest of config
}
```

Then use dark mode prefixes:

```tsx
<div className="bg-white dark:bg-gray-900 text-gray-900 dark:text-white">
  Adaptive content
</div>
```

---

## Best Practices

### Animations

1. **Keep it subtle:** Animations should enhance, not distract
2. **Respect preferences:** Check `prefers-reduced-motion` for accessibility
3. **Use consistent timing:** Stick to the defined transition presets
4. **Test performance:** Ensure animations run at 60fps

### Logging

1. **Use appropriate levels:** DEBUG for dev only, ERROR for production issues
2. **Include context:** Add relevant data to log messages
3. **Don't log sensitive data:** Avoid passwords, tokens, personal info
4. **Structure logs:** Use consistent format for easy parsing

### Class Names

1. **Use `cn()` for all dynamic classes:** Prevents Tailwind conflicts
2. **Put user classes last:** Allows overriding component defaults
3. **Keep it readable:** Break complex `cn()` calls across multiple lines
4. **Extract common patterns:** Create reusable class combinations

### Theme

1. **Test both themes:** Ensure UI looks good in light and dark
2. **Use semantic colors:** Let Tailwind handle color adaptation
3. **Respect system preference:** Default to user's system theme
4. **Provide toggle:** Always give users control over theme

---

## Related Documentation

- [Design System](./DESIGN_SYSTEM.md) - Visual design guidelines
- [Error Handling Guide](./ERROR_HANDLING_GUIDE.md) - Error handling patterns
- [Testing Guide](./TESTING.md) - Testing strategies
