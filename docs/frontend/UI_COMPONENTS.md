# UI Components Library

## Overview

The Fantasanremo frontend includes a comprehensive library of reusable UI components built with React, TypeScript, and Framer Motion. All components follow the Fantasanremo design system with consistent styling, animations, and accessibility.

## Table of Contents

- [Installation](#installation)
- [Core Components](#core-components)
- [Form Components](#form-components)
- [Feedback Components](#feedback-components)
- [Navigation Components](#navigation-components)
- [Data Display Components](#data-display-components)
- [Specialized Components](#specialized-components)

---

## Installation

All components are exported from `/frontend/src/components/ui/index.ts`:

```tsx
import {
  Button,
  Card,
  Modal,
  Badge,
  Input,
  Avatar,
  // ... and more
} from '@/components/ui'
```

---

## Core Components

### Button

Primary action button with variants and sizes.

**Props:**
```tsx
interface ButtonProps {
  variant?: 'primary' | 'secondary' | 'ghost' | 'danger'
  size?: 'sm' | 'md' | 'lg'
  loading?: boolean
  disabled?: boolean
  className?: string
  children: ReactNode
}
```

**Usage:**
```tsx
import { Button } from '@/components/ui'

<Button variant="primary" size="md">
  Create Team
</Button>

<Button variant="secondary" size="sm">
  Cancel
</Button>

<Button variant="danger" loading={true}>
  Deleting...
</Button>
```

**Variants:**
- `primary` - Gold gradient with shadow (default)
- `secondary` - Navy background with border
- `ghost` - Transparent with hover effect
- `danger` - Red for destructive actions

---

### Card

Container component with variants and sub-components.

**Props:**
```tsx
interface CardProps {
  variant?: 'default' | 'elevated' | 'bordered' | 'interactive'
  hover?: boolean
  delay?: number
  className?: string
  children: ReactNode
}
```

**Usage:**
```tsx
import { Card, CardHeader, CardBody, CardFooter, CardTitle, CardDescription } from '@/components/ui'

<Card variant="elevated">
  <CardHeader>
    <CardTitle>Artist Name</CardTitle>
    <CardDescription>Predizione 2026</CardDescription>
  </CardHeader>
  <CardBody>
    <p>Card content here</p>
  </CardBody>
  <CardFooter>
    <Button>View Details</Button>
  </CardFooter>
</Card>
```

**Sub-components:**
- `CardHeader` - Header section
- `CardBody` - Main content area
- `CardFooter` - Footer with actions
- `CardTitle` - Title heading
- `CardDescription` - Descriptive text

---

### Modal

Dialog overlay with animations and accessibility features.

**Props:**
```tsx
interface ModalProps {
  isOpen: boolean
  onClose: () => void
  title?: string
  size?: 'sm' | 'md' | 'lg' | 'xl'
  showCloseButton?: boolean
  children: ReactNode
}
```

**Usage:**
```tsx
import { Modal, ModalFooter } from '@/components/ui'

<Modal isOpen={showModal} onClose={() => setShowModal(false)} title="Confirm" size="md">
  <p>Are you sure you want to continue?</p>
  <ModalFooter>
    <Button variant="ghost" onClick={() => setShowModal(false)}>Cancel</Button>
    <Button variant="primary">Confirm</Button>
  </ModalFooter>
</Modal>
```

**Features:**
- ESC key to close
- Click outside to close
- Body scroll lock
- Accessibility attributes
- Smooth animations

**ConfirmModal Variant:**
```tsx
import { ConfirmModal } from '@/components/ui'

<ConfirmModal
  isOpen={showConfirm}
  onClose={() => setShowConfirm(false)}
  onConfirm={handleDelete}
  title="Delete Artist"
  message="This action cannot be undone."
  variant="danger"
  confirmLabel="Delete"
  cancelLabel="Cancel"
/>
```

---

### Badge

Small status or label indicator.

**Props:**
```tsx
interface BadgeProps {
  variant?: 'default' | 'success' | 'warning' | 'danger' | 'info' | 'gold'
  size?: 'sm' | 'md'
  className?: string
  children: ReactNode
}
```

**Usage:**
```tsx
import { Badge } from '@/components/ui'

<Badge variant="success">HIGH</Badge>
<Badge variant="warning">MEDIUM</Badge>
<Badge variant="danger">LOW</Badge>
<Badge variant="gold">PREMIUM</Badge>
```

---

### Avatar

User or artist avatar with fallback.

**Usage:**
```tsx
import { Avatar } from '@/components/ui'

<Avatar
  src="/artist-photo.jpg"
  alt="Artist Name"
  size="md"
  fallback="AN"
/>
```

---

## Form Components

### Input

Text input with validation support.

**Props:**
```tsx
interface InputProps {
  type?: string
  placeholder?: string
  value?: string
  onChange?: (e: ChangeEvent) => void
  error?: string
  disabled?: boolean
  label?: string
}
```

**Usage:**
```tsx
import { Input } from '@/components/ui'

<Input
  type="text"
  label="Artist Name"
  placeholder="Search artists..."
  value={search}
  onChange={(e) => setSearch(e.target.value)}
  error={error}
/>
```

---

### Select

Dropdown select component.

**Usage:**
```tsx
import { Select } from '@/components/ui'

<Select
  value={selected}
  onChange={setSelected}
  options={[
    { value: '1', label: 'Option 1' },
    { value: '2', label: 'Option 2' },
  ]}
  placeholder="Select option"
/>
```

---

### Slider

Range slider for numeric values.

**Usage:**
```tsx
import { Slider } from '@/components/ui'

<Slider
  min={0}
  max={100}
  value={value}
  onChange={setValue}
  label="Budget"
/>
```

---

## Feedback Components

### Loading

Loading indicators and overlays.

**Components:**
- `LoadingSpinner` - Rotating spinner
- `LoadingOverlay` - Full-screen overlay
- `Loading` - Flexible loading component
- `Skeleton` - Content placeholder
- `ProgressBar` - Progress indicator

**Usage:**
```tsx
import { LoadingSpinner, LoadingOverlay, Skeleton, ProgressBar } from '@/components/ui'

// Spinner
<LoadingSpinner size="md" />

// Overlay
<LoadingOverlay isLoading={true} message="Loading artists..." />

// Skeleton
<Skeleton variant="text" width="100%" height={20} />
<Skeleton variant="circle" size={40} />
<Skeleton variant="rect" width="100%" height={200} />

// Progress bar
<ProgressBar value={75} max={100} />
```

---

### Toast

Notification system for user feedback.

**Usage:**
```tsx
import { useToasts, ToastContainer } from '@/components/ui'

function App() {
  return (
    <>
      <YourApp />
      <ToastContainer />
    </>
  )
}

function Component() {
  const { addToast } = useToasts()

  const showToast = () => {
    addToast('Success message', 'success')
    addToast('Error message', 'error')
    addToast('Warning message', 'warning')
    addToast('Info message', 'info')
  }

  return <Button onClick={showToast}>Show Toast</Button>
}
```

---

### EmptyState

Placeholder for empty lists or states.

**Usage:**
```tsx
import { EmptyState } from '@/components/ui'

<EmptyState
  title="No artists found"
  description="Try adjusting your filters"
  icon={<SearchIcon />}
  action={<Button>Clear Filters</Button>}
/>
```

---

## Navigation Components

### Tabs

Tab navigation component.

**Usage:**
```tsx
import { Tabs, TabList, Tab, TabPanel } from '@/components/ui'

<Tabs defaultValue="tab1">
  <TabList>
    <Tab value="tab1">Overview</Tab>
    <Tab value="tab2">Details</Tab>
    <Tab value="tab3">History</Tab>
  </TabList>

  <TabPanel value="tab1">
    <p>Overview content</p>
  </TabPanel>

  <TabPanel value="tab2">
    <p>Details content</p>
  </TabPanel>

  <TabPanel value="tab3">
    <p>History content</p>
  </TabPanel>
</Tabs>
```

---

### ThemeToggle

Dark/light theme toggle button.

**Usage:**
```tsx
import { ThemeToggle } from '@/components/ui'

<ThemeToggle />
```

---

## Data Display Components

### StatCard

Card for displaying statistics with optional counter animation.

**Usage:**
```tsx
import { StatCard } from '@/components/ui'

<StatCard
  title="Total Artists"
  value={30}
  icon={<UsersIcon />}
  trend="+5%"
  trendUp={true}
/>
```

---

### ConfidenceMeter

Visual meter for confidence levels.

**Usage:**
```tsx
import { ConfidenceMeter } from '@/components/ui'

<ConfidenceMeter
  value={85}
  label="Confidence"
  max={100}
/>
```

---

### PerformanceChart

Chart for displaying performance over time.

**Usage:**
```tsx
import { PerformanceChart } from '@/components/ui'

<PerformanceChart
  data={performanceData}
  xKey="year"
  yKey="score"
  color="#ffd700"
/>
```

---

### Timeline

Vertical timeline component.

**Usage:**
```tsx
import { Timeline } from '@/components/ui'

<Timeline
  events={[
    { date: '2020', title: 'First Participation', description: '...' },
    { date: '2023', title: 'Winner', description: '...' },
  ]}
/>
```

---

### PerformanceLevelBadge

Badge showing performance level with icon.

**Usage:**
```tsx
import { PerformanceLevelBadge } from '@/components/ui'

<PerformanceLevelBadge level="high" />
<PerformanceLevelBadge level="medium" />
<PerformanceLevelBadge level="low" />
```

---

## Specialized Components

### AnimatedCounter

Animated number counter.

**Usage:**
```tsx
import { AnimatedCounter } from '@/components/ui'

<AnimatedCounter
  value={123}
  duration={1.5}
  prefix="€"
  decimals={0}
/>
```

---

### ProgressBar

Progress bar with animations.

**Usage:**
```tsx
import { ProgressBar } from '@/components/ui'

<ProgressBar
  value={75}
  max={100}
  label="Loading..."
  color="gold"
/>
```

---

## Component Variants

### Button Variants

| Variant | Description |
|---------|-------------|
| `primary` | Gold gradient, main CTAs |
| `secondary` | Navy background, secondary actions |
| `ghost` | Transparent, subtle actions |
| `danger` | Red, destructive actions |

### Card Variants

| Variant | Description |
|---------|-------------|
| `default` | Standard card with border |
| `elevated` | Card with shadow |
| `bordered` | Prominent border |
| `interactive` | Hover effects, clickable |

### Badge Variants

| Variant | Color | Usage |
|---------|-------|-------|
| `default` | Gray | Neutral |
| `success` | Green | Success states |
| `warning` | Yellow | Warnings |
| `danger` | Red | Errors |
| `info` | Blue | Information |
| `gold` | Gold | Premium/highlight |

---

## Animation Features

Most components include built-in animations using Framer Motion:

- **Entrance animations** - Fade in, slide up
- **Hover effects** - Scale, shadow changes
- **Interactive feedback** - Tap/click animations
- **Loading states** - Spinners, skeletons

Example with custom animations:
```tsx
import { motion } from 'framer-motion'
import { fadeInVariants } from '@/utils/animations'

<motion.div variants={fadeInVariants} initial="hidden" animate="visible">
  <Card>Your content</Card>
</motion.div>
```

---

## Accessibility

All components follow WCAG 2.1 AA standards:

- **Keyboard navigation** - All interactive elements accessible via Tab
- **ARIA attributes** - Proper labels and roles
- **Focus indicators** - Visible focus states
- **Screen reader support** - Semantic HTML

---

## Best Practices

1. **Import from index:** Use `@/components/ui` for imports
2. **Use variants:** Leverage built-in variants for consistency
3. **Compose components:** Combine sub-components (e.g., Card + CardHeader)
4. **Handle loading:** Use Loading/Skeleton components for async states
5. **Error states:** Use error props and Badge variants for feedback
6. **Responsive:** Components are mobile-first by default

---

## TypeScript Support

All components are fully typed with TypeScript:

```tsx
import type { ButtonProps, CardProps } from '@/components/ui'

// Props are fully typed
const MyButton = (props: ButtonProps) => {
  return <Button {...props} />
}
```

---

## Related Documentation

- [Design System](./DESIGN_SYSTEM.md) - Visual guidelines
- [Utilities](./UTILITIES.md) - Animation and utility functions
- [Error Handling](./ERROR_HANDLING_GUIDE.md) - Error patterns
- [Testing](./TESTING.md) - Testing strategies
