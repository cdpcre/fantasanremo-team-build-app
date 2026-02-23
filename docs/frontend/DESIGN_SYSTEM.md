# Fantasanremo Design System

## Overview

The Fantasanremo design system captures the visual identity of the FantaSanremo 2026 fantasy game, featuring a navy blue and gold color palette with orange accents. This system ensures consistency across all UI components while maintaining accessibility and performance standards.

## Table of Contents

- [Color Palette](#color-palette)
- [Typography](#typography)
- [Spacing](#spacing)
- [Border Radius](#border-radius)
- [Shadows & Elevation](#shadows--elevation)
- [Component Library](#component-library)
- [Accessibility Guidelines](#accessibility-guidelines)
- [Responsive Breakpoints](#responsive-breakpoints)

---

## Color Palette

### Primary Colors

Based on the Fantasanremo brand identity, featuring deep navy blues and gold accents.

| Tailwind Class | Hex Code | Usage |
|----------------|----------|-------|
| `fs-primary` | `#1a237e` | Main primary blue, CTAs, important actions |
| `fs-primary-dark` | `#0d1347` | Hover states, deeper backgrounds |
| `fs-primary-light` | #534bae` | Tinted backgrounds, subtle highlights |

### Navy Colors (Extended Palette)

| Tailwind Class | Hex Code | Usage |
|----------------|----------|-------|
| `navy-50` | `#f0f4f8` | Light backgrounds, subtle overlays |
| `navy-100` | `#d9e2ec` | Borders, dividers |
| `navy-200` | `#bcccdc` | Disabled states |
| `navy-300` | `#9fb3c8` | Muted text |
| `navy-400` | `#829ab1` | Secondary text |
| `navy-500` | `#627d98` | Body text |
| `navy-600` | `#486581` | Standard text |
| `navy-650` | `#3d5670` | Navigation active |
| `navy-700` | `#334e68` | Headers, emphasis |
| `navy-750` | `#284066` | Card backgrounds |
| `navy-800` | `#243b53` | Dark backgrounds |
| `navy-850` | `#1e3a5f` | Section backgrounds |
| `navy-900` | `#172b4d` | Deep backgrounds |
| `navy-950` | `#050f1a` | Primary background |

### Gold/Accent Colors

| Tailwind Class | Hex Code | Usage |
|----------------|----------|-------|
| `gold-400` | `#ffd700` | Primary accent, highlights |
| `gold-500` | `#ffb800` | Gold standard, emphasis |
| `gold-600` | `#e5a600` | Darker gold, hover states |

### Orange Accent (Fanta Brand)

| Tailwind Class | Hex Code | Usage |
|----------------|----------|-------|
| `orange-primary` | `#ff6b00` | Primary orange, banners |
| `orange-light` | `#ff9500` | Lighter orange, gradients |

### Semantic Colors

| Tailwind Class | Hex Code | Usage |
|----------------|----------|-------|
| `success` | `#4caf50` | Success states, confirmations |
| `error` | `#d32f2f` | Error states, destructive actions |
| `warning` | `#ff9800` | Warnings, alerts |
| `info` | `#2196f3` | Information, help text |

### Neutral Colors

| Tailwind Class | Hex Code | Usage |
|----------------|----------|-------|
| `white` | `#ffffff` | Primary text, highlights |
| `gray-50` | `#f9fafb` | Subtle backgrounds |
| `gray-100` | `#f3f4f6` | Cards, containers |
| `gray-200` | `#e5e7eb` | Borders, dividers |
| `gray-300` | `#d1d5db` | Disabled borders |
| `gray-400` | `#9ca3af` | Secondary text |
| `gray-500` | `#6b7280` | Muted text |
| `gray-600` | `#4b5563` | Body text |
| `gray-700` | `#374151` | Headers |
| `gray-800` | `#1f2937` | Dark backgrounds |
| `gray-900` | `#111827` | Deepest backgrounds |

### Semi-Transparent Colors

| Token | Value | Usage |
|-------|-------|-------|
| `card-bg` | `rgba(255, 255, 255, 0.1)` | Card backgrounds on dark |
| `button-bg` | `rgba(26, 35, 126, 0.8)` | Button backgrounds |
| `border-light` | `rgba(255, 255, 255, 0.2)` | Light borders on dark |
| `border-medium` | `rgba(255, 255, 255, 0.3)` | Medium borders |

---

## Typography

### Font Families

| Token | Value | Usage |
|-------|-------|-------|
| `font-sans` | `system-ui, -apple-system, sans-serif` | Body text, UI elements |
| `font-display` | `'Arial', sans-serif` | Headings, display text |

### Font Sizes

| Tailwind Class | Size | Line Height | Usage |
|----------------|------|-------------|-------|
| `text-xs` | 12px | 1.3 | Tiny text, metadata |
| `text-sm` | 14px | 1.4 | Small text, labels |
| `text-base` | 16px | 1.5 | Body text |
| `text-lg` | 18px | 1.5 | Emphasized body |
| `text-xl` | 20px | 1.4 | Subheadings |
| `text-2xl` | 24px | 1.3 | Section headings |
| `text-3xl` | 30px | 1.2 | Page headings |
| `text-4xl` | 36px | 1.2 | Large headings |
| `text-5xl` | 48px | 1.2 | Display headings |
| `text-6xl` | 60px | 1.1 | Hero text |

### Font Weights

| Tailwind Class | Weight | Usage |
|----------------|--------|-------|
| `font-normal` | 400 | Body text |
| `font-medium` | 500 | Emphasized text |
| `font-semibold` | 600 | Subheadings |
| `font-bold` | 700 | Headings, CTAs |

### Letter Spacing

| Tailwind Class | Value | Usage |
|----------------|-------|-------|
| `tracking-tight` | -0.025em | Large headings |
| `tracking-normal` | 0 | Body text |
| `tracking-wide` | 0.025em | Small text, labels |

---

## Spacing

### Spacing Scale

| Tailwind Class | Value | Usage |
|----------------|-------|-------|
| `space-0` | 0 | None |
| `space-px` | 1px | Hairline |
| `space-0.5` | 2px | Tight |
| `space-1` | 4px | Extra small |
| `space-2` | 8px | Small |
| `space-3` | 12px | Medium small |
| `space-4` | 16px | Medium |
| `space-5` | 20px | Medium large |
| `space-6` | 24px | Large |
| `space-8` | 32px | Extra large |
| `space-10` | 40px | XL |
| `space-12` | 48px | 2XL |
| `space-16` | 64px | 3XL |
| `space-20` | 80px | 4XL |
| `space-24` | 96px | 5XL |

### Component Padding

| Component | Padding |
|-----------|---------|
| Button (sm) | `py-1.5 px-3` |
| Button (md) | `py-2 px-4` |
| Button (lg) | `py-3 px-6` |
| Card | `p-4 md:p-6` |
| Input | `py-2 px-3` |
| Modal | `p-6` |

---

## Border Radius

| Tailwind Class | Value | Usage |
|----------------|-------|-------|
| `rounded-none` | 0 | Sharp corners |
| `rounded-sm` | 2px | Subtle rounding |
| `rounded` | 4px | Small elements |
| `rounded-md` | 6px | Cards, buttons |
| `rounded-lg` | 8px | Large cards |
| `rounded-xl` | 12px | Modals, panels |
| `rounded-2xl` | 16px | Hero elements |
| `rounded-3xl` | 24px | Special containers |
| `rounded-full` | 9999px | Circular elements |

---

## Shadows & Elevation

### Shadow Scale

| Tailwind Class | Value | Usage |
|----------------|-------|-------|
| `shadow-sm` | `0 1px 2px rgba(0,0,0,0.05)` | Subtle elevation |
| `shadow` | `0 1px 3px rgba(0,0,0,0.1)` | Default elevation |
| `shadow-md` | `0 4px 6px rgba(0,0,0,0.1)` | Cards, dropdowns |
| `shadow-lg` | `0 10px 15px rgba(0,0,0,0.1)` | Modals, popovers |
| `shadow-xl` | `0 20px 25px rgba(0,0,0,0.1)` | Heavy elevation |
| `shadow-2xl` | `0 25px 50px rgba(0,0,0,0.15)` | Hero elements |

### Glows

| Tailwind Class | Value | Usage |
|----------------|-------|-------|
| `glow-gold` | `0 0 20px rgba(255, 184, 0, 0.3)` | Gold highlights |
| `glow-primary` | `0 0 20px rgba(26, 35, 126, 0.4)` | Primary highlights |
| `glow-orange` | `0 0 20px rgba(255, 107, 0, 0.3)` | Orange accents |

---

## Component Library

### Button

**Variants:**
- `primary` - Main CTAs with gold gradient
- `secondary` - Secondary actions with navy
- `ghost` - Transparent backgrounds
- `danger` - Destructive actions

**Sizes:**
- `sm` - Small buttons
- `md` - Default size
- `lg` - Large buttons

**States:**
- Hover: Slight scale (1.02) and color shift
- Active: Scale (0.98) with darker color
- Focus: Visible ring outline
- Disabled: Reduced opacity, no pointer events

### Card

**Types:**
- Default - Standard card with subtle border
- Elevated - Card with shadow
- Interactive - Hover effects
- Bordered - Prominent border

**Structure:**
- Header - Title and actions
- Body - Main content
- Footer - Metadata and secondary actions

### Input

**Types:**
- Text
- Number
- Search (with icon)
- Select
- Textarea

**States:**
- Default
- Focus
- Error (red border)
- Disabled (grayed out)

### Badge/Tag

**Variants:**
- Default - Navy background
- Success - Green
- Warning - Orange
- Error - Red
- Info - Blue
- Gold - Accent

**Sizes:**
- `sm` - Small badges
- `md` - Default size

### Avatar

**Types:**
- Image
- Initials
- Fallback icon

**Sizes:**
- `xs` - 24px
- `sm` - 32px
- `md` - 40px
- `lg` - 48px
- `xl` - 64px

**States:**
- Online (green dot)
- Offline (gray dot)

### Modal/Dialog

**Features:**
- Backdrop overlay
- Close button
- Keyboard ESC to close
- Click outside to close
- Focus trap
- Animation on mount/unmount

**Sizes:**
- `sm` - 400px max width
- `md` - 600px max width
- `lg` - 800px max width
- `xl` - 1200px max width

### Tabs

**Types:**
- Underline - Bottom border
- Pill - Rounded container
- Card - Card-based tabs

### Loading/Spinner

**Types:**
- Spinner - Rotating circle
- Dots - Bouncing dots
- Skeleton - Content placeholder
- Progress - Progress bar

**Sizes:**
- `sm` - Small spinner
- `md` - Default size
- `lg` - Large spinner

### Toast/Notification

**Types:**
- Success
- Error
- Warning
- Info

**Positions:**
- Top-right
- Top-center
- Bottom-right
- Bottom-center

**Features:**
- Auto-dismiss
- Manual close
- Action buttons
- Progress indicator

---

## Accessibility Guidelines

### Color Contrast

All text must meet WCAG 2.1 AA standards:
- Normal text: Minimum 4.5:1 contrast ratio
- Large text (18px+): Minimum 3:1 contrast ratio
- UI components: Minimum 3:1 contrast ratio

### Keyboard Navigation

- All interactive elements must be keyboard accessible
- Visible focus indicators on all focusable elements
- Logical tab order
- Skip links for main content

### Screen Reader Support

- Semantic HTML elements
- ARIA labels for icons and buttons
- ARIA live regions for dynamic content
- Descriptive link text

### Motion

- Respect `prefers-reduced-motion`
- No content flashes below 3 times per second
- Provide controls for autoplaying content

---

## Responsive Breakpoints

| Breakpoint | Width | Devices |
|------------|-------|---------|
| `xs` | 0px | Small phones |
| `sm` | 640px | Phones |
| `md` | 768px | Tablets |
| `lg` | 1024px | Small laptops |
| `xl` | 1280px | Desktops |
| `2xl` | 1536px | Large screens |

### Mobile-First Approach

All styles start with mobile (default) and use `md:`, `lg:`, `xl:` prefixes for larger screens.

```tsx
// Example
<div className="p-4 md:p-6 lg:p-8">
  {/* Mobile: 16px padding, Tablet: 24px, Desktop: 32px */}
</div>
```

---

## Animation Guidelines

### Duration

| Speed | Duration | Usage |
|-------|----------|-------|
| Fast | 150ms | Micro-interactions |
| Base | 200ms | Default transitions |
| Slow | 300ms | Complex animations |

### Easing

| Tailwind Class | Cubic Bezier | Usage |
|----------------|--------------|-------|
| `ease-linear` | linear | Continuous movement |
| `ease-in` | cubic-bezier(0.4, 0, 1, 1) | Entering |
| `ease-out` | cubic-bezier(0, 0, 0.2, 1) | Leaving |
| `ease-in-out` | cubic-bezier(0.4, 0, 0.2, 1) | Default |

---

## Usage Examples

### Primary Button
```tsx
<Button variant="primary" size="md">
  Create Team
</Button>
```

### Card with Artist
```tsx
<Card>
  <CardHeader>
    <ArtistName name="Artist Name" />
  </CardHeader>
  <CardBody>
    <ArtistStats score={123} bonus={45} />
  </CardBody>
  <CardFooter>
    <Button variant="ghost">View Details</Button>
  </CardFooter>
</Card>
```

### Modal
```tsx
<Modal isOpen={showModal} onClose={() => setShowModal(false)}>
  <ModalHeader>Confirm Selection</ModalHeader>
  <ModalBody>
    Are you sure you want to select this artist?
  </ModalBody>
  <ModalFooter>
    <Button variant="ghost" onClick={() => setShowModal(false)}>
      Cancel
    </Button>
    <Button variant="primary">Confirm</Button>
  </ModalFooter>
</Modal>
```

---

## Design Principles

1. **Clarity First** - UI should be intuitive and self-explanatory
2. **Consistency** - Use established patterns consistently
3. **Accessibility** - Design for all users from the start
4. **Performance** - Optimize for Core Web Vitals
5. **Mobile-First** - Design for smallest screens first
6. **Fantasanremo Identity** - Maintain brand consistency throughout

---

## Resources

- [Tailwind CSS Documentation](https://tailwindcss.com/docs)
- [WCAG 2.1 Guidelines](https://www.w3.org/WAI/WCAG21/quickref/)
- [Figma Design File](#) - (When available)
- [Component Storybook](#) - (When implemented)

---

## Changelog

| Version | Date | Changes |
|---------|------|---------|
| 1.1.0 | 2026-02-08 | Add dark mode support, new utility modules (animations, logger, cn), UI component library documentation |
| 1.0.0 | 2026-02-07 | Initial design system from Fantasanremo screenshot analysis |
