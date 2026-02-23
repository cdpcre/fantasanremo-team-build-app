import { vi, expect, afterEach } from 'vitest'
import React from 'react'
import { cleanup } from '@testing-library/react'
import * as matchers from '@testing-library/jest-dom/matchers'

// Extend Vitest's expect with jest-dom matchers
expect.extend(matchers)

// Cleanup DOM after each test
afterEach(() => {
  cleanup()
})

// Filter out framer-motion specific props
const motionProps = [
  'whileHover', 'whileTap', 'whileFocus', 'whileInView', 'whileDrag',
  'animate', 'initial', 'exit', 'transition', 'variants',
  'drag', 'dragConstraints', 'dragElastic', 'dragMomentum', 'dragPropagation',
  'layout', 'layoutId', 'layoutScroll', 'layoutRoot',
  'onAnimationComplete', 'onDragStart', 'onDrag', 'onDragEnd',
  'onHoverStart', 'onHoverEnd', 'onTap', 'onTapStart', 'onTapCancel',
  'd', 'pathLength', 'pathOffset', 'pathSpacing'
]

// Create a wrapper that filters out motion props
const createMotionComponent = (tag: string) => {
  return ({ children, ...props }: any) => {
    const filteredProps: any = {}
    for (const key in props) {
      if (!motionProps.includes(key)) {
        filteredProps[key] = props[key]
      }
    }
    return React.createElement(tag, filteredProps, children)
  }
}

// Mock framer-motion
vi.mock('framer-motion', () => ({
  motion: {
    button: createMotionComponent('button'),
    div: createMotionComponent('div'),
    span: createMotionComponent('span'),
    p: createMotionComponent('p'),
    section: createMotionComponent('section'),
    h1: createMotionComponent('h1'),
    h2: createMotionComponent('h2'),
    h3: createMotionComponent('h3'),
    h4: createMotionComponent('h4'),
    h5: createMotionComponent('h5'),
    h6: createMotionComponent('h6'),
    a: createMotionComponent('a'),
    img: createMotionComponent('img'),
    svg: createMotionComponent('svg'),
    path: createMotionComponent('path'),
    rect: createMotionComponent('rect'),
    circle: createMotionComponent('circle'),
    tr: createMotionComponent('tr'),
    td: createMotionComponent('td'),
    th: createMotionComponent('th'),
    thead: createMotionComponent('thead'),
    tbody: createMotionComponent('tbody'),
    ul: createMotionComponent('ul'),
    li: createMotionComponent('li'),
    form: createMotionComponent('form'),
    input: createMotionComponent('input'),
    label: createMotionComponent('label'),
  },
  AnimatePresence: ({ children }: any) => React.createElement(React.Fragment, null, children),
  useSpring: () => ({
    set: vi.fn(),
    get: () => 0,
    on: vi.fn(() => vi.fn()),
  }),
  useMotionValue: () => ({ get: () => 0, set: vi.fn() }),
  useTransform: () => ({
    on: vi.fn(() => vi.fn()),
    get: () => '',
  }),
  useAnimation: () => ({ start: vi.fn(), stop: vi.fn() }),
  useMotionTemplate: () => '',
}))

// Mock ThemeContext
vi.mock('@/contexts/ThemeContext', () => ({
  useTheme: () => ({
    theme: 'dark',
    toggleTheme: vi.fn()
  }),
  ThemeProvider: ({ children }: any) => React.createElement(React.Fragment, null, children)
}))
