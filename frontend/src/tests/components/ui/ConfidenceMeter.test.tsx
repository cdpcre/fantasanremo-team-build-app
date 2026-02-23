import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import ConfidenceMeter from '@/components/ui/ConfidenceMeter'

describe('ConfidenceMeter Component', () => {
  describe('Zero Confidence Edge Case', () => {
    it('should render "N/A" when confidence is 0', () => {
      render(<ConfidenceMeter confidence={0} label="Confidenza ML" />)

      expect(screen.getByText('N/A')).toBeInTheDocument()
    })

    it('should render "Dati insufficienti" in the meter when confidence is 0', () => {
      render(<ConfidenceMeter confidence={0} label="Confidenza ML" />)

      expect(screen.getByText('Dati insufficienti')).toBeInTheDocument()
    })

    it('should render label when confidence is 0', () => {
      render(<ConfidenceMeter confidence={0} label="Confidenza ML" />)

      expect(screen.getByText('Confidenza ML')).toBeInTheDocument()
    })

    it('should render without label when confidence is 0 and label is not provided', () => {
      const { container } = render(<ConfidenceMeter confidence={0} />)

      expect(screen.queryByText('Confidenza ML')).not.toBeInTheDocument()
      expect(screen.getByText('Dati insufficienti')).toBeInTheDocument()
    })

    it('should apply reduced opacity when confidence is 0', () => {
      const { container } = render(<ConfidenceMeter confidence={0} label="Confidenza ML" />)

      const meterBar = container.querySelector('.bg-gray-200, .dark\\:\\bg-navy-900')
      expect(meterBar).toBeInTheDocument()
      expect(meterBar?.className).toContain('opacity-40')
    })

    it('should render all size variants when confidence is 0', () => {
      const { container: smContainer } = render(<ConfidenceMeter confidence={0} size="sm" />)
      const { container: mdContainer } = render(<ConfidenceMeter confidence={0} size="md" />)
      const { container: lgContainer } = render(<ConfidenceMeter confidence={0} size="lg" />)

      // Each should have the meter bar with different heights
      expect(smContainer.querySelector('[style*="height: 8px"]')).toBeInTheDocument()
      expect(mdContainer.querySelector('[style*="height: 12px"]')).toBeInTheDocument()
      expect(lgContainer.querySelector('[style*="height: 16px"]')).toBeInTheDocument()
    })
  })

  describe('Low Confidence Values', () => {
    it('should render red bar for confidence < 0.4', () => {
      const { container } = render(<ConfidenceMeter confidence={0.3} label="Confidenza ML" />)

      const bar = container.querySelector('.bg-red-500')
      expect(bar).toBeInTheDocument()
    })

    it('should render 30% for confidence 0.3', () => {
      render(<ConfidenceMeter confidence={0.3} label="Confidenza ML" />)

      expect(screen.getByText('30%')).toBeInTheDocument()
    })

    it('should render 1% for confidence 0.01', () => {
      render(<ConfidenceMeter confidence={0.01} label="Confidenza ML" />)

      expect(screen.getByText('1%')).toBeInTheDocument()
    })

    it('should not show "Dati insufficienti" for confidence > 0', () => {
      render(<ConfidenceMeter confidence={0.01} label="Confidenza ML" />)

      expect(screen.queryByText('Dati insufficienti')).not.toBeInTheDocument()
    })
  })

  describe('Medium Confidence Values', () => {
    it('should render yellow bar for confidence >= 0.4 and < 0.7', () => {
      const { container } = render(<ConfidenceMeter confidence={0.5} label="Confidenza ML" />)

      const bar = container.querySelector('.bg-yellow-500')
      expect(bar).toBeInTheDocument()
    })

    it('should render 40% for confidence 0.4', () => {
      render(<ConfidenceMeter confidence={0.4} label="Confidenza ML" />)

      expect(screen.getByText('40%')).toBeInTheDocument()
    })

    it('should render 50% for confidence 0.5', () => {
      render(<ConfidenceMeter confidence={0.5} label="Confidenza ML" />)

      expect(screen.getByText('50%')).toBeInTheDocument()
    })

    it('should render 69% for confidence 0.69', () => {
      render(<ConfidenceMeter confidence={0.69} label="Confidenza ML" />)

      expect(screen.getByText('69%')).toBeInTheDocument()
    })
  })

  describe('High Confidence Values', () => {
    it('should render green bar for confidence >= 0.7', () => {
      const { container } = render(<ConfidenceMeter confidence={0.85} label="Confidenza ML" />)

      const bar = container.querySelector('.bg-green-500')
      expect(bar).toBeInTheDocument()
    })

    it('should render 70% for confidence 0.7', () => {
      render(<ConfidenceMeter confidence={0.7} label="Confidenza ML" />)

      expect(screen.getByText('70%')).toBeInTheDocument()
    })

    it('should render 100% for confidence 1.0', () => {
      render(<ConfidenceMeter confidence={1.0} label="Confidenza ML" />)

      expect(screen.getByText('100%')).toBeInTheDocument()
    })
  })

  describe('Label Prop', () => {
    it('should render label when provided', () => {
      render(<ConfidenceMeter confidence={0.75} label="Confidenza ML" />)

      expect(screen.getByText('Confidenza ML')).toBeInTheDocument()
    })

    it('should not render label section when not provided', () => {
      render(<ConfidenceMeter confidence={0.75} />)

      expect(screen.queryByText('Confidenza ML')).not.toBeInTheDocument()
    })

    it('should render custom label', () => {
      render(<ConfidenceMeter confidence={0.6} label="Affidabilità" />)

      expect(screen.getByText('Affidabilità')).toBeInTheDocument()
    })
  })

  describe('Size Prop', () => {
    it('should render small size correctly', () => {
      const { container } = render(<ConfidenceMeter confidence={0.5} size="sm" />)

      const bar = container.querySelector('[style*="height: 8px"]')
      expect(bar).toBeInTheDocument()
    })

    it('should render medium size correctly (default)', () => {
      const { container } = render(<ConfidenceMeter confidence={0.5} size="md" />)

      const bar = container.querySelector('[style*="height: 12px"]')
      expect(bar).toBeInTheDocument()
    })

    it('should render large size correctly', () => {
      const { container } = render(<ConfidenceMeter confidence={0.5} size="lg" />)

      const bar = container.querySelector('[style*="height: 16px"]')
      expect(bar).toBeInTheDocument()
    })

    it('should use medium size when not specified', () => {
      const { container } = render(<ConfidenceMeter confidence={0.5} />)

      const bar = container.querySelector('[style*="height: 12px"]')
      expect(bar).toBeInTheDocument()
    })
  })

  describe('Edge Cases', () => {
    it('should round confidence to nearest integer', () => {
      render(<ConfidenceMeter confidence={0.678} label="Confidenza ML" />)

      expect(screen.getByText('68%')).toBeInTheDocument()
    })

    it('should handle confidence very close to 0', () => {
      render(<ConfidenceMeter confidence={0.001} label="Confidenza ML" />)

      expect(screen.getByText('0%')).toBeInTheDocument()
      expect(screen.queryByText('Dati insufficienti')).not.toBeInTheDocument()
    })

    it('should handle confidence very close to 1', () => {
      render(<ConfidenceMeter confidence={0.999} label="Confidenza ML" />)

      expect(screen.getByText('100%')).toBeInTheDocument()
    })
  })

  describe('Accessibility', () => {
    it('should have proper text size for different sizes', () => {
      const { container: smContainer } = render(<ConfidenceMeter confidence={0.5} size="sm" label="Test" />)
      const { container: lgContainer } = render(<ConfidenceMeter confidence={0.5} size="lg" label="Test" />)

      const smText = smContainer.querySelector('.text-xs')
      const lgText = lgContainer.querySelector('.text-base')

      expect(smText).toBeInTheDocument()
      expect(lgText).toBeInTheDocument()
    })
  })
})
