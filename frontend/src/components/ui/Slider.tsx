import { ChangeEvent } from 'react'
import { motion } from 'framer-motion'

export interface SliderProps {
  min?: number
  max?: number
  value?: number
  onChange?: (value: number) => void
  label?: string
  showValue?: boolean
  suffix?: string
}

export default function Slider({
  min = 0,
  max = 100,
  value = 50,
  onChange,
  label,
  showValue = true,
  suffix = ''
}: SliderProps) {
  const percentage = ((value - min) / (max - min)) * 100

  const handleChange = (e: ChangeEvent<HTMLInputElement>) => {
    const newValue = Number(e.target.value)
    onChange?.(newValue)
  }

  return (
    <div className="space-y-2">
      {(label || showValue) && (
        <div className="flex items-center justify-between">
          {label && (
            <label className="text-sm font-medium text-gray-300">{label}</label>
          )}
          {showValue && (
            <span className="text-sm font-bold text-amber-400">
              {value}{suffix}
            </span>
          )}
        </div>
      )}
      <div className="relative">
        <div className="absolute top-1/2 left-0 right-0 h-2 bg-navy-900 rounded-full -translate-y-1/2" />
        <motion.div
          className="absolute top-1/2 left-0 h-2 bg-gradient-to-r from-amber-600 to-amber-400 rounded-full -translate-y-1/2"
          initial={{ width: 0 }}
          animate={{ width: `${percentage}%` }}
          transition={{ type: 'spring', stiffness: 300, damping: 20 }}
        />
        <input
          type="range"
          min={min}
          max={max}
          value={value}
          onChange={handleChange}
          className="relative w-full h-2 bg-transparent appearance-none cursor-pointer focus:outline-none"
          style={{
            WebkitAppearance: 'none',
            background: 'transparent'
          }}
        />
        <style>{`
          input[type="range"]::-webkit-slider-thumb {
            -webkit-appearance: none;
            appearance: none;
            width: 20px;
            height: 20px;
            border-radius: 50%;
            background: linear-gradient(135deg, #ffd700, #ffb800);
            cursor: pointer;
            box-shadow: 0 2px 8px rgba(255, 184, 0, 0.4);
            transition: transform 0.2s;
          }
          input[type="range"]::-webkit-slider-thumb:hover {
            transform: scale(1.2);
          }
          input[type="range"]::-moz-range-thumb {
            width: 20px;
            height: 20px;
            border-radius: 50%;
            background: linear-gradient(135deg, #ffd700, #ffb800);
            cursor: pointer;
            border: none;
            box-shadow: 0 2px 8px rgba(255, 184, 0, 0.4);
            transition: transform 0.2s;
          }
          input[type="range"]::-moz-range-thumb:hover {
            transform: scale(1.2);
          }
        `}</style>
      </div>
      <div className="flex justify-between text-xs text-gray-500">
        <span>{min}{suffix}</span>
        <span>{max}{suffix}</span>
      </div>
    </div>
  )
}
