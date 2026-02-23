import { motion } from 'framer-motion'
import { Moon, Sun } from 'lucide-react'
import { useTheme } from '@/contexts/ThemeContext'

export default function ThemeToggle() {
  const { theme, toggleTheme } = useTheme()

  return (
    <motion.button
      whileHover={{ scale: 1.05 }}
      whileTap={{ scale: 0.95 }}
      onClick={toggleTheme}
      className="relative w-14 h-7 bg-gray-200 dark:bg-navy-700 rounded-full p-1 transition-colors duration-300"
      aria-label="Toggle theme"
    >
      <motion.div
        animate={{ x: theme === 'dark' ? 28 : 0 }}
        transition={{ type: 'spring', stiffness: 500, damping: 30 }}
        className="relative w-5 h-5 bg-white dark:bg-amber-400 rounded-full shadow-md"
      >
        {theme === 'dark' ? (
          <Moon className="w-3 h-3 text-navy-900 absolute top-1 left-1" />
        ) : (
          <Sun className="w-3 h-3 text-amber-500 absolute top-1 left-1" />
        )}
      </motion.div>
    </motion.button>
  )
}
