/** @type {import('tailwindcss').Config} */
export default {
  darkMode: 'class',
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        navy: {
          50: '#f0f4f8',
          100: '#d9e2ec',
          200: '#bcccdc',
          300: '#9fb3c8',
          400: '#829ab1',
          500: '#627d98',
          600: '#486581',
          650: '#3d5670',
          700: '#334e68',
          750: '#284066',
          800: '#243b53',
          850: '#1e3a5f',
          900: '#172b4d',
          950: '#050f1a',
        },
        gold: {
          400: '#ffd700',
          500: '#ffb800',
          600: '#e5a600',
        },
        // Fantasanremo brand colors
        fs: {
          primary: '#1a237e',
          'primary-dark': '#0d1347',
          'primary-light': '#534bae',
        },
        orange: {
          primary: '#ff6b00',
          light: '#ff9500',
        },
        // Semantic colors
        success: '#4caf50',
        error: '#d32f2f',
        warning: '#ff9800',
        info: '#2196f3',
      },
      fontFamily: {
        sans: [
          'system-ui',
          '-apple-system',
          'BlinkMacSystemFont',
          'Segoe UI',
          'Roboto',
          'sans-serif',
        ],
        display: [
          'Arial',
          'sans-serif',
        ],
      },
      fontSize: {
        'xxs': ['10px', { lineHeight: '1.3', letterSpacing: '0.025em' }],
      },
      spacing: {
        '18': '4.5rem',
        '88': '22rem',
        '128': '32rem',
      },
      borderRadius: {
        '4xl': '2rem',
      },
      boxShadow: {
        'glow-gold': '0 0 20px rgba(255, 184, 0, 0.3)',
        'glow-primary': '0 0 20px rgba(26, 35, 126, 0.4)',
        'glow-orange': '0 0 20px rgba(255, 107, 0, 0.3)',
        'navy-sm': '0 1px 2px rgba(5, 15, 26, 0.05)',
        'navy-md': '0 4px 6px rgba(5, 15, 26, 0.1)',
        'navy-lg': '0 10px 15px rgba(5, 15, 26, 0.1)',
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'bounce-slow': 'bounce 1s infinite',
        'spin-slow': 'spin 3s linear infinite',
      },
      transitionDuration: {
        '250': '250ms',
        '350': '350ms',
      },
      zIndex: {
        '60': '60',
        '70': '70',
        '80': '80',
        '90': '90',
        '100': '100',
      },
    },
  },
  plugins: [],
}
