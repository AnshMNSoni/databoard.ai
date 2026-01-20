/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        primary: {
          DEFAULT: "#2563eb",
          dark: "#1d4ed8",
          light: "#3b82f6",
        },
        background: {
          DEFAULT: "#09090b",
          secondary: "#18181b",
          tertiary: "#27272a",
        },
        text: {
          DEFAULT: "#fafafa",
          muted: "#a1a1aa",
          dim: "#71717a",
        }
      },
      fontFamily: {
        sans: ['Inter', 'sans-serif'],
        display: ['Cal Sans', 'Inter', 'sans-serif'],
      },
    },
  },
  plugins: [],
}
