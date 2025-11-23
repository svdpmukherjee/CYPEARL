/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'outlook-blue': '#0078d4',
        'outlook-bg': '#f0f0f0',
        'outlook-hover': '#e1dfdd',
      }
    },
  },
  plugins: [],
}
