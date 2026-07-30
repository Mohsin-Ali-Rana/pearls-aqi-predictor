/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        brand: {
          dark: "#0B132B",
          card: "#1C2541",
          accent: "#48CAE4",
          success: "#06D6A0",
          warning: "#FFD166",
          danger: "#EF476F"
        }
      }
    },
  },
  plugins: [],
}