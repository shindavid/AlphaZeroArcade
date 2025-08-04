import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  clearScreen: false,
  server: {
    // listen on all interfaces (0.0.0.0)
    host: '0.0.0.0',
    // hard‑code your desired port just once
    port: 5173,
    // optional: fail if 5173 is busy instead of picking the next free one
    strictPort: false
  },
})
