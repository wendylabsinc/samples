import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import path from 'path'

// During `npm run dev`, proxy the API + WebSocket streams to the FastAPI server
// (port 3010). In production the SPA is served by FastAPI itself, so no proxy.
const BACKEND = 'http://localhost:3010'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  server: {
    proxy: {
      '/metrics': BACKEND,
      '/restart': BACKEND,
      '/stream': { target: BACKEND, ws: true },
    },
  },
})
