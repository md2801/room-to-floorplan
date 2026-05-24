import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      // Any request from the frontend to /inferFloorplan or /outputs
      // is forwarded to the Flask backend on port 5000.
      '/inferFloorplan': {
        target: 'http://localhost:5000',
        changeOrigin: true,
      },
      '/outputs': {
        target: 'http://localhost:5000',
        changeOrigin: true,
      },
      '/health': {
        target: 'http://localhost:5000',
        changeOrigin: true,
      },
    },
  },
})
