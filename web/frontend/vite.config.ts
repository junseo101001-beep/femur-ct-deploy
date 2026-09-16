import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5180,
    strictPort: false,
    proxy: {
      // 백엔드(FastAPI)가 떠 있으면 /api 는 그대로 전달, 없으면 프론트는 Demo Mode 로 동작
      '/api': { target: 'http://127.0.0.1:8018', changeOrigin: true },
    },
  },
  preview: { port: 5181 },
  build: { outDir: 'dist', assetsInlineLimit: 0 },
})
