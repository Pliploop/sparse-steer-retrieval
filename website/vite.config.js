import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// GitHub Pages serves from https://<user>.github.io/sparse-steer-retrieval/
// `base` must equal the repo name exactly.
export default defineConfig({
  plugins: [react()],
  base: '/sparse-steer-retrieval/',
  define: {
    // Compile-time literal so rollup can drop the live-demo chunk from the
    // static build. Gating on Boolean(import.meta.env.X) is NOT constant-folded.
    __HAS_BACKEND__: JSON.stringify(Boolean(process.env.VITE_API_BASE)),
  },
  build: {
    outDir: 'dist',
    assetsDir: 'assets',
    chunkSizeWarningLimit: 1200,
  },
})
