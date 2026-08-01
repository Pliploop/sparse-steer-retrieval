import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Server build of the SSR smoke test: renders every page with renderToString in
// Node (no browser), proving each route mounts without throwing. See ssr-check.jsx.
export default defineConfig({
  plugins: [react()],
  define: {
    __HAS_BACKEND__: JSON.stringify(false),
  },
  build: {
    ssr: 'src/ssr-check.jsx',
    outDir: 'dist-ssr',
    rollupOptions: { output: { entryFileNames: 'ssr-check.js' } },
  },
})
