/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  darkMode: 'class',
  theme: {
    extend: {
      fontFamily: {
        sans: ['Inter', 'Helvetica Neue', 'Helvetica', 'Arial', 'ui-sans-serif', 'system-ui'],
        mono: ['ui-monospace', 'SFMono-Regular', 'Menlo', 'monospace'],
      },
      colors: {
        // Single source of truth, mirrored in src/theme.js. Colours are lifted
        // from the paper: the two modality manifolds and the three attribution
        // methods.
        manifold: {
          audio: '#69E0F5', // audio manifold A
          text: '#F06EA2', // text manifold T
        },
        method: {
          dtn: '#2E6FD6', // Discover-Then-Name cosine probing
          adam: '#FB8B24', // Adam inversion
          fista: '#1FA347', // FISTA inversion
        },
        steer: '#7B3FF2', // brand / concept-steering accent
      },
      keyframes: {
        'fade-up': {
          '0%': { opacity: '0', transform: 'translateY(8px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
      },
      animation: { 'fade-up': 'fade-up 0.5s ease-out both' },
    },
  },
  plugins: [],
}
