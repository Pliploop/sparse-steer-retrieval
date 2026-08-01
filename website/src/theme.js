// Single source of truth for the site's visual language, mirrored in
// tailwind.config.js. Colours are lifted from the paper so the site, figures,
// and (later) the video never disagree.

// The two modality manifolds in the joint music--text space.
export const MANIFOLD = { audio: '#69E0F5', text: '#F06EA2' }

// The three concept-attribution methods.
export const METHOD = {
  dtn: { id: 'dtn', label: 'DTN cosine probing', color: '#2E6FD6' },
  adam: { id: 'adam', label: 'Adam inversion', color: '#FB8B24' },
  fista: { id: 'fista', label: 'FISTA inversion', color: '#1FA347' },
}

export const STEER = '#7B3FF2' // brand / concept accent

// `null` renders as a disabled "soon" pill rather than a dead link. Never use
// '#' as a placeholder href.
export const LINKS = {
  conference: 'https://ismir2026.ismir.net/',
  huggingface: 'https://huggingface.co/Pliploop/steerable-retrieval-sae',
  demo: 'https://huggingface.co/spaces/Pliploop/steerable-retrieval',
  github: 'https://github.com/Pliploop/sparse-steer-retrieval',
  website: 'https://pliploop.github.io/sparse-steer-retrieval/',
}

export const hexToRgba = (hex, a) => {
  const h = hex.replace('#', '')
  const r = parseInt(h.slice(0, 2), 16)
  const g = parseInt(h.slice(2, 4), 16)
  const b = parseInt(h.slice(4, 6), 16)
  return `rgba(${r}, ${g}, ${b}, ${a})`
}

// Blend a hex colour toward another (0 = from, 1 = to). Used for the diverging
// suppress <-> amplify slider track.
export function mix(from, to, t) {
  const a = from.replace('#', '')
  const b = to.replace('#', '')
  const ch = (s, i) => parseInt(s.slice(i, i + 2), 16)
  const r = Math.round(ch(a, 0) + (ch(b, 0) - ch(a, 0)) * t)
  const g = Math.round(ch(a, 2) + (ch(b, 2) - ch(a, 2)) * t)
  const bl = Math.round(ch(a, 4) + (ch(b, 4) - ch(a, 4)) * t)
  return `rgb(${r}, ${g}, ${bl})`
}

// Suppress (negative alpha) is cool, amplify (positive alpha) is warm.
export const SUPPRESS_COLOR = '#2E6FD6'
export const AMPLIFY_COLOR = '#FB8B24'
