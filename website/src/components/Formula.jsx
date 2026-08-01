import { useEffect, useState } from 'react'

/**
 * A LaTeX formula typeset by KaTeX, imported dynamically (~300KB of fonts stays
 * out of the landing chunk). Until it loads -- and under SSR, where the effect
 * never runs -- we render `fallback`, a plain-text spelling of the expression.
 */
export default function Formula({ tex, fallback, display = false, className = '' }) {
  const [html, setHtml] = useState(null)

  useEffect(() => {
    let cancelled = false
    Promise.all([import('katex'), import('katex/dist/katex.min.css')])
      .then(([katex]) => {
        if (cancelled) return
        setHtml(katex.default.renderToString(tex, { throwOnError: false, displayMode: display }))
      })
      .catch(() => {
        /* keep the fallback */
      })
    return () => {
      cancelled = true
    }
  }, [tex, display])

  if (!html) return <span className={`font-mono text-[0.9em] ${className}`}>{fallback}</span>
  return <span className={className} dangerouslySetInnerHTML={{ __html: html }} />
}
