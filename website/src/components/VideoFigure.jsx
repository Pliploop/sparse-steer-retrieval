import { useRef, useState } from 'react'
import { motion } from 'framer-motion'

/**
 * The explainer film. Deliberately not autoplaying: `preload="metadata"` fetches
 * only enough to size the player and show the poster; bytes arrive on play.
 *
 * When `src` is falsy the figure renders a designed "coming soon" placeholder
 * rather than a broken player -- honest until the Manim walkthrough is rendered.
 */
export default function VideoFigure({ src, poster, caption }) {
  const video = useRef(null)
  const [started, setStarted] = useState(false)
  const play = () => {
    setStarted(true)
    video.current?.play()
  }

  return (
    <motion.figure
      initial={{ opacity: 0, y: 14 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, amount: 0.25 }}
      transition={{ duration: 0.5, ease: 'easeOut' }}
      className="group relative"
    >
      <div className="relative overflow-hidden rounded-3xl border border-neutral-200 bg-white shadow-2xl shadow-neutral-900/5 dark:border-neutral-800 dark:bg-neutral-900 dark:shadow-black/40">
        {src ? (
          <>
            <video
              ref={video}
              src={src}
              poster={poster}
              controls={started}
              preload="metadata"
              playsInline
              onPlay={() => setStarted(true)}
              className="block aspect-video w-full bg-white dark:bg-neutral-950"
            />
            {!started && (
              <button
                type="button"
                onClick={play}
                aria-label="Play the walkthrough"
                className="absolute inset-0 flex items-center justify-center bg-neutral-950/[0.04] transition-colors hover:bg-neutral-950/10 dark:bg-neutral-950/20 dark:hover:bg-neutral-950/30"
              >
                <span className="flex h-16 w-16 items-center justify-center rounded-full bg-white/90 shadow-xl ring-1 ring-black/5 backdrop-blur transition-transform duration-200 group-hover:scale-105">
                  <svg className="ml-1 h-6 w-6 text-neutral-900" fill="currentColor" viewBox="0 0 24 24">
                    <path d="M8 5v14l11-7z" />
                  </svg>
                </span>
              </button>
            )}
          </>
        ) : (
          <div className="flex aspect-video w-full flex-col items-center justify-center gap-3 bg-gradient-to-br from-neutral-50 to-neutral-100 dark:from-neutral-900 dark:to-neutral-950">
            <span className="flex h-16 w-16 items-center justify-center rounded-full border border-neutral-200 bg-white/80 text-neutral-400 shadow-sm dark:border-neutral-700 dark:bg-neutral-900/70">
              <svg className="ml-1 h-6 w-6" fill="currentColor" viewBox="0 0 24 24">
                <path d="M8 5v14l11-7z" />
              </svg>
            </span>
            <p className="text-sm font-medium text-neutral-500 dark:text-neutral-400">Explainer video</p>
            <span className="chip bg-neutral-200/70 text-neutral-500 dark:bg-neutral-800 dark:text-neutral-400">
              coming soon
            </span>
          </div>
        )}
      </div>
      {caption && <figcaption className="mt-3 text-xs text-neutral-500 dark:text-neutral-400">{caption}</figcaption>}
    </motion.figure>
  )
}
