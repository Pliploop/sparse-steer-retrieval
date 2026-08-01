import { AnimatePresence, motion, useReducedMotion } from 'framer-motion'
import { hexToRgba } from '../theme.js'

/**
 * A retrieved track. Fully hidden until `show` flips true (the slider reaches
 * its point), then plops down into place. Stripped to essentials: rank, title,
 * a tag, and a full-width Spotify preview. No border — the glass body and its
 * soft accent shadow give it presence. Spotify height is 152 so the embed never
 * grows its own scrollbar.
 */
export default function PlayCard({ rank, track, accent, show, seed = false }) {
  const reduce = useReducedMotion()
  const badge = seed ? '#8b8b93' : accent

  return (
    <AnimatePresence>
      {show && (
        <motion.div
          initial={reduce ? { opacity: 0 } : { opacity: 0, y: -20, scale: 0.9 }}
          animate={reduce ? { opacity: 1 } : { opacity: 1, y: 0, scale: 1 }}
          exit={reduce ? { opacity: 0 } : { opacity: 0, y: -12, scale: 0.94 }}
          transition={reduce ? { duration: 0.2 } : { type: 'spring', stiffness: 430, damping: 27 }}
          className="overflow-hidden rounded-2xl bg-white/60 backdrop-blur-md dark:bg-neutral-900/55"
          style={{ boxShadow: `0 16px 44px ${hexToRgba(badge, 0.18)}` }}
        >
          <div className="p-2.5">
            <div className="mb-2 flex items-center gap-2 px-0.5">
              <span
                className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full text-[10px] font-semibold text-white"
                style={{ backgroundColor: badge }}
              >
                {seed ? '◦' : rank}
              </span>
              <div className="min-w-0 flex-1">
                <p className="truncate text-[13px] font-semibold text-neutral-900 dark:text-neutral-100">{track.title}</p>
                <p className="truncate text-[11px] text-neutral-500 dark:text-neutral-400">{track.artist}</p>
              </div>
              <span
                className="chip shrink-0"
                style={{ backgroundColor: hexToRgba(badge, 0.14), color: badge }}
              >
                {seed ? 'seed' : track.genre}
              </span>
            </div>
            <iframe
              title={`Spotify ${track.title}`}
              src={`https://open.spotify.com/embed/track/${track.spotify}?theme=0`}
              width="100%"
              height="152"
              frameBorder="0"
              scrolling="no"
              loading="lazy"
              allow="autoplay; clipboard-write; encrypted-media; fullscreen; picture-in-picture"
              className="block w-full rounded-xl border-0"
            />
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  )
}
