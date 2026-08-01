import { useEffect, useMemo, useRef, useState } from 'react'
import { motion, useInView } from 'framer-motion'
import { EXAMPLES, PLACEHOLDER } from '../data/examples.js'
import { hexToRgba } from '../theme.js'
import PlayCard from './PlayCard.jsx'
import SimilarityAxis from './SimilarityAxis.jsx'

// (threshold along the slider, which staggered row, seed?). The seed sits at the
// very start; the three most relevant results reveal in order. Kept to four cards
// so each is wide enough for the Spotify play button to show.
const SLOTS = [
  { t: 0.0, row: 0, seed: true },
  { t: 0.3, row: 1 },
  { t: 0.68, row: 0 },
  { t: 0.9, row: 1 },
]

// Vertical geometry of the wide diagram (px). Horizontal is measured at runtime.
const AXIS_H = 92
const SLIDER_Y = AXIS_H + 14
const ROW0 = SLIDER_Y + 78
const ROW_GAP = 236
const CARD_H = 210
const STAGE_H = ROW0 + ROW_GAP + CARD_H + 8
const rowTop = (row) => (row === 0 ? ROW0 : ROW0 + ROW_GAP)

export default function SteerExplorer() {
  const [idx, setIdx] = useState(0)
  const [alpha, setAlpha] = useState(0)
  const [typed, setTyped] = useState(EXAMPLES[0].concept)
  const [ready, setReady] = useState(true)
  const ex = EXAMPLES[idx]
  const accent = ex.accent

  const stageRef = useRef(null)
  const inView = useInView(stageRef, { once: false, amount: 0.25 })

  // measure the diagram column so card positions are responsive (no fixed stage)
  const wrapRef = useRef(null)
  const [W, setW] = useState(0)
  useEffect(() => {
    const el = wrapRef.current
    if (!el) return
    const ro = new ResizeObserver(([e]) => setW(e.contentRect.width))
    ro.observe(el)
    setW(el.getBoundingClientRect().width)
    return () => ro.disconnect()
  }, [])

  // Entrance: once in view (and on every concept switch), type the concept then
  // "draw" the slider in. Re-runs on idx so a new concept replays the build.
  useEffect(() => {
    if (!inView) return
    setReady(false)
    setAlpha(0)
    setTyped('')
    const full = ex.concept
    let i = 0
    const id = setInterval(() => {
      i += 1
      setTyped(full.slice(0, i))
      if (i >= full.length) {
        clearInterval(id)
        setReady(true)
      }
    }, 55)
    return () => clearInterval(id)
  }, [inView, idx, ex.concept])

  const ranked = useMemo(() => [...ex.results].sort((a, b) => b.affinity - a.affinity), [ex])

  // Shared score domain across all examples so the concept-similarity plots show
  // real relative strength (a weak concept rises less than a strong one).
  const scoreDomain = useMemo(() => {
    let lo = Infinity, hi = -Infinity
    for (const e of EXAMPLES) for (const s of e.scores || []) {
      if (s < lo) lo = s
      if (s > hi) hi = s
    }
    return lo < hi ? { lo, hi } : null
  }, [])

  const cardW = Math.max(240, Math.min(300, W / 2 - 30))
  const padX = cardW / 2 + 6
  const xAt = (t) => Math.max(padX, Math.min(W - padX, t * W))
  const compact = W < 640 // covers SSR (W=0) and mobile/tablet → stacked fallback

  const items = SLOTS.map((sl, i) => ({
    ...sl,
    rank: i,
    track: sl.seed ? ex.seed : ranked[i - 1],
    // the seed hugs the very start of the slider; results sit at their threshold.
    x: sl.seed ? 0 : xAt(sl.t),
    top: rowTop(sl.row),
    // the seed (original track) is present from the very start; results reveal
    // as the slider reaches them, once the concept is typed and the slider drawn.
    show: sl.seed ? true : ready && alpha >= sl.t,
  })).filter((c) => c.track) // examples may carry fewer than the max results

  const trackStyle = {
    '--accent': accent,
    '--track': `linear-gradient(90deg, ${accent} ${alpha * 100}%, ${hexToRgba(accent, 0.16)} ${alpha * 100}%)`,
  }

  // The bare range input. `ticks` is a list of {key, left, on} placed on top,
  // so the caller can align them to pixel (wide) or percentage (compact) x.
  const slider = (ticks = []) => (
    <div className="relative w-full">
      <input
        type="range"
        min="0"
        max="1"
        step="0.01"
        value={alpha}
        onChange={(e) => setAlpha(parseFloat(e.target.value))}
        className="steer-range w-full"
        style={trackStyle}
        aria-label={`Amplify ${ex.concept}`}
      />
      {ticks.map((tk) => (
        <span
          key={tk.key}
          className="pointer-events-none absolute top-1/2 -translate-x-1/2 -translate-y-1/2"
          style={{ left: tk.left }}
        >
          <span className="block h-2.5 w-[2px] rounded-full" style={{ backgroundColor: hexToRgba(accent, tk.on ? 0.9 : 0.3) }} />
        </span>
      ))}
    </div>
  )

  return (
    <div className="glass p-5 sm:p-6">
      {/* concept chips + precomputed badge */}
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex flex-wrap gap-1.5">
          {EXAMPLES.map((e, i) => {
            const on = i === idx
            return (
              <button
                key={e.id}
                type="button"
                onClick={() => setIdx(i)}
                className={`inline-flex items-center gap-1.5 rounded-md px-2.5 py-1 text-xs font-medium transition-colors ${
                  on ? 'text-white' : 'text-neutral-600 hover:bg-neutral-100 dark:text-neutral-300 dark:hover:bg-neutral-800'
                }`}
                style={on ? { backgroundColor: e.accent } : undefined}
              >
                <span
                  className="h-1.5 w-1.5 rounded-full"
                  style={{ backgroundColor: on ? 'rgba(255,255,255,0.85)' : e.accent }}
                />
                {e.concept}
              </button>
            )
          })}
        </div>
        <span
          className="inline-flex items-center rounded-md border border-neutral-200 px-2.5 py-1 text-[11px] font-medium text-neutral-400 dark:border-neutral-700 dark:text-neutral-500"
          title="Retrievals are computed offline for a few seed tracks; the live demo runs on any track."
        >
          precomputed{PLACEHOLDER ? ' · sample data' : ''}
        </span>
      </div>

      {/* typed concept headline — the "slider being created" cue */}
      <div className="mt-4 text-center text-2xl font-semibold tracking-tight sm:text-3xl">
        <span className="text-neutral-400">steer toward </span>
        <span style={{ color: accent }}>{typed}</span>
        <span className="ml-0.5 inline-block h-[0.9em] w-[2px] align-middle" style={{ backgroundColor: accent, opacity: ready ? 0 : 1 }} />
      </div>

      <div ref={stageRef} className="mt-5">
        <div ref={wrapRef} className="min-w-0">
          {compact ? (
            <div>
              <SimilarityAxis alpha={alpha} accent={accent} active={ready} scores={ex.scores} domain={scoreDomain} />
              <div className="mt-1 flex items-center gap-3">
                <span className="text-[11px] font-medium text-neutral-400">seed</span>
                <motion.div
                  className="flex-1"
                  initial={{ clipPath: 'inset(0 100% 0 0)' }}
                  animate={{ clipPath: ready ? 'inset(0 0% 0 0)' : 'inset(0 100% 0 0)' }}
                  transition={{ duration: 0.55, ease: 'easeOut' }}
                >
                  {slider(items.filter((c) => !c.seed).map((c) => ({ key: c.rank, left: `${c.t * 100}%`, on: alpha >= c.t })))}
                </motion.div>
                <span className="text-xs font-semibold tabular-nums" style={{ color: accent }}>
                  α {alpha.toFixed(2)}
                </span>
              </div>
              <div className="mx-auto mt-4 flex max-w-sm flex-col gap-3">
                {items.map((c) => (
                  <PlayCard key={c.rank} rank={c.rank} track={c.track} accent={accent} show={c.show} seed={c.seed} />
                ))}
              </div>
            </div>
          ) : (
            <div className="relative" style={{ height: STAGE_H }}>
              {/* concept-similarity as a second axis, right above the slider */}
              <div className="absolute" style={{ left: 0, right: 0, top: 0 }}>
                <SimilarityAxis alpha={alpha} accent={accent} active={ready} scores={ex.scores} domain={scoreDomain} />
              </div>

              {/* the slider, drawn in left→right once the concept is typed */}
              <motion.div
                className="absolute"
                style={{ left: 0, right: 0, top: SLIDER_Y, transform: 'translateY(-50%)' }}
                initial={{ clipPath: 'inset(0 100% 0 0)' }}
                animate={{ clipPath: ready ? 'inset(0 0% 0 0)' : 'inset(0 100% 0 0)' }}
                transition={{ duration: 0.6, ease: 'easeOut' }}
              >
                {slider(items.map((c) => ({ key: c.rank, left: c.x, on: alpha >= c.t })))}
              </motion.div>
              <span
                className="absolute text-[11px] font-medium text-neutral-400 transition-opacity"
                style={{ left: 0, top: SLIDER_Y + 10, opacity: ready ? 1 : 0 }}
              >
                seed
              </span>
              <span
                className="absolute text-xs font-semibold tabular-nums transition-opacity"
                style={{ right: 0, top: SLIDER_Y + 10, color: accent, opacity: ready ? 1 : 0 }}
              >
                α {alpha.toFixed(2)}
              </span>

              {/* connectors: absent at rest, grow down to a card as it reveals */}
              {items.map((c) => (
                <motion.div
                  key={`conn-${c.rank}`}
                  className="absolute origin-top"
                  style={{
                    left: c.x,
                    top: SLIDER_Y + 6,
                    height: c.top - (SLIDER_Y + 6),
                    width: 2,
                    transform: c.seed ? 'translateX(0)' : 'translateX(-50%)',
                    backgroundColor: hexToRgba(accent, 0.4),
                  }}
                  initial={{ scaleY: 0, opacity: 0 }}
                  animate={{ scaleY: ready && c.show ? 1 : 0, opacity: ready && c.show ? 1 : 0 }}
                  transition={{ duration: 0.35, ease: 'easeOut' }}
                />
              ))}

              {/* floating cards — seed is left-anchored to the slider start */}
              {items.map((c) => (
                <div
                  key={c.rank}
                  className="absolute"
                  style={{ left: c.x, top: c.top, width: cardW, transform: c.seed ? 'translateX(0)' : 'translateX(-50%)' }}
                >
                  <PlayCard rank={c.rank} track={c.track} accent={accent} show={c.show} seed={c.seed} />
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      <p className="mt-4 text-center text-[11px] text-neutral-400">
        Drag right to amplify the concept; each track plops in as the slider reaches it.
      </p>
    </div>
  )
}
