import { hexToRgba } from '../theme.js'

const AXIS_H = 92

// Fallback curve if an example carries no measured `scores` (older data).
const FALLBACK = Array.from({ length: 21 }, (_, i) => {
  const t = i / 20
  return 0.3 * (t * t * (3 - 2 * t))
})

// Map a slider position t in [0,1] to an interpolated value in a sampled curve.
function sampleAt(curve, t) {
  const f = Math.max(0, Math.min(1, t)) * (curve.length - 1)
  const i = Math.floor(f)
  const frac = f - i
  const a = curve[i]
  const b = curve[Math.min(curve.length - 1, i + 1)]
  return a + (b - a) * frac
}

/**
 * A second axis for the slider: the steered query's concept similarity (y) against
 * alpha (x), drawn right above the slider and sharing its scale. The curve is the real
 * per-example steering response measured by demo/gen_examples.py; it fills in left→right
 * as alpha rises, with a dot on the current point and a dashed guide to the slider thumb.
 */
export default function SimilarityAxis({ alpha, accent, active, scores, domain }) {
  const curve = Array.isArray(scores) && scores.length > 1 ? scores : FALLBACK
  // A shared (global) domain keeps the plots comparable: a weakly-steerable concept
  // rises less than a strongly-steerable one. Falls back to this curve's own range.
  const lo = domain ? domain.lo : Math.min(...curve)
  const hi = domain ? domain.hi : Math.max(...curve)
  const span = hi - lo || 1
  const norm = (s) => Math.max(0, Math.min(1, (s - lo) / span)) // 0..1, higher = more on-concept

  const N = curve.length
  const line = curve.map((s, i) => `${i ? 'L' : 'M'} ${((i / (N - 1)) * 100).toFixed(2)} ${((1 - norm(s)) * 100).toFixed(2)}`).join(' ')
  const area = `${line} L 100 100 L 0 100 Z`

  const sNow = sampleAt(curve, alpha)
  const dotTop = (1 - norm(sNow)) * AXIS_H
  const clipId = `clip-${accent.replace('#', '')}`
  const gid = `simfill-${accent.replace('#', '')}`

  return (
    <div className="relative w-full" style={{ height: AXIS_H }}>
      <span className="absolute left-0 top-0 z-10 text-[10px] font-semibold uppercase tracking-wide text-neutral-400">
        concept similarity
      </span>

      <svg className="absolute inset-0 h-full w-full" viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden>
        <defs>
          <clipPath id={clipId}>
            <rect x="0" y="0" width={Math.max(0, alpha * 100)} height="100" />
          </clipPath>
          <linearGradient id={gid} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={accent} stopOpacity="0.34" />
            <stop offset="100%" stopColor={accent} stopOpacity="0.03" />
          </linearGradient>
        </defs>
        <path d={line} fill="none" stroke={accent} strokeOpacity="0.22" strokeWidth="2" vectorEffect="non-scaling-stroke" />
        <g clipPath={`url(#${clipId})`}>
          <path d={area} fill={`url(#${gid})`} stroke="none" />
          <path d={line} fill="none" stroke={accent} strokeWidth="2.5" vectorEffect="non-scaling-stroke" />
        </g>
      </svg>

      <div
        className="absolute"
        style={{ left: `${alpha * 100}%`, top: dotTop, height: AXIS_H - dotTop, width: 0, borderLeft: `1px dashed ${hexToRgba(accent, 0.5)}` }}
      />
      <div
        className="absolute h-3 w-3 rounded-full border-2 border-white dark:border-neutral-900"
        style={{ left: `${alpha * 100}%`, top: dotTop, backgroundColor: accent, transform: 'translate(-50%,-50%)' }}
      />
      <div
        className="absolute text-[11px] font-semibold tabular-nums"
        style={{
          left: `clamp(16px, ${alpha * 100}%, calc(100% - 16px))`,
          top: Math.max(0, dotTop - 18),
          color: active ? accent : '#9ca3af',
          transform: 'translateX(-50%)',
        }}
      >
        {Math.round(norm(sNow) * 100)}%
      </div>
    </div>
  )
}
