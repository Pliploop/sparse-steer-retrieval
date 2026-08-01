/**
 * Mark: a small mixing console of concept faders. Each fader is a concept you can
 * push up (amplify) or down (suppress); the coloured caps use the paper's method
 * palette. Reads as a console at any size.
 */
export default function Logo({ size = 32, className = '' }) {
  // x positions and cap heights (y of the cap centre) for four faders.
  const faders = [
    { x: 13, y: 40, fill: '#2E6FD6' },
    { x: 27, y: 22, fill: '#FB8B24' },
    { x: 41, y: 46, fill: '#1FA347' },
    { x: 55, y: 30, fill: '#EC4F9E' },
  ]
  return (
    <svg viewBox="0 0 68 64" width={size} height={size} className={className} role="img" aria-label="Steerable Retrieval">
      {/* console body */}
      <rect x="3" y="8" width="62" height="48" rx="9" fill="none" stroke="currentColor" strokeWidth="2.4" opacity="0.5" />
      {/* fader tracks */}
      <g stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" opacity="0.3">
        {faders.map((f) => (
          <line key={f.x} x1={f.x} y1="16" x2={f.x} y2="48" />
        ))}
      </g>
      {/* fader caps */}
      <g>
        {faders.map((f) => (
          <rect key={f.x} x={f.x - 5} y={f.y - 3.5} width="10" height="7" rx="3.5" fill={f.fill} stroke="white" strokeWidth="1.6" />
        ))}
      </g>
    </svg>
  )
}

export function Wordmark({ className = '' }) {
  return (
    <span className={`font-semibold tracking-tight ${className}`}>
      Steerable<span className="text-steer"> Retrieval</span>
    </span>
  )
}
