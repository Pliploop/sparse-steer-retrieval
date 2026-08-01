import { useEffect, useRef, useState } from 'react'
import { animate, motion, useMotionValue, useTransform } from 'framer-motion'
import { hexToRgba } from '../theme.js'

// Only the concept word is coloured and steered. `dir` is +1 amplify / -1 suppress.
const PHRASES = [
  { pre: 'have more ', word: 'piano', dir: +1, color: '#7B3FF2' },
  { pre: 'be ', word: 'grittier', dir: +1, color: '#FB8B24' },
  { pre: 'be ', word: 'punchier', dir: +1, color: '#E23B34' },
  { pre: 'sound more ', word: 'ambient', dir: +1, color: '#00B4D8' },
  { pre: 'have ', word: 'female vocals', dir: +1, color: '#EC4F9E' },
  { pre: 'be ', word: 'less loud', dir: -1, color: '#2E6FD6' },
  { pre: 'lose the ', word: 'guitar', dir: -1, color: '#1FA347' },
]

const wait = (ms) => new Promise((r) => setTimeout(r, ms))

/**
 * The paper's idea in one loop: a concept is typed, then a slider steers it. The
 * concept word grows/brightens as the slider goes up (amplify) and shrinks/dims
 * as it goes down (suppress). One shared motion value drives the thumb, the
 * filled track, and the word so they move together. SSR-safe: renders the first
 * concept steered.
 */
export default function SteerHeadline() {
  const [idx, setIdx] = useState(0)
  const [sub, setSub] = useState(PHRASES[0].pre.length + PHRASES[0].word.length)
  const [typing, setTyping] = useState(false)
  const pos = useMotionValue(PHRASES[0].dir > 0 ? 0.9 : 0.1) // 0 suppress .. 1 amplify

  const thumbLeft = useTransform(pos, [0, 1], ['0%', '100%'])
  const fillWidth = useTransform(pos, (v) => `${v * 100}%`)
  const wordScale = useTransform(pos, [0, 0.5, 1], [0.8, 1, 1.22])
  const wordOpacity = useTransform(pos, [0, 0.5, 1], [0.5, 0.85, 1])

  const idxRef = useRef(0)
  useEffect(() => {
    let cancelled = false
    async function loop() {
      let first = true
      while (!cancelled) {
        const i = idxRef.current
        const p = PHRASES[i]
        const full = p.pre.length + p.word.length
        if (!first) {
          // type the next concept in from the centre
          pos.set(0.5)
          setTyping(true)
          for (let s = 0; s <= full && !cancelled; s++) {
            setSub(s)
            await wait(45)
          }
          setTyping(false)
        }
        first = false
        await wait(320)
        // steer toward the concept's direction
        await animate(pos, p.dir > 0 ? 0.94 : 0.06, { duration: 0.95, ease: 'easeInOut' }).finished
        await wait(1000)
        // ease back to neutral so the movement is unmistakable
        await animate(pos, 0.5, { duration: 0.7, ease: 'easeInOut' }).finished
        await wait(250)
        // delete
        setTyping(true)
        for (let s = full; s >= p.pre.length && !cancelled; s--) {
          setSub(s)
          await wait(22)
        }
        idxRef.current = (i + 1) % PHRASES.length
        setIdx(idxRef.current)
      }
    }
    loop()
    return () => {
      cancelled = true
    }
  }, [pos])

  const p = PHRASES[idx]
  const shown = (p.pre + p.word).slice(0, sub)
  const preShown = shown.slice(0, Math.min(sub, p.pre.length))
  const wordShown = sub > p.pre.length ? shown.slice(p.pre.length) : ''

  return (
    <span className="whitespace-pre-wrap">
      {preShown}
      <span className="relative inline-block pb-5 align-baseline">
        <motion.span
          className="inline-block font-semibold"
          style={{ color: p.color, scale: wordScale, opacity: wordOpacity, transformOrigin: 'left center' }}
        >
          {wordShown}
        </motion.span>
        <span
          aria-hidden
          className="ml-[3px] inline-block h-[0.9em] w-[2px] translate-y-[0.1em] rounded-full"
          style={{ backgroundColor: p.color, opacity: typing ? 1 : 0, animation: typing ? 'none' : undefined }}
        />
        {wordShown && (
          <span
            className="absolute inset-x-0 bottom-1.5 block h-1 rounded-full"
            style={{ backgroundColor: hexToRgba(p.color, 0.16) }}
          >
            <motion.span className="absolute inset-y-0 left-0 rounded-full" style={{ width: fillWidth, backgroundColor: hexToRgba(p.color, 0.55) }} />
            <motion.span
              className="absolute top-1/2 h-4 w-4 -translate-x-1/2 -translate-y-1/2 rounded-full border-[3px] border-white shadow-[0_1px_5px_rgba(0,0,0,0.3)] dark:border-neutral-900"
              style={{ left: thumbLeft, backgroundColor: p.color }}
            />
          </span>
        )}
      </span>
    </span>
  )
}
