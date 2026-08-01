import { useEffect, useState } from 'react'
import Logo, { Wordmark } from './Logo.jsx'

export function useTheme() {
  const [dark, setDark] = useState(false)
  useEffect(() => {
    const saved = localStorage.getItem('ssr-theme')
    const prefers = window.matchMedia('(prefers-color-scheme: dark)').matches
    setDark(saved ? saved === 'dark' : prefers)
  }, [])
  useEffect(() => {
    document.documentElement.classList.toggle('dark', dark)
    localStorage.setItem('ssr-theme', dark ? 'dark' : 'light')
  }, [dark])
  return [dark, setDark]
}

/** Highlights whichever section is currently under the nav. */
export function useActiveSection(ids) {
  const [active, setActive] = useState(ids[0])
  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        const visible = entries
          .filter((e) => e.isIntersecting)
          .sort((a, b) => b.intersectionRatio - a.intersectionRatio)
        if (visible[0]) setActive(visible[0].target.id)
      },
      { rootMargin: '-20% 0px -60% 0px', threshold: [0.1, 0.4, 0.8] },
    )
    ids.forEach((id) => {
      const el = document.getElementById(id)
      if (el) observer.observe(el)
    })
    return () => observer.disconnect()
  }, [ids.join(',')])
  return active
}

/**
 * Scroll to a section without touching the URL hash. Even without a router, a
 * plain `href="#id"` jumps abruptly and fights `scroll-behavior: smooth`; we
 * intercept and scroll ourselves. `scroll-mt-*` on each section clears the nav.
 */
export function scrollToSection(e, id) {
  const el = document.getElementById(id)
  if (!el) return
  e.preventDefault()
  el.scrollIntoView({ behavior: 'smooth', block: 'start' })
}

export default function Nav({ dark, setDark, sections = [], active }) {
  return (
    <header className="sticky top-0 z-50 border-b border-neutral-200/70 bg-white/70 backdrop-blur-xl dark:border-white/10 dark:bg-neutral-950/70">
      <div className="mx-auto flex h-14 max-w-6xl items-center justify-between px-4 sm:px-6">
        <a
          href="#top"
          onClick={(e) => scrollToSection(e, 'top')}
          className="group flex items-center gap-2.5"
        >
          <Logo size={26} className="text-neutral-900 dark:text-neutral-100" />
          <Wordmark className="text-lg text-neutral-900 dark:text-neutral-100" />
        </a>

        <nav className="hidden items-center gap-7 text-sm md:flex">
          {sections.map((s) => (
            <a
              key={s.id}
              href={`#${s.id}`}
              onClick={(e) => scrollToSection(e, s.id)}
              className={
                active === s.id
                  ? 'font-medium text-steer'
                  : 'text-neutral-600 transition-colors hover:text-neutral-900 dark:text-neutral-400 dark:hover:text-neutral-100'
              }
            >
              {s.label}
            </a>
          ))}
        </nav>

        <button
          type="button"
          onClick={() => setDark(!dark)}
          aria-label="Toggle theme"
          className="flex h-8 w-8 items-center justify-center rounded-md border border-neutral-200 text-neutral-700 transition-colors hover:bg-neutral-50 dark:border-neutral-700 dark:text-neutral-300 dark:hover:bg-neutral-800"
        >
          {dark ? (
            <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2">
              <path strokeLinecap="round" strokeLinejoin="round" d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.36 6.36-.7-.7M6.34 6.34l-.7-.7m12.72 0-.7.7M6.34 17.66l-.7.7M16 12a4 4 0 1 1-8 0 4 4 0 0 1 8 0z" />
            </svg>
          ) : (
            <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2">
              <path strokeLinecap="round" strokeLinejoin="round" d="M20.354 15.354A9 9 0 0 1 8.646 3.646 9.003 9.003 0 0 0 12 21a9.003 9.003 0 0 0 8.354-5.646z" />
            </svg>
          )}
        </button>
      </div>
    </header>
  )
}
