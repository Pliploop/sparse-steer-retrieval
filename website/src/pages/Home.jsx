import { motion } from 'framer-motion'
import Nav, { useActiveSection, useTheme, scrollToSection } from '../components/Nav.jsx'
import Logo from '../components/Logo.jsx'
import SteerHeadline from '../components/SteerHeadline.jsx'
import Affiliations from '../components/Affiliations.jsx'
import VideoFigure from '../components/VideoFigure.jsx'
import SteerExplorer from '../components/SteerExplorer.jsx'
import Formula from '../components/Formula.jsx'
import ErrorBoundary from '../components/ErrorBoundary.jsx'
import { LINKS, METHOD, hexToRgba } from '../theme.js'

const SECTIONS = [
  { id: 'idea', label: 'Idea' },
  { id: 'method', label: 'Method' },
  { id: 'examples', label: 'Try it' },
  { id: 'results', label: 'Results' },
]

const asset = (p) => `${import.meta.env.BASE_URL}${p}`

function Section({ id, eyebrow, title, lede, children }) {
  return (
    <section id={id} className="scroll-mt-20 py-16 md:py-20">
      <motion.div
        initial={{ opacity: 0, y: 12 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true, amount: 0.2 }}
        transition={{ duration: 0.45, ease: 'easeOut' }}
      >
        {eyebrow && <p className="text-xs font-semibold uppercase tracking-[0.14em] text-steer">{eyebrow}</p>}
        <h2 className="mt-2 text-2xl font-semibold tracking-tight text-neutral-900 dark:text-neutral-100 md:text-3xl">{title}</h2>
        {lede && <p className="mt-3 max-w-2xl text-base leading-relaxed text-neutral-600 dark:text-neutral-400">{lede}</p>}
        <div className="mt-8">{children}</div>
      </motion.div>
    </section>
  )
}

function LinkPill({ href, children, soon }) {
  if (!href) {
    return (
      <span
        className="inline-flex cursor-not-allowed items-center gap-2 rounded-full border border-dashed border-neutral-300 px-4 py-2 text-sm text-neutral-400 dark:border-neutral-700 dark:text-neutral-500"
        title={`${soon}: not public yet`}
      >
        {children}
        <span className="text-[10px] uppercase tracking-wide">soon</span>
      </span>
    )
  }
  return (
    <a href={href} target="_blank" rel="noreferrer" className="pill">
      {children}
    </a>
  )
}

function PaperFigure({ src, alt, className = '' }) {
  // Always-white glass frame so white figures have no jarring edge in dark mode.
  return (
    <figure className="rounded-2xl bg-white p-4 shadow-lg ring-1 ring-black/5 dark:shadow-black/40 dark:ring-white/10">
      <img src={src} alt={alt} className={`mx-auto block w-full ${className}`} />
    </figure>
  )
}

export default function Home() {
  const [dark, setDark] = useTheme()
  const active = useActiveSection(SECTIONS.map((s) => s.id))

  return (
    <div id="top" className="min-h-screen bg-white text-neutral-900 antialiased transition-colors duration-300 dark:bg-neutral-950 dark:text-neutral-100">
      <Nav dark={dark} setDark={setDark} sections={SECTIONS} active={active} />

      <main className="mx-auto max-w-6xl px-4 sm:px-6">
        {/* Hero */}
        <section className="relative py-16 md:py-24">
          <div
            className="pointer-events-none absolute inset-x-0 top-0 -z-10 h-72 opacity-70"
            style={{ background: 'radial-gradient(60% 60% at 50% 0%, rgba(123,63,242,0.12), transparent 70%)' }}
            aria-hidden
          />
          <motion.div initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5 }}>
            <Logo size={56} className="text-neutral-900 dark:text-neutral-100" />
            <p className="mt-6 text-xs font-semibold uppercase tracking-[0.16em] text-neutral-400">Sparse Steerable Retrieval · ISMIR 2026</p>
            <h1 className="mt-3 text-3xl font-semibold leading-tight tracking-tight sm:text-4xl md:text-5xl">
              I want this track to <SteerHeadline />
            </h1>
            <p className="mt-5 max-w-2xl text-lg leading-relaxed text-neutral-600 dark:text-neutral-300">
              Open-vocabulary concept control for dense music retrieval. A sparse autoencoder factorises the
              embedding into concept features; sparse inversion identifies the ones a free-form concept maps to,
              so it can be amplified or suppressed before nearest-neighbour search.
            </p>

            <Affiliations />

            <div className="mt-8 flex flex-wrap gap-2.5">
              <LinkPill href={LINKS.conference} soon="Conference">ISMIR 2026</LinkPill>
              <LinkPill href={LINKS.huggingface} soon="Model">Model</LinkPill>
              <LinkPill href={LINKS.demo} soon="Live demo">Demo</LinkPill>
              <a href={LINKS.github} target="_blank" rel="noreferrer" className="pill">Code</a>
              <a
                href="#examples"
                onClick={(e) => scrollToSection(e, 'examples')}
                className="inline-flex items-center gap-2 rounded-full bg-neutral-900 px-4 py-2 text-sm font-medium text-white transition-transform hover:scale-[1.03] dark:bg-neutral-100 dark:text-neutral-900"
              >
                Try the slider ↓
              </a>
            </div>
          </motion.div>
        </section>

        <div className="pb-4">
          <VideoFigure
            src={asset('ssr-walkthrough.mp4')}
            poster={asset('ssr-walkthrough-poster.jpg')}
            caption="A walkthrough of sparse steerable retrieval, from audio concepts to controllable retrieval sliders."
          />
        </div>

        <Section
          id="idea"
          eyebrow="The idea"
          title="Dense retrieval has no control surface."
          lede="Dense retrieval finds similar tracks but exposes no dial for a single attribute. Steerable retrieval adds one: edit the query along a concept axis and keep the rest of it intact."
        >
          <div className="grid gap-4 md:grid-cols-2">
            <div className="card border-neutral-200 bg-neutral-50 dark:border-neutral-800 dark:bg-neutral-900/50">
              <p className="text-xs font-semibold uppercase tracking-wide text-neutral-500">Dense retrieval</p>
              <p className="mt-2 text-sm leading-relaxed text-neutral-600 dark:text-neutral-400">
                One embedding, one nearest-neighbour list. Changing an attribute means rewriting the whole query
                and starting over; nothing about the result is disentangled.
              </p>
            </div>
            <div className="card" style={{ borderColor: hexToRgba('#7B3FF2', 0.35), backgroundColor: hexToRgba('#7B3FF2', 0.06) }}>
              <p className="text-xs font-semibold uppercase tracking-wide text-steer">Steerable retrieval</p>
              <p className="mt-2 text-sm leading-relaxed text-neutral-700 dark:text-neutral-300">
                A sparse autoencoder factorises the embedding into concept features. Amplifying or suppressing the
                features a concept maps to edits that attribute and preserves the others.
              </p>
            </div>
          </div>
        </Section>

        <Section
          id="method"
          eyebrow="How it works"
          title="Concept attribution as sparse inversion."
          lede="The open question is which sparse features carry a concept. Discover-Then-Name (DTN) ranks decoder atoms by text alignment. We instead recover a sparse code that reconstructs the concept and stays consistent with the audio manifold."
        >
          <PaperFigure src={asset('method.png')} alt="SAE training, DTN cosine probing, and sparse inversion" className="max-w-4xl" />
          <div className="mt-8 grid gap-6 sm:gap-8 md:grid-cols-3">
            {[METHOD.dtn, METHOD.adam, METHOD.fista].map((m) => (
              <div key={m.id} className="card" style={{ borderColor: hexToRgba(m.color, 0.3), backgroundColor: hexToRgba(m.color, 0.05) }}>
                <p className="text-sm font-semibold" style={{ color: m.color }}>{m.label}</p>
                <p className="mt-1 text-xs leading-relaxed text-neutral-600 dark:text-neutral-400">
                  {m.id === 'dtn' && 'Baseline. Selects the decoder atom whose direction best matches the concept text. Sensitive to feature splitting and the modality gap.'}
                  {m.id === 'adam' && 'Gradient inversion through the SAE operator with a Mahalanobis manifold prior. General and differentiable.'}
                  {m.id === 'fista' && 'Fast linear inverse exploiting the linear decoder. About 20 ms on CPU, used for the live demo.'}
                </p>
              </div>
            ))}
          </div>
          <p className="mt-5 text-sm text-neutral-600 dark:text-neutral-400">
            Steering edits the sparse code,{' '}
            <Formula tex="s' = s + \alpha\,\Pi(u_c^\star)" fallback="s' = s + α·Π(u*_c)" className="text-neutral-800 dark:text-neutral-200" />.
            Positive <span className="font-mono">α</span> amplifies the concept and negative <span className="font-mono">α</span> suppresses it; the code is then decoded and retrieved.
          </p>
        </Section>

        <Section
          id="examples"
          eyebrow="Try it"
          title="Steer a seed track along a concept."
          lede="Pick a concept and amplify it. Relevant tracks surface as the steered query moves toward the concept. These retrievals are precomputed over real music4all tracks; the live demo runs steerable retrieval on any track."
        >
          <ErrorBoundary label="Steer explorer">
            <SteerExplorer />
          </ErrorBoundary>
          <div className="mt-5">
            <LinkPill href={LINKS.demo} soon="Live demo">Open the live demo</LinkPill>
          </div>
        </Section>

        <Section
          id="results"
          eyebrow="Does it work?"
          title="Stronger edits, better preservation."
          lede="At matched support size and edit strength, inversion produces stronger concept edits that stay closer to the audio manifold than DTN cosine probing, dominating the edit versus preservation frontier for amplification and suppression."
        >
          <div className="grid items-center gap-6 md:grid-cols-2">
            <PaperFigure src={asset('pareto_tradeoff.png')} alt="Edit versus preservation trade-off" className="max-w-md" />
            <ul className="space-y-3 text-sm text-neutral-600 dark:text-neutral-300">
              <li className="flex gap-2">
                <span className="mt-1 h-2 w-2 shrink-0 rounded-full" style={{ backgroundColor: METHOD.fista.color }} />
                Inversion supports overlap the sparse supports of concept-bearing audio far more than cosine-probed ones.
              </li>
              <li className="flex gap-2">
                <span className="mt-1 h-2 w-2 shrink-0 rounded-full" style={{ backgroundColor: METHOD.adam.color }} />
                A weak probe on the recovered neurons rises from about 50% AUROC (cosine) to 65–80% (inversion).
              </li>
              <li className="flex gap-2">
                <span className="mt-1 h-2 w-2 shrink-0 rounded-full" style={{ backgroundColor: '#7B3FF2' }} />
                Edits stay closer to the audio manifold, so the rest of the query is preserved.
              </li>
            </ul>
          </div>

          <div className="mt-10">
            <p className="text-sm font-semibold text-neutral-900 dark:text-neutral-100">Bundle recall &amp; hit rate</p>
            <p className="mt-1 max-w-2xl text-sm text-neutral-600 dark:text-neutral-400">
              How much of a concept's recovered support K(c) is active in concept-bearing audio S(x), at matched cardinality across sparsity levels L₀. Both inversion variants overlap real audio far more than DTN.
            </p>
            <div className="mt-4">
              <PaperFigure src={asset('bundle_recall.png')} alt="Bundle recall and hit rate: Adam and FISTA inversion versus DTN across L0" className="max-w-3xl" />
            </div>
          </div>
        </Section>

        <footer className="border-t border-neutral-200 py-10 text-sm dark:border-neutral-800">
          <div className="flex flex-col gap-6 md:flex-row md:items-start md:justify-between">
            <div className="flex items-center gap-2.5">
              <Logo size={22} className="text-neutral-900 dark:text-neutral-100" />
              <span className="text-neutral-500 dark:text-neutral-400">
                Sparse Steerable Retrieval · Queen Mary University of London · Universal Music Group
              </span>
            </div>
            <div className="max-w-md text-xs leading-relaxed text-neutral-500 dark:text-neutral-400">
              <p>
                No audio is hosted or redistributed by this site. Playback of music4all tracks is streamed through Spotify's embed.
              </p>
            </div>
          </div>
        </footer>
      </main>
    </div>
  )
}
