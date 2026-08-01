const asset = (p) => `${import.meta.env.BASE_URL}${p}`

/** Author line + institution logos for the paper header. */
export default function Affiliations() {
  return (
    <div className="mt-6">
      <p className="text-base text-neutral-800 dark:text-neutral-200">
        <a href="https://www.julienguinot.com/" target="_blank" rel="noreferrer" className="font-medium underline decoration-neutral-300 underline-offset-2 hover:decoration-neutral-800 dark:decoration-neutral-600">
          Julien Guinot
        </a>
        <sup className="text-neutral-400">1,2</sup>
        <span className="text-neutral-400">, </span>
        Alain Riou<sup className="text-neutral-400">2</sup>
        <span className="text-neutral-400">, </span>
        Elio Quinton<sup className="text-neutral-400">2</sup>
        <span className="text-neutral-400">, </span>
        György Fazekas<sup className="text-neutral-400">1</sup>
      </p>
      <p className="mt-1 text-xs text-neutral-500 dark:text-neutral-400">
        <sup>1</sup> Centre for Digital Music, Queen Mary University of London &nbsp;·&nbsp;
        <sup>2</sup> Music &amp; Audio Machine Learning Lab, Universal Music Group
      </p>
      <div className="mt-4 flex items-center gap-6">
        <img src={asset('img/qmul.svg')} alt="Queen Mary University of London" className="h-8 w-auto opacity-80 dark:opacity-90 dark:invert" />
        <img src={asset('img/umg.svg')} alt="Universal Music Group" className="h-7 w-auto opacity-80 dark:opacity-90 dark:invert" />
      </div>
    </div>
  )
}
