// Headless render check: renders every page with renderToString in Node (no
// browser), proving each mounts without throwing. Effects don't run under SSR,
// so this also proves the pre-effect fallbacks (typing headline, KaTeX, fetch)
// are sane. Exits non-zero if any page throws.
import { renderToString } from 'react-dom/server'
import Home from './pages/Home.jsx'

const PAGES = [['Home', <Home key="home" />]]

let failed = false
for (const [name, el] of PAGES) {
  try {
    const html = renderToString(el)
    console.log(`${name}  OK  ${html.length} chars`)
  } catch (err) {
    failed = true
    console.error(`${name}  FAILED\n`, err)
  }
}
process.exit(failed ? 1 : 0)
