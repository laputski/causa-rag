/* The generator for ui/src/fonts.css.
 *
 * The @fontsource packages ship every subset at once: seven for Inter, eight
 * per weight for Source Serif 4. Greek and Vietnamese are never needed by an
 * interface written in Latin and Cyrillic, and they weigh 144 KB of 480.
 *
 * Hence a hand-built set of @font-face rules instead of
 * `import '@fontsource/...'`: three subsets per family, woff2 only (no browser
 * that can run this build needs woff), and the upright style only.
 *
 * The output is generated. Edit this generator, and not the result:
 *   node scripts/gen-fonts-css.mjs
 */
import { readFileSync, writeFileSync } from 'node:fs'

const SUBSETS = ['latin', 'latin-ext', 'cyrillic']

// Each entry is one package's source css and what gets taken from it.
const SOURCES = [
  { pkg: '@fontsource-variable/inter',          css: 'wght.css', label: 'Inter Variable — the interface' },
  { pkg: '@fontsource-variable/jetbrains-mono',  css: 'wght.css', label: 'JetBrains Mono Variable — numbers, codes, paths' },
  { pkg: '@fontsource/source-serif-4',           css: '400.css',  label: 'Source Serif 4 400 — guide prose' },
  { pkg: '@fontsource/source-serif-4',           css: '600.css',  label: 'Source Serif 4 600 — headings' },
]

/** Every subset @fontsource ships at all. The full list is needed, and not
 *  just the wanted part: without it `cyrillic-ext-...` matches the start of
 *  `cyrillic` and four surplus files travel silently into the build. */
const ALL_SUBSETS = [
  'cyrillic-ext', 'cyrillic', 'greek-ext', 'greek',
  'latin-ext', 'latin', 'vietnamese', 'math', 'symbols',
]

/** Which subset a block belongs to is read from the woff2 filename and not
 *  from the comment: a comment is a description, a filename is a fact. */
function subsetOf(fileName, pkg) {
  const stem = fileName.replace(/\.woff2$/, '')
  const family = pkg.split('/')[1]
  const rest = stem.startsWith(family + '-') ? stem.slice(family.length + 1) : stem
  // what remains is e.g. `latin-ext-wght-normal` or `cyrillic-400-normal`
  const found = ALL_SUBSETS.find(s => rest.startsWith(s + '-'))
  return found && SUBSETS.includes(found) ? found : null
}

const blocks = []
for (const { pkg, css, label } of SOURCES) {
  const raw = readFileSync(`node_modules/${pkg}/${css}`, 'utf8')
  const kept = []
  for (const chunk of raw.split('@font-face').slice(1)) {
    const body = chunk.slice(chunk.indexOf('{'), chunk.indexOf('}') + 1)
    const woff2 = body.match(/url\(\.\/files\/([^)]+\.woff2)\)/)
    if (!woff2) continue
    const subset = subsetOf(woff2[1], pkg)
    if (!subset) continue
    if (/font-style:\s*italic/.test(body)) continue
    const rewritten = body
      // a relative path into node_modules: Vite resolves it and adds the hash
      .replace(/src:[^;]+;/, `src: url(../node_modules/${pkg}/files/${woff2[1]}) format('woff2');`)
      .split('\n').map(l => l.trim()).filter(Boolean).join('\n  ').replace(/\n  }$/, '\n}')
    kept.push({ subset, rule: `@font-face ${rewritten}` })
  }
  const order = new Map(SUBSETS.map((s, i) => [s, i]))
  kept.sort((a, b) => order.get(a.subset) - order.get(b.subset))
  blocks.push(`/* ── ${label} ── */\n\n` + kept.map(k => k.rule).join('\n\n'))
}

const header = `/* Interface fonts. Generated file: edit scripts/gen-fonts-css.mjs.
 *
 * Three subsets (${SUBSETS.join(', ')}) of the seven or eight the @fontsource
 * packages ship. Greek and Vietnamese are needed by neither interface language
 * and weigh a third of the set. woff2 only, upright only.
 *
 * The fonts are bundled instead of pulled from a CDN, because production runs
 * in a closed network, where a request to fonts.gstatic.com simply never
 * completes and the text quietly falls back to a system font.
 */\n\n`

writeFileSync('src/fonts.css', header + blocks.join('\n\n') + '\n')
const n = blocks.join('\n\n').split('@font-face').length - 1
console.log(`src/fonts.css: ${n} declarations, subsets: ${SUBSETS.join(', ')}`)
