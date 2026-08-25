import { describe, it, expect } from 'vitest'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'

// The stylesheet is read through `fs` rather than a `?raw` import: under vitest
// CSS is not processed at all, and `import cssRaw from '../styles.css?raw'`
// returns an empty string. Three of the rules below passed against that empty
// string having checked nothing — a failure that looks like a success.
const cssRaw = readFileSync(resolve(process.cwd(), 'src/styles.css'), 'utf-8')

/**
 * A fitness function over the design language, modelled on i18nCoverage.test.ts:
 * it reads the sources as raw text and checks properties that must hold
 * everywhere, rather than the behaviour of one screen.
 *
 * Each rule lists every violation rather than failing on the first: the work
 * proceeds screen by screen, and a list shows how much is left.
 */

const SOURCE_FILES = import.meta.glob('../{pages,components}/**/*.tsx', {
  query: '?raw',
  import: 'default',
  eager: true,
}) as Record<string, string>

const short = (path: string) => path.replace(/^\.\.\//, '')

function violations(re: RegExp, skip: (path: string) => boolean = () => false): string[] {
  const out: string[] = []
  for (const [path, src] of Object.entries(SOURCE_FILES)) {
    if (skip(path)) continue
    for (const m of src.matchAll(re)) {
      const line = src.slice(0, m.index).split('\n').length
      out.push(`${short(path)}:${line}  ${m[0].trim()}`)
    }
  }
  return out
}

// @lat: [[design-language#Фитнес-функция языка оформления#Кегль только со шкалы]]
describe('the design language', () => {
  it('font size comes from the scale rather than a literal', () => {
    // 13px against 12px reads as a different size but not as a different step;
    // the scale exists so that choice is made once.
    expect(violations(/fontSize:\s*\d+/g)).toEqual([])
  })

  // @lat: [[design-language#Фитнес-функция языка оформления#Цвет только из токенов]]
  it('colour comes from a token rather than a literal', () => {
    // The single exception is themeTokens.ts, where the literals are fallbacks
    // for when no document exists, and that is written down there.
    const found = violations(
      /['"]#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6})['"]|rgba?\([\d\s.,]+\)/g,
    )
    expect(found).toEqual([])
  })

  // @lat: [[design-language#Фитнес-функция языка оформления#Токен, которого нет]]
  it('every token the markup references is defined in the styles', () => {
    // Found live: `--color-text-secondary` and `--color-bg-secondary` are
    // defined in no theme at all. An unresolvable variable inherits the parent's
    // value in `color` and gives transparency in `background`, so the failure is
    // visible only by eye, and only to somebody who knows what it should look
    // like.
    const defined = new Set(
      [...cssRaw.matchAll(/^\s*(--[a-z0-9-]+)\s*:/gm)].map(m => m[1]),
    )
    const missing = new Set<string>()
    for (const src of Object.values(SOURCE_FILES)) {
      for (const m of src.matchAll(/var\((--[a-z0-9-]+)/g)) {
        if (!defined.has(m[1])) missing.add(m[1])
      }
    }
    expect([...missing].sort()).toEqual([])
  })

  // @lat: [[design-language#Фитнес-функция языка оформления#Заголовок не заворачивается в карточку]]
  it('a section heading is not wrapped in a card', () => {
    // A card still belongs around a self-contained object: a resource, a pack,
    // a judgment. That object's own name inside the frame is legitimate and
    // carries its own class (`.jd-detail-q` and the like). What this catches is
    // different: a heading with no class, or with `.section-title`, which makes
    // it a section heading, and a section is a rule rather than a card.
    const found = violations(
      /className="card"[\s\S]{0,160}?<h[23](?:>|[^>]*className="section-title")/g,
    )
    expect(found).toEqual([])
  })

  // @lat: [[design-language#Фитнес-функция языка оформления#Снятые классы не возвращаются]]
  it('retired classes do not come back into the markup', () => {
    expect(violations(/className=["'`][^"'`]*\b(?:stats-row|stat-card)\b/g)).toEqual([])
    // Their rules are gone from the styles too: a rule that outlived its markup
    // is an invitation to bring the markup back.
    expect(cssRaw).not.toMatch(/^\.stat-card[\s{,]/m)
    expect(cssRaw).not.toMatch(/^\.stats-row[\s{,]/m)
  })

  // @lat: [[design-language#Фитнес-функция языка оформления#Периwinkle снятого оформления]]
  it('no periwinkle from the retired design is left in the styles', () => {
    // It coloured the hover and active states of the whole shell: the highlight
    // followed no palette and looked identical under all five.
    expect(cssRaw).not.toContain('rgba(108,142,245')
    expect(cssRaw).not.toContain('rgba(108, 142, 245')
  })

  // @lat: [[design-language#Фитнес-функция языка оформления#Храповик по инлайновым стилям]]
  it('inline styles do not increase', () => {
    // A ratchet rather than a ban: an inline style is legitimate where the value
    // is computed from data (a bar's width, a colour chosen by threshold). The
    // budget is today's count per file; it comes down as screens are reworked
    // and never goes up. A file absent from this table is allowed none.
    const BUDGET: Record<string, number> = {
      // Everything left is computed from data: a column's width from the width
      // map, a metric's colour by threshold, a bar's height, a stage's share of
      // the latency — plus one shade handed to CSS as a property.
      'pages/ExperimentsPage.tsx': 5,
      'components/RunCharts.tsx': 2,
      'pages/DatasetsPage.tsx': 2,
      'pages/RunPage.tsx': 2,
      'components/Notifications.tsx': 1,
      'components/PipelineDiagram.tsx': 1,
      'pages/CorpusPage.tsx': 1,
      'pages/NewExperimentPage.tsx': 1,
      'pages/ProductionPage.tsx': 1,
      'pages/GuidePage.tsx': 1,
    }

    const over: string[] = []
    for (const [path, src] of Object.entries(SOURCE_FILES)) {
      const name = short(path)
      const count = (src.match(/style=\{\{/g) ?? []).length
      const budget = BUDGET[name] ?? 0
      if (count > budget) over.push(`${name}: ${count} > ${budget}`)
    }
    expect(over).toEqual([])

    // And the reverse: a budget that has risen above the fact must come down, or
    // the ratchet stops holding.
    const slack: string[] = []
    for (const [name, budget] of Object.entries(BUDGET)) {
      const src = SOURCE_FILES[`../${name}`]
      if (src === undefined) { slack.push(`${name}: no such file`); continue }
      const count = (src.match(/style=\{\{/g) ?? []).length
      if (count < budget) slack.push(`${name}: ${count} < ${budget}, lower the budget`)
    }
    expect(slack).toEqual([])
  })
})
