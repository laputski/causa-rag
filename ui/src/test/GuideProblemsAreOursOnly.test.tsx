import { describe, it, expect } from 'vitest'
import { readFileSync } from 'node:fs'
import { join } from 'node:path'
import en from '../i18n/locales/en.json'
import ru from '../i18n/locales/ru.json'

/**
 * Two catalogues, and the guide used to hold both.
 *
 * Sixteen cards stood in one list: twelve failures any RAG can have, two more
 * that belong to a graph, and two defects of this platform and of running it.
 * A reader had no way to tell which kind a card was, and the twelve had
 * entries in the atlas already, where an entry claims a signal catches the
 * failure and the build refuses that claim without a bait. The guide claimed
 * nothing of the sort about the same failures, one page away.
 *
 * The fourteen are gone from here and the section says where they went. What
 * is left is this platform's own incident log, which is what the section was
 * always meant to be.
 */
const source = readFileSync(join(__dirname, '..', 'pages', 'GuidePage.tsx'), 'utf8')

/** The two that stay, named so a third has to be argued for and cannot
 *  merely appear. Both are about the infrastructure around the retrieval, and
 *  neither is a failure a served system could have. */
const OURS = {
  problem5: 'an async driver called from a synchronous handler',
  problem14: "the container host's disk filling up",
}

describe('the guide keeps only this platform’s own incidents', () => {
  it('renders a card for each of them and for nothing else', () => {
    const rendered = [...source.matchAll(/problemsSection\.(problem\d+)Title/g)]
      .map(m => m[1])
    expect(new Set(rendered)).toEqual(new Set(Object.keys(OURS)))
  })

  it('carries no text for a card it no longer renders', () => {
    // A key left behind is a card somebody can restore without deciding to.
    for (const [name, bundle] of [['en', en], ['ru', ru]] as const) {
      const section = bundle.guidePage.problemsSection as Record<string, string>
      const orphaned = Object.keys(section)
        .filter(k => /^problem\d+/.test(k))
        .filter(k => !Object.keys(OURS).some(kept => k.startsWith(kept)))
      expect(orphaned, `${name} keeps text for cards nobody renders`).toEqual([])
    }
  })

  it('says where the failures a served system can have went', () => {
    expect(source).toContain('problemsSection.elsewhere')
    expect(source).toMatch(/<Link to="\/atlas"/)
    for (const [name, bundle] of [['en', en], ['ru', ru]] as const) {
      const section = bundle.guidePage.problemsSection as Record<string, string>
      expect(section.elsewhere, `${name} does not say where they went`).toBeTruthy()
      expect(section.elsewhere).toContain('<0>')
    }
  })
})
