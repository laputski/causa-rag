import { describe, it, expect } from 'vitest'

// A screen answers "what is happening now" and the guide answers "what is this
// and what is it for", and the path between them should be one click. A link
// travels by section name, and given an unfamiliar name the guide silently opens
// on its first section: only somebody who remembers where they meant to land
// notices the miss.

const SOURCES = import.meta.glob('../{pages,components}/**/*.tsx', {
  query: '?raw', import: 'default', eager: true,
}) as Record<string, string>

const guide = SOURCES['../pages/GuidePage.tsx']
const sectionIds = new Set([...guide.matchAll(/id: '([\w-]+)', label: 'guidePage\.nav\./g)].map(m => m[1]))

function linkedSections(): { file: string; section: string }[] {
  const out: { file: string; section: string }[] = []
  for (const [path, src] of Object.entries(SOURCES)) {
    for (const m of src.matchAll(/<GuideLink section="([\w-]+)"/g)) out.push({ file: path, section: m[1] })
    for (const m of src.matchAll(/\/guide\?section=([\w-]+)/g)) out.push({ file: path, section: m[1] })
  }
  return out
}

describe('links from screens into the guide', () => {
  it('the guide has sections that can be named at all', () => {
    // Insurance for the check below: against an empty set it would pass while
    // checking nothing.
    expect(sectionIds.size).toBeGreaterThan(20)
  })

  it('every link leads to a section that exists', () => {
    const broken = linkedSections()
      .filter(({ section }) => !sectionIds.has(section))
      .map(({ file, section }) => `${file.replace('../', '')} → ${section}`)
    expect(broken).toEqual([])
  })

  it('screens that have a section link to it', () => {
    // Not all of them: question sets and presets have no section of their own,
    // so there is nowhere to invent a link to. Listed here are the ones whose
    // correspondence is unambiguous.
    const expected: Record<string, string> = {
      '../pages/FrontierPage.tsx': 'frontier',
      '../pages/ProductionPage.tsx': 'production',
      '../pages/JudgmentsPage.tsx': 'judgments',
      '../pages/ComparisonPage.tsx': 'regression',
      '../pages/NewExperimentPage.tsx': 'config',
      '../pages/CorpusPage.tsx': 'corpus',
    }
    const found = new Map(linkedSections().map(({ file, section }) => [file, section]))
    for (const [file, section] of Object.entries(expected)) {
      expect(found.get(file), `${file} should link to "${section}"`).toBe(section)
    }
  })
})
