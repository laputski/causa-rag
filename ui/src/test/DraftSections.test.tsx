import { describe, it, expect } from 'vitest'

// Three sections are marked as drafts: they work, and they are not fully
// verified. This test keeps the list consistent in two places at once — the mark
// in the menu and the mark in the section's own header must sit on the same
// screens, or a reader arriving by direct link never sees the caveat.

const SOURCES = import.meta.glob('../{App.tsx,pages/*.tsx}', {
  query: '?raw', import: 'default', eager: true,
}) as Record<string, string>

const DRAFT_PAGES = ['FrontierPage', 'JudgmentsPage', 'ProductionPage']

describe('draft sections', () => {
  it('are marked in the menu', () => {
    const app = SOURCES['../App.tsx']
    for (const route of ['/frontier', '/data/judgments', '/production']) {
      const line = app.split('\n').find(l => l.includes(`<NavItem to="${route}"`))
      expect(line, `no menu item for ${route}`).toBeTruthy()
      expect(line, `${route} carries no draft mark`).toMatch(/\bdraft\b/)
    }
  })

  it("are marked in the section's own header", () => {
    for (const page of DRAFT_PAGES) {
      const src = SOURCES[`../pages/${page}.tsx`]
      expect(src, `no such file: ${page}`).toBeTruthy()
      expect(src, `${page} has no badge`).toContain('<DraftBadge />')
      expect(src, `${page} has no note`).toContain('<DraftNote />')
    }
  })

  it('and nowhere else: the mark denotes a list rather than decoration', () => {
    const marked = Object.entries(SOURCES)
      .filter(([path, src]) => path.startsWith('../pages/') && src.includes('<DraftBadge'))
      .map(([path]) => path.replace('../pages/', '').replace('.tsx', ''))
    expect(marked.sort()).toEqual([...DRAFT_PAGES].sort())
  })
})
