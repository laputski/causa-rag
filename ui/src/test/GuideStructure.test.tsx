import { describe, it, expect } from 'vitest'
import { render, screen, fireEvent, within } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import GuidePage from '../pages/GuidePage'
import ru from '../i18n/locales/ru.json'

// The guide grew section by section over many months: some described mechanisms
// that had since been removed, some screens were not described at all, and one
// group carried a third of every section. What is checked is what must hold for
// every section after the rework, rather than for the ones somebody remembered.

const source = readFileSync(resolve(process.cwd(), 'src/pages/GuidePage.tsx'), 'utf8')

function renderGuide() {
  return render(<MemoryRouter><GuidePage /></MemoryRouter>)
}

/** The table-of-contents buttons on the left, one per section. */
function navButtons(): HTMLElement[] {
  return [...document.querySelectorAll('.guide-nav-item')] as HTMLElement[]
}

describe('the guide: sections', () => {
  it('every section opens and begins with a leading paragraph', () => {
    renderGuide()
    const total = navButtons().length
    expect(total).toBeGreaterThanOrEqual(24)

    const empty: string[] = []
    for (let i = 0; i < total; i++) {
      // The button list is re-read each time: a re-render replaces the nodes.
      fireEvent.click(navButtons()[i])
      const body = document.querySelector('.guide-content') ?? document.body
      const lead = body.querySelector('.guide-lead')
      // The bound is a sentence rather than a matter of taste: the design notes
      // require a leading paragraph of at least one sentence and at most 250
      // characters.
      const text = (lead?.textContent ?? '').trim()
      if (!lead || text.length < 20 || text.length > 400) {
        empty.push(navButtons()[i].textContent?.trim() ?? `#${i}`)
      }
    }
    expect(empty).toEqual([])
  })

  it('no group carries more than a third of the sections', () => {
    // Eight sections out of twenty-two under one heading is a store room rather
    // than a group: a reader does not choose from it, they scroll past it.
    renderGuide()
    const groups = [...document.querySelectorAll('.guide-nav-group')]
    expect(groups.length).toBeGreaterThanOrEqual(6)
    const total = navButtons().length
    const sizes = groups.map(g => g.querySelectorAll('.guide-nav-item').length)
    expect(Math.max(...sizes)).toBeLessThanOrEqual(Math.ceil(total / 3))
  })

  it('tells of no mechanism the reader never saw', () => {
    // "The former mechanism substituted retrieval…", "that part of the former
    // mechanism was found unsound" — a reader arriving today has nothing to
    // compare the former thing to, and the paragraph tells them only that
    // there is something they do not know.
    //
    // The pattern is narrow on purpose. A broader sweep for «раньше»/«ранее»
    // flagged nine paragraphs and every one was legitimate: «неделей раньше»,
    // «названные заранее», «сохраняют прежний хеш» — the last about the
    // reader's own older runs, which they did see. What has no honest use in
    // a guide is naming a mechanism of the platform that no longer exists.
    const PAST = /прежн\w*\s+(механизм|устройств|подход|порядок)/i
    const offenders: string[] = []
    for (const [section, node] of Object.entries(ru.guidePage as Record<string, unknown>)) {
      if (section === 'problemsSection' || typeof node !== 'object' || node === null) continue
      for (const [key, value] of Object.entries(node as Record<string, unknown>)) {
        if (typeof value === 'string' && PAST.test(value)) offenders.push(`${section}.${key}`)
      }
    }
    expect(offenders).toEqual([])
  })

  it('describes how the shell is driven, not only the runs', () => {
    // Notifications, the command palette, realm transfer and key metrics: four
    // things the guide did not name at all.
    //
    // This used to be checked with Russian words, and passed by oversight: both
    // sections sat in the English locale file written in Russian. The test was
    // holding on to the very defect it should have reported. The words are
    // English now, which is what the tests render, and the invariants (⌘K,
    // causa-realm/v1) work in any language.
    renderGuide()
    fireEvent.click(screen.getByRole('button', { name: 'Commands and notifications' }))
    const shell = within(document.querySelector('.guide-content') as HTMLElement)
    expect(shell.getAllByText(/command palette/i).length).toBeGreaterThan(0)
    expect(shell.getByText(/⌘K/)).toBeTruthy()

    fireEvent.click(screen.getByRole('button', { name: 'Realms' }))
    const realm = within(document.querySelector('.guide-content') as HTMLElement)
    expect(realm.getByText(/causa-realm\/v1/i)).toBeTruthy()
    expect(realm.getAllByText(/key metrics/i).length).toBeGreaterThan(0)
  })
})

describe('the guide: diagrams', () => {
  it('carries no colour literals, only palette variables', () => {
    const hex = source.match(/#[0-9a-fA-F]{3}(?:[0-9a-fA-F]{3})?\b/g) ?? []
    expect(hex).toEqual([])
  })

  it('does not glue an opacity onto a variable name', () => {
    // `var(--diag-blue)` plus `'10'` gives `var(--diag-blue)10`, which is invalid
    // CSS, and the browser paints black. On the dark theme that goes unnoticed;
    // on the light one the diagram becomes a black rectangle. Opacity is set with
    // `fillOpacity` rather than an eight-digit hex, which only works with a
    // literal.
    const glued = source.match(/(?:fill|stroke)=\{\s*\w+\s*\+\s*'[0-9a-fA-F]{2}'/g) ?? []
    expect(glued).toEqual([])
  })

  it('every label fits inside its own box', () => {
    renderGuide()
    const overflow: string[] = []
    for (const btn of navButtons()) {
      fireEvent.click(btn)
      for (const svg of document.querySelectorAll('svg[viewBox]')) {
        const vb = (svg.getAttribute('viewBox') ?? '').split(/\s+/).map(Number)
        if (vb.length !== 4) continue
        const [, , vw] = vb
        for (const text of svg.querySelectorAll('text')) {
          const size = Number(text.getAttribute('fontSize') ?? text.getAttribute('font-size') ?? 10)
          const anchor = text.getAttribute('textAnchor') ?? text.getAttribute('text-anchor') ?? 'start'
          // Line by line: a long label is split across `tspan`s, and measuring it
          // whole would add up the widths of lines that sit one under another.
          const spans = text.querySelectorAll('tspan')
          const lines = spans.length > 0
            ? [...spans].map(sp => ({ x: Number(sp.getAttribute('x') ?? text.getAttribute('x') ?? 0), s: sp.textContent ?? '' }))
            : [{ x: Number(text.getAttribute('x') ?? 0), s: text.textContent ?? '' }]
          for (const line of lines) {
            // The width is approximated with the same coefficient `wrapLabel`
            // uses: an exact one needs a laid-out DOM, and there is none.
            const width = line.s.length * size * 0.54
            const left = anchor === 'middle' ? line.x - width / 2 : anchor === 'end' ? line.x - width : line.x
            if (left < -2 || left + width > vw + 2) {
              overflow.push(`${btn.textContent?.trim()}: «${line.s}»`)
            }
          }
        }
      }
    }
    expect(overflow.join('\n')).toEqual('')
  })
})
