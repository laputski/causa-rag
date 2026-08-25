import { describe, it, expect } from 'vitest'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { screen } from '@testing-library/react'
// The stylesheet is read from disk rather than imported.
//
// Both `import css from '../styles.css?raw'` and `import.meta.glob` with the
// same suffix return an empty string for CSS: Vite intercepts css before `?raw`
// gets a turn. Assertions about a substring being absent then passed over
// nothing — the test claimed there was no defect without having looked.
//
// Vitest runs under Node, so reading a file here is legitimate even though the
// project itself is a browser one. Under vitest `import.meta.url` is the
// module's http address rather than a file path, so the path is built from the
// process's working directory, which is `ui/`.
const css = readFileSync(resolve(process.cwd(), 'src/styles.css'), 'utf8')
import { LanguageMenu } from '../components/ShellControls'
import { renderWithRealm } from './realmTestUtils'

// The shell still rested on a colour retired with the old design.
//
// Eleven occurrences of `rgba(108,142,245, …)` coloured the hover and active
// states of the menu items, the realm switcher, the icon buttons and both
// dropdowns. There are five palettes, and the highlight was identical under all
// five, which means the menu took no part in the theming at all. That is
// visible only by eye and only after switching palette, so the test reads the
// stylesheet itself.

/** A rule's body, from `{` to its matching `}`.
 *
 *  A fixed-length slice will not do: these rules contain comments, and the
 *  first version of this test cut declarations off along with them, failing
 *  against correct CSS. */
function ruleBody(selector: string): string {
  const start = css.indexOf(selector)
  if (start < 0) throw new Error(`rule ${selector} not found`)
  const open = css.indexOf('{', start)
  return css.slice(open, css.indexOf('}', open) + 1)
}

describe('shell styling', () => {
  it('the stylesheet was actually read', () => {
    // A guard for the other assertions in this file: every one of them looks
    // for an absent substring, and against an empty string all of them pass.
    expect(css.length).toBeGreaterThan(10_000)
    expect(css).toContain('.nav-link')
  })

  it('no colour from the retired design is left in the styles', () => {
    const hits = [...css.matchAll(/rgba\(108,\s*142,\s*245/g)]
    expect(hits.length, 'the highlight must come from --color-primary').toBe(0)
  })

  it('a table row highlight is not white over transparent', () => {
    // White is invisible on the light theme: the row highlight existed only in
    // the dark one, and on the light one the table did not react to the cursor
    // at all.
    expect(css).not.toContain('rgba(255,255,255,.02)')
  })

  it('the active menu item is an inset rather than a three-pixel border', () => {
    const rule = ruleBody('.nav-link.active')
    expect(rule).toContain('inset 2px 0 0')
    expect(rule).not.toContain('border-left-color')
  })

  it('the menu sits on the page ground rather than its own fill', () => {
    expect(ruleBody('.sidebar {')).toContain('background: var(--color-bg)')
  })
})

describe('the language switcher', () => {
  it('shows the current language code beside the icon', async () => {
    renderWithRealm(<LanguageMenu />, '/overview', 'demo')
    // The icon says "language is changed here" without saying which one is
    // current, and in a multilingual interface somebody who opened the platform
    // in a language they do not read recognises the right control by exactly
    // those two letters.
    expect(await screen.findByText('EN')).toBeInTheDocument()
  })
})
