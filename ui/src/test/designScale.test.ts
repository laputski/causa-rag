import { describe, it, expect } from 'vitest'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'

// The sizes had drifted from the design: a field's label was the same size as
// its value, a section heading the same as the page title, and `.form-grid` and
// `.chips` were not defined at all, so forms ran as one column and chips sat
// flush against the rule.
const css = readFileSync(resolve(process.cwd(), 'src/styles.css'), 'utf8')

// The file is read from disk rather than through a `?raw` import: under vitest
// that returns an empty string, and every assertion below would pass over
// nothing.
it('the styles were read in full', () => {
  expect(css.length).toBeGreaterThan(10_000)
})

function rule(selector: string): string {
  const escaped = selector.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
  const m = css.match(new RegExp(`(^|\\n)${escaped}\\s*\\{([^}]*)\\}`))
  return m ? m[2] : ''
}

describe('the scale and density against the design', () => {
  it('a field label is smaller than its value', () => {
    // The design: 9.5px against the field's 12px. On the scale that is --fs-3xs
    // against --fs-xs.
    expect(rule('label')).toContain('font-size: var(--fs-3xs)')
    // The selector also catches an `<input>` with no `type` attribute: such a
    // field is textual, and `input[type="text"]` does not select it. And
    // `textarea`, which lived outside this rule and took the browser's white
    // fill — a white rectangle on the dark theme, invisible on the light one.
    const fieldRule = rule('select,\ntextarea,\ninput:not([type]),\ninput[type="text"], input[type="number"], input[type="search"], input[type="password"]')
    expect(fieldRule).toContain('font-size: var(--fs-xs)')
    expect(fieldRule).toContain('background: var(--color-bg)')
  })

  it('a section heading is smaller than the page title', () => {
    expect(rule('.section-title')).toContain('font-size: var(--fs-md)')
    expect(rule('h1, .page-title')).toContain('font-size: var(--fs-2xl)')
  })

  it('the form grid and the chip row are defined', () => {
    // Both classes were used in the markup and existed in no stylesheet: forms
    // ran as one column and chips sat flush against the rule.
    expect(rule('.form-grid')).toContain('grid-template-columns')
    expect(rule('.chips')).toContain('padding')
  })

  it('no periwinkle from the retired design is left in the styles', () => {
    expect(css).not.toContain('rgba(108,142,245')
  })
})
