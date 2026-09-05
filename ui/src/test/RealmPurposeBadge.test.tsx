import { describe, expect, it } from 'vitest'
import { render, screen } from '@testing-library/react'
import { readFileSync } from 'node:fs'
import { join } from 'node:path'
import '../i18n'
import { PROVING_GROUND, RealmPurposeBadge } from '../components/RealmPurposeBadge'
import en from '../i18n/locales/en.json'
import ru from '../i18n/locales/ru.json'

/**
 * A realm whose data is broken on purpose has to say so where it is named.
 *
 * Without the mark its red diagnostics read as a broken installation, and the
 * proving ground exists precisely to be red.
 *
 * The component itself is rendered here. A first version rendered a copy of
 * the markup written inside the test, which measures the copy and says nothing
 * about what the application draws.
 */
describe('the mark on a realm broken on purpose', () => {
  it('appears when the realm carries the purpose', () => {
    render(<RealmPurposeBadge purpose={PROVING_GROUND} />)
    expect(screen.getByText(en.realm.provingGround.label)).toBeTruthy()
  })

  it('is absent on a realm that carries none', () => {
    const { container } = render(<RealmPurposeBadge />)
    expect(container.innerHTML).toEqual('')
  })

  it('is absent on a realm carrying some other purpose', () => {
    const { container } = render(<RealmPurposeBadge purpose="something_else" />)
    expect(container.innerHTML).toEqual('')
  })

  it('carries the explanation as well as the label', () => {
    // A label alone renames the realm. What a reader needs is why red is
    // expected here, and that is what the title carries.
    render(<RealmPurposeBadge purpose={PROVING_GROUND} />)
    expect(screen.getByTitle(en.realm.provingGround.hint)).toBeTruthy()
  })
})

describe('every place that names a realm shows the mark', () => {
  const source = (relative: string) => readFileSync(join(__dirname, '..', relative), 'utf8')

  it.each([
    ['pages/OverviewPage.tsx', 'the overview names the realm in its title'],
    ['App.tsx', 'the realm switcher names it, and so does its menu'],
  ])('%s uses the component', (relative) => {
    expect(source(relative)).toContain('<RealmPurposeBadge')
  })

  it('the switcher marks the menu as well as the trigger', () => {
    // Two places in one file: the realm on screen now, and every realm on
    // offer. Marking only the first leaves a reader choosing blind.
    const occurrences = source('App.tsx').split('<RealmPurposeBadge').length - 1
    expect(occurrences).toBeGreaterThanOrEqual(2)
  })

  it('nobody spells the condition out a second time', () => {
    // The string belongs to the component. Three copies of a condition drift.
    for (const relative of ['pages/OverviewPage.tsx', 'App.tsx']) {
      expect(source(relative)).not.toContain("'proving_ground'")
    }
  })
})

describe('the wording exists in both languages', () => {
  it('is translated, and not left as a key', () => {
    for (const [name, bundle] of [['en', en], ['ru', ru]] as const) {
      expect(bundle.realm?.provingGround?.label, `${name} has no label`).toBeTruthy()
      expect(bundle.realm?.provingGround?.hint, `${name} has no hint`).toBeTruthy()
      expect(String(bundle.realm.provingGround.hint).length).toBeGreaterThan(40)
    }
  })

  it('says something different in each language', () => {
    // Pinned because a locale file filled by copying the other passes every
    // parity check there is.
    expect(en.realm.provingGround.label).not.toEqual(ru.realm.provingGround.label)
  })
})
