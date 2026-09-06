import { describe, expect, it } from 'vitest'
import en from '../i18n/locales/en.json'
import ru from '../i18n/locales/ru.json'
import { readFileSync } from 'node:fs'
import { join } from 'node:path'

/**
 * Reported from the running interface: a finding whose title was Russian
 * carried two English sentences under it, and a gap in the run's trace was
 * named by its field identifier, `candidate_source_refs`.
 *
 * The title had been translated by identifier for a while; the action and the
 * gap had not. Both are static prose on the server, so both can be. The
 * detail cannot: it carries the numbers the detector measured, interpolated
 * into a sentence before it leaves the server, and translating it would need
 * those numbers to arrive separately.
 */
const source = (relative: string) =>
  readFileSync(join(__dirname, '..', relative), 'utf8')

// Read out of the detectors themselves, never listed here. The list used to
// be typed out, and a detector added afterwards was covered by nothing: its
// title and its action reached the reader in English while everything around
// them was translated, and no test said so.
const DETECTORS = Array.from(
  readFileSync(join(__dirname, '..', '..', '..', 'core', 'eval', 'detectors.py'), 'utf8')
    .matchAll(/DiagnosticItem\(\s*\n\s*id="([a-z_]+)"/g),
).map(match => match[1])
const GAPS = ['sources', 'stage_trace', 'pre_rerank_source_refs', 'candidate_source_refs']

describe('a finding speaks the reader’s language', () => {
  it('finds the detectors to check, so a new one cannot slip past', () => {
    expect(DETECTORS.length).toBeGreaterThanOrEqual(9)
    expect(DETECTORS).toContain('unverified_coverage')
  })

  it.each(DETECTORS)('%s has a title and an action in both languages', (id) => {
    for (const [name, bundle] of [['en', en], ['ru', ru]] as const) {
      expect((bundle.runDiagnostics.finding as Record<string, string>)[id], `${name} title`).toBeTruthy()
      expect((bundle.runDiagnostics.findingAction as Record<string, string>)[id], `${name} action`).toBeTruthy()
    }
  })

  it('the two languages say different things', () => {
    // A locale file filled by copying the other passes every parity check.
    for (const id of DETECTORS) {
      const a = (en.runDiagnostics.findingAction as Record<string, string>)[id]
      const b = (ru.runDiagnostics.findingAction as Record<string, string>)[id]
      expect(b, `${id} is not translated`).not.toEqual(a)
    }
  })

  it('the panel looks the action up by identifier', () => {
    expect(source('components/RunDiagnostics.tsx')).toContain('runDiagnostics.findingAction.')
  })

  it('the detail is left in the server’s words, and says why', () => {
    // Asserted so that translating it becomes a deliberate act with the
    // server change it requires, and not a key somebody adds that silently
    // drops the numbers.
    const text = source('components/RunDiagnostics.tsx')
    expect(text).toContain('{item.detail}')
    expect(text).toContain('cannot be translated by')
  })
})

describe('a gap in the trace is named for a reader', () => {
  it.each(GAPS)('%s carries a label, an explanation and a remedy', (field) => {
    for (const [name, bundle] of [['en', en], ['ru', ru]] as const) {
      const gap = (bundle.runDiagnostics.traceGap as Record<string, Record<string, string>>)[field]
      expect(gap, `${name} has no entry for ${field}`).toBeTruthy()
      expect(gap.label, `${name} label`).toBeTruthy()
      expect(gap.unavailable.length, `${name} explanation`).toBeGreaterThan(40)
      expect(gap.remedy, `${name} remedy`).toBeTruthy()
    }
  })

  it('the field identifier is still shown, beside the name and not instead of it', () => {
    // Whoever has to return the field needs to know what it is called.
    const text = source('components/RunDiagnostics.tsx')
    expect(text).toContain('{gap.field}')
  })

  it('each of the three lines is looked up separately', () => {
    // Written loosely first: it asserted the file mentioned the key prefix at
    // all, so breaking one of the three lookups left it green.
    const text = source('components/RunDiagnostics.tsx')
    for (const part of ['label', 'unavailable', 'remedy']) {
      expect(text, `the ${part} is not looked up`).toContain(
        `runDiagnostics.traceGap.\${gap.field}.${part}`)
    }
  })

  it('the two languages say different things', () => {
    for (const field of GAPS) {
      const a = (en.runDiagnostics.traceGap as Record<string, Record<string, string>>)[field]
      const b = (ru.runDiagnostics.traceGap as Record<string, Record<string, string>>)[field]
      expect(b.label, `${field} label is not translated`).not.toEqual(a.label)
      expect(b.remedy, `${field} remedy is not translated`).not.toEqual(a.remedy)
    }
  })
})
