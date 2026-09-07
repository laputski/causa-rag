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
 * gap had not. Both are static prose on the server, so both can be.
 *
 * The detail was the one line that could not be, because it carries the
 * numbers the detector measured, interpolated into a sentence before it left
 * the server. It sends those numbers beside it now, and a check further down
 * used to assert the opposite: it was written so that translating the detail
 * would have to be a deliberate act with the server change it needs, and not
 * a key somebody adds that silently drops the numbers. That is what happened,
 * and the assertion is now the other way round.
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

  it('the panel composes the detail from what the server measured', () => {
    const text = source('components/RunDiagnostics.tsx')
    expect(text).toContain('runDiagnostics.findingDetail.')
    // The server's English is what a finding renders when this side has no
    // sentence for it, which is every run stored before the values travelled.
    expect(text).toContain('defaultValue: item.detail')
    // Two findings are the same finding on different evidence and share an
    // identifier, so the detail is keyed by its own name where they differ.
    expect(text).toContain('item.detail_key')
  })

  it('every detail carries the numbers, in both languages', () => {
    // A sentence translated without its placeholders reads perfectly and tells
    // the reader nothing the detector measured. Which placeholders belong to
    // which finding is checked against the detectors themselves, in
    // tests/fitness/test_finding_details_are_translatable.py; what is checked
    // here is that a translation carries any at all.
    for (const [name, bundle] of [['en', en], ['ru', ru]] as const) {
      const details = bundle.runDiagnostics.findingDetail as Record<string, unknown>
      const sentences: string[] = []
      const collect = (node: unknown) => {
        if (typeof node === 'string') sentences.push(node)
        else if (node && typeof node === 'object') Object.values(node).forEach(collect)
      }
      collect(details)
      expect(sentences.length, `${name} has no detail sentences`).toBeGreaterThanOrEqual(20)
      for (const sentence of sentences) {
        expect(sentence, `${name}: "${sentence}" carries no measured value`).toMatch(/\{\{\w+\}\}/)
      }
    }
  })

  it('the two languages say different things in the detail as well', () => {
    const flatten = (node: unknown, into: Record<string, string>, at = ''): Record<string, string> => {
      if (typeof node === 'string') into[at] = node
      else if (node && typeof node === 'object')
        for (const [k, v] of Object.entries(node)) flatten(v, into, at ? `${at}.${k}` : k)
      return into
    }
    const english = flatten(en.runDiagnostics.findingDetail, {})
    const russian = flatten(ru.runDiagnostics.findingDetail, {})
    for (const key of Object.keys(english)) {
      expect(russian[key], `${key} is not translated`).not.toEqual(english[key])
    }
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
