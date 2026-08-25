import { describe, it, expect, vi, beforeEach } from 'vitest'
import { screen, waitFor, fireEvent } from '@testing-library/react'
import NewExperimentPage from '../pages/NewExperimentPage'
import { renderWithRealm } from './realmTestUtils'

// The form ran as one sheet of fields: eleven selects in a row with no division
// between what is being run and what it runs on, and the start button hid
// beneath them.

const REGISTRY = {
  pipeline: ['dense', 'hybrid_rrf', 'graph_rag'],
  reranker: ['bge_reranker'],
  grounder: [], route_policy: [], scorer: [], mask_engine: [], refusal: [],
}

vi.mock('../api/client', () => ({
  api: {
    registry: () => Promise.resolve(REGISTRY),
    models: () => Promise.resolve([]),
    datasets: { list: () => Promise.resolve([{ filename: 'set-a.json', name: 'Set A', version: 'v1', count: 12, speed: 'fast' }]) },
    externalRags: { list: () => Promise.resolve([]) },
    corpus: {
      list: () => Promise.resolve([]),
      collections: () => Promise.resolve([{ corpus_id: 'demo_corpus', chunks: 120 }]),
    },
    experiments: { get: () => Promise.resolve(null), create: vi.fn() },
  },
}))

beforeEach(() => {
  localStorage.clear()
  vi.stubGlobal('fetch', vi.fn(async () => ({
    ok: true,
    json: async () => [{ id: 'demo', name: 'Demo', resources: [], key_metrics: [], created_at: '2026-01-01' }],
  })))
})

// @lat: [[design-language#Новый прогон — the form's two halves#Разделы формы]]
describe('the new-run form layout', () => {
  it('divides the form into three ruled sections', async () => {
    renderWithRealm(<NewExperimentPage />, '/new', 'demo')

    await waitFor(() => {
      const titles = [...document.querySelectorAll('.section-rule .section-title')]
        .map(n => n.textContent?.trim())
      expect(titles).toEqual(['Presets', 'What we run', 'Pipeline'])
    })
  })

  it('leaves no field outside a section grid', async () => {
    renderWithRealm(<NewExperimentPage />, '/new', 'demo')

    // A field outside `.form-grid` is one that fell out of the columns and
    // stretched full width: that is exactly how the layout drifts from the
    // design.
    await waitFor(() => expect(document.querySelector('.form-grid')).toBeTruthy())
    const stray = [...document.querySelectorAll('.form-group')]
      .filter(g => !g.closest('.form-grid'))
      .map(g => g.querySelector('label')?.textContent)
    expect(stray).toEqual([])
  })

  it('puts the start button in the header, above the form', async () => {
    renderWithRealm(<NewExperimentPage />, '/new', 'demo')

    await waitFor(() => expect(document.querySelector('form')).toBeTruthy())
    const form = document.querySelector('form') as HTMLFormElement
    const submit = document.querySelector('button[type="submit"]') as HTMLButtonElement
    // The button sits outside the form, tied to it by id, and earlier in the
    // document.
    expect(submit.getAttribute('form')).toBe(form.id)
    expect(form.compareDocumentPosition(submit) & Node.DOCUMENT_POSITION_PRECEDING).toBeTruthy()
    expect(form.querySelector('button[type="submit"]')).toBeNull()
  })

  it('clicking a dataset chip changes the pipeline fields', async () => {
    renderWithRealm(<NewExperimentPage />, '/new', 'demo')

    await waitFor(() => expect(document.querySelectorAll('.chip').length).toBeGreaterThan(0))
    const topK = screen.getByLabelText('top-k') as HTMLInputElement
    const before = topK.value
    const chip = [...document.querySelectorAll('.chip')]
      .find(c => c.textContent?.includes('High Recall')) as HTMLElement
    fireEvent.click(chip)

    expect(topK.value).not.toBe(before)
    expect(chip.classList.contains('active')).toBe(true)
  })
})
