import { describe, it, expect, vi, beforeEach } from 'vitest'
import { screen, waitFor, fireEvent } from '@testing-library/react'
import ProductionPage from '../pages/ProductionPage'
import { stubRealmFetch, renderWithRealm } from './realmTestUtils'
import type { ProductionTrace } from '../api/client'

// Phase 7 — the reviewer-facing half of production traffic. What is pinned
// here is the distinctions the page must not collapse: a candidate is not
// feedback, recording switched off is not silence, and an already-promoted
// trace is not promotable again.

const listMock = vi.fn()
const collectMock = vi.fn()
const promoteMock = vi.fn()
const coverageMock = vi.fn()

vi.mock('../api/client', () => ({
  api: {
    production: {
      list: (realmId: string, corpusId: string) => listMock(realmId, corpusId),
      collect: (body: unknown) => collectMock(body),
      promote: (body: unknown) => promoteMock(body),
      coverage: (realmId: string, corpusId: string, dataset: string, threshold?: number) =>
        coverageMock(realmId, corpusId, dataset, threshold),
    },
    corpus: { collections: () => Promise.resolve([{ corpus_id: 'handbook_01' }]) },
    datasets: { list: () => Promise.resolve([{ name: 'Cosmos 1', filename: 'cosmos1.jsonl', id: 'ds1', count: 50 }]) },
  },
}))

const TRACE: ProductionTrace = {
  trace_id: 'tr-1', realm_id: 'acme', corpus_id: 'handbook_01',
  query: "What are the product's dimensions?",
  sources: [], answer_preview: 'The system answer',
  created_at: '2026-08-10T10:00:00Z', collected_at: '2026-08-10T11:00:00Z',
  promoted_question_id: null,
}

function renderPage() {
  return renderWithRealm(<ProductionPage />, '/production', 'acme')
}

async function selectCorpus() {
  await waitFor(() => expect(screen.getByText('Choose a corpus')).toBeInTheDocument())
  // The prompt above renders before the corpus ids do, and a change to an
  // option the select does not offer yet is dropped in silence, so the wait
  // is on the option rather than on the invitation to choose one.
  // findAllByRole, because the page carries more than one select and the
  // corpus one is the first.
  const [select] = await screen.findAllByRole('combobox') as HTMLSelectElement[]
  await waitFor(() =>
    expect([...select.options].map(o => o.value)).toContain('handbook_01'))
  fireEvent.change(select, { target: { value: 'handbook_01' } })
}

describe('ProductionPage', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    stubRealmFetch('acme')
    listMock.mockResolvedValue([TRACE])
    collectMock.mockResolvedValue({ collected: 4, stored: 4, recording_enabled: true })
    promoteMock.mockResolvedValue({ question_id: 'q123', dataset: 'Cosmos 1', count: 51 })
  })

  // @lat: [[production-loop#Closing the loop with production#Reviewer-facing surface#A corpus scopes the page]]
  it('says why the list is empty before a corpus is chosen', async () => {
    renderPage()
    await waitFor(() => expect(screen.getByText('Choose a corpus')).toBeInTheDocument())
    expect(listMock).not.toHaveBeenCalled()
  })

  // @lat: [[production-loop#Closing the loop with production#Reviewer-facing surface#A trace with no sources says so on the row]]
  it('lists collected queries with the source count visible on the row', async () => {
    renderPage()
    await selectCorpus()
    await waitFor(() => expect(document.querySelector('.pick-row-title')).toHaveTextContent("What are the product's dimensions?"))
    // A served answer with no retrieved source is itself the finding, so the
    // count is on the row rather than hidden one click away.
    expect(screen.getByText('sources: 0')).toBeInTheDocument()
  })

  // @lat: [[production-loop#Closing the loop with production#Reviewer-facing surface#Recording switched off is not the same as no traffic]]
  it('distinguishes recording switched off from an empty result', async () => {
    collectMock.mockResolvedValue({ collected: 0, stored: 0, recording_enabled: false })
    renderPage()
    await selectCorpus()
    await waitFor(() => expect(document.querySelector('.pick-row-title')).toHaveTextContent("What are the product's dimensions?"))

    fireEvent.click(screen.getAllByRole('button', { name: /Fetch the log/ })[0])
    fireEvent.change(screen.getByLabelText(/Log address/), {
      target: { value: 'http://localhost:8020/traces' },
    })
    fireEvent.click(screen.getAllByRole('button', { name: /^Fetch the log$/ }).pop()!)

    await waitFor(() => expect(screen.getByText(/its logging is switched off/)).toBeInTheDocument())
  })

  // @lat: [[production-loop#Closing the loop with production#Reviewer-facing surface#Promotion demands a reference answer and refuses a second time]]
  it('will not promote without a reference answer typed by a reviewer', async () => {
    renderPage()
    await selectCorpus()
    await waitFor(() => expect(document.querySelector('.pick-row-title')).toHaveTextContent("What are the product's dimensions?"))
    fireEvent.click(document.querySelector('.pick-row-title') as HTMLElement)

    const button = await screen.findByRole('button', { name: 'Add to the set' })
    expect(button).toBeDisabled()

    // Two fields carry this label on the page — the promote panel's and the
    // coverage panel's below it. The first is the one in the detail.
    fireEvent.change(screen.getAllByLabelText(/Golden set/)[0], { target: { value: 'ds1' } })
    fireEvent.change(screen.getByLabelText(/Reference answer/), { target: { value: '600×400 mm' } })
    expect(button).not.toBeDisabled()
    fireEvent.click(button)

    await waitFor(() => expect(promoteMock).toHaveBeenCalledTimes(1))
    const body = promoteMock.mock.calls[0][0] as Record<string, unknown>
    // The user's own wording travels; the answer never comes from the trace.
    expect(body.reference_answer).toBe('600×400 mm')
    expect(body.trace_id).toBe('tr-1')
  })

  it('offers no second promotion for a trace already turned into a question', async () => {
    listMock.mockResolvedValue([{ ...TRACE, promoted_question_id: 'q999' }])
    renderPage()
    await selectCorpus()
    await waitFor(() => expect(document.querySelector('.pick-row-title')).toHaveTextContent("What are the product's dimensions?"))
    fireEvent.click(document.querySelector('.pick-row-title') as HTMLElement)

    expect(await screen.findByText(/already added as question q999/)).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Add to the set' })).not.toBeInTheDocument()
  })

  // @lat: [[production-loop#Closing the loop with production#Reviewer-facing surface#A promotion that no longer points at anything]]
  it('offers promotion again when the question it became was deleted', async () => {
    // Found on the platform's own data: a question removed after verification
    // left its trace refusing forever, with no way back short of editing the
    // database by hand.
    listMock.mockResolvedValue([{ ...TRACE, promoted_question_id: 'gone42', promotion_live: false }])
    renderPage()
    await selectCorpus()
    await waitFor(() => expect(document.querySelector('.pick-row-title')).toHaveTextContent("What are the product's dimensions?"))
    fireEvent.click(document.querySelector('.pick-row-title') as HTMLElement)

    expect(await screen.findByRole('button', { name: 'Add to the set' })).toBeInTheDocument()
    // The mark itself is kept and explained rather than quietly dropped.
    expect(screen.getByText(/already turned into question gone42/)).toBeInTheDocument()
    expect(screen.getByText('question deleted')).toBeInTheDocument()
  })

  // @lat: [[production-loop#Closing the loop with production#Reviewer-facing surface#Coverage reports the shape beside the share]]
  it('reports the share and the distribution, not one without the other', async () => {
    coverageMock.mockResolvedValue({
      threshold: 0.6, n_production: 4, n_golden: 50,
      uncovered_share: 1.0, uncovered_total: 4,
      uncovered: ["What are the product's dimensions?"],
      deciles: [0.55, 0.56, 0.57, 0.58, 0.59, 0.59, 0.59, 0.59, 0.59, 0.59],
    })
    renderPage()
    await selectCorpus()
    await waitFor(() => expect(document.querySelector('.pick-row-title')).toHaveTextContent("What are the product's dimensions?"))

    // The measure computes itself as soon as a dataset is chosen: there is no
    // "Measure" button any more, because a section showing two fields until a
    // press read as a form rather than an answer.

    await waitFor(() => expect(screen.getByText('100%')).toBeInTheDocument())
    // The filename, since that is what the server matches on first.
    expect(coverageMock).toHaveBeenCalledWith('acme', 'handbook_01', 'cosmos1.jsonl', 0.6)
    expect(screen.getByText(/by decile/)).toBeInTheDocument()
  })
})
