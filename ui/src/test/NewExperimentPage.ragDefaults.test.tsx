import { describe, it, expect, vi } from 'vitest'
import { render, screen, waitFor, fireEvent } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { MemoryRouter } from 'react-router-dom'
import NewExperimentPage from '../pages/NewExperimentPage'

// Selecting an external RAG must auto-fill its own
// default_corpus_id/pipeline_id/reranker_id/params instead of leaving
// whatever the form had before (previously corpus_id stayed hardcoded to
// 'handbook' regardless of which RAG was picked — a real experiment
// silently ran the wrong corpus_id against an unrelated RAG as a direct
// result, see the design notes).
vi.mock('../api/client', () => ({
  api: {
    registry: vi.fn().mockResolvedValue({
      pipeline: ['naive', 'hybrid_rrf', 'hybrid_weighted', 'graph'],
      reranker: ['cross_encoder_stub'],
      grounder: [], route_policy: [], scorer: [], mask_engine: [], refusal: [],
    }),
    datasets: { list: vi.fn().mockResolvedValue([]) },
    corpus: { list: vi.fn().mockResolvedValue([]), collections: vi.fn().mockResolvedValue([]) },
    externalRags: {
      list: vi.fn().mockResolvedValue([
        {
          id: 'rag1', name: 'demo-rag', url: 'http://localhost:8002/platform/query',
          default_corpus_id: 'handbook', default_pipeline_id: 'hybrid_weighted',
          default_reranker_id: 'none', default_params: { alpha: 0.7 },
        },
      ]),
      listDatasets: vi.fn().mockResolvedValue([]),
    },
    experiments: { create: vi.fn() },
  },
}))

function wrap(children: React.ReactNode) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return (
    <QueryClientProvider client={qc}>
      <MemoryRouter>{children}</MemoryRouter>
    </QueryClientProvider>
  )
}

describe('NewExperimentPage — external RAG default-config auto-fill', () => {
  it('fills corpus_id/pipeline_id/reranker/params from the selected RAG default_*', async () => {
    render(wrap(<NewExperimentPage />))

    const source = await screen.findByLabelText(/implementation/i) as HTMLSelectElement
    fireEvent.change(source, { target: { value: 'http' } })

    const ragSelect = await screen.findByLabelText(/external rag/i) as HTMLSelectElement
    await waitFor(() => expect([...ragSelect.options].some(o => o.value === 'rag1')).toBe(true))
    fireEvent.change(ragSelect, { target: { value: 'rag1' } })

    // Queried fresh inside waitFor, not captured beforehand: 'handbook'
    // isn't a known corpus_id for this Realm, so CorpusIdField swaps from a
    // <select> to a plain <input> (custom-value escape hatch) once the
    // auto-fill lands — a stale reference to the pre-swap <select> node
    // would never observe the new value.
    await waitFor(() => {
      const corpusInput = screen.getByLabelText(/corpus/i) as HTMLInputElement
      expect(corpusInput.value).toBe('handbook')
    })

    const paramsField = screen.getByLabelText(/params/i) as HTMLTextAreaElement
    expect(JSON.parse(paramsField.value)).toEqual({ alpha: 0.7 })
  })

  it('leaves the form untouched when the selected RAG has no default_* set', async () => {
    render(wrap(<NewExperimentPage />))

    const source = await screen.findByLabelText(/implementation/i) as HTMLSelectElement
    fireEvent.change(source, { target: { value: 'http' } })

    const ragSelect = await screen.findByLabelText(/external rag/i) as HTMLSelectElement
    const corpusInput = screen.getByLabelText(/corpus/i) as HTMLInputElement
    const before = corpusInput.value

    fireEvent.change(ragSelect, { target: { value: '' } })  // "— select from saved —"

    expect(corpusInput.value).toBe(before)
  })
})
