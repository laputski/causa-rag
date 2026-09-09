import { describe, it, expect, vi } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { MemoryRouter } from 'react-router-dom'
import NewExperimentPage from '../pages/NewExperimentPage'

vi.mock('../api/client', () => ({
  api: {
    registry: vi.fn().mockResolvedValue({
      pipeline: ['naive', 'hybrid_rrf', 'hybrid_weighted', 'graph'],
      reranker: ['cross_encoder_stub'],
      grounder: ['token_overlap'],
      route_policy: ['naive'],
      scorer: [],
      mask_engine: [],
      refusal: [],
    }),
    datasets: { list: vi.fn().mockResolvedValue([]) },
    corpus: { list: vi.fn().mockResolvedValue([]), collections: vi.fn().mockResolvedValue([]) },
    externalRags: { list: vi.fn().mockResolvedValue([]) },
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

describe('NewExperimentPage', () => {
  it('renders the form', () => {
    render(wrap(<NewExperimentPage />))
    expect(screen.getByRole('heading', { name: /new run/i })).toBeInTheDocument()
  })

  it('shows the submit button', () => {
    render(wrap(<NewExperimentPage />))
    expect(screen.getByRole('button', { name: /start run/i })).toBeInTheDocument()
  })

  it('offers a real "Retrieval type" selector built from the pipeline registry', async () => {
    render(wrap(<NewExperimentPage />))
    const select = await screen.findByLabelText(/retrieval type/i) as HTMLSelectElement
    // Options populate once the registry query resolves — wait for that
    // rather than asserting on the first (empty) render.
    await waitFor(() => expect(select.options.length).toBeGreaterThan(0))
    const values = [...select.options].map(o => o.value)
    // 'naive' (dense-only) must be reachable — it used to be impossible to
    // select from this form even though the registry always had it (see
    // the design notes).
    expect(values).toEqual(['naive', 'hybrid_rrf', 'hybrid_weighted', 'graph'])
    expect(select.value).toBe('hybrid_rrf')
  })

  it('offers no field this form cannot honestly vary', () => {
    // The reason has narrowed twice and the list has not moved. Of the four,
    // only the chunking strategy is still read by nothing: a corpus is cut at
    // load time and a run reads what the cut produced. The embedder, the
    // generator and the seed are all applied now, and each is a control that
    // has to be designed before it is written, so this form sends what the
    // gateway registers instead of drawing a lever nobody agreed on.
    render(wrap(<NewExperimentPage />))
    expect(screen.queryByLabelText(/seed/i)).not.toBeInTheDocument()
    expect(screen.queryByLabelText(/chunking strategy/i)).not.toBeInTheDocument()
    expect(screen.queryByLabelText(/embedder/i)).not.toBeInTheDocument()
    expect(screen.queryByLabelText(/^generator$/i)).not.toBeInTheDocument()
  })
})
