import { describe, it, expect, vi, beforeEach } from 'vitest'
import { waitFor, fireEvent } from '@testing-library/react'
import NewExperimentPage from '../pages/NewExperimentPage'
import { renderWithRealm } from './realmTestUtils'

// The form is documented as building itself from the registry, and it did for
// the list of pipelines and for nothing else. The retriever it recorded came
// from `pipeline_id === 'graph' ? 'graph_hybrid' : 'qdrant_dense'` and the
// merge from `pipeline_id === 'hybrid_weighted' ? 'weighted' : 'rrf'`, so an
// architecture outside those names was recorded as a dense retriever fusing by
// rank, and the merge is applied by the runner and not merely recorded.

const REGISTRY = {
  pipeline: ['naive', 'hybrid_weighted', 'not_yet_invented'],
  reranker: [], grounder: [], route_policy: [], scorer: [], mask_engine: [], refusal: [],
}

const PIPELINES = {
  naive: { retriever: 'qdrant_dense', merge_strategy: '' },
  hybrid_weighted: { retriever: 'hybrid', merge_strategy: 'weighted' },
  not_yet_invented: { retriever: 'something_new', merge_strategy: 'score_normalization' },
}

const createMock = vi.fn(async (..._args: unknown[]) => ({ run_id: 'r1' }))

vi.mock('../api/client', () => ({
  api: {
    registry: () => Promise.resolve(REGISTRY),
    pipelines: () => Promise.resolve(PIPELINES),
    models: () => Promise.resolve([]),
    datasets: { list: () => Promise.resolve([{ filename: 'set-a.json', name: 'Set A', version: 'v1', count: 12, speed: 'fast' }]) },
    externalRags: { list: () => Promise.resolve([]) },
    corpus: {
      list: () => Promise.resolve([]),
      collections: () => Promise.resolve([{ corpus_id: 'demo_corpus', chunks: 120 }]),
    },
    experiments: { get: () => Promise.resolve(null), create: (...args: unknown[]) => createMock(...args) },
  },
}))

beforeEach(() => {
  localStorage.clear()
  createMock.mockClear()
  vi.stubGlobal('fetch', vi.fn(async () => ({
    ok: true,
    json: async () => [{ id: 'demo', name: 'Demo', resources: [], key_metrics: [], created_at: '2026-01-01' }],
  })))
})

async function submitWith(pipelineId: string) {
  const { container } = renderWithRealm(<NewExperimentPage />, '/new', 'demo')
  // Wait for the registry to arrive: until it does, no select offers a
  // pipeline at all and the assertion below would fail for that reason.
  await waitFor(() => {
    const offered = [...container.querySelectorAll('select')].some(
      s => [...s.options].some(o => o.value === pipelineId),
    )
    expect(offered).toBe(true)
  })

  const retrieval = [...container.querySelectorAll('select')].find(
    s => [...s.options].some(o => o.value === pipelineId),
  )
  expect(retrieval, `no select offers ${pipelineId}`).toBeTruthy()
  fireEvent.change(retrieval!, { target: { value: pipelineId } })

  // The start button stays disabled until a corpus is chosen, which is
  // deliberate: no corpus is auto-picked.
  const corpus = [...container.querySelectorAll('select')].find(
    s => [...s.options].some(o => o.value === 'demo_corpus'),
  )
  expect(corpus, 'no select offers a corpus').toBeTruthy()
  fireEvent.change(corpus!, { target: { value: 'demo_corpus' } })

  const start = [...container.querySelectorAll('button')].find(b => !b.disabled && b.className.includes('primary'))
  expect(start, 'no enabled start button').toBeTruthy()
  fireEvent.click(start!)
  await waitFor(() => expect(createMock).toHaveBeenCalled())
  return createMock.mock.calls[0]![0] as Record<string, unknown>
}

describe('the form records what a pipeline is made of', () => {
  it('names the retriever the server holds under that pipeline', async () => {
    const cfg = await submitWith('hybrid_weighted')
    expect((cfg.retrievers as { component_id: string }[])[0].component_id).toBe('hybrid')
  })

  it('records the merge the pipeline was built with', async () => {
    const cfg = await submitWith('hybrid_weighted')
    expect(cfg.merge_strategy).toBe('weighted')
  })

  it('records no merge for a retriever that merges nothing', async () => {
    // Not 'rrf'. A dense pipeline fuses no sources, and the runner applies
    // this field, so recording a fusion it never did is not bookkeeping.
    const cfg = await submitWith('naive')
    expect(cfg.merge_strategy).toBeUndefined()
  })

  it('describes an architecture this file has never heard of', async () => {
    const cfg = await submitWith('not_yet_invented')
    expect((cfg.retrievers as { component_id: string }[])[0].component_id).toBe('something_new')
    expect(cfg.merge_strategy).toBe('score_normalization')
  })
})
