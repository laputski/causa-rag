import { describe, it, expect, vi, beforeEach } from 'vitest'
import { screen, waitFor, fireEvent } from '@testing-library/react'
import NewExperimentPage from '../pages/NewExperimentPage'
import { renderWithRealm } from './realmTestUtils'

// Which generator and which embedder a run uses.
//
// The form sent both as fixed names, so a generator or an embedder the
// registry held could be run only through the API. And the list of models was
// asked for with no Realm named, which the server answers with the gateway's
// local models whatever the Realm's own server is.
//
// Pinned: both come from the registry, the choice reaches the configuration,
// the models listed are those of the chosen generator's server in this Realm,
// and a corpus loaded with another embedder is said to be.

const REGISTRY = {
  pipeline: ['naive', 'hybrid_rrf'],
  generator: ['ollama', 'openai_compatible'],
  embedder: ['bge_m3', 'openai_compatible'],
  reranker: [], grounder: [], route_policy: [], scorer: [], mask_engine: [], refusal: [],
}

const modelsMock = vi.fn()
const createMock = vi.fn()

vi.mock('../api/client', () => ({
  api: {
    registry: () => Promise.resolve(REGISTRY),
    models: (completionOnly?: boolean, realmId?: string | null, provider?: string) => modelsMock(completionOnly, realmId, provider),
    datasets: { list: () => Promise.resolve([{ filename: 'set-a.jsonl', name: 'Set A', version: 'v1', count: 12, speed: 'fast' }]) },
    externalRags: { list: () => Promise.resolve([]) },
    corpus: {
      list: () => Promise.resolve([]),
      collections: () => Promise.resolve([
        { corpus_id: 'demo_corpus', backends: { qdrant: { embedder_id: 'bge_m3' } } },
        { corpus_id: 'external_docs', backends: { qdrant: { embedder_id: 'openai_compatible__embeds-3' } } },
      ]),
    },
    experiments: { get: () => Promise.resolve(null), create: (config: unknown, dataset: string, realm: string) => createMock(config, dataset, realm) },
  },
}))

const select = (id: string) => document.getElementById(id) as HTMLSelectElement

// The Realm's resources, as the realm list returns them.
let resources: Record<string, unknown>[] = []

async function open() {
  renderWithRealm(<NewExperimentPage />, '/new', 'demo')
  await waitFor(() => expect([...select('generator').options].map(o => o.value)).toContain('openai_compatible'))
}

beforeEach(() => {
  vi.clearAllMocks()
  localStorage.clear()
  modelsMock.mockImplementation(async (_c: boolean, _r: string, provider: string) =>
    provider === 'openai_compatible' ? [{ name: 'answers-large' }] : [{ name: 'qwen3:8b' }])
  createMock.mockResolvedValue({ run_id: 'r1', config_hash: 'h', status: 'started' })
  resources = []
  vi.stubGlobal('fetch', vi.fn(async () => ({
    ok: true,
    json: async () => [{ id: 'demo', name: 'Demo', resources, key_metrics: [], created_at: '2026-01-01' }],
  })))
})

describe('the new-run form — generator and embedder', () => {
  it('offers the generators and the embedders the registry holds, the local ones chosen', async () => {
    await open()
    expect([...select('generator').options].map(o => o.value).filter(Boolean)).toEqual(['ollama', 'openai_compatible'])
    expect([...select('embedder').options].map(o => o.value).filter(Boolean)).toEqual(['bge_m3', 'openai_compatible'])
    expect(select('generator').value).toBe('ollama')
    expect(select('embedder').value).toBe('bge_m3')
  })

  it('lists the models of the chosen generator in this Realm', async () => {
    await open()
    await waitFor(() => expect(modelsMock).toHaveBeenCalledWith(true, 'demo', 'ollama'))
    fireEvent.change(select('generator'), { target: { value: 'openai_compatible' } })
    await waitFor(() => expect(modelsMock).toHaveBeenCalledWith(true, 'demo', 'openai_compatible'))
    await waitFor(() => expect([...select('model').options].map(o => o.value)).toContain('answers-large'))
    expect([...select('model').options].map(o => o.value)).not.toContain('qwen3:8b')
  })

  it('drops the chosen model when another generator is chosen', async () => {
    await open()
    await waitFor(() => expect([...select('model').options].map(o => o.value)).toContain('qwen3:8b'))
    fireEvent.change(select('model'), { target: { value: 'qwen3:8b' } })
    expect(select('model').value).toBe('qwen3:8b')
    fireEvent.change(select('generator'), { target: { value: 'openai_compatible' } })
    // What is sent decides it, and not what the select shows: a select whose
    // options no longer hold the value shows it empty while the form still
    // carries it, and the run would then ask the other server for a model
    // it never had.
    fireEvent.change(select('corpus_id'), { target: { value: 'demo_corpus' } })
    fireEvent.click(document.querySelector('button[type="submit"]') as HTMLButtonElement)
    await waitFor(() => expect(createMock).toHaveBeenCalled())
    const config = createMock.mock.calls[0][0] as { params?: { model?: string } }
    expect(config.params?.model).toBeUndefined()
  })

  it('sends the chosen generator and embedder in the configuration', async () => {
    await open()
    fireEvent.change(select('generator'), { target: { value: 'openai_compatible' } })
    fireEvent.change(select('embedder'), { target: { value: 'openai_compatible' } })
    fireEvent.change(select('corpus_id'), { target: { value: 'external_docs' } })
    fireEvent.click(document.querySelector('button[type="submit"]') as HTMLButtonElement)
    await waitFor(() => expect(createMock).toHaveBeenCalled())
    const config = createMock.mock.calls[0][0] as { generator: { component_id: string }; embedder: { component_id: string } }
    expect(config.generator.component_id).toBe('openai_compatible')
    expect(config.embedder.component_id).toBe('openai_compatible')
    expect(createMock.mock.calls[0][2]).toBe('demo')
  })

  it('sends the local generator and embedder when neither was touched', async () => {
    await open()
    fireEvent.change(select('corpus_id'), { target: { value: 'demo_corpus' } })
    fireEvent.click(document.querySelector('button[type="submit"]') as HTMLButtonElement)
    await waitFor(() => expect(createMock).toHaveBeenCalled())
    const config = createMock.mock.calls[0][0] as { generator: { component_id: string }; embedder: { component_id: string } }
    expect([config.generator.component_id, config.embedder.component_id]).toEqual(['ollama', 'bge_m3'])
  })

  it('says so when the corpus was loaded with another embedder than the one chosen', async () => {
    await open()
    fireEvent.change(select('corpus_id'), { target: { value: 'external_docs' } })
    const note = await screen.findByTestId('embedder-mismatch')
    expect(note).toHaveTextContent('loaded with the embedder openai_compatible')
    fireEvent.change(select('embedder'), { target: { value: 'openai_compatible' } })
    await waitFor(() => expect(screen.queryByTestId('embedder-mismatch')).toBeNull())
  })

  it('says nothing of the kind when the embedder is the one the corpus was loaded with', async () => {
    await open()
    fireEvent.change(select('corpus_id'), { target: { value: 'demo_corpus' } })
    await waitFor(() => expect(select('corpus_id').value).toBe('demo_corpus'))
    expect(screen.queryByTestId('embedder-mismatch')).toBeNull()
  })

  it('says so when the Realm has no server for the generator chosen', async () => {
    await open()
    expect(screen.queryByTestId('generator-no-server')).toBeNull()
    fireEvent.change(select('generator'), { target: { value: 'openai_compatible' } })
    expect(await screen.findByTestId('generator-no-server')).toHaveTextContent('the run gets no answers')
  })

  it('says nothing of the kind when the Realm declares that server', async () => {
    resources = [{ type: 'openai_compatible', base_url: 'https://models.example/v1', model: 'answers-large', embedding_model: 'embeds-3' }]
    await open()
    fireEvent.change(select('generator'), { target: { value: 'openai_compatible' } })
    fireEvent.change(select('embedder'), { target: { value: 'openai_compatible' } })
    await waitFor(() => expect(select('embedder').value).toBe('openai_compatible'))
    expect(screen.queryByTestId('generator-no-server')).toBeNull()
    expect(screen.queryByTestId('embedder-no-model')).toBeNull()
  })

  it('says so when the Realm server names no model that embeds', async () => {
    resources = [{ type: 'openai_compatible', base_url: 'https://models.example/v1', model: 'answers-large', embedding_model: '' }]
    await open()
    fireEvent.change(select('embedder'), { target: { value: 'openai_compatible' } })
    expect(await screen.findByTestId('embedder-no-model')).toHaveTextContent('names no model that embeds')
  })
})
