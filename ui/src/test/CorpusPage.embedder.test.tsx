import { describe, it, expect, vi, beforeEach } from 'vitest'
import { act, screen, waitFor, fireEvent } from '@testing-library/react'
import CorpusPage from '../pages/CorpusPage'
import { renderWithRealm } from './realmTestUtils'

// Which model fills the dense index of a corpus being loaded.
//
// Pinned: the Realm's own server is offered only when the Realm names an
// embedding model for it, the local model stays the default, and the load is
// told which one was chosen. A choice the server never received would load
// the corpus with the local model under a form that said otherwise.

vi.mock('../components/GraphCommunityView', () => ({ default: () => null }))

vi.mock('../api/client', () => ({
  api: {
    corpus: { list: () => Promise.resolve([]), collections: () => Promise.resolve([]) },
    registry: () => Promise.resolve({ chunking_strategy: ['structure_aware'] }),
    health: () => Promise.resolve({ status: 'ok', components: {} }),
  },
}))

let resources: Record<string, unknown>[]
// The progress sockets the page opened, so a test can speak for the loader.
const sockets: { onmessage: ((e: { data: string }) => void) | null }[] = []
const posted: FormData[] = []

beforeEach(() => {
  posted.length = 0
  resources = [{ type: 'openai_compatible', base_url: 'https://models.example/v1', model: 'answers', embedding_model: 'embeds-3' }]
  sockets.length = 0
  // jsdom lays nothing out, so an element has no scrollTo; the log scrolls itself.
  Element.prototype.scrollTo = () => {}
  vi.stubGlobal('WebSocket', class {
    onmessage: ((e: { data: string }) => void) | null = null
    onerror: (() => void) | null = null
    constructor() { sockets.push(this) }
    close() {}
  })
  vi.stubGlobal('fetch', vi.fn(async (_url: string, init?: RequestInit) => {
    if (init?.method === 'POST') { posted.push(init.body as FormData); return { ok: true, json: async () => ({ job_id: 'j1' }) } }
    return { ok: true, json: async () => [{ id: 'acme', name: 'acme', resources, created_at: '2026-01-01' }] }
  }))
})

const show = () => renderWithRealm(<CorpusPage />, '/data/upload', 'acme')

function pickFile() {
  const input = document.querySelector('input[type="file"]') as HTMLInputElement
  fireEvent.change(input, { target: { files: [new File(['text'], 'a.txt', { type: 'text/plain' })] } })
}

describe('CorpusPage — the embedder of a load', () => {
  it('offers the Realm server by the model the Realm names, with the local one as default', async () => {
    show()
    const select = await screen.findByLabelText('Embedder') as HTMLSelectElement
    expect(select.value).toBe('bge_m3')
    expect([...select.options].map(o => o.textContent)).toEqual(['Local (BGE-M3)', "The Realm's server: embeds-3"])
  })

  it('offers no choice when the Realm names no embedding model', async () => {
    resources = [{ type: 'openai_compatible', base_url: 'https://models.example/v1', model: 'answers', embedding_model: '' }]
    show()
    await screen.findByText('Chunking strategy')
    expect(screen.queryByLabelText('Embedder')).toBeNull()
  })

  it('says the texts will be sent away once the Realm server is chosen', async () => {
    show()
    const select = await screen.findByLabelText('Embedder')
    expect(screen.queryByText(/will be sent to the Realm's server/)).toBeNull()
    fireEvent.change(select, { target: { value: 'openai_compatible' } })
    expect(screen.getByText(/will be sent to the Realm's server/)).toBeInTheDocument()
  })

  it('tells the load which embedder was chosen', async () => {
    show()
    fireEvent.change(await screen.findByLabelText('Embedder'), { target: { value: 'openai_compatible' } })
    pickFile()
    fireEvent.click(await screen.findByRole('button', { name: /Start|Upload|Ingest|Load/i }))
    await waitFor(() => expect(posted).toHaveLength(1))
    expect(posted[0].get('embedder')).toBe('openai_compatible')
  })

  it('loads with the local model when nothing was chosen', async () => {
    show()
    await screen.findByLabelText('Embedder')
    pickFile()
    fireEvent.click(await screen.findByRole('button', { name: /Start|Upload|Ingest|Load/i }))
    await waitFor(() => expect(posted).toHaveLength(1))
    expect(posted[0].get('embedder')).toBe('bge_m3')
  })

  it('says when the lexical index was not written, and why', async () => {
    // Found live: the store refused to create an index, the loader went on
    // with the vector half, and the page said only "Done".
    show()
    await screen.findByLabelText('Embedder')
    pickFile()
    fireEvent.click(await screen.findByRole('button', { name: /Start|Upload|Ingest|Load/i }))
    await waitFor(() => expect(sockets).toHaveLength(1))
    const say = (event: unknown) => act(() => { sockets[0].onmessage?.({ data: JSON.stringify(event) }) })
    say({ type: 'warning', code: 'lexical_index_not_written', message: 'index_create_block_exception' })
    say({ type: 'done', n_chunks: 12, hit_ratio: 0 })
    const line = await screen.findByText(/The lexical index \(BM25\) was not created/)
    expect(line).toHaveTextContent('Reason: index_create_block_exception')
    expect(line).toHaveClass('log-warning')
  })
})
