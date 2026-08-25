import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, waitFor, fireEvent } from '@testing-library/react'
import { MemoryRouter, Routes, Route } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import CorpusPage from '../pages/CorpusPage'
import { RealmProvider } from '../context/RealmContext'

/** With a real route: `CorpusPage` takes its tab from `useParams`, and without
 *  `/data/:tab` it opens on upload while the test looks for the fragment
 *  table. */
function renderContent() {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(
    <QueryClientProvider client={qc}>
      <MemoryRouter initialEntries={['/data/content?realm=demo']}>
        <RealmProvider>
          <Routes>
            <Route path="/data/:tab" element={<CorpusPage />} />
          </Routes>
        </RealmProvider>
      </MemoryRouter>
    </QueryClientProvider>,
  )
}

// A fragment's text can run to a thousand characters. Expanded in full, it
// turned a fifty-row table into a ribbon: neighbouring rows could not be
// compared and the total was invisible.

const LONG = 'Section 2.1. '.repeat(60)

// The graph pulls in sigma, and sigma needs WebGL, which jsdom lacks. The
// content tab has nothing to do with the graph, so it is stubbed out.
vi.mock('../components/GraphCommunityView', () => ({ default: () => null }))

vi.mock('../api/client', async () => {
  const actual = await vi.importActual<typeof import('../api/client')>('../api/client')
  return {
    ...actual,
    api: {
      ...actual.api,
      registry: () => Promise.resolve({ chunker: ['structure_aware'] }),
      corpus: {
        list: () => Promise.resolve([{ job_id: 'j1', corpus_id: 'demo', status: 'done', n_chunks: 2 }]),
        collections: () => Promise.resolve([{ id: 'c1', corpus_id: 'demo', deleted_at: null }]),
        chunks: () => Promise.resolve({
          items: [
            { chunk_id: 'aaaabbbbcccc', structural_path: 'root', text: LONG, length: LONG.length },
            { chunk_id: 'ddddeeeeffff', structural_path: 'sec. 1', text: 'Short', length: 8 },
          ],
        }),
      },
    },
  }
})

beforeEach(() => {
  localStorage.clear()
  vi.stubGlobal('fetch', vi.fn(async () => ({
    ok: true,
    json: async () => [{ id: 'demo', name: 'Demo', resources: [], key_metrics: [], created_at: '2026-01-01' }],
  })))
})

describe('the corpus fragment table', () => {
  it('keeps the text on one line and expands it on click', async () => {
    renderContent()

    const cells = () => [...document.querySelectorAll('.chunk-table .c-text')]
    await waitFor(() => expect(document.querySelectorAll('.chunk-table tbody tr')).toHaveLength(2))
    expect(cells().some(c => c.classList.contains('open'))).toBe(false)

    fireEvent.click(document.querySelector('.chunk-table tbody tr') as HTMLElement)
    expect(cells()[0].classList.contains('open')).toBe(true)
    // Exactly one is ever open: otherwise the table becomes a ribbon again.
    expect(cells()[1].classList.contains('open')).toBe(false)

    fireEvent.click(document.querySelector('.chunk-table tbody tr') as HTMLElement)
    expect(cells()[0].classList.contains('open')).toBe(false)
  })

  it('marks both a `root` path and an empty one as unstructured', async () => {
    // `root` is what the chunker marks a document with when it parsed no tree.
    renderContent()

    await waitFor(() => expect(document.querySelectorAll('.chunk-table tbody tr')).toHaveLength(2))
    expect(document.querySelectorAll('.q-noref')).toHaveLength(1)
  })
})
