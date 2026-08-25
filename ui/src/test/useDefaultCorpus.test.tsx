import { describe, it, expect, vi, beforeEach } from 'vitest'
import { renderHook, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import type { ReactNode } from 'react'
import { useDefaultCorpus } from '../hooks/useDefaultCorpus'

// Screens used to open on `default`, an id always present in the list and only
// sometimes holding anything, so the first thing a reader saw was "no data" with
// a populated corpus on the line below.

const list = vi.fn()
const collections = vi.fn()
vi.mock('../api/client', () => ({
  api: {
    corpus: {
      list: (...a: unknown[]) => list(...a),
      collections: (...a: unknown[]) => collections(...a),
    },
  },
}))

function wrapper({ children }: { children: ReactNode }) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return <QueryClientProvider client={qc}>{children}</QueryClientProvider>
}

const REGISTRY = [
  { id: 'r1', corpus_id: 'default', deleted_at: null },
  { id: 'r2', corpus_id: 'demo_corpus', deleted_at: null },
]

beforeEach(() => { list.mockReset(); collections.mockReset() })

// @lat: [[design-language#Экраны выбора: суждения, реальные запросы, ресурсы, граница#Корпус по умолчанию]]
describe('the corpus a screen opens on', () => {
  it('takes the corpus of the last ingest that indexed something', async () => {
    list.mockResolvedValue([
      { job_id: 'j2', corpus_id: 'demo_corpus', status: 'done', n_chunks: 16 },
      { job_id: 'j1', corpus_id: 'other', status: 'done', n_chunks: 4 },
    ])
    collections.mockResolvedValue(REGISTRY)
    const onPick = vi.fn()
    renderHook(() => useDefaultCorpus('demo', '', onPick), { wrapper })

    await waitFor(() => expect(onPick).toHaveBeenCalledWith('demo_corpus'))
  })

  it('skips an ingest that indexed nothing', async () => {
    // A job marked done with zero chunks is a failed ingest, and opening on its
    // corpus means showing an empty screen.
    list.mockResolvedValue([{ job_id: 'j1', corpus_id: 'empty_corpus', status: 'done', n_chunks: 0 }])
    collections.mockResolvedValue(REGISTRY)
    const onPick = vi.fn()
    renderHook(() => useDefaultCorpus('demo', '', onPick), { wrapper })

    // The fallback is the first live entry in the registry; the screen never
    // lands on a failed ingest's corpus.
    await waitFor(() => expect(onPick).toHaveBeenCalledWith('default'))
    expect(onPick).not.toHaveBeenCalledWith('empty_corpus')
  })

  it('leaves a choice the reader already made alone, `default` included', async () => {
    list.mockResolvedValue([{ job_id: 'j1', corpus_id: 'demo_corpus', status: 'done', n_chunks: 16 }])
    collections.mockResolvedValue(REGISTRY)
    const onPick = vi.fn()
    renderHook(() => useDefaultCorpus('demo', 'default', onPick), { wrapper })

    await waitFor(() => expect(list).toHaveBeenCalled())
    expect(onPick).not.toHaveBeenCalled()
  })
})
