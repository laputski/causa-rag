import { describe, it, expect, vi } from 'vitest'
import { waitFor } from '@testing-library/react'
import RealmResourcesPage from '../pages/RealmResourcesPage'
import { stubRealmFetch, renderWithRealm } from './realmTestUtils'

// The embedder mismatch/hint badge on the RAG list must be
// visible from the LAST stored test() result, without requiring a fresh
// "Test" click (see the design notes).

const ragWithMismatch = {
  id: 'rag1', name: 'demo-rag', url: 'http://localhost:8002/platform/query',
  description: '', headers: {}, realm_id: 'demo', default_corpus_id: 'handbook',
  created_at: '2026-01-01',
  capabilities: {
    supports_trace: true, supports_retrieval_only: false, retrieve_endpoint: null,
    max_top_k: null, source_ref_granularity: 'chunk', supported_params: [],
    reported_embedders: ['paraphrase-multilingual-mpnet-base-v2'],
    embedder_mismatch_warning: 'The RAG reports embedder(s) [\'paraphrase-multilingual-mpnet-base-v2\'] while corpus \'handbook\' is registered with [\'bge_m3\']',
    embedder_hint: null,
  },
}

const ragWithHint = {
  id: 'rag2', name: 'other-rag', url: 'http://localhost:8003/platform/query',
  description: '', headers: {}, realm_id: 'demo', default_corpus_id: 'handbook',
  created_at: '2026-01-01',
  capabilities: {
    supports_trace: true, supports_retrieval_only: false, retrieve_endpoint: null,
    max_top_k: null, source_ref_granularity: 'chunk', supported_params: [],
    reported_embedders: null, embedder_mismatch_warning: null,
    embedder_hint: 'This RAG does not report which embedder it used',
  },
}

vi.mock('../api/client', () => ({
  api: {
    externalRags: { list: vi.fn(async () => [ragWithMismatch, ragWithHint]) },
    corpus: { collections: vi.fn(async () => []) },
    datasets: { list: vi.fn(async () => []) },
  },
}))

describe('RealmResourcesPage embedder badge', () => {
  it('shows the mismatch warning from stored capabilities without a Test click', async () => {
    stubRealmFetch('demo')
    const { getByText } = renderWithRealm(<RealmResourcesPage />, '/settings/resources', 'demo')

    await waitFor(() => expect(getByText(/embedder mismatch/)).toBeTruthy())
  })

  it('shows the soft hint when the RAG reported no embedder at all', async () => {
    stubRealmFetch('demo')
    const { getByText } = renderWithRealm(<RealmResourcesPage />, '/settings/resources', 'demo')

    await waitFor(() => expect(getByText(/embedder not declared/)).toBeTruthy())
  })
})
