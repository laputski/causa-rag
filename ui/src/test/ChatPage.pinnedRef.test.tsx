import { describe, it, expect, vi, beforeAll, beforeEach } from 'vitest'
import { screen, waitFor, fireEvent } from '@testing-library/react'
import ChatPage from '../pages/ChatPage'
import { renderWithRealm } from './realmTestUtils'

// A chunk injected by an active retrieval pin is
// flagged `pinned` in the /query response's source_refs; ChatPage renders
// a small badge for it, the same auditability the run detail page already
// gives via RunPage.tsx's RetrievalPanel.

beforeAll(() => { Element.prototype.scrollIntoView = vi.fn() })

vi.mock('../api/client', () => ({
  api: {
    models: vi.fn().mockResolvedValue([]),
    settings: { get: vi.fn().mockResolvedValue({ active_model: 'qwen3:8b', active_packs: [] }) },
    externalRags: { list: vi.fn().mockResolvedValue([]) },
    corpus: { list: vi.fn().mockResolvedValue([]), collections: vi.fn().mockResolvedValue([]) },
    registry: vi.fn().mockResolvedValue({ pipeline: ['naive'] }),
  },
}))

function stubRealmAndQueryFetch(realmId: string) {
  vi.stubGlobal('fetch', vi.fn(async (url: string) => {
    if (typeof url === 'string' && url.includes('/query')) {
      return {
        ok: true,
        json: async () => ({
          text: 'An answer with one ordinary and one pinned source.',
          source_refs: [
            { doc_id: 'd1', chunk_id: 'c1', structural_path: 'Article 1', score: 0.9, pinned: false },
            { doc_id: 'd2', chunk_id: 'c2', structural_path: 'Article 42', score: 0.8, pinned: true },
          ],
          computed_citations: [],
          metadata: {},
        }),
      }
    }
    // /api/realms
    return { ok: true, json: async () => [{ id: realmId, name: realmId, resources: [], created_at: '2026-01-01' }] }
  }))
}

beforeEach(() => {
  vi.clearAllMocks()
})

describe('ChatPage — pinned source badge', () => {
  it('shows a pin badge only on the source_ref flagged pinned=true', async () => {
    stubRealmAndQueryFetch('acme')
    renderWithRealm(<ChatPage />, '/chat', 'acme')

    const textarea = await screen.findByPlaceholderText(/.+/)
    fireEvent.change(textarea, { target: { value: 'What is the procedure?' } })
    fireEvent.click(screen.getByText((_, el) => el?.tagName === 'BUTTON' && el.classList.contains('chat-send')))

    await waitFor(() => expect(screen.getByText('Article 42')).toBeInTheDocument())

    const pinnedRef = screen.getByText('Article 42').closest('.chat-ref-badge') as HTMLElement
    const ordinaryRef = screen.getByText('Article 1').closest('.chat-ref-badge') as HTMLElement

    expect(pinnedRef.querySelector('svg')).not.toBeNull()
    expect(ordinaryRef.querySelector('svg')).toBeNull()
  })
})
