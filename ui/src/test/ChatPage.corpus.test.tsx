import { describe, it, expect, vi, beforeAll } from 'vitest'
import { screen, waitFor, fireEvent } from '@testing-library/react'
import ChatPage from '../pages/ChatPage'
import { stubRealmFetch, renderWithRealm } from './realmTestUtils'

// jsdom doesn't implement scrollIntoView — ChatPage calls it on every
// messages update, unrelated to what this file tests.
beforeAll(() => { Element.prototype.scrollIntoView = vi.fn() })

// Found live: "corpus can't be selected" — the corpus field used to be a
// plain `<input list="...">` that read as an empty text box, not an
// intentional dropdown, unlike the native `<select>`s right next to it
// (see the design notes "Chat routing via Realm"). Now a real `<select>` of
// known corpus_ids + a custom-ID escape hatch, mirroring
// CorpusPage.tsx#UploadCorpusIdField/NewExperimentPage.tsx#CorpusIdField.

vi.mock('../api/client', () => ({
  api: {
    models: vi.fn().mockResolvedValue([]),
    settings: { get: vi.fn().mockResolvedValue({ active_model: 'qwen3:8b', active_packs: [] }) },
    externalRags: { list: vi.fn().mockResolvedValue([]) },
    corpus: {
      list: vi.fn().mockResolvedValue([]),
      collections: vi.fn().mockResolvedValue([{ corpus_id: 'handbook_01', description: '' }]),
    },
    registry: vi.fn().mockResolvedValue({ pipeline: ['naive', 'hybrid_rrf', 'hybrid_weighted', 'graph'] }),
  },
}))

describe('ChatPage corpus selector', () => {
  it('renders a real <select> of known corpus_ids, not a free-text input', async () => {
    stubRealmFetch('acme')
    renderWithRealm(<ChatPage />, '/chat', 'acme')

    const corpusSelect = await screen.findByLabelText(/corpus/i) as HTMLSelectElement
    expect(corpusSelect.tagName).toBe('SELECT')
    await waitFor(() => expect([...corpusSelect.options].some(o => o.value === 'handbook_01')).toBe(true))
    // The list no longer offers 'default': the name was always in it and the
    // content behind the name was not, so whoever picked it queried nothing.
    expect([...corpusSelect.options].map(o => o.value)).not.toContain('default')
  })

  it('opens on a corpus that exists, not on a synthetic id', async () => {
    stubRealmFetch('acme')
    renderWithRealm(<ChatPage />, '/chat', 'acme')

    const corpusSelect = await screen.findByLabelText(/corpus/i) as HTMLSelectElement
    await waitFor(() => expect(corpusSelect.value).toBe('handbook_01'))
  })

  it('picking a known corpus_id from the select updates the value directly', async () => {
    stubRealmFetch('acme')
    renderWithRealm(<ChatPage />, '/chat', 'acme')

    const corpusSelect = await screen.findByLabelText(/corpus/i) as HTMLSelectElement
    await waitFor(() => expect([...corpusSelect.options].some(o => o.value === 'handbook_01')).toBe(true))
    fireEvent.change(corpusSelect, { target: { value: 'handbook_01' } })
    expect(corpusSelect.value).toBe('handbook_01')
  })

  it('the "+ custom ID..." option swaps to a free-text input for an unregistered corpus_id', async () => {
    stubRealmFetch('acme')
    renderWithRealm(<ChatPage />, '/chat', 'acme')

    const corpusSelect = await screen.findByLabelText(/corpus/i) as HTMLSelectElement
    fireEvent.change(corpusSelect, { target: { value: '__custom_corpus_id__' } })

    const corpusInput = await screen.findByLabelText(/corpus/i) as HTMLInputElement
    expect(corpusInput.tagName).toBe('INPUT')
    fireEvent.change(corpusInput, { target: { value: 'my-own-corpus' } })
    expect(corpusInput.value).toBe('my-own-corpus')
  })
})
