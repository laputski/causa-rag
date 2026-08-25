import { describe, it, expect, vi, beforeEach } from 'vitest'
import { screen, waitFor, fireEvent } from '@testing-library/react'
import PromptsPage from '../pages/PromptsPage'
import { stubRealmFetch, renderWithRealm } from './realmTestUtils'

// Regression guard: prompts had zero realm_id awareness at all (same bug
// shape as datasets) — see the design notes "Prompts scoping".
const listMock = vi.fn().mockResolvedValue([])
const generateMock = vi.fn()
const modelsMock = vi.fn().mockResolvedValue([])
const collectionsMock = vi.fn().mockResolvedValue([])

vi.mock('../api/client', () => ({
  api: {
    prompts: {
      list: (realmId?: string | null) => listMock(realmId),
      get: vi.fn(),
      create: vi.fn(),
      activate: vi.fn(),
      delete: vi.fn(),
      generate: (body: unknown) => generateMock(body),
    },
    models: (completionOnly?: boolean) => modelsMock(completionOnly),
    corpus: {
      collections: (realmId?: string | null) => collectionsMock(realmId),
    },
  },
}))

describe('PromptsPage realm scoping', () => {
  it('passes activeRealmId into api.prompts.list', async () => {
    stubRealmFetch('acme')
    renderWithRealm(<PromptsPage />, '/prompts', 'acme')
    await waitFor(() => expect(listMock).toHaveBeenCalledWith('acme'))
  })
})

// ── AI-drafted prompt (found live: prompts were only ever written by hand —
// no way to get a first draft tailored to a specific corpus/model) ─────────

describe('PromptsPage AI draft', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    listMock.mockResolvedValue([])
    modelsMock.mockResolvedValue([{ name: 'qwen3:8b' }])
    collectionsMock.mockResolvedValue([{ corpus_id: 'handbook_01', description: '' }])
  })

  it('pre-fills name/description/template from the generated draft', async () => {
    generateMock.mockResolvedValue({
      name: 'Technical documentation',
      description: 'Answers from the manuals',
      template: 'Context:\n{context}\n\nQuestion: {query}\nAnswer:',
    })
    stubRealmFetch('acme')
    renderWithRealm(<PromptsPage />, '/prompts', 'acme')

    fireEvent.click(await screen.findByRole('button', { name: /\+ New/i }))
    fireEvent.click(await screen.findByRole('button', { name: /Fill in with AI/i }))

    const modelSelect = await screen.findByLabelText(/Model/i) as HTMLSelectElement
    await waitFor(() => expect([...modelSelect.options].some(o => o.value === 'qwen3:8b')).toBe(true))
    fireEvent.change(modelSelect, { target: { value: 'qwen3:8b' } })
    const corpusSelect = await screen.findByLabelText(/Corpus/i) as HTMLSelectElement
    await waitFor(() => expect([...corpusSelect.options].some(o => o.value === 'handbook_01')).toBe(true))
    fireEvent.change(corpusSelect, { target: { value: 'handbook_01' } })

    fireEvent.click(screen.getByRole('button', { name: /Generate/i }))

    await waitFor(() => expect(generateMock).toHaveBeenCalledWith({
      realm_id: 'acme', corpus_id: 'handbook_01', model: 'qwen3:8b',
    }))
    const nameInput = await screen.findByDisplayValue('Technical documentation')
    expect(nameInput).toBeInTheDocument()
    expect(screen.getByDisplayValue('Answers from the manuals')).toBeInTheDocument()
    expect(screen.getByDisplayValue(/Context:/)).toBeInTheDocument()
  })

  it('disables Generate until both model and corpus are chosen', async () => {
    stubRealmFetch('acme')
    renderWithRealm(<PromptsPage />, '/prompts', 'acme')

    fireEvent.click(await screen.findByRole('button', { name: /\+ New/i }))
    fireEvent.click(await screen.findByRole('button', { name: /Fill in with AI/i }))

    const generateButton = await screen.findByRole('button', { name: /Generate/i })
    expect(generateButton).toBeDisabled()
  })
})
