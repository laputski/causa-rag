import { describe, it, expect, vi, beforeEach } from 'vitest'
import { screen, waitFor, fireEvent } from '@testing-library/react'
import PromptsPage from '../pages/PromptsPage'
import { stubRealmFetch, renderWithRealm } from './realmTestUtils'

// Prompts and presets are one screen with two tabs. Each keeps its own URL,
// because links to them have already spread into other people's notes, and both
// are laid out identically, because they do the same thing: pick from a list on
// the left, read and edit on the right.

const promptsList = vi.fn()
const presetsList = vi.fn()

vi.mock('../api/client', async () => {
  const actual = await vi.importActual<typeof import('../api/client')>('../api/client')
  return {
    ...actual,
    api: {
      ...actual.api,
      models: () => Promise.resolve(['qwen3:8b']),
      corpus: { ...actual.api.corpus, collections: () => Promise.resolve([]) },
      prompts: {
        ...actual.api.prompts,
        list: (...a: unknown[]) => promptsList(...a),
      },
      generationPresets: {
        ...actual.api.generationPresets,
        list: (...a: unknown[]) => presetsList(...a),
      },
    },
  }
})

const PROMPTS = [
  { id: 'p1', name: 'Strict', description: 'no guessing', template: 'Answer from the fragments: {context}', is_active: true, version: 3, created_at: '2026-01-02T00:00:00Z' },
  { id: 'p2', name: 'Loose', description: '', template: 'Answer: {context}', is_active: false, version: 1, created_at: '2026-01-01T00:00:00Z' },
]

const PRESETS = [
  { id: 'g1', name: 'Basic', description: 'one fragment at a time', template: '{chunk_text}', realm_id: 'acme', created_at: '2026-01-01T00:00:00Z' },
]

beforeEach(() => {
  vi.clearAllMocks()
  stubRealmFetch('acme')
  promptsList.mockResolvedValue(PROMPTS)
  presetsList.mockResolvedValue(PRESETS)
})

describe('Prompts and presets as two tabs of one screen', () => {
  it('opens on prompts, with the second tab unselected', async () => {
    const { container } = renderWithRealm(<PromptsPage />, '/prompts', 'acme')
    await screen.findAllByText('Strict').then(els => expect(els.length).toBeGreaterThan(0))

    const tabs = [...container.querySelectorAll('[role="tab"]')]
    expect(tabs.length).toBe(2)
    expect(tabs[0].getAttribute('aria-selected')).toBe('true')
    expect(tabs[1].getAttribute('aria-selected')).toBe('false')
  })

  it('clicking the second tab changes both the content and the selection', async () => {
    const { container } = renderWithRealm(<PromptsPage />, '/prompts', 'acme')
    await screen.findAllByText('Strict').then(els => expect(els.length).toBeGreaterThan(0))

    fireEvent.click(container.querySelectorAll('[role="tab"]')[1])

    // The content changed: a list of presets rather than prompts.
    await waitFor(() => expect(screen.getAllByText('Basic').length).toBeGreaterThan(0))
    expect(screen.queryAllByText('Strict').length).toBe(0)
    const tabs = [...container.querySelectorAll('[role="tab"]')]
    expect(tabs[1].getAttribute('aria-selected')).toBe('true')
    // And the tab fetched its own data rather than reusing the other's response.
    expect(presetsList).toHaveBeenCalled()
  })

  it('the presets tab opened by URL is the one selected', async () => {
    // Each tab has its own URL: links to them already sit in other people's notes.
    const { container } = renderWithRealm(<PromptsPage />, '/data/presets', 'acme')
    await screen.findAllByText('Basic').then(els => expect(els.length).toBeGreaterThan(0))

    const tabs = [...container.querySelectorAll('[role="tab"]')]
    expect(tabs[1].getAttribute('aria-selected')).toBe('true')
    expect(tabs[0].getAttribute('aria-selected')).toBe('false')
  })

  it('exactly one prompt is marked active', async () => {
    promptsList.mockResolvedValue([
      ...PROMPTS,
      { id: 'p3', name: 'Third', description: '', template: '{context}', is_active: false, version: 1, created_at: '2026-01-03T00:00:00Z' },
    ])
    const { container } = renderWithRealm(<PromptsPage />, '/prompts', 'acme')
    await screen.findAllByText('Strict').then(els => expect(els.length).toBeGreaterThan(0))

    // One prompt is in force per realm; two marked would mean a reader cannot
    // tell which one is answering.
    await waitFor(() => {
      const marks = container.querySelectorAll('.pick-list .badge-success')
      expect(marks.length).toBe(1)
    })
  })

  it('both tabs share one layout', async () => {
    const classesOf = (el: HTMLElement) =>
      new Set([...el.querySelectorAll('[class]')].flatMap(n => [...n.classList]))

    const prompts = renderWithRealm(<PromptsPage />, '/prompts', 'acme')
    await screen.findAllByText('Strict').then(els => expect(els.length).toBeGreaterThan(0))
    const promptClasses = classesOf(prompts.container)
    prompts.unmount()

    const presets = renderWithRealm(<PromptsPage />, '/data/presets', 'acme')
    await screen.findAllByText('Basic').then(els => expect(els.length).toBeGreaterThan(0))
    const presetClasses = classesOf(presets.container)

    // Two screens built the same way had drifted apart in layout, and that read
    // as two different mechanisms. One skeleton: list on the left, detail on the
    // right, a rule between them.
    for (const shared of ['page-flush', 'split', 'split-aside', 'pick-list', 'pick-row']) {
      expect(promptClasses.has(shared), `prompts lack .${shared}`).toBe(true)
      expect(presetClasses.has(shared), `presets lack .${shared}`).toBe(true)
    }
  })
})
