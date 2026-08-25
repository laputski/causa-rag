import { describe, it, expect, vi, beforeEach } from 'vitest'
import { screen, waitFor, fireEvent } from '@testing-library/react'
import RealmResourcesPage from '../pages/RealmResourcesPage'
import { stubRealmFetch, renderWithRealm } from './realmTestUtils'

// The section showed the same thing twice: a table row expanded into a card,
// and the same cards also sat in a list below, with a comment on the first one
// saying outright why the second should not exist. The add form meanwhile stood
// open for everybody, always, collecting two more disclosures beneath it.

const RAGS = [
  {
    id: 'r1', name: 'ms-rag', url: 'http://localhost:8002/platform/query',
    description: '', realm_id: 'acme',
    capabilities: {
      supports_trace: true, supports_retrieval_only: false, retrieve_endpoint: null,
      source_ref_granularity: 'chunk', supported_params: [],
      embedder_mismatch_warning: 'RAG says e5-large, the realm indexes with bge-m3',
      embedder_hint: null,
    },
  },
]

vi.mock('../api/client', async () => {
  const actual = await vi.importActual<typeof import('../api/client')>('../api/client')
  return {
    ...actual,
    api: {
      ...actual.api,
      realms: { ...actual.api.realms, get: () => Promise.resolve({ id: 'acme', name: 'Acme', resources: [] }) },
      externalRags: { ...actual.api.externalRags, list: () => Promise.resolve(RAGS) },
      panels: () => Promise.resolve({}),
      panelsStatus: () => Promise.resolve([]),
    },
  }
})

beforeEach(() => { vi.clearAllMocks(); stubRealmFetch('acme') })

const render = () => renderWithRealm(<RealmResourcesPage />, '/settings/resources', 'acme')

describe('RAG endpoints', () => {
  it('an endpoint card stays hidden until its row is expanded', async () => {
    const { container } = render()
    await screen.findByText('ms-rag')
    // Not one card: the table has already listed every endpoint, and
    // repeating that list as expanded blocks shows one thing twice.
    expect(container.querySelectorAll('.rag-card').length).toBe(0)
  })

  it('clicking a row expands exactly one card', async () => {
    const { container } = render()
    fireEvent.click(await screen.findByText('ms-rag'))
    await waitFor(() => expect(container.querySelectorAll('.rag-card').length).toBe(1))
  })

  it('an embedder mismatch shows on the row, and not in the card alone', async () => {
    // A warning you have to click to find is a warning nobody sees. Under a
    // mismatch the retrieval metrics are comparing different things.
    const { container } = render()
    await screen.findByText('ms-rag')
    const row = container.querySelector('tbody tr')!
    expect(row.querySelector('.badge-warn')).toBeTruthy()
  })

  it('the add form appears on a button press instead of standing open', async () => {
    const { container } = render()
    await screen.findByText('ms-rag')
    expect(container.querySelector('.rag-add')).toBeNull()

    const add = [...container.querySelectorAll('.section-rule button')]
      .find(b => /RAG/i.test(b.textContent ?? ''))!
    fireEvent.click(add)
    await waitFor(() => expect(container.querySelector('.rag-add')).toBeTruthy())
  })

  it('the reference sections inside the form start collapsed', async () => {
    const { container } = render()
    await screen.findByText('ms-rag')
    fireEvent.click([...container.querySelectorAll('.section-rule button')]
      .find(b => /RAG/i.test(b.textContent ?? ''))!)
    await waitFor(() => expect(container.querySelector('.rag-add')).toBeTruthy())

    // The contract requirements and the format mapping are needed by a
    // minority and once; open, they took more space than the form itself.
    const refs = [...container.querySelectorAll('.rag-add details')] as HTMLDetailsElement[]
    expect(refs.length).toBeGreaterThanOrEqual(2)
    expect(refs.every(d => !d.open)).toBe(true)
  })
})
