import { describe, it, expect, vi, beforeEach } from 'vitest'
import { screen, waitFor } from '@testing-library/react'
import NewExperimentPage from '../pages/NewExperimentPage'
import { stubRealmFetch, renderWithRealm } from './realmTestUtils'

// Found live: the Corpus field on this page defaulted to a hardcoded
// 'handbook' (a demo-only corpus) regardless of which Realm was
// active, and never corrected itself when switching to a Realm that never
// had it, unlike the `dataset` field, which already had a
// realm-switch reset effect. Regression guard for both the hardcoded
// default and the missing realm-scoping of the shown options.
const collectionsMock = vi.fn().mockResolvedValue([])

vi.mock('../api/client', () => ({
  api: {
    registry: vi.fn().mockResolvedValue({
      pipeline: ['naive', 'hybrid_rrf', 'hybrid_weighted', 'graph'],
      reranker: [], grounder: [], route_policy: [], scorer: [], mask_engine: [], refusal: [],
    }),
    datasets: { list: vi.fn().mockResolvedValue([]) },
    corpus: {
      list: vi.fn().mockResolvedValue([]),
      collections: (realmId?: string | null) => collectionsMock(realmId),
    },
    externalRags: { list: vi.fn().mockResolvedValue([]) },
    experiments: { create: vi.fn() },
  },
}))

beforeEach(() => {
  vi.clearAllMocks()
  collectionsMock.mockResolvedValue([])
})

describe('NewExperimentPage corpus field realm scoping', () => {
  it('passes activeRealmId into api.corpus.collections', async () => {
    stubRealmFetch('acme')
    renderWithRealm(<NewExperimentPage />, '/new', 'acme')
    await waitFor(() => expect(collectionsMock).toHaveBeenCalledWith('acme'))
  })

  it('only offers the active Realm\'s own corpus_ids, never a hardcoded handbook', async () => {
    collectionsMock.mockResolvedValue([
      { realm_id: 'acme', corpus_id: 'handbook_01', storage_type: 'dense_only', backends: {}, owner: 'platform', description: '' },
    ])
    stubRealmFetch('acme')
    renderWithRealm(<NewExperimentPage />, '/new', 'acme')

    const corpusSelect = await screen.findByLabelText(/corpus/i) as HTMLSelectElement
    await waitFor(() => {
      const values = [...corpusSelect.options].map(o => o.value)
      expect(values).toContain('handbook_01')
    })
    const values = [...corpusSelect.options].map(o => o.value)
    expect(values).not.toContain('handbook')
  })

  // No safe universal default corpus_id exists across Realms (demo's
  // own 'default' is an older/smaller demo corpus, not what its own
  // pre-filled handbook.v2 dataset was actually built against — picking
  // either hardcoded value risks silently running against the wrong
  // index). The field starts unselected and the submit button stays
  // disabled until the user makes an explicit choice.
  it('starts unselected and disables submit until a corpus_id is chosen', async () => {
    collectionsMock.mockResolvedValue([
      { realm_id: 'acme', corpus_id: 'handbook_01', storage_type: 'dense_only', backends: {}, owner: 'platform', description: '' },
    ])
    stubRealmFetch('acme')
    renderWithRealm(<NewExperimentPage />, '/new', 'acme')

    const corpusSelect = await screen.findByLabelText(/corpus/i) as HTMLSelectElement
    await waitFor(() => expect([...corpusSelect.options].some(o => o.value === 'handbook_01')).toBe(true))
    expect(corpusSelect.value).toBe('')
    const submitButton = screen.getByRole('button', { name: /start run/i }) as HTMLButtonElement
    expect(submitButton.disabled).toBe(true)
  })
})
