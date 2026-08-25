import { describe, it, expect, vi, beforeEach } from 'vitest'
import { screen } from '@testing-library/react'
import OverviewPage from '../pages/OverviewPage'
import { stubRealmFetch, renderWithRealm } from './realmTestUtils'

// The setup band is derived from whether five lists are non-empty, and stored
// nowhere. A stored "step complete" flag disagrees with reality at the exact
// moment somebody deletes the only corpus, and disagrees silently.

// `vi.mock` rather than `vi.spyOn(api.…)`; see ExperimentsPage.columns.test.tsx.
// Stubbing a method on the shared module instance interferes with other test
// files under parallel execution.
const collections = vi.fn()
const datasets = vi.fn()
const prompts = vi.fn()
const presets = vi.fn()
const experiments = vi.fn()
const panels = vi.fn()

vi.mock('../api/client', () => ({
  api: {
    corpus: { collections: (...a: unknown[]) => collections(...a) },
    datasets: { list: (...a: unknown[]) => datasets(...a) },
    prompts: { list: (...a: unknown[]) => prompts(...a) },
    generationPresets: { list: (...a: unknown[]) => presets(...a) },
    experiments: { list: (...a: unknown[]) => experiments(...a) },
    externalRags: { list: () => Promise.resolve([]) },
    realms: { testResource: () => Promise.resolve({ status: 'ok' }) },
    panelsStatus: () => panels(),
  },
}))

function setup({ corpora = 0, sets = 0, prompt = 0, preset = 0, runs = 0 }) {
  collections.mockResolvedValue(Array.from({ length: corpora }, (_, i) => ({
    id: `c${i}`, realm_id: 'demo', corpus_id: `corpus-${i}`, storage_type: 'dense',
    backends: {}, owner: 'platform', description: '',
  })))
  datasets.mockResolvedValue(Array.from({ length: sets }, (_, i) => ({ filename: `d${i}.jsonl`, name: `d${i}` })) as never)
  prompts.mockResolvedValue(Array.from({ length: prompt }, (_, i) => ({ id: `p${i}` })) as never)
  presets.mockResolvedValue(Array.from({ length: preset }, (_, i) => ({ id: `g${i}` })) as never)
  experiments.mockResolvedValue(Array.from({ length: runs }, (_, i) => ({
    run_id: `r${i}`, name: `run ${i}`, config_hash: 'h', aggregate_metrics: {},
    started_at: '2026-08-18T10:00:00Z', finished_at: '', n_questions: 10,
    dataset_name: 'd0.jsonl', status: 'done' as const,
  })))
  panels.mockResolvedValue([])
}

describe('the realm overview', () => {
  beforeEach(() => { vi.clearAllMocks(); localStorage.clear() })

  it('counts a step complete when its list is non-empty', async () => {
    setup({ corpora: 1, sets: 2, prompt: 1 })
    stubRealmFetch('demo')
    renderWithRealm(<OverviewPage />, '/overview', 'demo')

    expect(await screen.findByText('3 of 5')).toBeInTheDocument()
  })

  it('at five of five the steps stay on screen and carry their content', async () => {
    setup({ corpora: 1, sets: 1, prompt: 1, preset: 1, runs: 1 })
    stubRealmFetch('demo')
    renderWithRealm(<OverviewPage />, '/overview', 'demo')

    // The heading stays: five completed steps is a realm's state, and somebody
    // opening a realm they did not build should see that it is configured.
    expect(await screen.findByText('All five steps are done.')).toBeInTheDocument()
    expect(screen.getByText('Realm setup')).toBeInTheDocument()
    // The steps do not collapse: collapsed, they left an empty column beneath
    // them, and a tick with no content does not say what the realm is configured
    // with.
    expect(screen.getByText('Corpus loaded')).toBeInTheDocument()
    expect(document.querySelectorAll('.step-row')).toHaveLength(5)
    // A completed step carries a caption saying what completed it.
    expect(document.querySelectorAll('.step-detail').length).toBeGreaterThan(0)
  })

  it('an empty realm shows all five steps incomplete', async () => {
    setup({})
    stubRealmFetch('demo')
    renderWithRealm(<OverviewPage />, '/overview', 'demo')

    expect(await screen.findByText('0 of 5')).toBeInTheDocument()
    expect(screen.getByText('No runs yet')).toBeInTheDocument()
  })
})
