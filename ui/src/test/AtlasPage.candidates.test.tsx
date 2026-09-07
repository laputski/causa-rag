import { describe, it, expect, vi, beforeEach } from 'vitest'
import { screen, waitFor, fireEvent, within } from '@testing-library/react'
import AtlasPage from '../pages/AtlasPage'
import { renderWithRealm } from './realmTestUtils'

// The catalogue is read-only from here, and that is why its guarantees hold:
// an entry claims a signal catches a failure, and the build refuses such a
// claim without a bait. A candidate claims nothing of the kind, so it can be
// written from the interface, and it has to be impossible to mistake for an
// entry while it sits on the same page.

const ATLAS = {
  entries: [{
    id: 'F40', title: 'One half of the retrieval is absent', title_key: 'atlas.F40.title',
    atlas_rows: [], stage_origin: 'ingest', stage_visible: 'retrieval',
    severity: { quiet: 3, cost: 3, prevalence: 2, total: 18 },
    origin: 'ours-2', detection: 'detector', state: 'caught', instrument: 'ingest',
    signals: [], applies_when: [], applies_here: true, applies_to_points: ['hybrid'],
    bait: 'tests/unit/test_atlas_baits.py::test_bait[F40]', bait_level: 'unit',
    not_detected_reason: '', scope_caveat: '', shares_signals_with: [], superseded_by: [],
  }],
  schema: { source: 'https://ragworld.org', release: '2026-08-14', verified_on: '', dimensions: 28 },
  points: { hybrid: { coordinates: [], applicable: 1 } },
  point: 'hybrid',
  uncovered_coordinates: [],
}

const CANDIDATES = [
  {
    candidate_id: 'C7f21ab4', realm_id: 'demo',
    title: 'The reranker drops the only fragment that answers',
    looked_like: 'Retrieval found the right section and the answer never used it.',
    observed_on: 'run 9da7259a', suspected_signal: 'pre_rerank_recall_at_k',
    status: 'proposed' as const, note: '', promoted_to: '',
    created_at: '2026-09-06T10:00:00Z', updated_at: '2026-09-06T10:00:00Z',
    confirmed_by_a_bait: false,
  },
  {
    candidate_id: 'C0d4e8f1', realm_id: 'demo',
    title: 'One half of the retrieval is absent',
    looked_like: 'Every setting still said two.',
    observed_on: 'realm proving-ground', suspected_signal: '',
    status: 'accepted' as const, note: '', promoted_to: 'F40',
    created_at: '2026-08-28T10:00:00Z', updated_at: '2026-08-30T10:00:00Z',
    confirmed_by_a_bait: false,
  },
  {
    candidate_id: 'C1c90e02', realm_id: 'demo',
    title: 'Two halves return the same ten fragments',
    looked_like: 'Both retrievers looked healthy.',
    observed_on: 'corpus base-ru', suspected_signal: '',
    status: 'accepted' as const, note: '', promoted_to: '',
    created_at: '2026-09-04T10:00:00Z', updated_at: '2026-09-04T10:00:00Z',
    confirmed_by_a_bait: false,
  },
]

const reportMock = vi.fn(async (..._args: unknown[]) => CANDIDATES[0])
const decideMock = vi.fn(async (..._args: unknown[]) => CANDIDATES[0])
const promoteMock = vi.fn(async (..._args: unknown[]) => CANDIDATES[1])

vi.mock('../api/client', () => ({
  api: {
    atlas: {
      read: () => Promise.resolve(ATLAS),
      signals: () => Promise.resolve({ signals: [] }),
      candidates: () => Promise.resolve({ candidates: CANDIDATES }),
      report: (...args: unknown[]) => reportMock(...args),
      decide: (...args: unknown[]) => decideMock(...args),
      promote: (...args: unknown[]) => promoteMock(...args),
    },
  },
}))

beforeEach(() => {
  reportMock.mockClear()
  decideMock.mockClear()
  promoteMock.mockClear()
  vi.stubGlobal('fetch', vi.fn(async () => ({
    ok: true,
    json: async () => [{ id: 'demo', name: 'Demo', resources: [], key_metrics: [], created_at: '2026-01-01' }],
  })))
})

function render() {
  return renderWithRealm(<AtlasPage />, '/atlas', 'demo')
}

describe('the atlas keeps reported failures apart from its entries', () => {
  it('shows them under a heading of their own, below the catalogue', async () => {
    const { container } = render()
    await screen.findByText('Reported, not yet in the catalogue')

    const headings = [...container.querySelectorAll('.section-title')].map(h => h.textContent)
    const catalogue = headings.findIndex(h => h?.includes('Applies to this point'))
    const reported = headings.findIndex(h => h?.includes('Reported, not yet'))
    expect(catalogue).toBeGreaterThanOrEqual(0)
    expect(reported).toBeGreaterThan(catalogue)
  })

  it('says on the row itself that no bait has confirmed it', async () => {
    // Not a column and not a tooltip: a reader who skims has to meet it
    // without reading a legend.
    const { container } = render()
    const row = await screen.findByText('The reranker drops the only fragment that answers')
    const cell = row.closest('td')!
    expect(within(cell).getByText('no bait has confirmed this')).toBeTruthy()
    expect(container.querySelectorAll('.cand-unbaited').length).toBe(2)
  })

  it('drops that mark once a candidate became an entry, and names the entry', async () => {
    const { container } = render()
    await screen.findByText('One half of the retrieval is absent', { selector: '.cand-title' })

    const state = container.querySelector('.cand-promoted')!
    expect(state.textContent).toContain('promoted')
    expect(state.textContent).toContain('F40')
    const cell = screen.getByText('One half of the retrieval is absent', { selector: '.cand-title' })
      .closest('td')!
    expect(within(cell).queryByText('no bait has confirmed this')).toBeNull()
  })

  it('refuses to report until the two things the server requires are written', async () => {
    render()
    fireEvent.click(await screen.findByText('Report a failure you met'))

    const submit = screen.getByRole('button', { name: 'Report it' }) as HTMLButtonElement
    expect(submit.disabled).toBe(true)

    fireEvent.change(screen.getByLabelText(/What went wrong/), {
      target: { value: 'The reranker drops the only fragment that answers' },
    })
    expect(submit.disabled).toBe(true)

    fireEvent.change(screen.getByLabelText(/What looked like it worked while/), {
      target: { value: 'Retrieval found the right section and the answer never used it.' },
    })
    expect(submit.disabled).toBe(false)

    fireEvent.click(submit)
    await waitFor(() => expect(reportMock).toHaveBeenCalled())
    const body = reportMock.mock.calls[0]![0] as Record<string, string>
    expect(body.title).toContain('reranker')
    expect(body.looked_like).toContain('never used it')
  })
})

describe('a report says where it was seen, and the platform can open it', () => {
  it('links a run and a corpus it recognises in the free text', async () => {
    const { container } = render()
    await screen.findByText('The reranker drops the only fragment that answers')

    const links = [...container.querySelectorAll('a[href]')]
      .map(a => a.getAttribute('href'))
      .filter(h => h?.includes('/experiments/') || h?.includes('corpus_id='))
    expect(links.some(h => h!.includes('/experiments/9da7259a'))).toBe(true)
  })

  it('leaves text it does not recognise exactly as it was written', async () => {
    // The field is free text on purpose: somebody who has just met a failure
    // should not have to know which of the platform's nouns applies. Guessing
    // at an identifier would send a reader to a page about something else.
    const { container } = render()
    await screen.findByText('One half of the retrieval is absent', { selector: '.cand-title' })

    const cells = [...container.querySelectorAll('td')]
      .filter(td => td.textContent?.includes('realm proving-ground'))
    expect(cells.length).toBe(1)
    expect(cells[0]!.querySelector('a[href*="realm-proving"]')).toBeNull()
  })
})

describe('deciding a report and recording what it became', () => {
  it('refuses a rejection until its reason is written', async () => {
    render()
    fireEvent.click(await screen.findByText('Decide'))

    const reject = screen.getByRole('button', { name: 'Reject as ours' }) as HTMLButtonElement
    const accept = screen.getByRole('button', { name: "A served system's failure" }) as HTMLButtonElement
    expect(reject.disabled).toBe(true)
    expect(accept.disabled).toBe(false)

    fireEvent.change(screen.getByLabelText(/Why, if you are rejecting/), {
      target: { value: 'Ours, not a served system’s.' },
    })
    expect(reject.disabled).toBe(false)

    fireEvent.click(reject)
    await waitFor(() => expect(decideMock).toHaveBeenCalled())
    const [id, body] = decideMock.mock.calls[0]! as [string, Record<string, string>]
    expect(id).toBe('C7f21ab4')
    expect(body.status).toBe('rejected')
    expect(body.note).toContain('Ours')
  })

  it('hands over the command that drafts the entry, and refuses a pointer to one nobody wrote', async () => {
    promoteMock.mockRejectedValueOnce(new Error("The catalogue holds no 'F42'."))
    render()
    fireEvent.click(await screen.findByText('Promote'))

    expect(screen.getByText(/--scaffold C1c90e02/)).toBeTruthy()

    const record = screen.getByRole('button', { name: 'Record the pointer' }) as HTMLButtonElement
    expect(record.disabled).toBe(true)
    fireEvent.change(screen.getByLabelText(/The entry it became/), { target: { value: 'F42' } })
    expect(record.disabled).toBe(false)

    fireEvent.click(record)
    // The server's own sentence, shown as it arrives: a second version of the
    // rule would be one more thing for the reader to reconcile.
    expect(await screen.findByText(/The catalogue holds no 'F42'/)).toBeTruthy()
  })
})
