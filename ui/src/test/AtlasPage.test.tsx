import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import AtlasPage from '../pages/AtlasPage'
import type { Atlas, AtlasEntry } from '../api/client'

// The page answers one question: how a reader tells a caught failure from an
// uncaught one, and both from a claim nobody has run. Everything below tests a
// decision that was argued over, never a shape that happened.

vi.mock('../api/client', async () => {
  const actual = await vi.importActual<typeof import('../api/client')>('../api/client')
  return { ...actual, api: { ...actual.api, atlas: { read: vi.fn(), signals: vi.fn() } } }
})
const { api } = await import('../api/client')

function entry(over: Partial<AtlasEntry>): AtlasEntry {
  return {
    id: 'F01', title: 'Re-ingestion adds instead of updating', title_key: 'atlas.F01.title',
    atlas_rows: [1], stage_origin: 'ingest', stage_visible: 'retrieval',
    severity: { quiet: 3, cost: 2, prevalence: 3, total: 18 },
    origin: 'ours-2', detection: 'detector', state: 'caught', instrument: 'config',
    signals: [{ id: 'detector:duplicates', kind: 'detector', name: 'duplicates', side: 'core' }],
    applies_when: [], applies_here: true, applies_to_points: ['dense', 'hybrid', 'graph'],
    bait: 'tests/unit/test_atlas_baits.py::test_bait[F01]', bait_level: 'unit',
    not_detected_reason: '', scope_caveat: '', shares_signals_with: [], superseded_by: [],
    ...over,
  }
}

const ATLAS: Atlas = {
  entries: [
    entry({}),
    entry({ id: 'F31', title: 'The model answers from its own knowledge', detection: 'visible',
            state: 'visible', severity: { quiet: 3, cost: 3, prevalence: 3, total: 27 } }),
    entry({ id: 'F19', title: 'No confidence threshold', detection: 'visible',
            state: 'unproven', bait_level: 'proving_ground',
            bait: 'tests/proving_ground/test_atlas_baits_live.py::test_bait[F19]' }),
    entry({ id: 'F09', title: 'A stale index', detection: 'none', state: 'none', bait: '', bait_level: null,
            not_detected_reason: 'no fingerprint of the source is recorded at ingest' }),
    entry({ id: 'F99', title: 'Not applicable here', applies_here: false, applies_to_points: ['graph'] }),
    entry({ id: 'F98', title: 'Undetermined here', applies_here: null, applies_to_points: [] }),
  ],
  schema: { source: 'https://ragworld.org', release: '2026-08-14', dimensions: 28 },
  points: {
    dense: { coordinates: [{ code: 'C3', value: 'none', url: 'https://ragworld.org/article?d=C3#schema' }], applicable: 27 },
    hybrid: { coordinates: [{ code: 'C3', value: 'rrf', url: 'https://ragworld.org/article?d=C3#schema' }], applicable: 34 },
    graph: { coordinates: [{ code: 'A4', value: 'graph', url: 'https://ragworld.org/article?d=A4#schema' }], applicable: 32 },
  },
  point: 'hybrid',
  uncovered_coordinates: [
    { code: 'A3', value: 'extracted_triples', dimension: 'Unit enrichment', url: 'https://ragworld.org/article?d=A3#schema' },
  ],
}

function renderPage() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter><AtlasPage /></MemoryRouter>
    </QueryClientProvider>,
  )
}

beforeEach(() => {
  vi.mocked(api.atlas.read).mockResolvedValue(ATLAS)
})

describe('the failure atlas', () => {
  it('counts only what applies to the chosen architecture', async () => {
    // A count over the whole catalogue answers no question anybody has: half
    // the entries cannot occur in a dense system. Four of the six fixtures
    // apply; one does not and one is undetermined, and neither may be counted.
    renderPage()
    expect(await screen.findByText(/4.*6|4 of 6|4 из 6/)).toBeTruthy()
  })

  it('a failure nothing catches is not shown as a failure ruled out', async () => {
    renderPage()
    await screen.findAllByRole('table')
    // Collapsed, and counted where the number stays in sight.
    expect(screen.getByText(/1 (отказов, которые не ловятся ничем|failures nothing catches)/)).toBeTruthy()
  })

  it('a claim whose bait lives on a proving ground reads as unproven', async () => {
    // The state that must not be folded into "caught": nobody has watched the
    // signal fire on this failure.
    renderPage()
    await screen.findAllByRole('table')
    expect(screen.getAllByText(/не доказано|unproven/).length).toBeGreaterThan(0)
  })

  it('names the coordinates of this point that no entry speaks about', async () => {
    // The state the graph point is in, and nothing anywhere used to say so.
    renderPage()
    expect(await screen.findByText(/A3=extracted_triples/)).toBeTruthy()
  })

  it('every coordinate carries a link to whoever defines it', async () => {
    // The codes are borrowed from a published schema, so a reader meeting one
    // must be able to reach its definition instead of guessing. The anchor is
    // asserted because a link naming a fragment that does not exist opens the
    // page and drops the reader at its top with no sign it missed.
    renderPage()
    await screen.findAllByRole('table')
    const coords = [...document.querySelectorAll('.coord a')] as HTMLAnchorElement[]
    expect(coords.length).toBeGreaterThan(0)
    for (const link of coords) {
      expect(link.href).toContain('ragworld.org')
      expect(link.href).toContain('#schema')
    }
  })

  it('opening an entry shows what is missing when nothing catches it', async () => {
    renderPage()
    await screen.findAllByRole('table')
    fireEvent.click(screen.getByText(/1 (отказов, которые не ловятся ничем|failures nothing catches)/))
    fireEvent.click(screen.getByRole('button', { name: 'F09' }))
    expect(screen.getByText(/no fingerprint of the source/)).toBeTruthy()
  })

  it('switching architecture asks the server again for that point', async () => {
    renderPage()
    await screen.findAllByRole('table')
    fireEvent.click(screen.getByRole('button', { name: /Графовый|Graph/ }))
    expect(vi.mocked(api.atlas.read)).toHaveBeenCalledWith('graph')
  })

  it('the signal reference says which entries a signal cannot tell apart', async () => {
    // The reverse direction, and the fact that must survive it: a signal
    // standing for two entries is evidence for either and for neither in
    // particular.
    vi.mocked(api.atlas.signals).mockResolvedValue({
      signals: [
        { id: 'detector:duplicates', side: 'core', kind: 'detector',
          failures: ['F01'], singles_out: true, bait_level: ['unit'] },
        { id: 'detector:bm25_dominance', side: 'core', kind: 'detector',
          failures: ['F17', 'F21'], singles_out: false, bait_level: ['unit'] },
        { id: 'ui:dense_score_low', side: 'ui', kind: 'ui',
          failures: ['F10'], singles_out: true, bait_level: ['unit'] },
      ],
    })
    renderPage()
    await screen.findAllByRole('table')
    fireEvent.click(screen.getByRole('tab', { name: /Сигналы|Signals/ }))
    expect(await screen.findByText('detector:bm25_dominance')).toBeTruthy()
    expect(screen.getByText(/не различает их|does not separate them/)).toBeTruthy()
    // Both sides are named: a reference showing one would describe half a
    // platform, and the two disagree on the same data.
    expect(screen.getByText(/интерфейс|interface/)).toBeTruthy()
  })

  it('a link to an entry that cannot occur here moves to one where it can', async () => {
    // The first version picked "any point other than this one", which sends a
    // reader back and forth for ever between two architectures that both
    // exclude the entry. The destination comes from the entry itself.
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    render(
      <QueryClientProvider client={client}>
        <MemoryRouter initialEntries={['/atlas?entry=F99']}><AtlasPage /></MemoryRouter>
      </QueryClientProvider>,
    )
    await screen.findAllByRole('table')
    expect(vi.mocked(api.atlas.read)).toHaveBeenCalledWith('graph')
  })

  it('says the catalogue is unavailable instead of showing an empty one', async () => {
    // An empty table reads as "nothing to report", which is the opposite of
    // what a failed request means.
    vi.mocked(api.atlas.read).mockRejectedValue(new Error('down'))
    renderPage()
    expect(await screen.findByText(/недоступен|unavailable/)).toBeTruthy()
  })
})
