import { describe, it, expect, vi, beforeEach } from 'vitest'
import { screen, waitFor, fireEvent, within } from '@testing-library/react'
import { Routes, Route } from 'react-router-dom'
import CorpusPage from '../pages/CorpusPage'
import { stubRealmFetch, renderWithRealm } from './realmTestUtils'
import { acceptConfirm, declineConfirm } from './confirmHelper'

// GraphCommunityView pulls in sigma.js, which needs a real WebGL2 context —
// unavailable in jsdom. Only the 'upload' tab (the default) is under test
// here, so the graph view is never actually rendered; it just can't be
// *imported* without this stub.
vi.mock('../components/GraphCommunityView', () => ({ default: () => null }))

// Regression guard: switching Realm used to leave Upload/Content/
// Health/Graph showing every other Realm's corpus_ids, since
// GET /corpus wasn't filtered at all (see the design notes "Realm routing
// for corpus inspection").
const corpusListMock = vi.fn().mockResolvedValue([])
const collectionsMock = vi.fn().mockResolvedValue([])
const updateCollectionMock = vi.fn()
const deleteCollectionMock = vi.fn()
const restoreCollectionMock = vi.fn()

vi.mock('../api/client', () => ({
  api: {
    corpus: {
      list: (realmId?: string | null) => corpusListMock(realmId),
      collections: (realmId?: string | null, includeDeleted?: boolean) => collectionsMock(realmId, includeDeleted),
      updateCollection: (id: string, body: unknown) => updateCollectionMock(id, body),
      deleteCollection: (id: string) => deleteCollectionMock(id),
      restoreCollection: (id: string) => restoreCollectionMock(id),
    },
    registry: vi.fn().mockResolvedValue({}),
    health: vi.fn().mockResolvedValue({ status: 'ok', components: {} }),
  },
}))

beforeEach(() => {
  vi.clearAllMocks()
  corpusListMock.mockResolvedValue([])
  collectionsMock.mockResolvedValue([])
})

describe('CorpusPage realm scoping', () => {
  it('passes activeRealmId into api.corpus.list', async () => {
    stubRealmFetch('acme')
    renderWithRealm(<CorpusPage />, '/data/upload', 'acme')
    await waitFor(() => expect(corpusListMock).toHaveBeenCalledWith('acme'))
  })
})

describe('CorpusPage tab bar', () => {
  // The four views were four sidebar items claiming to be four subjects while
  // being four views of one corpus. They are tabs again, and each keeps its own
  // address so a view stays linkable and the back button still works.
  it('offers all four views as tabs on one page', async () => {
    stubRealmFetch('acme')
    renderWithRealm(<CorpusPage />, '/data/content', 'acme')

    const tabs = await screen.findAllByRole('tab')
    expect(tabs.map(el => el.textContent?.trim())).toEqual(['Upload', 'Content', 'Health', 'Graph'])
  })

  // @lat: [[navigation#Navigation and the patterns shared across pages#The sidebar — four sections, at most five items each#One entry for four views of one corpus]]
  it('marks the tab matching the address, not always the first one', async () => {
    stubRealmFetch('acme')
    // Through a real route, because the tab is read from the address rather
    // than from component state — which is what keeps a view linkable.
    renderWithRealm(
      <Routes><Route path="/data/:tab" element={<CorpusPage />} /></Routes>,
      '/data/health', 'acme',
    )

    const selected = (await screen.findAllByRole('tab')).filter(el => el.getAttribute('aria-selected') === 'true')
    expect(selected).toHaveLength(1)
    expect(selected[0].textContent?.trim()).toBe('Health')
  })
})

describe('CorpusPage corpus management', () => {
  it('shows an empty state when the Realm has no registered corpora', async () => {
    stubRealmFetch('acme')
    renderWithRealm(<CorpusPage />, '/data/upload', 'acme')
    await screen.findByText('No corpora registered for this Realm yet.')
  })

  it('lists registered corpora with their storage type and backends', async () => {
    collectionsMock.mockResolvedValue([
      {
        id: 'c1', realm_id: 'acme', corpus_id: 'manuals', storage_type: 'dense_sparse',
        backends: { qdrant: { collection: 'x' }, opensearch: { index: 'y' } }, owner: 'platform',
        description: 'Install manuals', deleted_at: null,
      },
    ])
    stubRealmFetch('acme')
    renderWithRealm(<CorpusPage />, '/data/upload', 'acme')

    // "manuals" also appears as an option in the upload form's own Corpus ID
    // <select> (a known-ids list, not just the management table) — scope to
    // the table to avoid an ambiguous multi-match.
    const table = await screen.findByRole('table')
    expect(within(table).getByText('manuals')).toBeInTheDocument()
    expect(within(table).getByText('dense_sparse')).toBeInTheDocument()
    expect(within(table).getByText('qdrant, opensearch')).toBeInTheDocument()
    expect(within(table).getByText('Install manuals')).toBeInTheDocument()
  })

  it('edits a corpus description in place', async () => {
    collectionsMock.mockResolvedValue([
      { id: 'c1', realm_id: 'acme', corpus_id: 'manuals', storage_type: 'dense_sparse', backends: {}, owner: 'platform', description: 'old', deleted_at: null },
    ])
    updateCollectionMock.mockResolvedValue({ id: 'c1', description: 'new description' })
    stubRealmFetch('acme')
    renderWithRealm(<CorpusPage />, '/data/upload', 'acme')

    fireEvent.click(await screen.findByText('Edit'))
    const input = screen.getByDisplayValue('old')
    fireEvent.change(input, { target: { value: 'new description' } })
    fireEvent.click(screen.getByText('Save'))

    await waitFor(() => expect(updateCollectionMock).toHaveBeenCalledWith('c1', { description: 'new description' }))
  })

  it('soft-deletes a corpus after confirmation', async () => {
    collectionsMock.mockResolvedValue([
      { id: 'c1', realm_id: 'acme', corpus_id: 'manuals', storage_type: 'dense_sparse', backends: {}, owner: 'platform', description: '', deleted_at: null },
    ])
    deleteCollectionMock.mockResolvedValue(undefined)
    stubRealmFetch('acme')
    renderWithRealm(<CorpusPage />, '/data/upload', 'acme')

    fireEvent.click(await screen.findByText('Delete'))
    await acceptConfirm()
    await waitFor(() => expect(deleteCollectionMock).toHaveBeenCalledWith('c1'))
  })

  it('does not delete when the confirmation is declined', async () => {
    collectionsMock.mockResolvedValue([
      { id: 'c1', realm_id: 'acme', corpus_id: 'manuals', storage_type: 'dense_sparse', backends: {}, owner: 'platform', description: '', deleted_at: null },
    ])
    stubRealmFetch('acme')
    renderWithRealm(<CorpusPage />, '/data/upload', 'acme')

    fireEvent.click(await screen.findByText('Delete'))
    await declineConfirm()
    expect(deleteCollectionMock).not.toHaveBeenCalled()
  })

  it('restores a soft-deleted corpus when "show deleted" is checked', async () => {
    collectionsMock.mockResolvedValue([
      { id: 'c1', realm_id: 'acme', corpus_id: 'manuals', storage_type: 'dense_sparse', backends: {}, owner: 'platform', description: '', deleted_at: '2026-02-01T00:00:00' },
    ])
    restoreCollectionMock.mockResolvedValue({ id: 'c1', deleted_at: null })
    stubRealmFetch('acme')
    renderWithRealm(<CorpusPage />, '/data/upload', 'acme')

    fireEvent.click(await screen.findByLabelText(/Show deleted/))
    await waitFor(() => expect(collectionsMock).toHaveBeenCalledWith('acme', true))
    fireEvent.click(await screen.findByText('Restore'))
    await waitFor(() => expect(restoreCollectionMock).toHaveBeenCalledWith('c1'))
  })
})

describe('CorpusPage upload form Corpus ID field', () => {
  it('shows registered corpus_ids as real <select> options, not a datalist suggestion', async () => {
    // Found live: this used to be a plain text <input list="..."> with a
    // <datalist> — most browsers only surface datalist suggestions once you
    // start typing, so a registered corpus (e.g. "handbook_01") read as
    // invisible, not "only default exists". A real <select> makes every
    // known corpus_id visible without typing anything.
    collectionsMock.mockResolvedValue([
      { id: 'c1', realm_id: 'acme', corpus_id: 'handbook_01', storage_type: 'dense_only', backends: {}, owner: 'platform', description: '', deleted_at: null },
    ])
    stubRealmFetch('acme')
    renderWithRealm(<CorpusPage />, '/data/upload', 'acme')

    await screen.findByRole('table')
    const corpusIdSelect = document.querySelector('#upload-corpus-id') as HTMLSelectElement
    expect(corpusIdSelect.tagName).toBe('SELECT')
    const optionValues = Array.from(corpusIdSelect.options).map(o => o.value)
    expect(optionValues).toContain('handbook_01')
  })

  it('lets typing a brand-new corpus_id via the "+ custom ID..." escape hatch', async () => {
    stubRealmFetch('acme')
    renderWithRealm(<CorpusPage />, '/data/upload', 'acme')

    // The corpus id now also appears as a chip in the page header, so what has
    // to be found is the upload form's field rather than any element whose value
    // is `default`.
    await waitFor(() => expect(document.querySelector('#upload-corpus-id')).toBeTruthy())
    const corpusIdSelect = document.querySelector('#upload-corpus-id') as HTMLSelectElement
    fireEvent.change(corpusIdSelect, { target: { value: '__custom_corpus_id__' } })

    const customInput = screen.getByPlaceholderText('my-corpus-id')
    fireEvent.change(customInput, { target: { value: 'brand-new-corpus' } })
    expect(customInput).toHaveValue('brand-new-corpus')
  })
})

describe('CorpusPage dump-type upload form', () => {
  it('defaults to text files and switches file-input constraints per dump type', async () => {
    stubRealmFetch('acme')
    renderWithRealm(<CorpusPage />, '/data/upload', 'acme')

    const select = await screen.findByLabelText(/Source type/)
    expect((select as HTMLSelectElement).value).toBe('text')
    // The chunking-specific fields (size/overlap/exclude) only apply to text mode.
    expect(screen.getByText('Chunk size')).toBeInTheDocument()

    fireEvent.change(select, { target: { value: 'neo4j-cypher' } })
    expect(screen.queryByText('Chunk size')).not.toBeInTheDocument()
    expect(screen.getByText(/executes the file's Cypher statements verbatim/)).toBeInTheDocument()

    // Pre-made dumps aren't chunked and only ever have one embedder platform-
    // wide (bge_m3) — no chunking-strategy or embedder field for these, only
    // Corpus ID (which corpus this data is registered/addressable under).
    fireEvent.change(select, { target: { value: 'qdrant-snapshot' } })
    expect(screen.queryByText('Chunking strategy')).not.toBeInTheDocument()
    expect(screen.queryByText('Embedder')).not.toBeInTheDocument()
    expect(screen.getByText('Corpus ID (multiple corpora in parallel)')).toBeInTheDocument()

    fireEvent.change(select, { target: { value: 'opensearch-dump' } })
    expect(screen.queryByText('Chunking strategy')).not.toBeInTheDocument()
  })

  it('disables submit for a Cypher upload until the confirmation checkbox is ticked', async () => {
    stubRealmFetch('acme')
    renderWithRealm(<CorpusPage />, '/data/upload', 'acme')

    fireEvent.change(await screen.findByLabelText(/Source type/), { target: { value: 'neo4j-cypher' } })
    const submit = screen.getByText('Upload and index')
    expect(submit).toBeDisabled()

    const fileInput = document.querySelector('input[type="file"]') as HTMLInputElement
    const file = new File(['CREATE (n);'], 'dump.cypher', { type: 'text/plain' })
    fireEvent.change(fileInput, { target: { files: [file] } })
    // Still disabled — file alone isn't enough, the confirm checkbox is required too.
    expect(submit).toBeDisabled()

    fireEvent.click(screen.getByLabelText(/executes the file's Cypher statements verbatim/))
    expect(submit).toBeEnabled()
  })

  it('sends replace=true for an OpenSearch dump only when the checkbox is ticked', async () => {
    // Default upload is upsert-by-chunk_id, not a full index replace (found
    // live: a shorter re-upload left old chunks stranded in the index) —
    // this checkbox is the opt-in for "recreate the index instead".
    const fetchMock = vi.fn(async (url: string, opts?: RequestInit) => {
      void opts
      if (String(url).includes('/api/realms')) {
        return { ok: true, json: async () => [{ id: 'acme', name: 'acme', resources: [], created_at: '2026-01-01' }] }
      }
      return { ok: true, json: async () => ({ corpus: {}, index: 'i', n_indexed: 1, n_skipped: 0, n_errors: 0, replaced: true }) }
    })
    vi.stubGlobal('fetch', fetchMock)
    renderWithRealm(<CorpusPage />, '/data/upload', 'acme')

    fireEvent.change(await screen.findByLabelText(/Source type/), { target: { value: 'opensearch-dump' } })
    const checkbox = screen.getByLabelText(/Fully recreate the index/)
    expect(checkbox).not.toBeChecked()
    fireEvent.click(checkbox)

    const fileInput = document.querySelector('input[type="file"]') as HTMLInputElement
    const file = new File(['{"chunk_id":"c1","text":"hi"}'], 'dump.ndjson', { type: 'application/x-ndjson' })
    fireEvent.change(fileInput, { target: { files: [file] } })
    fireEvent.click(screen.getByText('Upload and index'))

    await waitFor(() => expect(fetchMock).toHaveBeenCalled())
    const lastCall = fetchMock.mock.calls[fetchMock.mock.calls.length - 1]
    const body = lastCall[1]!.body as FormData
    expect(body.get('replace')).toBe('true')
  })
})
