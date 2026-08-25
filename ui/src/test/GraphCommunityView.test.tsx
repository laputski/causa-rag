import { describe, it, expect, vi, beforeEach } from 'vitest'
import { waitFor, fireEvent } from '@testing-library/react'
import GraphCommunityView from '../components/GraphCommunityView'
import { renderWithRealm } from './realmTestUtils'

// A node with no chunk list does not say what a community is made of, and that
// is exactly what people open the graph for. Clicking a node expands a section
// under the graph which lives its own life: it loads on its own request, closes
// with a button, shows the first thirty rows and says plainly that there are
// more.

// Sigma draws in WebGL, which jsdom does not have. The whole wrapper is stubbed:
// the test checks our component's behaviour rather than a third-party library's.
// The click handler the component attaches to sigma is lifted out, because
// otherwise there is nothing to click a node with.
const clickHandlers: ((e: { node: string }) => void)[] = []

vi.mock('@react-sigma/core', () => ({
  SigmaContainer: ({ children }: { children: React.ReactNode }) => <div data-sigma>{children}</div>,
  useLoadGraph: () => () => {},
  useSigma: () => ({
    on: (event: string, handler: (e: { node: string }) => void) => {
      if (event === 'clickNode') clickHandlers.push(handler)
    },
    removeListener: () => {},
  }),
}))
vi.mock('@react-sigma/core/lib/react-sigma.min.css', () => ({}))

const communitiesMock = vi.fn()
const detailMock = vi.fn()

vi.mock('../api/client', async () => {
  const actual = await vi.importActual<typeof import('../api/client')>('../api/client')
  return {
    ...actual,
    api: {
      ...actual.api,
      corpus: {
        ...actual.api.corpus,
        graphCommunities: (...a: unknown[]) => communitiesMock(...a),
        graphCommunityDetail: (...a: unknown[]) => detailMock(...a),
      },
    },
  }
})

const COMMUNITIES = {
  algorithm: 'leiden',
  edge_type: 'lexical',
  realm_scoped: true,
  community_count: 3,
  modularity: 0.52,
  communities: [
    { community_id: 7, size: 12, doc_ids: ['d1'], distinct_source_count: 1, dominant_source_share: 1 },
    { community_id: 8, size: 4, doc_ids: ['d2'], distinct_source_count: 2, dominant_source_share: 0.5 },
    { community_id: 9, size: 1, doc_ids: ['d3'], distinct_source_count: 1, dominant_source_share: 1 },
  ],
  inter_community_edges: [{ community_a: 7, community_b: 8, weight: 3 }],
}

const node = (i: number) => ({ chunk_id: `chunk-${String(i).padStart(4, '0')}-abcdef`, doc_id: 'd1', path: `document/article[Article ${i}]` })

beforeEach(() => {
  vi.clearAllMocks()
  clickHandlers.length = 0
  communitiesMock.mockResolvedValue(COMMUNITIES)
  detailMock.mockResolvedValue({
    community_id: 7,
    nodes: Array.from({ length: 5 }, (_, i) => node(i + 1)),
    edges: [{ source: 'chunk-0001-abcdef', target: 'chunk-0002-abcdef' }],
  })
})

function render() {
  return renderWithRealm(<GraphCommunityView corpusId="handbook_01" realmId="acme" />, '/data/graph', 'acme')
}

async function clickNode(container: HTMLElement, id: number) {
  await waitFor(() => expect(clickHandlers.length).toBeGreaterThan(0))
  clickHandlers.forEach(h => h({ node: String(id) }))
  await waitFor(() => expect(container.querySelector('.graph-detail')).toBeTruthy())
}

describe('The community graph: expanding a node', () => {
  it('clicking a node opens a section with a chunk table', async () => {
    const { container } = render()
    await clickNode(container, 7)

    // The community requested is that one, rather than whichever came first.
    expect(detailMock).toHaveBeenCalledWith('handbook_01', 7, 'leiden', 'lexical', 'acme')

    const detail = container.querySelector('.graph-detail')!
    await waitFor(() => expect(detail.querySelector('tbody tr')).toBeTruthy())
    const headers = [...detail.querySelectorAll('th')].map(th => th.textContent)
    expect(headers[0]).toBe('chunk_id')
    expect(detail.querySelectorAll('tbody tr').length).toBe(5)
  })

  it('the close button removes the section', async () => {
    const { container } = render()
    await clickNode(container, 7)

    const close = container.querySelector('.graph-detail .icon-btn') as HTMLButtonElement
    expect(close).toBeTruthy()
    fireEvent.click(close)
    await waitFor(() => expect(container.querySelector('.graph-detail')).toBeNull())
  })

  it('beyond thirty nodes: thirty rows plus a truncation note', async () => {
    detailMock.mockResolvedValue({
      community_id: 7,
      nodes: Array.from({ length: 42 }, (_, i) => node(i + 1)),
      edges: [],
    })
    const { container } = render()
    await clickNode(container, 7)

    const detail = container.querySelector('.graph-detail')!
    await waitFor(() => expect(detail.querySelectorAll('tbody tr').length).toBe(30))
    // Truncating silently leaves a reader believing the community holds thirty
    // nodes, and that is the only number they would take away.
    expect(detail.querySelector('.hint-line')?.textContent?.trim().length).toBeGreaterThan(0)
  })

  it('exactly thirty nodes: no note, nothing to truncate', async () => {
    detailMock.mockResolvedValue({
      community_id: 7,
      nodes: Array.from({ length: 30 }, (_, i) => node(i + 1)),
      edges: [],
    })
    const { container } = render()
    await clickNode(container, 7)

    const detail = container.querySelector('.graph-detail')!
    await waitFor(() => expect(detail.querySelectorAll('tbody tr').length).toBe(30))
    expect(detail.querySelector('.hint-line')).toBeNull()
  })

  it('a structural path stays on one line and carries the full text in a tooltip', async () => {
    const { container } = render()
    await clickNode(container, 7)

    const detail = container.querySelector('.graph-detail')!
    await waitFor(() => expect(detail.querySelector('.path-cell')).toBeTruthy())
    const cell = detail.querySelector('.path-cell')!
    // A path can run past a hundred characters; wrapping stretched every row to
    // three, and any pattern across thirty fragments stopped being readable.
    expect(cell.getAttribute('title')).toBe('document/article[Article 1]')
  })

  it('a node with no path shows a dash rather than an empty cell', async () => {
    detailMock.mockResolvedValue({
      community_id: 7,
      nodes: [{ chunk_id: 'chunk-x-abcdef', doc_id: 'd1', path: '' }],
      edges: [],
    })
    const { container } = render()
    await clickNode(container, 7)

    const detail = container.querySelector('.graph-detail')!
    await waitFor(() => expect(detail.querySelector('.path-cell')).toBeTruthy())
    expect(detail.querySelector('.path-cell')!.textContent).toBe('—')
  })
})
