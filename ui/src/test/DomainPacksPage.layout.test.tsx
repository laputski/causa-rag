import { describe, it, expect, vi } from 'vitest'
import { screen, waitFor, fireEvent } from '@testing-library/react'
import DomainPacksPage from '../pages/DomainPacksPage'
import { renderWithRealm } from './realmTestUtils'

// The page existed for its toggle and opened with four paragraphs of
// explanation, so the one thing anybody does on it fell below the fold. The
// tiles moved up, the explanation collapsed, and the checkbox became a
// toggle.

const PACKS = [
  {
    id: 'manuals', version: '1.0.0', display_name: 'Manuals',
    description: 'Articles, clauses, cross-references.',
    exported_kinds: ['structure_parser', 'mask_engine', 'unknown_kind'],
    active: true,
  },
  {
    id: 'generic_qa', version: '0.1.0', display_name: 'Generic Q&A (demo)',
    description: 'A minimal demo pack.', exported_kinds: ['route_policy'], active: false,
  },
]

const setActive = vi.fn().mockResolvedValue({ active_packs: [], status: 'updated' })

vi.mock('../api/client', async () => {
  const actual = await vi.importActual<typeof import('../api/client')>('../api/client')
  return {
    ...actual,
    api: {
      ...actual.api,
      domainPacks: {
        list: () => Promise.resolve(PACKS),
        setActive: (ids: string[], realmId?: string | null) => setActive(ids, realmId),
      },
    },
  }
})

describe('Domain packs against the design', () => {
  it('the header count reports enabled packs rather than discovered ones', async () => {
    renderWithRealm(<DomainPacksPage />, '/domain-packs', 'acme')
    // "2 discovered" answered a question nobody asks of this page.
    expect(await screen.findByText('1 of 2 enabled')).toBeTruthy()
  })

  it('a pack is toggled by a real checkbox, and an enabled tile is marked', async () => {
    const { container } = renderWithRealm(<DomainPacksPage />, '/domain-packs', 'acme')
    const manuals = await screen.findByLabelText('Manuals') as HTMLInputElement
    expect(manuals.type).toBe('checkbox')
    expect(manuals.checked).toBe(true)
    expect(container.querySelectorAll('.pack-tile.on').length).toBe(1)

    fireEvent.click(manuals)
    // The active list is sent whole rather than as a delta: the server stores it
    // as given.
    await waitFor(() => expect(setActive).toHaveBeenCalledWith([], 'acme'))
  })

  it('a kind with no label is named by its raw id rather than skipped', async () => {
    renderWithRealm(<DomainPacksPage />, '/domain-packs', 'acme')
    // A pack can export a kind the label dictionary does not know yet.
    expect(await screen.findByText('parser')).toBeTruthy()
    expect(screen.getAllByText('unknown_kind').length).toBeGreaterThan(0)
  })

  it('the explanation is collapsed and takes no room above the tiles', async () => {
    const { container } = renderWithRealm(<DomainPacksPage />, '/domain-packs', 'acme')
    await screen.findByText('1 of 2 enabled')
    const details = container.querySelector('details.prose-details') as HTMLDetailsElement
    expect(details).toBeTruthy()
    expect(details.open).toBe(false)
    // And it sits after the tile grid rather than before it.
    const grid = container.querySelector('.res-grid')!
    expect(grid.compareDocumentPosition(details) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy()
  })
})
