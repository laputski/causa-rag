import { describe, it, expect, vi, beforeEach } from 'vitest'
import { screen, waitFor, fireEvent } from '@testing-library/react'
import { acceptConfirm, declineConfirm } from './confirmHelper'
import PresetsPage from '../pages/PresetsPage'
import { stubRealmFetch, renderWithRealm } from './realmTestUtils'

// Presets are realm-agnostic CRUD, managed through this page instead of the
// old auto-seeded, legal-flavored defaults (see the design notes "Presets
// are entirely optional" and generation.py's module comment above
// _BUILTIN_TEMPLATES for the live bug this page replaces).
const listMock = vi.fn().mockResolvedValue([])
const createMock = vi.fn()
const updateMock = vi.fn()
const deleteMock = vi.fn()

vi.mock('../api/client', () => ({
  api: {
    generationPresets: {
      list: (realmId?: string | null) => listMock(realmId),
      create: (body: unknown) => createMock(body),
      update: (id: string, body: unknown, realmId?: string | null) => updateMock(id, body, realmId),
      delete: (id: string, realmId?: string | null) => deleteMock(id, realmId),
    },
  },
}))

beforeEach(() => {
  vi.clearAllMocks()
  listMock.mockResolvedValue([])
})

describe('PresetsPage realm scoping', () => {
  it('passes activeRealmId into api.generationPresets.list', async () => {
    stubRealmFetch('acme')
    renderWithRealm(<PresetsPage />, '/data/presets', 'acme')
    await waitFor(() => expect(listMock).toHaveBeenCalledWith('acme'))
  })
})

describe('PresetsPage CRUD', () => {
  it('shows the optional-preset hint and an empty state when a Realm has none', async () => {
    stubRealmFetch('acme')
    renderWithRealm(<PresetsPage />, '/data/presets', 'acme')
    await screen.findByText(/No presets yet/)
    expect(screen.getAllByText(/works fine without them|built-in, domain-neutral template/).length).toBeGreaterThan(0)
  })

  it('creates a new preset scoped to the active Realm', async () => {
    createMock.mockResolvedValue({ id: 'p1', name: 'My preset', description: 'desc', template: '{chunk_text}', realm_id: 'acme' })
    stubRealmFetch('acme')
    renderWithRealm(<PresetsPage />, '/data/presets', 'acme')

    fireEvent.click(await screen.findByText('+ New'))
    fireEvent.change(screen.getByLabelText(/Name/), { target: { value: 'My preset' } })
    fireEvent.change(screen.getByLabelText(/Template/), { target: { value: '{chunk_text}' } })
    fireEvent.click(screen.getByText('Create'))

    await waitFor(() => expect(createMock).toHaveBeenCalledWith(expect.objectContaining({
      name: 'My preset', template: '{chunk_text}', realm_id: 'acme',
    })))
  })

  it('edits and saves an existing preset in place', async () => {
    listMock.mockResolvedValueOnce([
      { id: 'p1', name: 'Old name', description: '', template: '{chunk_text}', realm_id: 'acme' },
    ])
    updateMock.mockResolvedValue({ id: 'p1', name: 'New name', description: '', template: '{chunk_text}', realm_id: 'acme' })
    stubRealmFetch('acme')
    renderWithRealm(<PresetsPage />, '/data/presets', 'acme')

    fireEvent.click(await screen.findByText('Old name'))
    const nameInput = screen.getByLabelText(/Name/)
    fireEvent.change(nameInput, { target: { value: 'New name' } })
    fireEvent.click(screen.getByText('Save'))

    await waitFor(() => expect(updateMock).toHaveBeenCalledWith(
      'p1', expect.objectContaining({ name: 'New name' }), 'acme',
    ))
  })

  it('deletes a preset after confirmation', async () => {
    listMock.mockResolvedValueOnce([
      { id: 'p1', name: 'To delete', description: '', template: '{chunk_text}', realm_id: 'acme' },
    ])
    deleteMock.mockResolvedValue(undefined)
    stubRealmFetch('acme')
    renderWithRealm(<PresetsPage />, '/data/presets', 'acme')

    fireEvent.click(await screen.findByText('To delete'))
    fireEvent.click(screen.getByText('Delete'))
    await acceptConfirm()

    await waitFor(() => expect(deleteMock).toHaveBeenCalledWith('p1', 'acme'))
  })

  it('does not delete when the confirmation is declined', async () => {
    listMock.mockResolvedValueOnce([
      { id: 'p1', name: 'Keep me', description: '', template: '{chunk_text}', realm_id: 'acme' },
    ])
    stubRealmFetch('acme')
    renderWithRealm(<PresetsPage />, '/data/presets', 'acme')

    fireEvent.click(await screen.findByText('Keep me'))
    fireEvent.click(screen.getByText('Delete'))
    await declineConfirm()

    expect(deleteMock).not.toHaveBeenCalled()
  })
})
