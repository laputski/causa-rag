import { describe, it, expect, vi } from 'vitest'
import { waitFor } from '@testing-library/react'
import ExperimentsPage from '../pages/ExperimentsPage'
import { stubRealmFetch, renderWithRealm } from './realmTestUtils'

// Regression guard: experiment_runs is the original Realm-scoped
// entity — this pins the frontend side of it so a future refactor of
// ExperimentsPage can't silently drop the realmId it already threads
// through (see the design notes "Experiment and settings scoping").
const listMock = vi.fn().mockResolvedValue([])

vi.mock('../api/client', () => ({
  api: {
    experiments: {
      list: (params?: { realmId?: string | null }) => listMock(params),
    },
  },
}))

describe('ExperimentsPage realm scoping', () => {
  it('passes activeRealmId into api.experiments.list', async () => {
    stubRealmFetch('acme')
    renderWithRealm(<ExperimentsPage />, '/experiments', 'acme')
    await waitFor(() => expect(listMock).toHaveBeenCalledWith({ realmId: 'acme' }))
  })
})
