import { describe, it, expect, vi, beforeEach } from 'vitest'
import { screen, within } from '@testing-library/react'
import RealmsPage from '../pages/RealmsPage'
import { stubRealmFetch, renderWithRealm } from './realmTestUtils'

// The rule for keeping runs stands on the Realms page, under the list it
// counts "the last runs" of. The block has its own tests; this pins that the
// page carries it and that it shows the installation's rule.

vi.mock('../api/client', () => ({
  api: {
    experiments: { list: () => Promise.resolve([]) },
    settings: {
      retention: () => Promise.resolve({ keep_last_per_realm: 3, max_age_days: 45, max_mib: 256, automatic: false }),
      retentionPreview: () => Promise.resolve({ cleared: [{ run_id: 'a', reason: 'age' }], texts_removed: 0 }),
      setRetention: vi.fn(),
    },
  },
}))

describe('RealmsPage — keeping runs', () => {
  beforeEach(() => stubRealmFetch('acme'))

  it('shows the rule for keeping runs under the list of Realms', async () => {
    renderWithRealm(<RealmsPage />, '/settings/realms', 'acme')
    const block = await screen.findByTestId('run-retention')
    expect(within(block).getByText('Keeping runs')).toBeInTheDocument()
    expect((within(block).getByLabelText('Remove lists older than, days') as HTMLInputElement).value).toBe('45')
    expect(await within(block).findByText('1 run would be affected now.')).toBeInTheDocument()
    expect(within(block).getByText('Automatic clearing is off on this installation.')).toBeInTheDocument()
  })
})
