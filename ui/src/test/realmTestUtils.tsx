import { vi } from 'vitest'
import { render } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { MemoryRouter } from 'react-router-dom'
import { RealmProvider } from '../context/RealmContext'

/** Stubs the RealmProvider's `/api/realms` fetch with a single Realm whose
 * id matches `realmId` — enough for `useRealm().activeRealmId` to resolve
 * to it via the `?realm=` URL param without racing realms[0] auto-pick. */
export function stubRealmFetch(realmId: string) {
  vi.stubGlobal('fetch', vi.fn(async () => ({
    ok: true,
    json: async () => [{ id: realmId, name: realmId, resources: [], created_at: '2026-01-01' }],
  })))
}

/** Renders `children` inside RealmProvider + QueryClientProvider +
 * MemoryRouter with `?realm=<realmId>` already in the URL — the shape every
 * Realm-scoped page expects. Call `stubRealmFetch(realmId)` first. */
export function renderWithRealm(children: React.ReactNode, path: string, realmId: string) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  const sep = path.includes('?') ? '&' : '?'
  return render(
    <QueryClientProvider client={qc}>
      <MemoryRouter initialEntries={[`${path}${sep}realm=${realmId}`]}>
        <RealmProvider>{children}</RealmProvider>
      </MemoryRouter>
    </QueryClientProvider>,
  )
}
