import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { RealmProvider, useRealmPath, useRealm } from '../context/RealmContext'

// Two realms where realms[0] ("demo") is NOT the one the URL asked for
// ("acme") — reproduces the exact live bug: any code path that drops
// `?realm=` makes RealmProvider's "unset ⇒ auto-pick realms[0]" fallback
// silently bounce the user to whichever Realm sorts first (see
// the design notes "Realm selection & UI shell"). useRealmPath is the fix —
// these tests are the regression guard for it.
const REALMS = [
  { id: 'demo', name: 'Demo', resources: [], created_at: '2026-01-01' },
  { id: 'acme', name: 'Acme', resources: [], created_at: '2026-02-01' },
]

function Probe({ to }: { to: string }) {
  const toRealm = useRealmPath()
  const { activeRealmId, loaded } = useRealm()
  return (
    <div>
      <span data-testid="active-realm">{loaded ? activeRealmId ?? 'null' : 'loading'}</span>
      <span data-testid="path">{toRealm(to)}</span>
    </div>
  )
}

function renderWithRealm(initialEntry: string, to: string) {
  return render(
    <MemoryRouter initialEntries={[initialEntry]}>
      <RealmProvider>
        <Probe to={to} />
      </RealmProvider>
    </MemoryRouter>,
  )
}

describe('useRealmPath', () => {
  beforeEach(() => {
    vi.stubGlobal('fetch', vi.fn(async () => ({
      ok: true,
      json: async () => REALMS,
    })))
  })

  it('appends the active Realm to a plain path', async () => {
    renderWithRealm('/new?realm=acme', '/new')
    await waitFor(() => expect(screen.getByTestId('active-realm').textContent).toBe('acme'))
    expect(screen.getByTestId('path').textContent).toBe('/new?realm=acme')
  })

  it('does not silently switch to realms[0] — the exact live bug', async () => {
    // realms[0] is "demo", but the URL/active Realm is "acme" — a
    // regression here would show 'demo' in the generated path instead.
    renderWithRealm('/experiments?realm=acme', '/experiments')
    await waitFor(() => expect(screen.getByTestId('active-realm').textContent).toBe('acme'))
    expect(screen.getByTestId('path').textContent).toBe('/experiments?realm=acme')
    expect(screen.getByTestId('path').textContent).not.toContain('demo')
  })

  it('merges with a path that already has its own query string', async () => {
    renderWithRealm('/compare?realm=acme', '/compare?a=x1&b=x2')
    await waitFor(() => expect(screen.getByTestId('active-realm').textContent).toBe('acme'))
    const path = screen.getByTestId('path').textContent!
    expect(path).toMatch(/^\/compare\?/)
    expect(path).toContain('a=x1')
    expect(path).toContain('b=x2')
    expect(path).toContain('realm=acme')
  })

  it('leaves the path unchanged when there is no Realm to attach', async () => {
    // Zero Realms ⇒ the auto-pick effect never fires (guarded on
    // `realms.length === 0`) ⇒ activeRealmId stays null deterministically,
    // unlike racing the fetch resolution against a non-empty realms list.
    vi.stubGlobal('fetch', vi.fn(async () => ({ ok: true, json: async () => [] })))
    renderWithRealm('/guide', '/guide')
    await waitFor(() => expect(screen.getByTestId('active-realm').textContent).toBe('null'))
    expect(screen.getByTestId('path').textContent).toBe('/guide')
  })
})
