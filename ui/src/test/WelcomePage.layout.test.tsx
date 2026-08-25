import { describe, it, expect, vi, beforeEach } from 'vitest'
import { screen, fireEvent, waitFor } from '@testing-library/react'
import WelcomePage from '../pages/WelcomePage'
import { renderWithRealm } from './realmTestUtils'

// First start: there are no realms. The shell used to answer every path with
// the realm management screen, and the whole menu went dim with a single
// tooltip, so somebody opening the platform for the first time saw a grey
// interface and could not tell whether it was broken or deliberate. Three doors,
// because there genuinely are three ways in.

vi.mock('../api/client', async () => {
  const actual = await vi.importActual<typeof import('../api/client')>('../api/client')
  return { ...actual, api: { ...actual.api, realms: { ...actual.api.realms, importRealm: vi.fn() } } }
})

beforeEach(() => { vi.restoreAllMocks() })

describe('First start against the design', () => {
  it('three doors, the first one emphasised', () => {
    const { container } = renderWithRealm(<WelcomePage />, '/', '')
    const choices = container.querySelectorAll('.choice')
    expect(choices.length).toBe(3)
    // Starting your own is the primary route; the other two lead into somebody
    // else's work.
    expect(choices[0].className).toContain('choice-primary')
    expect(container.querySelectorAll('.choice-primary').length).toBe(1)
  })

  it('every door names both itself and what lies behind it', () => {
    const { container } = renderWithRealm(<WelcomePage />, '/', '')
    for (const choice of container.querySelectorAll('.choice')) {
      expect(choice.querySelector('.choice-title')?.textContent).toBeTruthy()
      expect(choice.querySelector('.choice-desc')?.textContent).toBeTruthy()
    }
  })

  it('"create" expands a form in place rather than navigating away', async () => {
    const { container } = renderWithRealm(<WelcomePage />, '/', '')
    fireEvent.click(container.querySelector('.choice-primary')!)
    await waitFor(() => expect(container.querySelector('.choice-list')).toBeNull())
    expect(screen.getByLabelText(/name/i)).toBeTruthy()
  })

  it('the demo realm loads by the same route as any export', async () => {
    const bundle = { format: 'causa-realm/v1', realm: { id: 'demo' } }
    // Answer by URL. A mock returning the bundle for every request also
    // answered RealmProvider's own GET /realms with it, so `realms` was an
    // object and `realms.find` threw inside the provider. The assertions below
    // did not touch that, so the suite reported a pass with an unhandled error
    // beside it.
    const fetchMock = vi.fn().mockImplementation((url: string) =>
      Promise.resolve({ ok: true, json: async () => (url === '/demo.realm.json' ? bundle : []) }))
    vi.stubGlobal('fetch', fetchMock)
    const { api } = await import('../api/client')
    const imported = vi.mocked(api.realms.importRealm).mockResolvedValue({ realm_id: 'demo' } as never)

    const { container } = renderWithRealm(<WelcomePage />, '/', '')
    fireEvent.click(container.querySelectorAll('.choice')[2])

    // No separate "create an example" mechanism: were it to drift from the
    // import path, the example would stop being a check that import works.
    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith('/demo.realm.json'))
    await waitFor(() => expect(imported).toHaveBeenCalled())
  })

  it('the layout does not rest on inline styles', () => {
    const { container } = renderWithRealm(<WelcomePage />, '/', '')
    expect(container.querySelector('.welcome-page')).toBeTruthy()
    expect(container.querySelectorAll('[style]').length).toBe(0)
  })
})
