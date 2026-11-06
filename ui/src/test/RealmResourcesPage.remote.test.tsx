import { describe, it, expect, vi, beforeEach } from 'vitest'
import { screen, waitFor, fireEvent, within } from '@testing-library/react'
import RealmResourcesPage from '../pages/RealmResourcesPage'
import { renderWithRealm } from './realmTestUtils'
import { STORED_SECRET } from '../components/ResourceForm'

// A Realm whose services are not all on this machine.
//
// Pinned: the page says once, with addresses, that data leaves the machine;
// each resource is marked by where the server placed it; and changing a
// resource that holds a secret sends the mask back, which the server reads as
// "keep what is stored". Before there was a way to change a resource the only
// route was to delete it and type the key again.

const CATALOGUE = [
  { type: 'qdrant', label: 'Qdrant', description: '', params_schema: {
    host: { type: 'string', default: 'localhost' }, port: { type: 'integer', default: 6333 },
    https: { type: 'boolean', default: false }, api_key: { type: 'string', secret: true }, url: { type: 'string' } } },
  { type: 'openai_compatible', label: 'OpenAI-compatible API', description: '', params_schema: {
    base_url: { type: 'string' }, api_key: { type: 'string', secret: true },
    model: { type: 'string' }, embedding_model: { type: 'string' } } },
]

const QDRANT = { type: 'qdrant', host: 'localhost', port: 6333, https: false, api_key: '', url: '' }
// `region` is no field of the type: something another writer put there.
const API = { type: 'openai_compatible', base_url: 'https://models.example/v1', api_key: STORED_SECRET, model: 'answers-large', embedding_model: '', region: 'eu' }

let realm: Record<string, unknown>
// What the server answers when a resource is tested.
let checked: Record<string, unknown>
const puts: unknown[] = []

vi.mock('../api/client', async () => {
  const actual = await vi.importActual<typeof import('../api/client')>('../api/client')
  return {
    ...actual,
    api: {
      ...actual.api,
      externalRags: { ...actual.api.externalRags, list: () => Promise.resolve([]) },
      panels: () => Promise.resolve({}),
      panelsStatus: () => Promise.resolve([]),
      settings: { ...actual.api.settings, judge: () => Promise.resolve({ provider: '', model: '', default_model: 'qwen3:32b' }) },
    },
  }
})

beforeEach(() => {
  puts.length = 0
  checked = { status: 'ok', type: 'qdrant', address: 'http://localhost:6333', unavailable_operations: {} }
  realm = {
    id: 'acme', name: 'Acme', created_at: '2026-01-01', resources: [QDRANT, API],
    leaves_this_machine: [{ type: 'openai_compatible', address: 'https://models.example' }],
  }
  vi.stubGlobal('fetch', vi.fn(async (url: string, init?: RequestInit) => {
    if (init?.method === 'PUT') puts.push(JSON.parse(String(init.body)))
    const body = String(url).includes('connector-types') ? CATALOGUE
      : String(url).endsWith('/api/realms') ? [realm]
      : String(url).endsWith('/resources/test') ? checked
      : {}
    return { ok: true, json: async () => body }
  }))
})

const show = () => renderWithRealm(<RealmResourcesPage />, '/settings/resources', 'acme')
// A tile by its name, and only a tile: the button that adds a resource of the
// same type carries the same name, and stands on the page for the moment
// before the Realm's resources arrive (found when CI met that moment).
const tile = (name: string) =>
  screen.getByText(name, { selector: '.resource-card-name' }).closest('.res-tile') as HTMLElement
// Every test waits for the tiles once before it reads them.
const tilesShown = () => screen.findByText('Qdrant', { selector: '.resource-card-name' })

describe('RealmResourcesPage — services elsewhere', () => {
  it('says once that data leaves this machine, and where to', async () => {
    show()
    const notice = await screen.findByTestId('leaves-this-machine')
    expect(notice).toHaveTextContent('Some data leaves this machine')
    expect(notice).toHaveTextContent('https://models.example')
  })

  it('says nothing of the kind for a Realm that is all on this machine', async () => {
    realm = { ...realm, resources: [QDRANT], leaves_this_machine: [] }
    show()
    await screen.findByText('Qdrant')
    expect(screen.queryByTestId('leaves-this-machine')).toBeNull()
  })

  it('marks each resource by where the server placed it', async () => {
    show()
    await screen.findByTestId('leaves-this-machine')
    await tilesShown()
    expect(within(tile('Model over the OpenAI-compatible API')).getByText('external')).toBeInTheDocument()
    expect(within(tile('Qdrant')).getByText('local')).toBeInTheDocument()
  })

  it('leaves the empty fields of a resource out of its tile', async () => {
    show()
    await screen.findByTestId('leaves-this-machine')
    await tilesShown()
    const params = tile('Qdrant').querySelector('.resource-params') as HTMLElement
    expect(within(params).getByText('host')).toBeInTheDocument()
    expect(within(params).queryByText('url')).toBeNull()
    expect(within(params).queryByText('api_key')).toBeNull()
  })

  it('changes a resource in place and keeps the secret nobody touched', async () => {
    show()
    await screen.findByTestId('leaves-this-machine')
    await tilesShown()
    fireEvent.click(within(tile('Model over the OpenAI-compatible API')).getByTitle('Change the resource'))

    const form = await screen.findByTestId('resource-form')
    expect(within(form).getByText('stored')).toBeInTheDocument()
    expect(within(form).getByText('Not tried with cloud services.')).toBeInTheDocument()
    fireEvent.change(within(form).getByLabelText('Model that answers'), { target: { value: 'answers-small' } })
    fireEvent.click(within(form).getByText('Save'))

    await waitFor(() => expect(puts).toHaveLength(1))
    const sent = puts[0] as Record<string, unknown>[]
    // Still one resource of each type: the changed one replaced the old.
    expect(sent.map(r => r.type).sort()).toEqual(['openai_compatible', 'qdrant'])
    expect(sent.find(r => r.type === 'openai_compatible')).toMatchObject({
      base_url: 'https://models.example/v1', model: 'answers-small', api_key: STORED_SECRET,
      // Kept although the form has no field for it.
      region: 'eu',
    })
  })

  it('shows a check that lacks nothing as empty, and one that lacks something by name', async () => {
    // Found by pressing Test on a healthy graph: the mapping of unavailable
    // operations was printed as "[object Object]", empty or not.
    show()
    await screen.findByTestId('leaves-this-machine')
    await tilesShown()
    fireEvent.click(within(tile('Qdrant')).getByText('Test'))
    await waitFor(() => expect(tile('Qdrant').querySelector('.conn-status')).toBeTruthy())
    expect(tile('Qdrant')).not.toHaveTextContent('[object Object]')
    const row = within(tile('Qdrant')).getByText('unavailable_operations').closest('.conn-status-row') as HTMLElement
    expect(row).toHaveTextContent('empty')

    checked = { ...checked, unavailable_operations: { gds: ['community detection', 'similarity'] } }
    fireEvent.click(within(tile('Qdrant')).getByText('Test'))
    await waitFor(() => expect(tile('Qdrant')).toHaveTextContent('gds: community detection, similarity'))
    expect(tile('Qdrant')).not.toHaveTextContent('[object Object]')
  })

  it('offers the judge the model servers the Realm declares', async () => {
    show()
    const judge = await screen.findByTestId('judge-setting')
    const server = within(judge).getByLabelText('Server') as HTMLSelectElement
    // The list follows the resources, which may arrive after the block.
    await waitFor(() => expect([...server.options].map(o => o.value)).toEqual(['', 'openai_compatible']))
  })
})
