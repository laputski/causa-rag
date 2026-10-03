import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import DatasetComposition from '../components/DatasetComposition'
import type { DatasetProfile } from '../api/client'

// A set's composition in one line. Pinned: a set of one kind is called that
// before any run, and a mixed set is not.

const profileMock = vi.fn()

vi.mock('../api/client', () => ({
  api: { datasets: { profile: (id: string, realmId?: string | null) => profileMock(id, realmId) } },
}))

function profile(overrides: Partial<DatasetProfile>): DatasetProfile {
  return {
    questions: 13, by_scope: { global: 13 }, by_type: { global: 13 }, by_answerability: { answerable: 13 },
    distinct_references: 25, with_assertions: 13, assertions: 60, ...overrides,
  }
}

function show(signature = 'a') {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  const view = render(
    <QueryClientProvider client={client}><DatasetComposition datasetId="set1" realmId="acme" signature={signature} /></QueryClientProvider>,
  )
  return { ...view, client }
}

describe('DatasetComposition', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('counts questions, sources and both kinds, and says when there is one kind only', async () => {
    profileMock.mockResolvedValue(profile({}))
    show()
    await screen.findByText(/13 questions · 25 sources · every question has statements/)
    expect(screen.getByText('whole corpus: 13')).toBeInTheDocument()
    expect(screen.getByText('ordinary: 0')).toBeInTheDocument()
    expect(screen.getByText('The set holds questions of one kind only.')).toBeInTheDocument()
  })

  it('does not call a mixed set uniform', async () => {
    profileMock.mockResolvedValue(profile({ questions: 20, by_scope: { global: 6, local: 14 }, with_assertions: 6 }))
    show()
    await screen.findByText('whole corpus: 6')
    expect(screen.getByText('ordinary: 14')).toBeInTheDocument()
    expect(screen.getByText(/6 of 20 have statements/)).toBeInTheDocument()
    expect(screen.queryByText('The set holds questions of one kind only.')).toBeNull()
  })

  it('says a set of ordinary questions has no statements, and that it is one kind', async () => {
    profileMock.mockResolvedValue(profile({ questions: 19, by_scope: { local: 19 }, with_assertions: 0, distinct_references: 19 }))
    show()
    await screen.findByText(/19 questions · 19 sources · no statements/)
    expect(screen.getByText('The set holds questions of one kind only.')).toBeInTheDocument()
  })

  it('asks for the set together with its Realm', async () => {
    // A set that arrived by import has no id and is found by its file name,
    // which is unique only within a Realm.
    profileMock.mockResolvedValue(profile({}))
    show()
    await screen.findByText('whole corpus: 13')
    expect(profileMock).toHaveBeenCalledWith('set1', 'acme')
  })

  it('shows nothing for an empty set', async () => {
    profileMock.mockResolvedValue(profile({ questions: 0, by_scope: {}, with_assertions: 0, distinct_references: 0 }))
    const { container, client } = show()
    // Until the answer has arrived the block is empty whatever it holds, so
    // the assertion waits for the answer and not for the request.
    await waitFor(() => expect(client.getQueryData(['dataset-profile', 'set1', 'acme', 'a'])).toBeDefined())
    expect(container).toBeEmptyDOMElement()
  })
})
