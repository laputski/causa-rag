import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import RunRetention from '../components/RunRetention'

// The rule by which old runs lose their intermediate lists.
//
// Pinned: what the rule would do now is shown beside it, an installation
// where the pass does not run by itself says so, and only the rule's three
// numbers are saved.

const retentionMock = vi.fn()
const setRetentionMock = vi.fn()
const previewMock = vi.fn()

vi.mock('../api/client', () => ({
  api: { settings: {
    retention: () => retentionMock(),
    setRetention: (rule: unknown) => setRetentionMock(rule),
    retentionPreview: () => previewMock(),
  } },
}))

const RULE = { keep_last_per_realm: 2, max_age_days: 30, max_mib: 512 }

function show() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(<QueryClientProvider client={client}><RunRetention /></QueryClientProvider>)
}

describe('RunRetention', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    retentionMock.mockResolvedValue({ ...RULE, automatic: true })
    setRetentionMock.mockImplementation(async (rule: unknown) => rule)
    previewMock.mockResolvedValue({ cleared: [], texts_removed: 0 })
  })

  it('shows the rule and how many runs it would affect now', async () => {
    previewMock.mockResolvedValue({ cleared: [{ run_id: 'a', reason: 'age' }, { run_id: 'b', reason: 'age' }], texts_removed: 5 })
    show()
    expect((await screen.findByLabelText('Remove lists older than, days') as HTMLInputElement).value).toBe('30')
    expect(await screen.findByText('2 runs would be affected now.')).toBeInTheDocument()
    expect(screen.getByText('The baseline run never loses its lists.')).toBeInTheDocument()
  })

  it('says so when the pass does not run by itself here', async () => {
    retentionMock.mockResolvedValue({ ...RULE, automatic: false })
    show()
    expect(await screen.findByText('Automatic clearing is off on this installation.')).toBeInTheDocument()
  })

  it('does not say the pass is off where it runs', async () => {
    show()
    await screen.findByLabelText('Volume limit, MiB')
    expect(screen.queryByText('Automatic clearing is off on this installation.')).toBeNull()
  })

  it('saves the three numbers and nothing else', async () => {
    show()
    fireEvent.change(await screen.findByLabelText('Remove lists older than, days'), { target: { value: '14' } })
    fireEvent.click(screen.getByText('Save'))
    await waitFor(() => expect(setRetentionMock).toHaveBeenCalledWith({ keep_last_per_realm: 2, max_age_days: 14, max_mib: 512 }))
  })

  it('refuses a rule below the bounds the server holds', async () => {
    show()
    fireEvent.change(await screen.findByLabelText('Volume limit, MiB'), { target: { value: '8' } })
    expect(screen.getByText('Save')).toBeDisabled()
    fireEvent.change(screen.getByLabelText('Volume limit, MiB'), { target: { value: '64' } })
    expect(screen.getByText('Save')).not.toBeDisabled()
    fireEvent.change(screen.getByLabelText('Leave the last runs of a Realm alone'), { target: { value: '0' } })
    expect(screen.getByText('Save')).toBeDisabled()
  })
})
