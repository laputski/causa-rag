import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import JudgeSetting from '../components/JudgeSetting'

// Which model server judges answers for a Realm, and with which model.

const judgeMock = vi.fn()
const setJudgeMock = vi.fn()

vi.mock('../api/client', () => ({
  api: { settings: {
    judge: (realmId: string) => judgeMock(realmId),
    setJudge: (realmId: string, judge: unknown) => setJudgeMock(realmId, judge),
  } },
}))

function show(servers: string[]) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(<QueryClientProvider client={client}><JudgeSetting realmId="acme" servers={servers} /></QueryClientProvider>)
}

describe('JudgeSetting', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    judgeMock.mockResolvedValue({ provider: '', model: '', default_model: 'qwen3:32b' })
    setJudgeMock.mockImplementation(async (_realm: string, judge: unknown) => judge)
  })

  it('offers the default and the model servers the Realm declares', async () => {
    show(['ollama'])
    const server = await screen.findByLabelText('Server') as HTMLSelectElement
    expect([...server.options].map(o => o.value)).toEqual(['', 'ollama'])
    expect((screen.getByLabelText("Judge's model") as HTMLInputElement).placeholder).toBe('qwen3:32b')
  })

  it('keeps a server that was chosen earlier and is no longer among the resources', async () => {
    judgeMock.mockResolvedValue({ provider: 'openai_compatible', model: 'judge-model', default_model: 'qwen3:32b' })
    show(['ollama'])
    const server = await screen.findByLabelText('Server') as HTMLSelectElement
    await waitFor(() => expect(server.value).toBe('openai_compatible'))
    expect([...server.options].map(o => o.value)).toEqual(['', 'ollama', 'openai_compatible'])
  })

  it('saves the chosen server and model for the Realm', async () => {
    show(['ollama', 'openai_compatible'])
    fireEvent.change(await screen.findByLabelText('Server'), { target: { value: 'openai_compatible' } })
    fireEvent.change(screen.getByLabelText("Judge's model"), { target: { value: ' judge-model ' } })
    fireEvent.click(screen.getByText('Save'))
    await waitFor(() => expect(setJudgeMock).toHaveBeenCalledWith('acme', { provider: 'openai_compatible', model: 'judge-model' }))
  })

  it('says an API server needs its judge model named', async () => {
    show(['openai_compatible'])
    fireEvent.change(await screen.findByLabelText('Server'), { target: { value: 'openai_compatible' } })
    expect(screen.getByText(/needs its model named/)).toBeInTheDocument()
    fireEvent.change(screen.getByLabelText("Judge's model"), { target: { value: 'judge-model' } })
    expect(screen.queryByText(/needs its model named/)).toBeNull()
  })

  it('offers nothing to save until something changed', async () => {
    show(['ollama'])
    await screen.findByLabelText('Server')
    expect(screen.getByText('Save')).toBeDisabled()
  })
})
