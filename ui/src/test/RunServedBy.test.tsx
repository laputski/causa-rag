import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent, within, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import RunServedBy from '../components/RunServedBy'
import type { ExperimentDetail } from '../api/client'

// What ran a run, read from what the run recorded about itself.
//
// Pinned: a service somewhere else is marked as external and one on this
// machine as local, a part that said nothing gets no mark at all, a run with
// no judge says so, and the size of an export is asked for only when somebody
// opens the control.

const exportSizeMock = vi.fn()

vi.mock('../api/client', () => ({
  api: {
    experiments: {
      exportSize: (id: string) => exportSizeMock(id),
      exportUrl: (id: string, withTexts: boolean) => `/api/experiments/${id}/export${withTexts ? '?with_texts=true' : ''}`,
    },
  },
}))

function run(overrides: Partial<ExperimentDetail> = {}): ExperimentDetail {
  return {
    run_id: 'run1', status: 'done', config: { name: 'run' }, aggregate_metrics: {}, question_results: [],
    served_by: {
      generator: { provider: 'openai_compatible', address: 'https://models.example/v1', model: 'answers-large', remote: true,
                   last_call: { seed_sent: true, served_model: 'answers-large-2026' } },
      embedder: { provider: 'in_process', model: 'bge_m3', remote: false },
      stores: { qdrant: { address: 'http://localhost:6333', remote: false } },
      leaves_this_machine: ['generator'],
    },
    ...overrides,
  } as ExperimentDetail
}

function show(detail: ExperimentDetail) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(<QueryClientProvider client={client}><RunServedBy run={detail} /></QueryClientProvider>)
}

const rowOf = (label: string) => screen.getByText(label).closest('tr') as HTMLElement

describe('RunServedBy', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    exportSizeMock.mockResolvedValue({ bytes_without_texts: 473_088, bytes_with_texts: 2_621_440, debug_cleared_at: null })
  })

  it('marks a service elsewhere as external and one on this machine as local', () => {
    show(run())
    expect(within(rowOf('Generator')).getByText('external')).toBeInTheDocument()
    expect(within(rowOf('Generator')).getByText('https://models.example/v1')).toBeInTheDocument()
    expect(within(rowOf('Vector index')).getByText('local')).toBeInTheDocument()
    expect(within(rowOf('Embedder')).getByText('inside the platform')).toBeInTheDocument()
  })

  it('gives the judge no mark, since the run did not record where it is', () => {
    show(run({ judge: { provider: 'ollama', model: 'qwen3:32b', base_url: 'http://localhost:11434', calls: 52, seconds: 88.4 } }))
    const judge = rowOf('Judge')
    expect(within(judge).getByText('qwen3:32b')).toBeInTheDocument()
    expect(within(judge).queryByText('local')).toBeNull()
    expect(within(judge).queryByText('external')).toBeNull()
    expect(screen.getByText(/The judge checked 52 statements in 88 s\./)).toBeInTheDocument()
  })

  it('says a seed was sent and which model the server named', () => {
    show(run())
    expect(screen.getByText(/A seed was sent\./)).toBeInTheDocument()
    expect(screen.getByText(/answered with the model answers-large-2026/)).toBeInTheDocument()
  })

  it('says nobody checked the answers when the run had no judge', () => {
    show(run({ judge: { gap: 'model_not_set', note: 'no model is named for the judge' } }))
    expect(screen.getByText(/Answers were not checked against their statements\. no model is named/)).toBeInTheDocument()
    expect(screen.queryByText('Judge')).toBeNull()
  })

  it('says the intermediate lists were removed, with the date', () => {
    show(run({ debug_cleared_at: '2026-09-30T10:00:00Z' }))
    expect(screen.getByText(/lists were removed on 30\.09\.2026/)).toBeInTheDocument()
  })

  it('says so when a run recorded nothing about its services', () => {
    show(run({ served_by: undefined }))
    expect(screen.getByText('This run did not record which services ran it.')).toBeInTheDocument()
  })

  it('says so as well when the judge is the only thing the run recorded', () => {
    // Found by opening a stored run: one row, the judge, with nothing to say
    // that the generator and the stores were simply never written down.
    show(run({ served_by: undefined, judge: { provider: 'ollama', model: 'qwen3:32b', base_url: 'http://localhost:11434' } }))
    expect(screen.getByText('This run did not record which services ran it.')).toBeInTheDocument()
    expect(rowOf('Judge')).toBeInTheDocument()
  })

  it('does not say so for a run that recorded its services', () => {
    show(run())
    expect(screen.queryByText('This run did not record which services ran it.')).toBeNull()
  })

  it('asks for the size of an export only when the control is opened', async () => {
    show(run())
    expect(exportSizeMock).not.toHaveBeenCalled()
    fireEvent.click(screen.getByText('Export the run'))
    await screen.findByText('473 KB')
    expect(screen.getByText('2.6 MB')).toBeInTheDocument()
    expect(exportSizeMock).toHaveBeenCalledWith('run1')
  })

  it('downloads without texts unless texts are chosen', async () => {
    show(run())
    fireEvent.click(screen.getByText('Export the run'))
    const link = () => screen.getByText('Download').closest('a') as HTMLAnchorElement
    expect(link().getAttribute('href')).toBe('/api/experiments/run1/export')
    fireEvent.click(screen.getByLabelText(/With fragment texts/))
    await waitFor(() => expect(link().getAttribute('href')).toBe('/api/experiments/run1/export?with_texts=true'))
  })
})
