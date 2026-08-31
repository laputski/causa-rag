import { describe, it, expect, vi, beforeEach } from 'vitest'
import { screen, waitFor, fireEvent, within } from '@testing-library/react'
import { Routes, Route } from 'react-router-dom'
import RunPage from '../pages/RunPage'
import { stubRealmFetch, renderWithRealm } from './realmTestUtils'
import type { ExperimentDetail, Dataset } from '../api/client'

// Reframed after a fair question: the question is
// already in a dataset (the run was scored against one), so this isn't
// "add to dataset" — it's "confirm or correct this question's ground
// truth", almost always in that same dataset. No dataset picker in the
// common case; it only appears behind "save elsewhere" or when the run's
// source dataset can't be resolved. See services/api_gateway/routers/
// feedback.py#promote_to_dataset for the backend side of this contract.

const getExperimentMock = vi.fn()
const getFeedbackMock = vi.fn()
const listDatasetsMock = vi.fn()
const promoteMock = vi.fn()

vi.mock('../api/client', () => ({
  api: {
    experiments: { get: (id: string) => getExperimentMock(id), setBaseline: vi.fn() },
    feedback: {
      get: (runId: string) => getFeedbackMock(runId),
      upsert: vi.fn(),
      promote: (runId: string, questionId: string, body: unknown, realmId: string | null) =>
        promoteMock(runId, questionId, body, realmId),
    },
    datasets: { list: (realmId: string | null) => listDatasetsMock(realmId) },
    prompts: { get: vi.fn() },
  },
}))

const RUN: ExperimentDetail = {
  status: 'done',
  config_hash: 'hash1',
  config_name: 'experiment',
  run_id: 'run1',
  aggregate_metrics: {},
  dataset_name: 'handbook.v0.fast.jsonl',
  question_results: [
    { question_id: 'q1', question: 'Does article 5 apply?', reference_answer: 'Yes', generated_answer: 'Yes', metrics: {} },
  ],
  config: { pipeline_id: 'naive', corpus_id: 'default' },
}

const RUN_UNKNOWN_SOURCE: ExperimentDetail = { ...RUN, dataset_name: 'deleted.v0.fast.jsonl' }

const DATASETS: Dataset[] = [
  { name: 'handbook', version: 'v0', speed: 'fast', filename: 'handbook.v0.fast.jsonl', id: 'ds1', count: 10 },
  { name: 'other', version: 'v0', speed: 'fast', filename: 'other.v0.fast.jsonl', id: 'ds2', count: 3 },
]

function renderRunPage() {
  return renderWithRealm(
    <Routes><Route path="/experiments/:runId" element={<RunPage />} /></Routes>,
    '/experiments/run1', 'demo',
  )
}

beforeEach(() => {
  vi.clearAllMocks()
  stubRealmFetch('demo')
  getExperimentMock.mockResolvedValue(RUN)
  getFeedbackMock.mockResolvedValue({})
  listDatasetsMock.mockResolvedValue(DATASETS)
  promoteMock.mockResolvedValue({ id: 'ds1', questions: [], created: true })
})

function getQ1Details(): HTMLDetailsElement {
  return screen.getByText(/Does article 5 apply/).closest('details') as HTMLDetailsElement
}

describe('RunPage — confirm ground truth', () => {
  it('shows the source dataset as a plain hint, no picker, in the common case', async () => {
    renderRunPage()
    await waitFor(() => expect(screen.getByText(/Does article 5 apply/)).toBeInTheDocument())

    const withinQ1 = within(getQ1Details())
    await waitFor(() => expect(withinQ1.getByText(/handbook\.v0\.fast\.jsonl/)).toBeInTheDocument())
    expect(withinQ1.queryByRole('combobox')).not.toBeInTheDocument()
    expect(withinQ1.getByRole('button', { name: 'Confirm' })).not.toBeDisabled()
  })

  it('clicking confirm calls promote with the source dataset and the pre-filled answer', async () => {
    renderRunPage()
    await waitFor(() => expect(screen.getByText(/Does article 5 apply/)).toBeInTheDocument())

    const withinQ1 = within(getQ1Details())
    await waitFor(() => expect(withinQ1.getByText(/handbook\.v0\.fast\.jsonl/)).toBeInTheDocument())
    fireEvent.click(withinQ1.getByRole('button', { name: 'Confirm' }))

    await waitFor(() => expect(promoteMock).toHaveBeenCalledWith(
      'run1', 'q1', { target_dataset_id: 'ds1', reference_answer: 'Yes', article_refs: undefined }, 'demo',
    ))
  })

  it('sends an edited reference answer instead of the original', async () => {
    renderRunPage()
    await waitFor(() => expect(screen.getByText(/Does article 5 apply/)).toBeInTheDocument())

    const withinQ1 = within(getQ1Details())
    await waitFor(() => expect(withinQ1.getByText(/handbook\.v0\.fast\.jsonl/)).toBeInTheDocument())
    fireEvent.change(withinQ1.getByDisplayValue('Yes'), { target: { value: 'A corrected answer' } })
    fireEvent.click(withinQ1.getByRole('button', { name: 'Confirm' }))

    await waitFor(() => expect(promoteMock).toHaveBeenCalledWith(
      'run1', 'q1', { target_dataset_id: 'ds1', reference_answer: 'A corrected answer', article_refs: undefined }, 'demo',
    ))
  })

  it('parses comma-separated reference ids when the reviewer overrides them', async () => {
    renderRunPage()
    await waitFor(() => expect(screen.getByText(/Does article 5 apply/)).toBeInTheDocument())

    const withinQ1 = within(getQ1Details())
    await waitFor(() => expect(withinQ1.getByText(/handbook\.v0\.fast\.jsonl/)).toBeInTheDocument())
    fireEvent.change(withinQ1.getByPlaceholderText(/Comma-separated/), { target: { value: 'art-5, art-6' } })
    fireEvent.click(withinQ1.getByRole('button', { name: 'Confirm' }))

    await waitFor(() => expect(promoteMock).toHaveBeenCalledWith(
      'run1', 'q1', { target_dataset_id: 'ds1', reference_answer: 'Yes', article_refs: ['art-5', 'art-6'] }, 'demo',
    ))
  })

  it('"save elsewhere" reveals the picker and overrides the target', async () => {
    renderRunPage()
    await waitFor(() => expect(screen.getByText(/Does article 5 apply/)).toBeInTheDocument())

    const withinQ1 = within(getQ1Details())
    await waitFor(() => expect(withinQ1.getByText(/handbook\.v0\.fast\.jsonl/)).toBeInTheDocument())
    fireEvent.click(withinQ1.getByText('save to a different dataset'))

    const select = withinQ1.getByDisplayValue('handbook.v0.fast.jsonl')
    fireEvent.change(select, { target: { value: 'ds2' } })
    fireEvent.click(withinQ1.getByRole('button', { name: 'Confirm' }))

    await waitFor(() => expect(promoteMock).toHaveBeenCalledWith(
      'run1', 'q1', { target_dataset_id: 'ds2', reference_answer: 'Yes', article_refs: undefined }, 'demo',
    ))
  })

  it('shows the picker directly (no hint) when the source dataset cannot be resolved', async () => {
    getExperimentMock.mockResolvedValue(RUN_UNKNOWN_SOURCE)
    renderRunPage()
    await waitFor(() => expect(screen.getByText(/Does article 5 apply/)).toBeInTheDocument())

    const withinQ1 = within(getQ1Details())
    // The picker is rendered from what the call returns, so waiting for the
    // call itself stops one step short of what is being asserted.
    expect(await withinQ1.findByRole('combobox')).toBeInTheDocument()
    expect(withinQ1.queryByText(/Dataset:/)).not.toBeInTheDocument()
    expect(withinQ1.getByRole('button', { name: 'Confirm' })).toBeDisabled()
  })

  it('shows "added" for a new question and "updated" for an existing one, without navigating away', async () => {
    renderRunPage()
    await waitFor(() => expect(screen.getByText(/Does article 5 apply/)).toBeInTheDocument())
    const withinQ1 = within(getQ1Details())
    await waitFor(() => expect(withinQ1.getByText(/handbook\.v0\.fast\.jsonl/)).toBeInTheDocument())

    fireEvent.click(withinQ1.getByRole('button', { name: 'Confirm' }))
    await waitFor(() => expect(withinQ1.getByText('Added to dataset')).toBeInTheDocument())

    promoteMock.mockResolvedValue({ id: 'ds1', questions: [], created: false })
    fireEvent.click(withinQ1.getByRole('button', { name: 'Confirm' }))
    await waitFor(() => expect(withinQ1.getByText('Updated in dataset (question already existed)')).toBeInTheDocument())

    expect(screen.getByText(/Does article 5 apply/)).toBeInTheDocument() // no navigation away
  })
})
