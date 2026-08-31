import { describe, it, expect, vi, beforeEach } from 'vitest'
import { screen, waitFor } from '@testing-library/react'
import { render } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { MemoryRouter } from 'react-router-dom'
import RunPrescription from '../components/RunPrescription'
import type { ExperimentDetail, Prescription } from '../api/client'

// Phase 4, whose stated deliverable was that the owner of an external system
// receives a document they can work from. Everything for it existed on the
// server and nothing showed it, which made the phase's own output the one
// thing a reader could not reach.

const prescriptionMock = vi.fn()

vi.mock('../api/client', () => ({
  api: { experiments: { prescription: (runId: string) => prescriptionMock(runId) } },
}))

const PRESCRIPTION: Prescription = {
  run_id: 'r1', dataset_name: 'Cosmos 1', n_questions: 50,
  diagnosis_depth: 'partial',
  fixes: [{
    cause: 'ranking', lever: 'ranking', entity: 'manual_07',
    questions: 12, examples: ["What are the product's dimensions?"],
    verification_question_ids: ['q1', 'q2', 'q3'],
  }],
  gaps: [{
    field: 'stage_trace', unavailable: 'slow retrieval cannot be told from slow generation',
    remedy: 'report per-stage timings',
  }],
  context_size_advice: { current_k: 5, recommended_k: 20, questions_gained: 23, unreachable: 4 },
  metrics: { retrieval_recall_at_k: 0.3 },
  markdown: '# Prescription\n\nFix manual_07.',
}

const RUN = { run_id: 'r1' } as ExperimentDetail

function renderPanel() {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(
    <QueryClientProvider client={qc}>
      <MemoryRouter><RunPrescription run={RUN} /></MemoryRouter>
    </QueryClientProvider>,
  )
}

describe('RunPrescription', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    prescriptionMock.mockResolvedValue(PRESCRIPTION)
  })

  // @lat: [[prescription#Prescription — a document for someone who cannot see the platform#Reviewer-facing surface#Both renderings come from one response]]
  it('shows each fix as a cause and the lever it points at', async () => {
    renderPanel()
    await waitFor(() => expect(screen.getByText('manual_07')).toBeInTheDocument())
    // The arrow disambiguates: the word "ranking" is both this cause's
    // name and part of the lever sentence next to it.
    expect(screen.getByText(/ranking →/)).toBeInTheDocument()
    expect(screen.getByText(/Raise the selection size/)).toBeInTheDocument()
  })

  // @lat: [[prescription#Prescription — a document for someone who cannot see the platform#Reviewer-facing surface#The acceptance set is named on the fix]]
  it('names how many questions the fix will be judged on', async () => {
    renderPanel()
    await waitFor(() => expect(screen.getByText('manual_07')).toBeInTheDocument())
    // Three ids fixed at the moment the document was written — a criterion the
    // recipient could still widen afterwards would not be a criterion.
    expect(screen.getByText(/verified on 3 question/)).toBeInTheDocument()
  })

  // @lat: [[prescription#Prescription — a document for someone who cannot see the platform#Reviewer-facing surface#A gap is stated with its remedy]]
  it('states what the run could not establish, and what would make it establishable', async () => {
    renderPanel()
    await waitFor(() => expect(screen.getByText('stage_trace')).toBeInTheDocument())
    expect(screen.getByText(/slow retrieval cannot be told/)).toBeInTheDocument()
    expect(screen.getByText(/report per-stage timings/)).toBeInTheDocument()
  })

  // The whole point of the tab: the document itself, ready to hand over.
  it('renders the document text alongside the panels', async () => {
    renderPanel()
    await waitFor(() => expect(screen.getByText(/Fix manual_07/)).toBeInTheDocument())
    expect(screen.getByRole('button', { name: /Copy the document/ })).toBeInTheDocument()
  })

  it('says plainly when there is nothing to prescribe', async () => {
    prescriptionMock.mockResolvedValue({
      ...PRESCRIPTION, fixes: [], gaps: [], context_size_advice: null,
    })
    renderPanel()
    await waitFor(() => expect(screen.getByText(/No fixes derived/)).toBeInTheDocument())
    expect(screen.getByText(/reported everything diagnosis needs/)).toBeInTheDocument()
  })
})
