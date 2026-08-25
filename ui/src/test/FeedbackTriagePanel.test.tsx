import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, waitFor, fireEvent } from '@testing-library/react'
import { QueryClient, QueryClientProvider, useQuery } from '@tanstack/react-query'
import { MemoryRouter } from 'react-router-dom'
import { RealmProvider } from '../context/RealmContext'
import FeedbackTriagePanel from '../components/FeedbackTriagePanel'
import { stubRealmFetch } from './realmTestUtils'
import type { Feedback } from '../api/client'

// read-only recommendation: this panel classifies
// a comment and lets a human confirm/reject/create-a-pin, never applies
// anything itself.

const triageMock = vi.fn()
const triageDecideMock = vi.fn()
const navigateMock = vi.fn()

vi.mock('react-router-dom', async () => {
  const actual = await vi.importActual('react-router-dom')
  return { ...actual, useNavigate: () => navigateMock }
})

vi.mock('../api/client', () => ({
  api: {
    feedback: {
      triage: (runId: string, questionId: string, realmId?: string | null) => triageMock(runId, questionId, realmId),
      triageDecide: (runId: string, questionId: string, body: unknown, realmId?: string | null) =>
        triageDecideMock(runId, questionId, body, realmId),
    },
  },
}))

const FEEDBACK_NO_COMMENT: Feedback = {
  id: 'fb1', run_id: 'run1', question_id: 'q1', scores: {}, created_at: '', updated_at: '',
}

const FEEDBACK_WITH_COMMENT: Feedback = {
  ...FEEDBACK_NO_COMMENT, comment: 'A citation to the wrong article',
}

const TRIAGE_RESULT: Feedback = {
  ...FEEDBACK_WITH_COMMENT,
  triage_result: {
    error_classes: ['wrong_citation'],
    target_spans: ['1 year'],
    expected_refs: ['art-10'],
    severity: 'medium',
    confidence: 0.85,
    verified_refs: { 'art-10': true },
    funnel_layer: 'ok',
    diagnosis_confirmed: true,
    needs_manual_review: false,
    parse_error: null,
    proposal: { primary_lever: 'fix_citation_mapping', alternative_levers: [], justification: 'wrong_citation: ...' },
  },
  triage_status: 'proposed',
}

// The component itself only ever receives `feedback` as a prop (RunPage.tsx
// owns the ['feedback', runId] query and passes the map down to every
// QuestionRow); its own mutations write back into that same cache entry on
// success. This harness mirrors that one level up, so a mutation's
// onSuccess is actually observable here exactly like it is in the real
// page, instead of writing into a cache entry nothing re-reads.
function Harness({ initial }: { initial: Feedback }) {
  const { data } = useQuery({
    queryKey: ['feedback', 'run1'],
    queryFn: async () => ({ q1: initial }) as Record<string, Feedback>,
    initialData: { q1: initial },
  })
  return (
    <FeedbackTriagePanel
      runId="run1" questionId="q1" feedback={data?.q1}
      corpusId="handbook" questionText="What is the deadline?" realmId="demo"
    />
  )
}

function renderPanel(feedback: Feedback) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(
    <QueryClientProvider client={qc}>
      <MemoryRouter initialEntries={['/experiments/run1?realm=demo']}>
        <RealmProvider>
          <Harness initial={feedback} />
        </RealmProvider>
      </MemoryRouter>
    </QueryClientProvider>,
  )
}

beforeEach(() => {
  vi.clearAllMocks()
  stubRealmFetch('demo')
})

describe('FeedbackTriagePanel', () => {
  it('renders nothing when the question has no feedback comment yet', () => {
    const { container } = renderPanel(FEEDBACK_NO_COMMENT)
    expect(container.textContent).toBe('')
  })

  it('shows an "analyze" button once a comment exists, and runs triage on click', async () => {
    triageMock.mockResolvedValue(TRIAGE_RESULT)
    renderPanel(FEEDBACK_WITH_COMMENT)

    const button = await screen.findByText('Analyze feedback')
    fireEvent.click(button)

    await waitFor(() => expect(triageMock).toHaveBeenCalledWith('run1', 'q1', 'demo'))
    expect(await screen.findByText('wrong_citation')).toBeInTheDocument()
    expect(screen.getByText('fix_citation_mapping')).toBeInTheDocument()
  })

  it('shows verified/unverified badges for expected refs', async () => {
    triageMock.mockResolvedValue(TRIAGE_RESULT)
    renderPanel(FEEDBACK_WITH_COMMENT)
    fireEvent.click(await screen.findByText('Analyze feedback'))
    await screen.findByText('art-10')
    expect(screen.getByText('verified')).toBeInTheDocument()
  })

  it('shows a manual-review flag when needs_manual_review is set', async () => {
    triageMock.mockResolvedValue({
      ...TRIAGE_RESULT,
      triage_result: { ...TRIAGE_RESULT.triage_result!, needs_manual_review: true, parse_error: 'no JSON object found' },
    })
    renderPanel(FEEDBACK_WITH_COMMENT)
    fireEvent.click(await screen.findByText('Analyze feedback'))
    expect(await screen.findByText(/Needs manual review/)).toBeInTheDocument()
  })

  it('confirm calls triageDecide with action=confirm', async () => {
    triageDecideMock.mockResolvedValue({ ...TRIAGE_RESULT, triage_status: 'confirmed' })
    renderPanel(TRIAGE_RESULT)

    fireEvent.click(await screen.findByText('Confirm'))
    await waitFor(() => expect(triageDecideMock).toHaveBeenCalledWith(
      'run1', 'q1', { action: 'confirm' }, 'demo',
    ))
  })

  it('reject calls triageDecide with action=reject', async () => {
    triageDecideMock.mockResolvedValue({ ...TRIAGE_RESULT, triage_status: 'rejected' })
    renderPanel(TRIAGE_RESULT)

    fireEvent.click(await screen.findByText('Reject'))
    await waitFor(() => expect(triageDecideMock).toHaveBeenCalledWith(
      'run1', 'q1', { action: 'reject' }, 'demo',
    ))
  })

  it('the triage action navigates to JudgmentsPage prefilled with corpus, question text, and feedback id', async () => {
    renderPanel(TRIAGE_RESULT)
    // The button was labelled "create a pin" long after pins were removed,
    // naming an action the platform no longer has on the button that performs
    // the one which replaced it.
    fireEvent.click(await screen.findByText('Record a judgment'))

    expect(navigateMock).toHaveBeenCalledTimes(1)
    const url = navigateMock.mock.calls[0][0] as string
    expect(url).toContain('/data/judgments')
    expect(url).toContain('corpus_id=handbook')
    expect(url).toContain('source_feedback_id=fb1')
  })
})
