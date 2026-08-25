import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, waitFor, fireEvent, within } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { MemoryRouter, Routes, Route } from 'react-router-dom'
import { RealmProvider } from '../context/RealmContext'
import RunPage from '../pages/RunPage'
import { stubRealmFetch } from './realmTestUtils'
import type { ExperimentDetail, Feedback } from '../api/client'

// Human feedback on run answers — see the design notes and
// services/api_gateway/routers/feedback.py's module docstring for the
// design (separate answer_feedback collection, merge-on-upsert semantics).
// Note on scope: hover-reveal (the CSS opacity transition on
// .qrow-feedback-hover) isn't verified here — jsdom doesn't apply an
// external stylesheet's :hover rules, so there's nothing meaningful to
// assert about visibility on hover in this environment (verified live in
// the browser instead). What's tested here is that the controls exist,
// are reachable, and drive the right API calls.

const getExperimentMock = vi.fn()
const getFeedbackMock = vi.fn()
const upsertFeedbackMock = vi.fn()

vi.mock('../api/client', () => ({
  api: {
    experiments: {
      get: (id: string) => getExperimentMock(id),
      setBaseline: vi.fn(),
    },
    feedback: {
      get: (runId: string) => getFeedbackMock(runId),
      upsert: (runId: string, questionId: string, body: unknown) => upsertFeedbackMock(runId, questionId, body),
    },
    prompts: { get: vi.fn() },
  },
}))

const RUN: ExperimentDetail = {
  status: 'done',
  config_hash: 'hash1',
  config_name: 'experiment',
  run_id: 'run1',
  aggregate_metrics: { correct_refusal: 0.8 },
  question_results: [
    { question_id: 'q1', question: 'Does article 5 apply?', reference_answer: 'Yes', generated_answer: 'Yes, it applies', metrics: { answer_similarity: 0.9 } },
    { question_id: 'q2', question: 'Is the equipment rule still in force?', reference_answer: 'Yes', generated_answer: 'Yes', metrics: {} },
  ],
  config: { pipeline_id: 'naive', corpus_id: 'default' },
}

function renderRunPage(runId = 'run1') {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(
    <QueryClientProvider client={qc}>
      <MemoryRouter initialEntries={[`/experiments/${runId}?realm=demo`]}>
        <RealmProvider>
          <Routes>
            <Route path="/experiments/:runId" element={<RunPage />} />
          </Routes>
        </RealmProvider>
      </MemoryRouter>
    </QueryClientProvider>,
  )
}

beforeEach(() => {
  vi.clearAllMocks()
  stubRealmFetch('demo')
  getExperimentMock.mockResolvedValue(RUN)
  getFeedbackMock.mockResolvedValue({})
  Object.defineProperty(navigator, 'clipboard', {
    value: { writeText: vi.fn().mockResolvedValue(undefined) },
    configurable: true,
  })
})

// FeedbackPanel is always in the DOM inside its <details> body (native
// <details> hides non-<summary> content via the UA stylesheet, not
// conditional rendering — see FeedbackPanel's own comment in RunPage.tsx),
// so tests scope queries to one question's <details> via `within` rather
// than relying on visibility.
function getQ1Details(): HTMLDetailsElement {
  return screen.getByText(/Does article 5 apply/).closest('details') as HTMLDetailsElement
}

describe('RunPage — binary feedback', () => {
  it('clicking the thumbs-up icon upserts rating=good for that question', async () => {
    renderRunPage()
    await waitFor(() => expect(screen.getByText(/Does article 5 apply/)).toBeInTheDocument())

    const goodButtons = screen.getAllByRole('button', { name: 'Good answer' })
    fireEvent.click(goodButtons[0])

    await waitFor(() => expect(upsertFeedbackMock).toHaveBeenCalledWith('run1', 'q1', { rating: 'good' }))
  })

  it('clicking the summary rating buttons does not also expand the row', async () => {
    renderRunPage()
    await waitFor(() => expect(screen.getByText(/Does article 5 apply/)).toBeInTheDocument())

    const details = getQ1Details()
    expect(details.open).toBe(false)

    const badButtons = screen.getAllByRole('button', { name: 'Bad answer' })
    fireEvent.click(badButtons[0])

    expect(details.open).toBe(false)
  })
})

describe('RunPage — score dimensions and comment', () => {
  it('clicking a star upserts that one dimension only', async () => {
    renderRunPage()
    await waitFor(() => expect(screen.getByText(/Does article 5 apply/)).toBeInTheDocument())

    const withinQ1 = within(getQ1Details())
    fireEvent.click(withinQ1.getByRole('button', { name: 'Accuracy: 4 of 5' }))

    await waitFor(() => expect(upsertFeedbackMock).toHaveBeenCalledWith('run1', 'q1', { scores: { accuracy: 4 } }))
  })

  it('saves the comment on blur, not on every keystroke', async () => {
    renderRunPage()
    await waitFor(() => expect(screen.getByText(/Does article 5 apply/)).toBeInTheDocument())

    const withinQ1 = within(getQ1Details())
    const textarea = withinQ1.getByPlaceholderText('Comment…')
    fireEvent.change(textarea, { target: { value: 'The answer is imprecise' } })
    expect(upsertFeedbackMock).not.toHaveBeenCalled()

    fireEvent.blur(textarea)
    await waitFor(() => expect(upsertFeedbackMock).toHaveBeenCalledWith('run1', 'q1', { comment: 'The answer is imprecise' }))
  })
})

describe('RunPage — copy feedback to clipboard', () => {
  const FEEDBACK: Record<string, Feedback> = {
    q1: {
      id: 'fb1', run_id: 'run1', question_id: 'q1', rating: 'bad',
      scores: { accuracy: 2 }, comment: 'Not enough detail', reviewer: 'reviewer',
      created_at: '2026-01-01T00:00:00+00:00', updated_at: '2026-01-01T00:00:00+00:00',
    },
    // q2 has no feedback at all — must be skipped in the export.
  }

  it('shows the no-feedback state when nothing has been annotated', async () => {
    renderRunPage()
    await waitFor(() => expect(screen.getByText(/Does article 5 apply/)).toBeInTheDocument())

    fireEvent.click(screen.getByRole('button', { name: /Copy feedback/ }))

    await waitFor(() => expect(screen.getByText('No feedback yet')).toBeInTheDocument())
    expect(navigator.clipboard.writeText).not.toHaveBeenCalled()
  })

  it('copies only annotated questions, including the question text for traceability', async () => {
    getFeedbackMock.mockResolvedValue(FEEDBACK)
    renderRunPage()
    await waitFor(() => expect(screen.getByText(/Does article 5 apply/)).toBeInTheDocument())

    fireEvent.click(screen.getByRole('button', { name: /Copy feedback/ }))

    await waitFor(() => expect(navigator.clipboard.writeText).toHaveBeenCalled())
    const copied = (navigator.clipboard.writeText as ReturnType<typeof vi.fn>).mock.calls[0][0] as string
    expect(copied).toContain('q1')
    expect(copied).toContain('Does article 5 apply?')
    expect(copied).toContain('bad')
    expect(copied).toContain('accuracy=2')
    expect(copied).toContain('Not enough detail')
    expect(copied).toContain('reviewer')
    expect(copied).not.toContain('Is the equipment rule still in force?')
  })
})
