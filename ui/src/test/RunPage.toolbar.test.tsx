import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, waitFor, within, fireEvent } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { MemoryRouter, Routes, Route } from 'react-router-dom'
import RunPage from '../pages/RunPage'
import { RealmProvider } from '../context/RealmContext'

// The run toolbar carried seven buttons, two of which navigated to other pages.
// Finding either of those two meant reading all seven labels.

const getMock = vi.fn()
const setBaselineMock = vi.fn()
const unsetBaselineMock = vi.fn()

vi.mock('../api/client', () => ({
  api: {
    experiments: {
      get: (id: string) => getMock(id),
      setBaseline: (id: string) => setBaselineMock(id),
      unsetBaseline: () => unsetBaselineMock(),
    },
    feedback: { get: () => Promise.resolve({}) },
    prompts: { get: () => Promise.resolve({ template: '' }) },
  },
}))

const RUN = {
  run_id: '15bc244b', config_name: 'experiment', config_hash: 'b188ed6',
  aggregate_metrics: { retrieval_recall_at_k: 0.5 },
  question_results: [], started_at: '2026-08-18T11:40:00Z', finished_at: '2026-08-18T11:46:00Z',
  n_questions: 10, dataset_name: 'handbook.v0.fast', status: 'done' as const,
}

/** Render with a real route: `RunPage` takes its id from `useParams`, and
 *  without `<Route path="/experiments/:runId">` that id is empty, so the page
 *  renders "run not found" and the test looks for buttons in an empty page. */
function renderRunPage() {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(
    <QueryClientProvider client={qc}>
      <MemoryRouter initialEntries={['/experiments/15bc244b?realm=demo']}>
        <RealmProvider>
          <Routes>
            <Route path="/experiments/:runId" element={<RunPage />} />
          </Routes>
        </RealmProvider>
      </MemoryRouter>
    </QueryClientProvider>,
  )
}

function stubRealm() {
  vi.stubGlobal('fetch', vi.fn(async () => ({
    ok: true,
    json: async () => [{ id: 'demo', name: 'Demo', resources: [], created_at: '2026-01-01' }],
  })))
}

describe('the run action toolbar', () => {
  beforeEach(() => { vi.clearAllMocks(); localStorage.clear() })

  it('carries five buttons plus an overflow, not seven', async () => {
    getMock.mockResolvedValue(RUN)
    stubRealm()
    renderRunPage()

    await waitFor(() => expect(document.querySelector('.run-toolbar')).toBeTruthy())
    const bar = document.querySelector('.run-toolbar') as HTMLElement
    // Five actions plus the overflow: "manage corpus" and "manage dataset"
    // moved under it, being navigation rather than actions on the run.
    expect(within(bar).getAllByRole('button')).toHaveLength(6)
    expect(within(bar).queryByText(/Manage corpus/)).toBeNull()
  })

  it('on the baseline run the button clears the mark', async () => {
    getMock.mockResolvedValue({ ...RUN, is_baseline: true })
    stubRealm()
    renderRunPage()

    // Setting a baseline was possible and clearing it was not: on a marked run
    // the button simply was not drawn, so there was no way back to "no
    // baseline".
    const button = await screen.findByText('Unset baseline')
    expect(button).toBeInTheDocument()
  })

  it('on an ordinary run the same button sets the mark', async () => {
    getMock.mockResolvedValue(RUN)
    stubRealm()
    renderRunPage()
    expect(await screen.findByText(/Set as baseline/)).toBeInTheDocument()
  })
})

describe('the run configuration block', () => {
  beforeEach(() => { vi.clearAllMocks(); localStorage.clear() })

  it('renders as pairs rather than as a table', async () => {
    getMock.mockResolvedValue(RUN)
    stubRealm()
    renderRunPage()

    await waitFor(() => expect(document.querySelector('.kv-list')).toBeTruthy())
    const list = document.querySelector('.kv-list') as HTMLElement
    // A two-column table with no headers and no sorting is a list of pairs
    // typeset as though somebody were going to compare down a column.
    expect(list.querySelector('table')).toBeNull()
    expect(list.querySelectorAll('.kv-row').length).toBeGreaterThan(10)
  })
})

describe('the loss funnel', () => {
  beforeEach(() => { vi.clearAllMocks(); localStorage.clear() })

  it('counts how many questions reached each layer', async () => {
    getMock.mockResolvedValue({
      ...RUN,
      // Shaped like a real response: a question row needs `metrics` and text,
      // or it throws during render and takes the whole page with it, leaving
      // the test looking for the funnel in an empty page with no clue why.
      question_results: [
        { question_id: 'q1', question: 'Q?', reference_answer: 'A', generated_answer: 'A', metrics: {}, funnel: { layer: 'retrieval' } },
        { question_id: 'q2', question: 'Q?', reference_answer: 'A', generated_answer: 'A', metrics: {}, funnel: { layer: 'rerank' } },
        { question_id: 'q3', question: 'Q?', reference_answer: 'A', generated_answer: 'A', metrics: {}, funnel: { layer: 'generation' } },
        { question_id: 'q4', question: 'Q?', reference_answer: 'A', generated_answer: 'A', metrics: {}, funnel: { layer: 'ok' } },
      ],
      root_cause_counts: { ranking: 2, data_missing: 1 },
    })
    stubRealm()
    renderRunPage()

    await waitFor(() => expect(document.querySelector('.funnel')).toBeTruthy())
    const funnel = within(document.querySelector('.funnel') as HTMLElement)
    expect(funnel.getByText('3')).toBeInTheDocument()   // found: 4 − 1 not retrieved
    // The leading cause comes from the server's count rather than being
    // recomputed: a second count on the client would diverge from the
    // prescription.
    expect(funnel.getByText('ranking')).toBeInTheDocument()
  })

  it('is not drawn when there are no questions', async () => {
    getMock.mockResolvedValue(RUN)
    stubRealm()
    renderRunPage()
    await waitFor(() => expect(document.querySelector('.kv-list')).toBeTruthy())
    expect(document.querySelector('.funnel')).toBeNull()
  })
})

const Q = (id: string, question: string, layer?: string) => ({
  question_id: id, question, reference_answer: 'A', generated_answer: 'A', metrics: {},
  ...(layer ? { funnel: { layer } } : {}),
})

// @lat: [[design-language#Сверка с макетом, второй заход#Экран прогона — шапка и порядок разделов]]
describe('filtering questions by failure layer', () => {
  beforeEach(() => { vi.clearAllMocks(); localStorage.clear() })

  it('shows only the failures by default rather than the whole list', async () => {
    getMock.mockResolvedValue({
      ...RUN,
      question_results: [Q('q1', 'A failing one', 'retrieval'), Q('q2', 'A passing one', 'ok')],
    })
    stubRealm()
    renderRunPage()

    await screen.findByText('Need attention')
    expect(screen.getByText(/A failing one/)).toBeInTheDocument()
    expect(screen.queryByText(/A passing one/)).toBeNull()

    // "All questions" restores the full list: the filter chooses where to
    // start rather than hiding data.
    fireEvent.click(screen.getByRole('button', { name: /All questions/ }))
    expect(screen.getByText(/A passing one/)).toBeInTheDocument()
  })

  it('shows every question when there are no funnel verdicts at all', async () => {
    // An older run, or an external system with no tracing: filtering by layer
    // would leave an empty screen, and that would read as "no questions".
    getMock.mockResolvedValue({ ...RUN, question_results: [Q('q1', 'No verdict')] })
    stubRealm()
    renderRunPage()

    await screen.findByText('Need attention')
    expect(screen.getByText(/No verdict/)).toBeInTheDocument()
  })
})
