import { describe, it, expect, vi, beforeEach } from 'vitest'
import { act, screen, waitFor, fireEvent, within } from '@testing-library/react'
import DatasetsPage from '../pages/DatasetsPage'
import { stubRealmFetch, renderWithRealm } from './realmTestUtils'

// The kind of a question and what its answer must contain, on the page that
// holds the set.
//
// Pinned: a row says which kind of question it is and how many statements it
// carries, the edit form shows both and sends both, and a draft the generator
// wrote shows its statements and the words it should be read for.

const listMock = vi.fn()
const getMock = vi.fn()
const updateQuestionMock = vi.fn()
const profileMock = vi.fn()
const generateMock = vi.fn()

vi.mock('../api/client', () => ({
  api: {
    datasets: {
      list: () => listMock(),
      get: () => getMock(),
      profile: (id: string, realmId?: string | null) => profileMock(id, realmId),
      updateQuestion: (id: string, qid: string, q: unknown) => updateQuestionMock(id, qid, q),
    },
    models: () => Promise.resolve([{ name: 'qwen3:8b', size_gb: 5, modified_at: '', capabilities: ['completion'] }]),
    corpus: { collections: () => Promise.resolve([{ id: 'c1', realm_id: 'demo', corpus_id: 'handbook', storage_type: 'dense_sparse', description: '' }]) },
    generationPresets: { list: () => Promise.resolve([]) },
    generate: { questions: (body: unknown) => generateMock(body) },
  },
  referenceAnswerOf: (q: { reference_answer?: string; ground_truth?: string } | undefined) =>
    q?.reference_answer ?? q?.ground_truth ?? '',
  datasetKeyOf: (set: { id?: string | null; filename: string }) => set.id || set.filename,
}))

class FakeWebSocket {
  static instances: FakeWebSocket[] = []
  onmessage: ((ev: { data: string }) => void) | null = null
  onerror: (() => void) | null = null
  constructor(public url: string) { FakeWebSocket.instances.push(this) }
  close() {}
  emit(event: unknown) { act(() => { this.onmessage?.({ data: JSON.stringify(event) }) }) }
}

const WHOLE = {
  id: 'q1', question: 'Which instruments are verified every 24 months?', reference_answer: 'The SF-210 and the IK-240.',
  question_type: 'global', article_refs: ['DOC1', 'DOC2'], scope: 'global',
  assertions: ['The SF-210 is verified every 24 months.', 'The IK-240 is verified every 24 months.'],
  provenance: { origin: 'manual' as const },
}
const ORDINARY = {
  id: 'q2', question: 'How often is the filter replaced?', reference_answer: 'Every 6 months.',
  question_type: 'closed', article_refs: ['DOC1'], provenance: { origin: 'manual' as const },
}
const DATASET = {
  id: 'ds1', name: 'handbook', version: 'v0', speed: 'fast', filename: 'handbook.v0.fast.jsonl',
  realm_id: 'demo', count: 2, questions: [WHOLE, ORDINARY],
}

async function open() {
  stubRealmFetch('demo')
  renderWithRealm(<DatasetsPage />, '/data/qa', 'demo')
  await screen.findByText(WHOLE.question)
}

const rowOf = (text: string) => screen.getByText(text).closest('tr') as HTMLElement

beforeEach(() => {
  vi.clearAllMocks()
  vi.unstubAllGlobals()
  FakeWebSocket.instances = []
  vi.stubGlobal('WebSocket', FakeWebSocket)
  listMock.mockResolvedValue([DATASET])
  getMock.mockResolvedValue(DATASET)
  profileMock.mockResolvedValue({
    questions: 2, by_scope: { global: 1, local: 1 }, by_type: {}, by_answerability: {},
    distinct_references: 2, with_assertions: 1, assertions: 2,
  })
  updateQuestionMock.mockImplementation(async () => DATASET)
})

describe('DatasetsPage — kind of question and statements', () => {
  it('says in a row which kind of question it is and how many statements it has', async () => {
    await open()
    expect(within(rowOf(WHOLE.question)).getByText('whole corpus')).toBeInTheDocument()
    expect(within(rowOf(WHOLE.question)).getByText('2 statements')).toBeInTheDocument()
    expect(within(rowOf(ORDINARY.question)).getByText('closed')).toBeInTheDocument()
    expect(within(rowOf(ORDINARY.question)).queryByText(/statement/)).toBeNull()
  })

  it('shows the composition of the set above its questions', async () => {
    await open()
    expect(await screen.findByText('whole corpus: 1')).toBeInTheDocument()
    expect(profileMock).toHaveBeenCalledWith('ds1', 'demo')
  })

  it('addresses a set that has no id by its file name, for its composition and for an edit', async () => {
    // Found live: a set written by a Realm import carries no id, the page
    // sent `null` in its place, and neither the line nor an edit worked.
    getMock.mockResolvedValue({ ...DATASET, id: null })
    await open()
    await screen.findByText('whole corpus: 1')
    expect(profileMock).toHaveBeenCalledWith('handbook.v0.fast.jsonl', 'demo')
    fireEvent.click(within(rowOf(ORDINARY.question)).getByText('Edit'))
    fireEvent.click(screen.getByText('Save'))
    await waitFor(() => expect(updateQuestionMock).toHaveBeenCalledWith('handbook.v0.fast.jsonl', 'q2', expect.anything()))
  })

  it('opens the edit form on the stored kind and statements, and sends them back', async () => {
    await open()
    fireEvent.click(within(rowOf(WHOLE.question)).getByText('Edit'))
    expect((screen.getByLabelText('Kind of question') as HTMLSelectElement).value).toBe('global')
    expect(screen.getByDisplayValue(WHOLE.assertions[1])).toBeInTheDocument()
    expect(screen.getByText(/Retrieval metrics are not computed/)).toBeInTheDocument()

    fireEvent.click(screen.getByText('Save'))
    await waitFor(() => expect(updateQuestionMock).toHaveBeenCalledWith('ds1', 'q1', expect.objectContaining({
      scope: 'global', assertions: WHOLE.assertions,
    })))
  })

  it('adds, changes and removes statements, and drops a blank one on save', async () => {
    await open()
    fireEvent.click(within(rowOf(ORDINARY.question)).getByText('Edit'))
    expect((screen.getByLabelText('Kind of question') as HTMLSelectElement).value).toBe('local')

    fireEvent.click(screen.getByText('+ Add a statement'))
    fireEvent.click(screen.getByText('+ Add a statement'))
    fireEvent.click(screen.getByText('+ Add a statement'))
    fireEvent.change(screen.getByLabelText('Statement 1'), { target: { value: 'The filter is replaced every 6 months.' } })
    fireEvent.change(screen.getByLabelText('Statement 2'), { target: { value: 'to be removed' } })
    fireEvent.click(within(screen.getByTestId('statements-editor')).getAllByText('Remove')[1])
    fireEvent.click(screen.getByText('Save'))

    await waitFor(() => expect(updateQuestionMock).toHaveBeenCalledWith('ds1', 'q2', expect.objectContaining({
      scope: 'local', assertions: ['The filter is replaced every 6 months.'],
    })))
  })

  it('sets the kind when the type that names it is chosen, and leaves it alone otherwise', async () => {
    await open()
    fireEvent.click(within(rowOf(ORDINARY.question)).getByText('Edit'))
    const kind = screen.getByLabelText('Kind of question') as HTMLSelectElement
    fireEvent.change(screen.getByLabelText('Question type'), { target: { value: 'open' } })
    expect(kind.value).toBe('local')
    fireEvent.change(screen.getByLabelText('Question type'), { target: { value: 'global' } })
    expect(kind.value).toBe('global')
    fireEvent.click(screen.getByText('Save'))
    await waitFor(() => expect(updateQuestionMock).toHaveBeenCalledWith('ds1', 'q2', expect.objectContaining({
      question_type: 'global', scope: 'global',
    })))
  })

  it('shows the statements of a draft and the words it should be read for', async () => {
    generateMock.mockResolvedValue({ job_id: 'job-1', status: 'started', n_groups: 1 })
    await open()
    fireEvent.click(screen.getByText('Generate'))
    // A select takes a value only once the option for it has arrived.
    await screen.findByRole('option', { name: 'qwen3:8b' })
    await waitFor(() => expect(within(screen.getByLabelText(/Corpus/)).getByText(/handbook/)).toBeInTheDocument())
    fireEvent.change(screen.getByLabelText(/Model/), { target: { value: 'qwen3:8b' } })
    fireEvent.change(screen.getByLabelText(/Corpus/), { target: { value: 'handbook' } })
    const editor = screen.getByTestId('type-counts-editor')
    fireEvent.change(within(editor).getAllByRole('combobox')[0], { target: { value: 'global' } })
    expect(screen.getByText(/written from four documents/)).toBeInTheDocument()
    const generate = screen.getByRole('button', { name: 'Generate' })
    await waitFor(() => expect(generate).toBeEnabled())
    fireEvent.click(generate)

    await waitFor(() => expect(FakeWebSocket.instances.length).toBe(1))
    FakeWebSocket.instances[0].emit({
      type: 'done',
      drafts: [{
        question: 'Within how many days is the report due?', reference_answer: 'Two days.', question_type: 'global',
        article_refs: ['DOC1', 'DOC2'], scope: 'global', assertions: ['The report is due within two days.'],
        provenance: { origin: 'generated', model: 'qwen3:8b', mixed_script_words: ['выезda'] },
      }],
      failed: [{ reason: 'not_enough_documents', chunk_ids: [], missing: 2 }, { reason: 'prompt_cut', chunk_ids: ['c1'] }],
    })

    expect(await screen.findByDisplayValue('The report is due within two days.')).toBeInTheDocument()
    expect(screen.getByText(/these words mix letters of different alphabets: выезda/)).toBeInTheDocument()
    // Two shortfalls and two sentences: one question the model failed to
    // write, and two that were never started.
    expect(screen.getByText(/For 2 whole-corpus questions there were no four documents/)).toBeInTheDocument()
    expect(screen.getByText(/Failed to generate 1 question/)).toBeInTheDocument()
    expect(generateMock).toHaveBeenCalledWith(expect.objectContaining({ type_counts: [{ question_type: 'global', n_questions: 10 }] }))
  })
})
