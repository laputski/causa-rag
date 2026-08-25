import { describe, it, expect, vi, beforeEach } from 'vitest'
import { act, screen, waitFor, fireEvent, within } from '@testing-library/react'
import DatasetsPage from '../pages/DatasetsPage'
import { stubRealmFetch, renderWithRealm } from './realmTestUtils'
import { acceptConfirm, declineConfirm } from './confirmHelper'

// Regression guard for the live bug: "Control questions" listed one Realm's
// golden datasets while a different Realm was active. GET /datasets had
// no realm_id parameter at all (see the design notes "Datasets scoping").
// This only checks the page threads activeRealmId through; the actual
// filtering behavior is covered server-side by
// tests/unit/test_realm_scoping.py.
const listMock = vi.fn().mockResolvedValue([])
const getMock = vi.fn()
const createMock = vi.fn()
const deleteDatasetMock = vi.fn()
const addQuestionMock = vi.fn()
const updateQuestionMock = vi.fn()
const deleteQuestionMock = vi.fn()
const addQuestionsBatchMock = vi.fn()
const modelsMock = vi.fn()
const corpusCollectionsMock = vi.fn().mockResolvedValue([])
const presetsListMock = vi.fn().mockResolvedValue([])
const generateMock = vi.fn()

vi.mock('../api/client', () => ({
  api: {
    datasets: {
      list: (realmId?: string | null) => listMock(realmId),
      get: (filename: string) => getMock(filename),
      create: (body: unknown) => createMock(body),
      delete: (id: string, realmId?: string | null) => deleteDatasetMock(id, realmId),
      addQuestion: (id: string, q: unknown, realmId?: string | null) => addQuestionMock(id, q, realmId),
      updateQuestion: (id: string, qid: string, q: unknown, realmId?: string | null) => updateQuestionMock(id, qid, q, realmId),
      deleteQuestion: (id: string, qid: string, realmId?: string | null) => deleteQuestionMock(id, qid, realmId),
      addQuestionsBatch: (id: string, qs: unknown, realmId?: string | null) => addQuestionsBatchMock(id, qs, realmId),
    },
    models: (completionOnly?: boolean) => modelsMock(completionOnly),
    corpus: { collections: (realmId?: string | null) => corpusCollectionsMock(realmId) },
    generationPresets: { list: (realmId?: string | null) => presetsListMock(realmId) },
    generate: { questions: (body: unknown) => generateMock(body) },
  },
  referenceAnswerOf: (q: { reference_answer?: string; ground_truth?: string } | undefined) =>
    q?.reference_answer ?? q?.ground_truth ?? '',
}))

// POST /generate/questions now only starts a background job (see
// generation.py's docstring); the actual drafts arrive as a "done" event on
// the paired progress WebSocket (generation.py#generate_questions_progress).
// jsdom has no real WebSocket server, so this fakes the constructor and lets
// each test drives the socket's onmessage callback directly to simulate the
// backend's event stream.
class FakeWebSocket {
  static instances: FakeWebSocket[] = []
  url: string
  onmessage: ((ev: { data: string }) => void) | null = null
  onerror: (() => void) | null = null
  closed = false
  constructor(url: string) {
    this.url = url
    FakeWebSocket.instances.push(this)
  }
  close() { this.closed = true }
  emit(event: unknown) {
    // The real onmessage handler updates React state; wrapping here (rather
    // than in every call site) keeps each test's emit() calls synchronous
    // and warning-free without repeating the act() boilerplate everywhere.
    act(() => { this.onmessage?.({ data: JSON.stringify(event) }) })
  }
}

async function lastSocket(): Promise<FakeWebSocket> {
  await waitFor(() => expect(FakeWebSocket.instances.length).toBeGreaterThan(0))
  return FakeWebSocket.instances[FakeWebSocket.instances.length - 1]
}

const DATASET = {
  id: 'ds1', name: 'handbook', version: 'v0', speed: 'fast', filename: 'handbook.v0.fast.jsonl',
  realm_id: 'demo', count: 1,
  questions: [
    { id: 'q1', question: 'An existing question?', reference_answer: 'Answer', question_type: 'closed', article_refs: [], provenance: { origin: 'manual' as const } },
  ],
}

beforeEach(() => {
  vi.clearAllMocks()
  vi.unstubAllGlobals()
  listMock.mockResolvedValue([])
  corpusCollectionsMock.mockResolvedValue([])
  presetsListMock.mockResolvedValue([])
  FakeWebSocket.instances = []
  vi.stubGlobal('WebSocket', FakeWebSocket)
})

describe('DatasetsPage realm scoping', () => {
  it('passes activeRealmId into api.datasets.list', async () => {
    stubRealmFetch('acme')
    renderWithRealm(<DatasetsPage />, '/data/qa', 'acme')
    await waitFor(() => expect(listMock).toHaveBeenCalledWith('acme'))
  })
})

describe('DatasetsPage create dataset', () => {
  it('composes filename from the three separate fields', async () => {
    createMock.mockResolvedValue({ ...DATASET, questions: [] })
    getMock.mockResolvedValue({ ...DATASET, questions: [] })
    stubRealmFetch('demo')
    renderWithRealm(<DatasetsPage />, '/data/qa', 'demo')

    fireEvent.click(await screen.findByText('New set'))
    fireEvent.change(screen.getByLabelText(/Name/), { target: { value: 'my_set' } })
    fireEvent.change(screen.getByLabelText(/Version/), { target: { value: 'v2' } })
    fireEvent.change(screen.getByLabelText(/Speed/), { target: { value: 'full' } })
    fireEvent.click(screen.getByText('Create'))

    await waitFor(() => expect(createMock).toHaveBeenCalledWith(expect.objectContaining({
      filename: 'my_set.v2.full.jsonl', realm_id: 'demo', questions: [],
    })))
  })
})

describe('DatasetsPage question CRUD', () => {
  it('adds a question via the add form', async () => {
    listMock.mockResolvedValueOnce([DATASET])
    getMock.mockResolvedValue(DATASET)
    addQuestionMock.mockResolvedValue({ ...DATASET, count: 2 })
    stubRealmFetch('demo')
    renderWithRealm(<DatasetsPage />, '/data/qa', 'demo')

    fireEvent.click(await screen.findByText('handbook'))
    fireEvent.click(await screen.findByText('+ Add question'))
    fireEvent.change(screen.getByLabelText(/^Question\s*\*/), { target: { value: 'A new question?' } })
    fireEvent.change(screen.getByLabelText(/Reference\ answer/), { target: { value: 'A new answer' } })
    fireEvent.click(screen.getByText('Save'))

    await waitFor(() => expect(addQuestionMock).toHaveBeenCalledWith(
      'ds1',
      expect.objectContaining({ question: 'A new question?', reference_answer: 'A new answer' }),
      'demo',
    ))
  })

  it('edits an existing question', async () => {
    listMock.mockResolvedValueOnce([DATASET])
    getMock.mockResolvedValue(DATASET)
    updateQuestionMock.mockResolvedValue(DATASET)
    stubRealmFetch('demo')
    renderWithRealm(<DatasetsPage />, '/data/qa', 'demo')

    fireEvent.click(await screen.findByText('handbook'))
    fireEvent.click(await screen.findByText('Edit'))
    fireEvent.change(screen.getByLabelText(/^Question\s*\*/), { target: { value: 'An edited question?' } })
    fireEvent.click(screen.getByText('Save'))

    await waitFor(() => expect(updateQuestionMock).toHaveBeenCalledWith(
      'ds1', 'q1', expect.objectContaining({ question: 'An edited question?' }), 'demo',
    ))
  })

  it('deletes a question after confirmation', async () => {
    listMock.mockResolvedValueOnce([DATASET])
    getMock.mockResolvedValue(DATASET)
    deleteQuestionMock.mockResolvedValue({ ...DATASET, questions: [], count: 0 })
    stubRealmFetch('demo')
    renderWithRealm(<DatasetsPage />, '/data/qa', 'demo')

    fireEvent.click(await screen.findByText('handbook'))
    fireEvent.click(await screen.findByText('Delete'))
    await acceptConfirm()

    await waitFor(() => expect(deleteQuestionMock).toHaveBeenCalledWith('ds1', 'q1', 'demo'))
  })

  it('does not delete when the confirmation is declined', async () => {
    listMock.mockResolvedValueOnce([DATASET])
    getMock.mockResolvedValue(DATASET)
    stubRealmFetch('demo')
    renderWithRealm(<DatasetsPage />, '/data/qa', 'demo')

    fireEvent.click(await screen.findByText('handbook'))
    fireEvent.click(await screen.findByText('Delete'))
    await declineConfirm()

    expect(deleteQuestionMock).not.toHaveBeenCalled()
  })

  it('shows and edits the answer of a question stored under the older ground_truth field name', async () => {
    const oldShapeDataset = {
      ...DATASET,
      questions: [{ id: 'q1', question: 'An old question?', ground_truth: 'An old answer', article_refs: [] }],
    }
    listMock.mockResolvedValueOnce([DATASET])
    getMock.mockResolvedValue(oldShapeDataset)
    stubRealmFetch('demo')
    renderWithRealm(<DatasetsPage />, '/data/qa', 'demo')

    fireEvent.click(await screen.findByText('handbook'))
    // Found live: this text was invisible in both the row and the edit form
    // because only `reference_answer` was read, never `ground_truth`.
    await screen.findByText('An old answer')
    fireEvent.click(screen.getByText('Edit'))
    expect((screen.getByLabelText(/Reference\ answer/) as HTMLTextAreaElement).value).toBe('An old answer')
  })

  it('deletes the whole dataset after confirmation', async () => {
    listMock.mockResolvedValueOnce([DATASET])
    getMock.mockResolvedValue(DATASET)
    deleteDatasetMock.mockResolvedValue(undefined)
    stubRealmFetch('demo')
    renderWithRealm(<DatasetsPage />, '/data/qa', 'demo')

    fireEvent.click(await screen.findByText('handbook'))
    fireEvent.click(await screen.findByText('Delete set'))
    await acceptConfirm()

    await waitFor(() => expect(deleteDatasetMock).toHaveBeenCalledWith('ds1', 'demo'))
  })
})

describe('DatasetsPage generator', () => {
  it('requests only completion-capable models', async () => {
    listMock.mockResolvedValueOnce([DATASET])
    getMock.mockResolvedValue(DATASET)
    modelsMock.mockResolvedValue([{ name: 'qwen3:8b', size_gb: 5, modified_at: '', capabilities: ['completion'] }])
    presetsListMock.mockResolvedValue([{ id: 'p1', name: 'Preset 1', description: '', template: '', realm_id: 'demo' }])
    stubRealmFetch('demo')
    renderWithRealm(<DatasetsPage />, '/data/qa', 'demo')

    fireEvent.click(await screen.findByText('handbook'))
    fireEvent.click(await screen.findByText('Generate'))

    await waitFor(() => expect(modelsMock).toHaveBeenCalledWith(true))
  })

  it('generates drafts and saves only the ones kept', async () => {
    listMock.mockResolvedValueOnce([DATASET])
    getMock.mockResolvedValue(DATASET)
    modelsMock.mockResolvedValue([{ name: 'qwen3:8b', size_gb: 5, modified_at: '', capabilities: ['completion'] }])
    corpusCollectionsMock.mockResolvedValue([{ id: 'c1', realm_id: 'demo', corpus_id: 'handbook', storage_type: 'dense_sparse', description: '' }])
    presetsListMock.mockResolvedValue([{ id: 'p1', name: 'Preset 1', description: '', template: '', realm_id: 'demo' }])
    generateMock.mockResolvedValue({ job_id: 'job-1', status: 'started', n_groups: 2 })
    addQuestionsBatchMock.mockResolvedValue(DATASET)
    stubRealmFetch('demo')
    renderWithRealm(<DatasetsPage />, '/data/qa', 'demo')

    fireEvent.click(await screen.findByText('handbook'))
    fireEvent.click(await screen.findByText('Generate'))
    // Wait for presets to load — the submit button only needs model/corpus
    // set (a preset is optional), but this test explicitly picks one.
    await screen.findByText('Preset 1')
    fireEvent.change(screen.getByLabelText(/Model/), { target: { value: 'qwen3:8b' } })
    fireEvent.change(screen.getByLabelText(/Corpus/), { target: { value: 'handbook' } })
    fireEvent.change(screen.getByLabelText(/Preset/), { target: { value: 'p1' } })
    // The tab toggle above (also textually "Generate") is role="tab",
    // not role="button" — this only matches the panel's submit button.
    const generateButton = screen.getByRole('button', { name: 'Generate' })
    await waitFor(() => expect(generateButton).toBeEnabled())
    fireEvent.click(generateButton)

    // An explicitly chosen preset must actually reach the request — found
    // live: an earlier bug sent preset_id: '' here instead of the effective
    // (then auto-selected-first) one. Presets are opt-in now, so this
    // asserts the deliberate choice is threaded through, not an auto-pick.
    await waitFor(() => expect(generateMock).toHaveBeenCalledWith(expect.objectContaining({
      preset_id: 'p1', model: 'qwen3:8b', corpus_id: 'handbook',
    })))

    // Generation itself is an async job — the drafts arrive as a "done"
    // event on the progress WebSocket, not in the POST response.
    const ws = await lastSocket()
    ws.emit({
      type: 'done',
      drafts: [
        { question: 'Q1?', reference_answer: 'A1', question_type: 'closed', article_refs: ['HK1/1'], provenance: { origin: 'generated', model: 'qwen3:8b' } },
        { question: 'Q2?', reference_answer: 'A2', question_type: 'closed', article_refs: ['HK1/2'], provenance: { origin: 'generated', model: 'qwen3:8b' } },
      ],
      failed: [],
    })

    await screen.findByText('Q1?')
    expect(screen.getByText('Q2?')).toBeInTheDocument()

    // Remove the second draft before saving.
    const removeButtons = screen.getAllByText('Remove')
    fireEvent.click(removeButtons[1])
    await waitFor(() => expect(screen.queryByText('Q2?')).not.toBeInTheDocument())

    fireEvent.click(screen.getByText('Save all'))
    await waitFor(() => expect(addQuestionsBatchMock).toHaveBeenCalledWith(
      'ds1',
      [expect.objectContaining({ question: 'Q1?' })],
      'demo',
    ))

    // Found live: after a successful save the generator's drafts just
    // disappeared with no indication of where they went, which read as data
    // loss even though the questions really were persisted. A successful
    // save must switch back to "Questions" and say where the saved question(s)
    // landed.
    await screen.findByText(/Added 1 question/)
    expect(screen.getByText('+ Add question')).toBeInTheDocument()
  })

  it('generates without picking a preset — preset_id is simply omitted', async () => {
    // Presets are optional: the "Preset" select defaults to a no-preset
    // option, and the submit button does not wait on presets loading or
    // being chosen — only model/corpus are required.
    listMock.mockResolvedValueOnce([DATASET])
    getMock.mockResolvedValue(DATASET)
    modelsMock.mockResolvedValue([{ name: 'qwen3:8b', size_gb: 5, modified_at: '', capabilities: ['completion'] }])
    corpusCollectionsMock.mockResolvedValue([{ id: 'c1', realm_id: 'demo', corpus_id: 'handbook', storage_type: 'dense_sparse', description: '' }])
    presetsListMock.mockResolvedValue([])
    generateMock.mockResolvedValue({ job_id: 'job-1', status: 'started', n_groups: 1 })
    stubRealmFetch('demo')
    renderWithRealm(<DatasetsPage />, '/data/qa', 'demo')

    fireEvent.click(await screen.findByText('handbook'))
    fireEvent.click(await screen.findByText('Generate'))
    // Wait for the model options to actually load before picking one —
    // otherwise fireEvent.change on a <select> with no matching <option>
    // yet is a no-op and the button never enables.
    await screen.findByText('qwen3:8b')
    fireEvent.change(screen.getByLabelText(/Model/), { target: { value: 'qwen3:8b' } })
    fireEvent.change(screen.getByLabelText(/Corpus/), { target: { value: 'handbook' } })

    const generateButton = screen.getByRole('button', { name: 'Generate' })
    await waitFor(() => expect(generateButton).toBeEnabled())
    fireEvent.click(generateButton)

    await waitFor(() => expect(generateMock).toHaveBeenCalled())
    expect(generateMock.mock.calls[0][0]).not.toHaveProperty('preset_id')
  })

  it('keeps drafts visible when saving them fails, instead of clearing them', async () => {
    // Found live: an earlier version cleared the draft list immediately
    // after firing the save call, regardless of whether it succeeded — a
    // failed save looked exactly like "the generated questions just
    // disappeared" (nothing was actually persisted).
    listMock.mockResolvedValueOnce([DATASET])
    getMock.mockResolvedValue(DATASET)
    modelsMock.mockResolvedValue([{ name: 'qwen3:8b', size_gb: 5, modified_at: '', capabilities: ['completion'] }])
    corpusCollectionsMock.mockResolvedValue([{ id: 'c1', realm_id: 'demo', corpus_id: 'handbook', storage_type: 'dense_sparse', description: '' }])
    presetsListMock.mockResolvedValue([{ id: 'p1', name: 'Preset 1', description: '', template: '', realm_id: 'demo' }])
    generateMock.mockResolvedValue({ job_id: 'job-1', status: 'started', n_groups: 1 })
    addQuestionsBatchMock.mockRejectedValue(new Error('500 network error'))
    stubRealmFetch('demo')
    renderWithRealm(<DatasetsPage />, '/data/qa', 'demo')

    fireEvent.click(await screen.findByText('handbook'))
    fireEvent.click(await screen.findByText('Generate'))
    await screen.findByText('Preset 1')
    fireEvent.change(screen.getByLabelText(/Model/), { target: { value: 'qwen3:8b' } })
    fireEvent.change(screen.getByLabelText(/Corpus/), { target: { value: 'handbook' } })
    const generateButton = screen.getByRole('button', { name: 'Generate' })
    await waitFor(() => expect(generateButton).toBeEnabled())
    fireEvent.click(generateButton)

    const ws = await lastSocket()
    ws.emit({
      type: 'done',
      drafts: [{ question: 'Q1?', reference_answer: 'A1', question_type: 'closed', article_refs: [], provenance: { origin: 'generated', model: 'qwen3:8b' } }],
      failed: [],
    })
    await screen.findByText('Q1?')

    fireEvent.click(screen.getByText('Save all'))
    await screen.findByText(/Failed to save/)
    expect(screen.getByText('Q1?')).toBeInTheDocument()
  })

  it('shows how many questions have been generated so far while a batch is running, not elapsed seconds', async () => {
    // Found live: an earlier version only showed elapsed seconds during
    // generation, giving no sense of how much of a multi-minute batch was
    // actually done. Progress now comes from generation.py's "progress"
    // events on the job's WebSocket.
    listMock.mockResolvedValueOnce([DATASET])
    getMock.mockResolvedValue(DATASET)
    modelsMock.mockResolvedValue([{ name: 'qwen3:8b', size_gb: 5, modified_at: '', capabilities: ['completion'] }])
    corpusCollectionsMock.mockResolvedValue([{ id: 'c1', realm_id: 'demo', corpus_id: 'handbook', storage_type: 'dense_sparse', description: '' }])
    presetsListMock.mockResolvedValue([{ id: 'p1', name: 'Preset 1', description: '', template: '', realm_id: 'demo' }])
    generateMock.mockResolvedValue({ job_id: 'job-1', status: 'started', n_groups: 3 })
    stubRealmFetch('demo')
    renderWithRealm(<DatasetsPage />, '/data/qa', 'demo')

    fireEvent.click(await screen.findByText('handbook'))
    fireEvent.click(await screen.findByText('Generate'))
    await screen.findByText('Preset 1')
    fireEvent.change(screen.getByLabelText(/Model/), { target: { value: 'qwen3:8b' } })
    fireEvent.change(screen.getByLabelText(/Corpus/), { target: { value: 'handbook' } })
    const generateButton = screen.getByRole('button', { name: 'Generate' })
    await waitFor(() => expect(generateButton).toBeEnabled())
    fireEvent.click(generateButton)

    const ws = await lastSocket()
    ws.emit({ type: 'start', total: 3 })
    ws.emit({ type: 'progress', processed: 1, total: 3, generated: 1, failed: 0 })
    await screen.findByText('Generated 1 of 3')

    ws.emit({ type: 'progress', processed: 2, total: 3, generated: 2, failed: 0 })
    await screen.findByText('Generated 2 of 3')

    expect(screen.queryByText(/\d+ s —/)).not.toBeInTheDocument()
  })

  it('sends a mix of question types with their own counts, not one uniform type', async () => {
    // A batch can mix types (e.g. 2 closed + 3 open) instead of forcing one
    // type for the whole request — generation.py#GenerateQuestionsRequest's
    // `type_counts`. This drives the actual multi-row editor rather than
    // asserting on component state, so a regression in the row add/remove
    // wiring itself would fail this test.
    listMock.mockResolvedValueOnce([DATASET])
    getMock.mockResolvedValue(DATASET)
    modelsMock.mockResolvedValue([{ name: 'qwen3:8b', size_gb: 5, modified_at: '', capabilities: ['completion'] }])
    corpusCollectionsMock.mockResolvedValue([{ id: 'c1', realm_id: 'demo', corpus_id: 'handbook', storage_type: 'dense_sparse', description: '' }])
    presetsListMock.mockResolvedValue([{ id: 'p1', name: 'Preset 1', description: '', template: '', realm_id: 'demo' }])
    generateMock.mockResolvedValue({ job_id: 'job-1', status: 'started', n_groups: 5 })
    stubRealmFetch('demo')
    renderWithRealm(<DatasetsPage />, '/data/qa', 'demo')

    fireEvent.click(await screen.findByText('handbook'))
    fireEvent.click(await screen.findByText('Generate'))
    await screen.findByText('Preset 1')
    fireEvent.change(screen.getByLabelText(/Model/), { target: { value: 'qwen3:8b' } })
    fireEvent.change(screen.getByLabelText(/Corpus/), { target: { value: 'handbook' } })

    const editor = screen.getByTestId('type-counts-editor')
    fireEvent.change(within(editor).getAllByRole('spinbutton')[0], { target: { value: '2' } })
    fireEvent.click(within(editor).getByText('+ add type'))
    const selects = within(editor).getAllByRole('combobox')
    fireEvent.change(selects[1], { target: { value: 'open' } })
    fireEvent.change(within(editor).getAllByRole('spinbutton')[1], { target: { value: '3' } })
    await screen.findByText('Total: 5')

    const generateButton = screen.getByRole('button', { name: 'Generate' })
    await waitFor(() => expect(generateButton).toBeEnabled())
    fireEvent.click(generateButton)

    await waitFor(() => expect(generateMock).toHaveBeenCalledWith(expect.objectContaining({
      type_counts: [
        { question_type: 'closed', n_questions: 2 },
        { question_type: 'open', n_questions: 3 },
      ],
    })))
  })

  it('does not let the same question type be picked twice', async () => {
    // Picking a type already used by another row would just mean two rows
    // silently add up to one type's count — hidden from each row's own
    // <select>, and "+ add type" disables once every known type is used.
    listMock.mockResolvedValueOnce([DATASET])
    getMock.mockResolvedValue(DATASET)
    modelsMock.mockResolvedValue([{ name: 'qwen3:8b', size_gb: 5, modified_at: '', capabilities: ['completion'] }])
    presetsListMock.mockResolvedValue([{ id: 'p1', name: 'Preset 1', description: '', template: '', realm_id: 'demo' }])
    stubRealmFetch('demo')
    renderWithRealm(<DatasetsPage />, '/data/qa', 'demo')

    fireEvent.click(await screen.findByText('handbook'))
    fireEvent.click(await screen.findByText('Generate'))
    await screen.findByText('Preset 1')

    const editor = screen.getByTestId('type-counts-editor')
    const addTypeButton = within(editor).getByText('+ add type')
    // 5 known types total; the first row starts on "closed" — 4 more adds
    // exhaust the list.
    fireEvent.click(addTypeButton)
    fireEvent.click(addTypeButton)
    fireEvent.click(addTypeButton)
    fireEvent.click(addTypeButton)

    expect(within(editor).getAllByRole('combobox')).toHaveLength(5)
    await waitFor(() => expect(addTypeButton).toBeDisabled())

    // Row 0 ("closed") must not offer any type already used by rows 1-4.
    const firstRowOptions = within(editor).getAllByRole('combobox')[0]
      .querySelectorAll('option')
    expect(firstRowOptions).toHaveLength(1)
    expect((firstRowOptions[0] as HTMLOptionElement).value).toBe('closed')
  })
})
