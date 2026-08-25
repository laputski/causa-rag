import { useEffect, useId, useRef, useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { Link, useSearchParams } from 'react-router-dom'
import { useTranslation } from 'react-i18next'
import type { TFunction } from 'i18next'
import { api, Dataset, DatasetDetail, Question, QuestionWrite, referenceAnswerOf } from '../api/client'
import { useRealm, useRealmPath } from '../context/RealmContext'
import { useConfirm } from '../components/ConfirmDialog'
import { CheckCircle2, Database, Sparkles, X } from 'lucide-react'

const KNOWN_QUESTION_TYPES = ['closed', 'open', 'clarifying', 'comparative', 'navigational']
const CUSTOM_TYPE = '__custom__'
const CUSTOM_CORPUS = '__custom_corpus__'

// Same convention as CorpusPage.tsx/ChatPage.tsx: the generation progress
// WebSocket talks straight to the gateway's own host:port rather than
// through Vite's /api rewrite proxy (which isn't configured for the
// WebSocket upgrade — see ui/vite.config.ts).
const API_BASE = import.meta.env.VITE_API_BASE ?? 'http://localhost:8081'
const WS_BASE = API_BASE.replace(/^http/, 'ws')

type Mode = 'view' | 'create-dataset' | 'generate'

export default function DatasetsPage() {
  const { confirm, dialog } = useConfirm()
  const { t } = useTranslation()
  const { activeRealmId } = useRealm()
  const qc = useQueryClient()

  const [selectedFilename, setSelectedFilename] = useState<string | null>(null)
  const [mode, setMode] = useState<Mode>('view')

  const { data: datasets = [], isLoading } = useQuery({
    queryKey: ['datasets', activeRealmId],
    queryFn: () => api.datasets.list(activeRealmId),
  })

  // "Manage dataset" (RunPage.tsx toolbar) deep-links here with
  // ?dataset=<filename> so a reviewer lands directly on the run's own
  // control-question set instead of the plain unfiltered list. Applied
  // once the list has actually loaded (so a match can be found at all) and
  // only once per page load — otherwise picking a different dataset by
  // hand and the list quietly refetching (e.g. after adding a question)
  // would keep snapping the selection back to the URL's dataset.
  const [searchParams] = useSearchParams()
  const deepLinkedDataset = searchParams.get('dataset')
  const didApplyDeepLinkRef = useRef(false)
  useEffect(() => {
    if (!deepLinkedDataset || didApplyDeepLinkRef.current || datasets.length === 0) return
    if (datasets.some(d => d.filename === deepLinkedDataset)) {
      didApplyDeepLinkRef.current = true
      setSelectedFilename(deepLinkedDataset)
    }
  }, [deepLinkedDataset, datasets])

  // Without this the right column arrives empty although there is something
  // to choose. The largest dataset opens: that is usually the one runs use,
  // and the small ones beside it are drafts and trials.
  useEffect(() => {
    if (selectedFilename || deepLinkedDataset || datasets.length === 0) return
    if (mode === 'create-dataset') return
    const biggest = [...datasets].sort((a, b) => (b.count ?? 0) - (a.count ?? 0))[0]
    setSelectedFilename(biggest.filename)
  }, [datasets, selectedFilename, deepLinkedDataset, mode])

  const { data: detail } = useQuery({
    queryKey: ['dataset', selectedFilename],
    queryFn: () => api.datasets.get(selectedFilename!),
    enabled: !!selectedFilename,
  })

  const setDetail = (d: DatasetDetail) => qc.setQueryData(['dataset', selectedFilename], d)

  const createDatasetMut = useMutation({
    mutationFn: (form: { name: string; version: string; speed: string }) =>
      api.datasets.create({
        filename: `${form.name}.${form.version}.${form.speed}.jsonl`,
        realm_id: activeRealmId || '',
        questions: [],
      }),
    onSuccess: (d: Dataset) => {
      qc.invalidateQueries({ queryKey: ['datasets'] })
      setSelectedFilename(d.filename)
      setMode('view')
    },
  })

  const deleteDatasetMut = useMutation({
    mutationFn: () => api.datasets.delete(detail!.id!, activeRealmId),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['datasets'] })
      setSelectedFilename(null)
      setMode('view')
    },
  })

  const addQuestionMut = useMutation({
    mutationFn: (q: QuestionWrite) => api.datasets.addQuestion(detail!.id!, q, activeRealmId),
    onSuccess: setDetail,
  })
  const updateQuestionMut = useMutation({
    mutationFn: ({ questionId, q }: { questionId: string; q: QuestionWrite }) =>
      api.datasets.updateQuestion(detail!.id!, questionId, q, activeRealmId),
    onSuccess: setDetail,
  })
  const deleteQuestionMut = useMutation({
    mutationFn: (questionId: string) => api.datasets.deleteQuestion(detail!.id!, questionId, activeRealmId),
    onSuccess: setDetail,
  })
  const addBatchMut = useMutation({
    mutationFn: (qs: QuestionWrite[]) => api.datasets.addQuestionsBatch(detail!.id!, qs, activeRealmId),
    onSuccess: (d: DatasetDetail) => {
      setDetail(d)
      // The sidebar's own "N questions" count comes from the separate
      // ['datasets', realm] list query, not this dataset's detail query —
      // without this it kept showing the pre-generation count until the
      // user navigated away and back (found live: looked like the saved
      // drafts hadn't actually been added anywhere).
      qc.invalidateQueries({ queryKey: ['datasets'] })
    },
  })

  return (
    <div className="page-flush split">
      {dialog}
      <aside className="split-aside">
        <div className="split-aside-head">
          <h1 className="split-title">{t('datasetsPage.title')}</h1>
          <button
            className="btn btn-sm btn-primary"
            onClick={() => { setSelectedFilename(null); setMode('create-dataset') }}
          >
            {t('datasetsPage.newSet')}
          </button>
        </div>

        {isLoading ? (
          <p className="split-empty">{t('datasetsPage.loading')}</p>
        ) : datasets.length === 0 ? (
          <p className="split-empty">{t('datasetsPage.noSets')}</p>
        ) : (
          <ul className="pick-list">
            {datasets.map(d => {
              const isSelected = selectedFilename === d.filename && mode !== 'create-dataset'
              return (
                <li key={d.filename}>
                  <button
                    type="button"
                    className={`pick-row${isSelected ? ' active' : ''}`}
                    onClick={() => { setSelectedFilename(d.filename); setMode('view') }}
                  >
                    <div className="pick-row-title">{d.name}</div>
                    <div className="pick-row-meta">
                      <span className="pick-ver">{d.version}</span>
                      <span className="pick-row-sub">{t('datasetsPage.questionsCount', { count: d.count ?? 0 })}</span>
                      <span className={`badge ${d.speed === 'fast' ? 'badge-success' : 'badge-warn'}`}>{d.speed}</span>
                    </div>
                  </button>
                </li>
              )
            })}
          </ul>
        )}
      </aside>

      <main className="split-main">
        {mode === 'create-dataset' && (
          <CreateDatasetForm
            onSave={f => createDatasetMut.mutate(f)}
            onCancel={() => setMode('view')}
            saving={createDatasetMut.isPending}
          />
        )}

        {mode !== 'create-dataset' && !detail && (
          <p className="split-empty split-empty-lg">{t('datasetsPage.selectOrCreate')}</p>
        )}

        {mode !== 'create-dataset' && detail && (
          <DatasetDetailView
            detail={detail}
            mode={mode}
            onModeChange={setMode}
            realmId={activeRealmId ?? null}
            onAddQuestion={q => addQuestionMut.mutate(q)}
            onUpdateQuestion={(id, q) => updateQuestionMut.mutate({ questionId: id, q })}
            onDeleteQuestion={async id => { if (await confirm({ title: t('datasetsPage.confirmDeleteQuestion'), danger: true })) deleteQuestionMut.mutate(id) }}
            onSaveDrafts={qs => addBatchMut.mutateAsync(qs)}
            onDeleteDataset={async () => {
              if (await confirm({ title: t('datasetsPage.confirmDeleteSet', { name: detail.name, count: detail.count ?? 0 }), danger: true })) {
                deleteDatasetMut.mutate()
              }
            }}
            deletingDataset={deleteDatasetMut.isPending}
          />
        )}
      </main>
    </div>
  )
}

// ── Create dataset ────────────────────────────────────────────────────────────

function CreateDatasetForm({ onSave, onCancel, saving }: {
  onSave: (f: { name: string; version: string; speed: string }) => void
  onCancel: () => void
  saving: boolean
}) {
  const { t } = useTranslation()
  const [name, setName] = useState('')
  const [version, setVersion] = useState('v0')
  const [speed, setSpeed] = useState<'fast' | 'full'>('fast')
  const valid = name.trim().length > 0

  return (
    <div className="ds-new">
      <div className="ds-new-head">
        <h1 className="ds-new-title">{t('datasetsPage.newSet')}</h1>
        <div className="ds-new-actions">
          <button className="btn" onClick={onCancel}>{t('datasetsPage.cancel')}</button>
          <button className="btn btn-primary" disabled={!valid || saving} onClick={() => onSave({ name: name.trim(), version, speed })}>
            {saving ? t('datasetsPage.creating') : t('datasetsPage.create')}
          </button>
        </div>
      </div>
      <div className="card ds-new-form">
        <FieldLabel htmlFor="ds-name" required>{t('datasetsPage.nameLabel')}</FieldLabel>
        <input id="ds-name" className="input" value={name} onChange={e => setName(e.target.value)} placeholder="handbook" autoFocus />

        <div className="ds-new-pair">
          <div>
            <FieldLabel htmlFor="ds-version">{t('datasetsPage.versionLabel')}</FieldLabel>
            <input id="ds-version" className="input" value={version} onChange={e => setVersion(e.target.value)} placeholder="v0" />
          </div>
          <div>
            <FieldLabel htmlFor="ds-speed">{t('datasetsPage.speedLabel')}</FieldLabel>
            <select id="ds-speed" className="input" value={speed} onChange={e => setSpeed(e.target.value as 'fast' | 'full')}>
              <option value="fast">fast</option>
              <option value="full">full</option>
            </select>
          </div>
        </div>

        <p className="ds-new-file">
          {t('datasetsPage.fileLabel')} <code className="inline-code">
            {name.trim() || '...'}.{version}.{speed}.jsonl
          </code>
        </p>
      </div>
    </div>
  )
}

function FieldLabel({ htmlFor, required, children }: { htmlFor: string; required?: boolean; children: React.ReactNode }) {
  return (
    <label htmlFor={htmlFor}>
      {children} {required && <span className="req">*</span>}
    </label>
  )
}

// ── Dataset detail: questions + generator ─────────────────────────────────────

function DatasetDetailView({
  detail, mode, onModeChange, realmId, onAddQuestion, onUpdateQuestion, onDeleteQuestion, onSaveDrafts,
  onDeleteDataset, deletingDataset,
}: {
  detail: DatasetDetail
  mode: Mode
  onModeChange: (m: Mode) => void
  realmId: string | null
  onAddQuestion: (q: QuestionWrite) => void
  onUpdateQuestion: (id: string, q: QuestionWrite) => void
  onDeleteQuestion: (id: string) => void
  onSaveDrafts: (qs: QuestionWrite[]) => Promise<unknown>
  onDeleteDataset: () => void
  deletingDataset: boolean
}) {
  const { t } = useTranslation()
  const [adding, setAdding] = useState(false)
  // Surfaced after a successful GeneratePanel save: drafts are cleared from
  // the generator (they're saved now, not lost) and the view switches to
  // "Questions" — but the switch alone still looks like the questions just
  // vanished from the generator panel, so this banner points at where they
  // actually landed (found live).
  const [justSaved, setJustSaved] = useState<number | null>(null)
  const [search, setSearch] = useState('')
  const [filter, setFilter] = useState<'all' | 'answered' | 'unlinked'>('all')

  // A question with no source references is one recall cannot be computed
  // from. The "no references" chip says how many the dataset holds, before a
  // run shows an empty metric.
  const ordered = sortedByProvenance(detail.questions)
  const counts = {
    all: ordered.length,
    answered: ordered.filter(q => referenceAnswerOf(q).trim().length > 0).length,
    unlinked: ordered.filter(q => (q.article_refs ?? []).length === 0).length,
  }
  const needle = search.trim().toLowerCase()
  const visible = ordered.filter(q => {
    if (filter === 'answered' && !referenceAnswerOf(q).trim()) return false
    if (filter === 'unlinked' && (q.article_refs ?? []).length > 0) return false
    if (!needle) return true
    return `${q.question} ${referenceAnswerOf(q)}`.toLowerCase().includes(needle)
  })

  useEffect(() => {
    if (justSaved === null) return
    const id = setTimeout(() => setJustSaved(null), 8000)
    return () => clearTimeout(id)
  }, [justSaved])

  return (
    <div>
      <div className="flex-between ds-head">
        <div>
          <h1 className="page-title tight">{detail.name}</h1>
          <p className="jd-provenance">
            <code className="inline-code">{detail.version}</code>
            <span>{t('datasetsPage.questionsCount', { count: detail.count ?? 0 })}</span>
            <span>{detail.speed}</span>
          </p>
        </div>
        <div className="detail-actions">
          <TabSwitch mode={mode} onModeChange={onModeChange} />
          <button className="btn btn-sm btn-danger" onClick={onDeleteDataset} disabled={deletingDataset}>
            {deletingDataset ? '...' : t('datasetsPage.deleteSet')}
          </button>
        </div>
      </div>

      {mode === 'view' && (
        <>
          {justSaved !== null && (
            <div className="save-success-banner">
              <CheckCircle2 size={16} />
              <span>
                {t('datasetsPage.savedBanner', { count: justSaved })}
              </span>
              <button className="icon-btn push" onClick={() => setJustSaved(null)} aria-label={t('datasetsPage.dismiss')}>
                <X size={14} />
              </button>
            </div>
          )}
          <div className="table-toolbar">
            <input
              className="toolbar-search" type="search" value={search}
              onChange={e => setSearch(e.target.value)}
              placeholder={t('datasetsPage.searchPlaceholder')}
              aria-label={t('datasetsPage.searchPlaceholder')}
            />
            <div className="chips">
              {(['all', 'answered', 'unlinked'] as const).map(f => (
                <button
                  key={f} type="button"
                  className={`chip${filter === f ? ' active' : ''}`}
                  onClick={() => setFilter(f)}
                >
                  {t(`datasetsPage.filter.${f}`)}
                  <span className="chip-count">{counts[f]}</span>
                </button>
              ))}
            </div>
            {!adding && (
              <button className="btn btn-sm btn-primary push" onClick={() => setAdding(true)}>
                {t('datasetsPage.addQuestion')}
              </button>
            )}
          </div>

          {adding && (
            <QuestionForm
              onSave={q => { onAddQuestion(q); setAdding(false) }}
              onCancel={() => setAdding(false)}
            />
          )}

          {visible.length === 0 ? (
            <p className="empty tight">
              {detail.questions.length === 0 ? t('datasetsPage.emptySet') : t('datasetsPage.noMatches')}
            </p>
          ) : (
            <table className="q-table">
              <thead>
                <tr>
                  <th>{t('datasetsPage.col.question')}</th>
                  <th>{t('datasetsPage.col.refs')}</th>
                  <th>{t('datasetsPage.col.origin')}</th>
                  <th />
                </tr>
              </thead>
              <tbody>
                {visible.map(q => (
                  <QuestionRow
                    key={q.id} question={q}
                    onSave={upd => onUpdateQuestion(q.id, upd)}
                    onDelete={() => onDeleteQuestion(q.id)}
                  />
                ))}
              </tbody>
            </table>
          )}
        </>
      )}

      {mode === 'generate' && (
        <GeneratePanel
          realmId={realmId}
          existingCount={detail.count ?? 0}
          onSaveDrafts={onSaveDrafts}
          onSaved={count => { setJustSaved(count); onModeChange('view') }}
        />
      )}
    </div>
  )
}

function TabSwitch({ mode, onModeChange }: { mode: Mode; onModeChange: (m: Mode) => void }) {
  const { t } = useTranslation()
  const tabs: { key: Mode; label: string }[] = [
    { key: 'view', label: t('datasetsPage.tabQuestions') },
    { key: 'generate', label: t('datasetsPage.tabGenerate') },
  ]
  return (
    <div className="chips" role="tablist">
      {tabs.map(tab => (
        <button
          key={tab.key} role="tab" type="button"
          aria-selected={mode === tab.key}
          className={`chip${mode === tab.key ? ' active' : ''}`}
          onClick={() => onModeChange(tab.key)}
        >
          {tab.label}
        </button>
      ))}
    </div>
  )
}

// ── One question — view / inline edit ─────────────────────────────────────────

// Reviewer-promoted questions surface first — a reviewer who just promoted
// a question (or is about to promote a duplicate) needs to see it without
// scrolling past everything else the dataset already had; stable otherwise
// (ties keep their original relative order).
function sortedByProvenance(questions: Question[]): Question[] {
  const priority = (q: Question) => (q.provenance?.origin === 'reviewer_feedback' ? 0 : 1)
  return questions
    .map((q, idx) => ({ q, idx }))
    .sort((a, b) => priority(a.q) - priority(b.q) || a.idx - b.idx)
    .map(({ q }) => q)
}

// Plain muted gray made every origin equally (in)visible — a reviewer
// scanning the list couldn't tell a curator-typed question from an
// LLM-drafted one from a reviewer-confirmed one without reading the text.
function provenanceColor(q: Question): string {
  const origin = q.provenance?.origin
  if (origin === 'reviewer_feedback') return 'var(--color-warning)'
  if (origin === 'generated') return 'var(--color-primary)'
  return 'var(--color-text-muted)'
}

function provenanceLabel(q: Question, t: TFunction): string {
  const p = q.provenance
  if (!p) return t('datasetsPage.provenance.manual')
  if (p.origin === 'generated') {
    const base = t('datasetsPage.provenance.generated', { model: p.model ?? '?', corpusId: p.corpus_id ?? '?' })
    return p.edited_manually ? t('datasetsPage.provenance.editedSuffix', { base }) : base
  }
  if (p.origin === 'reviewer_feedback') {
    const base = t('datasetsPage.provenance.reviewerFeedback', { runId: p.source_run_id ?? '?' })
    return p.previous_origin
      ? t('datasetsPage.provenance.reviewerFeedbackConfirmed', { base, previousOrigin: p.previous_origin })
      : base
  }
  return p.updated_at ? t('datasetsPage.provenance.manualEdited') : t('datasetsPage.provenance.manual')
}

// A golden question promoted from reviewer feedback
// links back to the run it came from (provenanceLabel above only renders
// text; the run_id needs to be a real link, so this stays a separate
// element rather than folding into that function's plain-string return).
function ProvenanceSourceLink({ question }: { question: Question }) {
  const toRealm = useRealmPath()
  const p = question.provenance
  if (p?.origin !== 'reviewer_feedback' || !p.source_run_id) return null
  return (
    <Link to={toRealm(`/experiments/${p.source_run_id}`)} className="q-ref">
      →
    </Link>
  )
}

function QuestionRow({ question, onSave, onDelete }: {
  question: Question
  onSave: (q: QuestionWrite) => void
  onDelete: () => void
}) {
  const { t } = useTranslation()
  const [editing, setEditing] = useState(false)

  // Editing expands into a row spanning the whole table: the form is wider
  // than any single column, and table markup does not tolerate a `div` in
  // `tbody`.
  if (editing) {
    return (
      <tr>
        <td colSpan={4} className="q-edit-cell">
          <QuestionForm
            initial={question}
            onSave={q => { onSave(q); setEditing(false) }}
            onCancel={() => setEditing(false)}
          />
        </td>
      </tr>
    )
  }

  return (
    <tr>
      <td className="q-text">
        <div className="q-question">{question.question}</div>
        {/* The reference answer on a second line, muted: it belongs beside the
            question, but the question is what eyes scan the list for. */}
        <div className="q-answer">{referenceAnswerOf(question)}</div>
        {question.question_type && <span className="badge badge-info">{question.question_type}</span>}
      </td>
      <td className="q-refs">
        {(question.article_refs ?? []).length > 0
          ? (question.article_refs ?? []).map(ref => <span key={ref} className="badge">{ref}</span>)
          : <span className="q-noref">{t('datasetsPage.noRefs')}</span>}
      </td>
      <td className="q-origin" style={{ color: provenanceColor(question) }}>
        {provenanceLabel(question, t)}
        <ProvenanceSourceLink question={question} />
      </td>
      <td className="q-actions">
        <button className="btn btn-sm" onClick={() => setEditing(true)}>{t('datasetsPage.edit')}</button>
        <button className="btn btn-sm btn-danger" onClick={onDelete}>{t('datasetsPage.delete')}</button>
      </td>
    </tr>
  )
}

function QuestionForm({ initial, onSave, onCancel }: {
  initial?: Partial<Question>
  onSave: (q: QuestionWrite) => void
  onCancel: () => void
}) {
  const { t } = useTranslation()
  const uid = useId()
  const [question, setQuestion] = useState(initial?.question ?? '')
  const [referenceAnswer, setReferenceAnswer] = useState(referenceAnswerOf(initial))
  const [questionType, setQuestionType] = useState(initial?.question_type ?? '')
  const [articleRefs, setArticleRefs] = useState((initial?.article_refs ?? []).join(', '))
  const valid = question.trim().length > 0 && referenceAnswer.trim().length > 0

  const handleSave = () => {
    onSave({
      question: question.trim(),
      reference_answer: referenceAnswer.trim(),
      question_type: questionType.trim(),
      article_refs: articleRefs.split(',').map(s => s.trim()).filter(Boolean),
    })
  }

  return (
    <div className="edit-form">
      <div className="edit-form-head">
        <h3 className="edit-form-title">
          {initial?.id ? t('datasetsPage.editQuestionTitle') : t('datasetsPage.newQuestionTitle')}
        </h3>
        <div className="detail-actions">
          <button className="btn btn-sm" onClick={onCancel}>{t('datasetsPage.cancel')}</button>
          <button className="btn btn-sm btn-primary" disabled={!valid} onClick={handleSave}>
            {t('datasetsPage.save')}
          </button>
        </div>
      </div>

      <div className="edit-grid">
        <div className="form-group form-span">
          <label htmlFor={`${uid}-question`}>
            {t('datasetsPage.questionLabel')} <span className="req-mark">*</span>
          </label>
          <textarea
            id={`${uid}-question`} className="edit-area"
            value={question} onChange={e => setQuestion(e.target.value)} autoFocus
            placeholder={t('datasetsPage.questionPlaceholder')}
          />
        </div>
        <div className="form-group form-span">
          <label htmlFor={`${uid}-answer`}>
            {t('datasetsPage.referenceAnswerLabel')} <span className="req-mark">*</span>
          </label>
          <textarea
            id={`${uid}-answer`} className="edit-area"
            value={referenceAnswer} onChange={e => setReferenceAnswer(e.target.value)}
            placeholder={t('datasetsPage.referenceAnswerPlaceholder')}
          />
          <p className="hint-line">{t('datasetsPage.referenceAnswerHint')}</p>
        </div>
        <div className="form-group">
          <label htmlFor={`${uid}-type`}>{t('datasetsPage.questionTypeLabel')}</label>
          <QuestionTypeSelect id={`${uid}-type`} value={questionType} onChange={setQuestionType} />
        </div>
        <div className="form-group">
          <label htmlFor={`${uid}-refs`}>{t('datasetsPage.articleRefsLabel')}</label>
          <input
            id={`${uid}-refs`} value={articleRefs}
            onChange={e => setArticleRefs(e.target.value)} placeholder="DOC0001/5, DOC0001/47"
          />
          {/* A question with no references takes no part in recall. Saying so
              here costs less than showing an empty metric after a run. */}
          <p className="hint-line">{t('datasetsPage.articleRefsHint')}</p>
        </div>
      </div>
    </div>
  )
}

function QuestionTypeSelect({ id, value, onChange }: { id: string; value: string; onChange: (v: string) => void }) {
  const { t } = useTranslation()
  const isKnown = KNOWN_QUESTION_TYPES.includes(value)
  const [customMode, setCustomMode] = useState(value !== '' && !isKnown)

  if (customMode) {
    return (
      <div className="field-with-btn">
        <input
          id={id} className="input" value={value}
          onChange={e => onChange(e.target.value)} placeholder={t('datasetsPage.customType')} autoFocus
        />
        <button
          type="button" className="btn-sm"
          onClick={() => { setCustomMode(false); onChange('') }}
        >
          {t('datasetsPage.fromList')}
        </button>
      </div>
    )
  }

  return (
    <select
      id={id} className="input" value={isKnown ? value : ''}
      onChange={e => {
        if (e.target.value === CUSTOM_TYPE) { setCustomMode(true); onChange('') }
        else onChange(e.target.value)
      }}
    >
      <option value="" disabled>{t('datasetsPage.selectPrompt')}</option>
      {KNOWN_QUESTION_TYPES.map(qt => <option key={qt} value={qt}>{qt}</option>)}
      <option value={CUSTOM_TYPE}>{t('datasetsPage.customTypeOption')}</option>
    </select>
  )
}

type TypeCount = { question_type: string; n_questions: number }

/** One row per question type + how many of that type to generate in this
 * batch (e.g. 5 closed + 3 open + 2 clarifying) — replaces a single
 * type+count pair so a batch isn't forced to be one uniform type
 * (services/api_gateway/routers/generation.py#GenerateQuestionsRequest's
 * `type_counts`, sampled/generated independently per entry then merged). */
function TypeCountsEditor({ value, onChange, disabled }: {
  value: TypeCount[]
  onChange: (v: TypeCount[]) => void
  disabled?: boolean
}) {
  const { t } = useTranslation()
  const total = value.reduce((sum, tc) => sum + tc.n_questions, 0)

  const update = (i: number, patch: Partial<TypeCount>) =>
    onChange(value.map((tc, idx) => (idx === i ? { ...tc, ...patch } : tc)))

  const usedTypes = new Set(value.map(tc => tc.question_type))
  const allTypesUsed = usedTypes.size >= KNOWN_QUESTION_TYPES.length

  const addRow = () => {
    const nextType = KNOWN_QUESTION_TYPES.find(qt => !usedTypes.has(qt))
    if (!nextType) return
    onChange([...value, { question_type: nextType, n_questions: 1 }])
  }

  const removeRow = (i: number) => onChange(value.filter((_, idx) => idx !== i))

  return (
    <div data-testid="type-counts-editor">
      {value.map((tc, i) => {
        // A type already picked by another row is hidden from this row's
        // options (each type can only appear once — picking "closed" twice
        // instead of just raising its count doesn't mean anything extra).
        const otherUsed = new Set(value.filter((_, idx) => idx !== i).map(o => o.question_type))
        const optionsForRow = KNOWN_QUESTION_TYPES.filter(qt => qt === tc.question_type || !otherUsed.has(qt))
        return (
          <div key={i} className="tc-row">
            <select
              className="input" value={tc.question_type} disabled={disabled}
              onChange={e => update(i, { question_type: e.target.value })}
            >
              {optionsForRow.map(qt => <option key={qt} value={qt}>{qt}</option>)}
            </select>
            <input
              type="number" className="input tc-count" min={1} max={50}
              value={tc.n_questions} disabled={disabled}
              onChange={e => update(i, { n_questions: Math.max(1, Math.min(50, Number(e.target.value) || 1)) })}
            />
            <button
              type="button" className="btn-sm"
              disabled={disabled || value.length <= 1} onClick={() => removeRow(i)}
              aria-label={t('datasetsPage.generator.removeType')}
            >
              <X size={14} />
            </button>
          </div>
        )
      })}
      <div className="flex-between">
        <button type="button" className="btn-sm" disabled={disabled || allTypesUsed} onClick={addRow}>
          + {t('datasetsPage.generator.addType')}
        </button>
        <span className="field-note">
          {t('datasetsPage.generator.totalLabel', { count: total })}
        </span>
      </div>
    </div>
  )
}

/** Real `<select>` of the Realm's known corpora + a "custom id" escape hatch —
 * same shape as QuestionTypeSelect above, replacing a free-text input with a
 * `<datalist>` that reads as a plain text box rather than an intentional
 * dropdown (its suggestions only show once you start typing). */
function CorpusSelect({ id, value, onChange, collections, disabled }: {
  id: string
  value: string
  onChange: (v: string) => void
  collections: { corpus_id: string; description?: string }[]
  disabled?: boolean
}) {
  const { t } = useTranslation()
  const knownIds = collections.map(c => c.corpus_id)
  const isKnown = knownIds.includes(value)
  const [customMode, setCustomMode] = useState(value !== '' && !isKnown)

  if (customMode) {
    return (
      <div className="field-with-btn">
        <input
          id={id} className="input" value={value} disabled={disabled}
          onChange={e => onChange(e.target.value)} placeholder="corpus_id" autoFocus
        />
        <button
          type="button" className="btn-sm" disabled={disabled}
          onClick={() => { setCustomMode(false); onChange('') }}
        >
          {t('datasetsPage.fromList')}
        </button>
      </div>
    )
  }

  return (
    <div className="gen-field-icon-wrap">
      <Database size={14} className="gen-field-icon" />
      <select
        id={id} className="input" value={isKnown ? value : ''} disabled={disabled}
        onChange={e => {
          if (e.target.value === CUSTOM_CORPUS) { setCustomMode(true); onChange('') }
          else onChange(e.target.value)
        }}
      >
        <option value="" disabled>{t('datasetsPage.selectCorpusPrompt')}</option>
        {collections.map(c => (
          <option key={c.corpus_id} value={c.corpus_id}>
            {c.corpus_id}{c.description ? ` — ${c.description}` : ''}
          </option>
        ))}
        <option value={CUSTOM_CORPUS}>{t('datasetsPage.customCorpusOption')}</option>
      </select>
    </div>
  )
}

// ── Generator ──────────────────────────────────────────────────────────────────

type Draft = QuestionWrite & { _key: string }

type GenProgressEvent =
  | { type: 'start'; total: number }
  | { type: 'progress'; processed: number; total: number; generated: number; failed: number }
  | { type: 'done'; drafts: (QuestionWrite & { id?: string })[]; failed: { reason: string; chunk_ids: string[] }[] }
  | { type: 'error'; message?: string }

/** Follows a generation job's progress WebSocket
 * (generation.py#generate_questions_progress) and reports how many
 * questions have actually been generated so far — replaces an earlier plain
 * elapsed-seconds timer that gave no sense of how much of a multi-minute
 * batch was actually done (found live). Same WS_BASE-not-/api-proxy
 * convention as CorpusPage.tsx's ingestion progress socket (see top of this
 * file for why). */
function useGenerationProgress(
  jobId: string | null,
  onDone: (drafts: (QuestionWrite & { id?: string })[], failed: { reason: string; chunk_ids: string[] }[]) => void,
) {
  const { t } = useTranslation()
  const [event, setEvent] = useState<GenProgressEvent | null>(null)
  const [wsError, setWsError] = useState<string | null>(null)

  useEffect(() => {
    setEvent(null)
    setWsError(null)
    if (!jobId) return

    const ws = new WebSocket(`${WS_BASE}/generate/questions/${jobId}/progress`)
    ws.onmessage = (e) => {
      const ev: GenProgressEvent = JSON.parse(e.data)
      setEvent(ev)
      if (ev.type === 'done') { onDone(ev.drafts, ev.failed); ws.close() }
      if (ev.type === 'error') { setWsError(ev.message ?? t('datasetsPage.generationError')); ws.close() }
    }
    ws.onerror = () => setWsError(t('datasetsPage.connectionLost'))
    return () => ws.close()
    // eslint-disable-next-line react-hooks/exhaustive-deps -- onDone/t are re-created each render, only jobId should re-run this
  }, [jobId])

  return { event, wsError }
}

function GeneratePanel({ realmId, existingCount, onSaveDrafts, onSaved }: {
  realmId: string | null
  existingCount: number
  onSaveDrafts: (qs: QuestionWrite[]) => Promise<unknown>
  onSaved: (count: number) => void
}) {
  const { t } = useTranslation()
  const { data: models = [] } = useQuery({
    queryKey: ['models', 'completion'],
    queryFn: () => api.models(true),
  })
  const { data: corpusCollections = [] } = useQuery({
    queryKey: ['corpus-collections', realmId],
    queryFn: () => api.corpus.collections(realmId),
  })
  const { data: presets = [] } = useQuery({
    queryKey: ['generation-presets', realmId],
    queryFn: () => api.generationPresets.list(realmId),
    enabled: !!realmId,
  })

  const [model, setModel] = useState('')
  const [corpusId, setCorpusId] = useState('')
  const [typeCounts, setTypeCounts] = useState<TypeCount[]>([{ question_type: 'closed', n_questions: 10 }])
  const [presetId, setPresetId] = useState('')
  const [drafts, setDrafts] = useState<Draft[]>([])
  const [failedCount, setFailedCount] = useState(0)
  const [saveError, setSaveError] = useState<string | null>(null)
  const [saving, setSaving] = useState(false)
  const [jobId, setJobId] = useState<string | null>(null)
  const [genError, setGenError] = useState<string | null>(null)

  const { event: progress, wsError } = useGenerationProgress(jobId, (newDrafts, failed) => {
    setDrafts(newDrafts.map((d, i) => ({ ...d, _key: `draft-${Date.now()}-${i}` })))
    setFailedCount(failed.length)
    setJobId(null)
  })

  useEffect(() => {
    if (wsError) { setGenError(wsError); setJobId(null) }
  }, [wsError])

  const generateMut = useMutation({
    // preset_id is optional — an empty selection means "no preset",
    // omitted from the request rather than sent as "" so the backend's
    // `if body.preset_id:` check (generation.py#generate_questions) takes
    // the built-in-template fallback path instead of a 404-on-empty-string.
    mutationFn: () => api.generate.questions({
      realm_id: realmId || '', corpus_id: corpusId, model,
      ...(presetId ? { preset_id: presetId } : {}),
      type_counts: typeCounts,
    }),
    onSuccess: (res) => {
      setGenError(null)
      setSaveError(null)
      setFailedCount(0)
      setJobId(res.job_id)
    },
    onError: (e) => setGenError((e as Error).message),
  })

  const isGenerating = generateMut.isPending || jobId !== null
  const canGenerate = !!model && !!corpusId && typeCounts.length > 0 && !isGenerating

  const generated = progress?.type === 'progress' ? progress.generated : 0
  const processed = progress?.type === 'progress' ? progress.processed : 0
  const requestedTotal = typeCounts.reduce((sum, tc) => sum + tc.n_questions, 0)
  const total = progress && (progress.type === 'start' || progress.type === 'progress') ? progress.total : requestedTotal
  const progressPct = total ? Math.round((processed / total) * 100) : 0

  // Drafts are only cleared once the save actually succeeds — found live: an
  // earlier version cleared them unconditionally right after firing the
  // save call, so a failed save looked exactly like "the questions just
  // disappeared" (they were never persisted, and the review copy was gone).
  // onSaved() lets the parent switch to "Questions" and point at where the
  // saved questions actually landed (found live: switching tabs silently
  // still read as "they vanished").
  const handleSaveAll = async () => {
    setSaving(true)
    setSaveError(null)
    const savedCount = drafts.length
    try {
      await onSaveDrafts(drafts.map(({ _key, ...q }) => q))
      setDrafts([])
      onSaved(savedCount)
    } catch (e) {
      setSaveError((e as Error).message)
    } finally {
      setSaving(false)
    }
  }

  return (
    <div>
      <div className="section">
        <div className="gen-section-title"><Sparkles size={16} /> {t('datasetsPage.generator.title')}</div>
        <p className="gen-section-sub">
          {t('datasetsPage.generator.subtitle')}
        </p>

        <div className="gen-pair">
          <div>
            <FieldLabel htmlFor="gen-model" required>{t('datasetsPage.generator.modelLabel')}</FieldLabel>
            <select id="gen-model" className="input" value={model} onChange={e => setModel(e.target.value)} disabled={isGenerating}>
              <option value="">{t('datasetsPage.selectPrompt')}</option>
              {models.map(m => <option key={m.name} value={m.name}>{m.name}</option>)}
            </select>
          </div>
          <div>
            <FieldLabel htmlFor="gen-corpus" required>{t('datasetsPage.generator.corpusLabel')}</FieldLabel>
            <CorpusSelect id="gen-corpus" value={corpusId} onChange={setCorpusId} collections={corpusCollections} disabled={isGenerating} />
          </div>
          <div>
            <FieldLabel htmlFor="gen-preset">{t('datasetsPage.generator.presetLabel')}</FieldLabel>
            <select id="gen-preset" className="input" value={presetId} onChange={e => setPresetId(e.target.value)} disabled={isGenerating}>
              <option value="">{t('datasetsPage.generator.noPreset')}</option>
              {presets.map(p => <option key={p.id} value={p.id}>{p.name}</option>)}
            </select>
          </div>
        </div>

        <div className="gen-block">
          <FieldLabel htmlFor="gen-type-counts">{t('datasetsPage.generator.typeCountsLabel')}</FieldLabel>
          <TypeCountsEditor value={typeCounts} onChange={setTypeCounts} disabled={isGenerating} />
        </div>

        <div className="gen-actions">
          <button className="btn btn-primary" disabled={!canGenerate} onClick={() => generateMut.mutate()}>
            <Sparkles size={14} /> {isGenerating ? t('datasetsPage.generator.generating') : t('datasetsPage.generator.generate')}
          </button>
        </div>

        {isGenerating && (
          <div className="gen-progress">
            <div className="gen-progress-head">
              <span className="flex-inline">
                <Spinner /> {t('datasetsPage.generator.generatingQuestions')}
              </span>
              <span className="gen-progress-count">{t('datasetsPage.generator.progressCount', { generated, total })}</span>
            </div>
            <div className="progress-bar"><div className="progress-fill" style={{ width: `${progressPct}%` }} /></div>
            <p className="field-note gen-note">
              {t('datasetsPage.generator.progressHint')}
            </p>
          </div>
        )}

        {genError && (
          <p className="form-error gen-block">
            {t('datasetsPage.generator.errorPrefix')} {genError}
          </p>
        )}
        {!isGenerating && failedCount > 0 && (
          <p className="gen-hint">
            {t('datasetsPage.generator.failedCount', { count: failedCount })}
          </p>
        )}
      </div>

      {drafts.length > 0 && (
        <>
          <div className="flex-between drafts-head">
            <div>
              <h3 className="section-title">{t('datasetsPage.generator.draftsHeading', { count: drafts.length })}</h3>
              <p className="field-note">
                {t('datasetsPage.generator.draftsHint', { count: existingCount })}
              </p>
            </div>
            <button className="btn btn-primary" disabled={saving} onClick={handleSaveAll}>
              {saving ? t('datasetsPage.generator.savingAll') : t('datasetsPage.generator.saveAll')}
            </button>
          </div>
          {saveError && (
            <p className="form-error">
              {t('datasetsPage.generator.saveFailed', { error: saveError })}
            </p>
          )}
          <div className="draft-list">
            {drafts.map(d => (
              <DraftRow
                key={d._key}
                draft={d}
                onChange={upd => setDrafts(prev => prev.map(x => x._key === d._key ? { ...upd, _key: d._key } : x))}
                onRemove={() => setDrafts(prev => prev.filter(x => x._key !== d._key))}
              />
            ))}
          </div>
        </>
      )}
    </div>
  )
}

function Spinner() {
  return (
    <span className="spinner" aria-hidden="true" />
  )
}

function DraftRow({ draft, onChange, onRemove }: {
  draft: Draft
  onChange: (d: QuestionWrite) => void
  onRemove: () => void
}) {
  const { t } = useTranslation()
  return (
    <div className="card draft-card">
      <div className="draft-row">
        <div className="draft-body">
          <textarea
            className="input draft-area strong"
            value={draft.question}
            onChange={e => onChange({ ...draft, question: e.target.value })}
          />
          <textarea
            className="input draft-area"
            value={draft.reference_answer}
            onChange={e => onChange({ ...draft, reference_answer: e.target.value })}
          />
          <div className="draft-meta">
            {draft.question_type && <span className="badge badge-info">{draft.question_type}</span>}
            {(draft.article_refs ?? []).map(ref => <span key={ref} className="badge">{ref}</span>)}
            <span className="field-note">
              {t('datasetsPage.generator.draftGenerated', { model: draft.provenance?.model })}
            </span>
          </div>
        </div>
        <button
          className="btn-sm draft-drop"
          onClick={onRemove}
        >
          {t('datasetsPage.generator.removeDraft')}
        </button>
      </div>
    </div>
  )
}
