import { useEffect, useId, useMemo, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { DraftBadge, DraftNote } from '../components/DraftNotice'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { useSearchParams, Link } from 'react-router-dom'
import {
  Scale, Trash2, FlaskConical, Package, Search, Plus, ThumbsUp, ThumbsDown,
  Archive, RotateCcw, BookOpen,
} from 'lucide-react'
import { api, type Judgment, type JudgmentWrite } from '../api/client'
import SelectBox from '../components/SelectBox'
import { useDefaultCorpus } from '../hooks/useDefaultCorpus'
import { useConfirm } from '../components/ConfirmDialog'
import { useRealm } from '../context/RealmContext'
import { copyToClipboard } from '../lib/clipboard'

// Replaces PinsPage.
//
// The reviewer states what is correct, not what the system should do. That is
// why this page has no threshold field, no rewrite field and no "validate
// blast radius" step: all three belonged to deciding which *other* questions
// a statement should also affect, which is the part of the retired
// retrieval-pin mechanism the objective review rejected.
//
// The layout answers three questions in order, because that is the order a
// reviewer actually has them in: what do I have (the stats row), which one am
// I looking at (the list), and what can I do with it (the detail). An earlier
// cut buried the corpus selector inside the list card, which made the page
// look empty on arrival for a reason that was not explained anywhere.

type FormState = {
  corpus_id: string
  question: string
  relevant: string
  irrelevant: string
  note: string
  source_feedback_id: string
  source_run_id: string
}

const EMPTY_FORM: FormState = {
  corpus_id: '', question: '', relevant: '', irrelevant: '', note: '',
  source_feedback_id: '', source_run_id: '',
}

type StatusFilter = 'all' | 'active' | 'retired'

function splitIds(raw: string): string[] {
  return raw.split(/[,\n]/).map(s => s.trim()).filter(Boolean)
}

function addIdToField(current: string, id: string): string {
  const ids = splitIds(current)
  if (ids.includes(id)) return current
  return ids.length === 0 ? id : `${ids.join(', ')}, ${id}`
}

function toWrite(form: FormState, realmId: string): JudgmentWrite {
  return {
    realm_id: realmId,
    corpus_id: form.corpus_id.trim(),
    question: form.question.trim(),
    relevant: splitIds(form.relevant),
    irrelevant: splitIds(form.irrelevant),
    note: form.note.trim(),
    source_feedback_id: form.source_feedback_id.trim(),
    source_run_id: form.source_run_id.trim(),
  }
}

function Field({ id, label, required, hint, children }: {
  id: string; label: string; required?: boolean; hint?: string; children: React.ReactNode
}) {
  return (
    <div className="form-group">
      <label className="jd-toolbar-label block-label" htmlFor={id}>
        {label} {required && <span className="req">*</span>}
      </label>
      {children}
      {hint && <div className="jd-section-hint tight-top">{hint}</div>}
    </div>
  )
}

function CorpusSelect({ id, value, realmId, onChange }: {
  id: string; value: string; realmId: string | null; onChange: (v: string) => void
}) {
  const { t } = useTranslation()
  const { data: collections = [], isLoading, isError } = useQuery({
    queryKey: ['corpus-collections', realmId],
    queryFn: () => api.corpus.collections(realmId),
  })
  const ids = Array.from(new Set(collections.map(c => c.corpus_id)))
  // Three states, not two. "The list has not arrived", "the request failed"
  // and "this Realm has no corpora" used to render as the same bare text
  // input, so a reader typing into it could not tell whether the corpus they
  // wanted was missing or merely unfetched. Only the failure keeps the input,
  // because that is the case where typing an id is still useful.
  if (isLoading) {
    return <input id={id} className="input" value={value} disabled placeholder={t('judgmentsPage.loading')} />
  }
  if (isError) {
    return (
      <>
        <input id={id} className="input" value={value} onChange={e => onChange(e.target.value)} />
        <div className="jd-section-hint tight-top">{t('judgmentsPage.corpusListFailed')}</div>
      </>
    )
  }
  if (ids.length === 0) {
    return (
      <>
        <input id={id} className="input" value={value} onChange={e => onChange(e.target.value)} />
        <div className="jd-section-hint tight-top">{t('judgmentsPage.corpusListEmpty')}</div>
      </>
    )
  }
  return (
    <SelectBox id={id} value={value} onChange={e => onChange(e.target.value)}
               aria-label={t('judgmentsPage.corpusLabel')}>
      <option value="">{t('judgmentsPage.corpusPlaceholder')}</option>
      {ids.map(cid => <option key={cid} value={cid}>{cid}</option>)}
    </SelectBox>
  )
}

// Looking a chunk up by its text beats having to know a chunk_id by heart.
// The two textareas stay the source of truth; this only ever appends.
function ChunkPicker({ corpusId, realmId, onAddRelevant, onAddIrrelevant }: {
  corpusId: string; realmId: string | null
  onAddRelevant: (chunkId: string) => void
  onAddIrrelevant: (chunkId: string) => void
}) {
  const { t } = useTranslation()
  const [q, setQ] = useState('')
  const { data, isLoading } = useQuery({
    queryKey: ['judgments-chunk-search', corpusId, q, realmId],
    queryFn: () => api.corpus.chunks(corpusId, { limit: 10, q, realmId }),
    enabled: !!corpusId,
  })

  if (!corpusId) return null

  return (
    <div className="jd-picker">
      <Field id="jd-chunk-search" label={t('judgmentsPage.chunkSearchLabel')} hint={t('judgmentsPage.chunkSearchHint')}>
        <input
          id="jd-chunk-search" className="input" value={q} onChange={e => setQ(e.target.value)}
          placeholder={t('judgmentsPage.chunkSearchPlaceholder')}
        />
      </Field>
      {isLoading && <div className="jd-section-hint flat">{t('judgmentsPage.loading')}</div>}
      {data && data.items.length === 0 && q && (
        <div className="jd-section-hint flat">{t('judgmentsPage.chunkSearchEmpty')}</div>
      )}
      {data && data.items.length > 0 && (
        <div className="chunk-picker">
          {data.items.map(c => (
            <div key={c.chunk_id} className="jd-picker-row">
              <div className="jd-picker-body">
                <div className="jd-picker-path">{c.structural_path || c.chunk_id}</div>
                <div className="jd-picker-text">{c.text}</div>
              </div>
              <button type="button" className="btn-sm" title={t('judgmentsPage.chunkAddRelevant')}
                      onClick={() => onAddRelevant(c.chunk_id)}>
                <ThumbsUp size={12} aria-hidden="true" /> {t('judgmentsPage.chunkAddRelevant')}
              </button>
              <button type="button" className="btn-sm" title={t('judgmentsPage.chunkAddIrrelevant')}
                      onClick={() => onAddIrrelevant(c.chunk_id)}>
                <ThumbsDown size={12} aria-hidden="true" /> {t('judgmentsPage.chunkAddIrrelevant')}
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

function CreateForm({ form, onChange, realmId, canSubmit, onCreate, creating, onCancel }: {
  form: FormState
  onChange: (patch: Partial<FormState>) => void
  realmId: string | null
  canSubmit: boolean
  onCreate: () => void
  creating: boolean
  onCancel: () => void
}) {
  const { t } = useTranslation()
  const uid = useId()

  return (
    <div className="edit-form jd-create">
      <div className="edit-form-head">
        <h3 className="edit-form-title">{t('judgmentsPage.createTitle')}</h3>
        <button type="button" className="btn btn-sm" onClick={onCancel}>{t('judgmentsPage.cancel')}</button>
      </div>
      <p className="hint-line bare">
        {t('judgmentsPage.createIntro')}
      </p>

      {/* `minmax(0, 1fr)` and not `1fr`: a grid item's default minimum
          width equals its content, so one long fragment line stretched the
          column past `max-width` and carried the form off the screen. */}
      <div className="jd-create-grid">
        <Field id={`${uid}-corpus`} label={t('judgmentsPage.corpusLabel')} required>
          <CorpusSelect id={`${uid}-corpus`} value={form.corpus_id} realmId={realmId}
                        onChange={v => onChange({ corpus_id: v })} />
        </Field>

        <Field id={`${uid}-question`} label={t('judgmentsPage.questionLabel')} required
               hint={t('judgmentsPage.questionHint')}>
          <textarea id={`${uid}-question`} className="input" rows={2} value={form.question}
                    onChange={e => onChange({ question: e.target.value })} />
        </Field>

        <ChunkPicker
          corpusId={form.corpus_id.trim()} realmId={realmId}
          onAddRelevant={id => onChange({ relevant: addIdToField(form.relevant, id) })}
          onAddIrrelevant={id => onChange({ irrelevant: addIdToField(form.irrelevant, id) })}
        />

        <div className="jd-verdict-cols">
          <Field id={`${uid}-rel`} label={t('judgmentsPage.relevantLabel')} hint={t('judgmentsPage.relevantHint')}>
            <textarea id={`${uid}-rel`} className="input" rows={2} value={form.relevant}
                      onChange={e => onChange({ relevant: e.target.value })} />
          </Field>
          <Field id={`${uid}-irr`} label={t('judgmentsPage.irrelevantLabel')} hint={t('judgmentsPage.irrelevantHint')}>
            <textarea id={`${uid}-irr`} className="input" rows={2} value={form.irrelevant}
                      onChange={e => onChange({ irrelevant: e.target.value })} />
          </Field>
        </div>

        <Field id={`${uid}-note`} label={t('judgmentsPage.noteLabel')} hint={t('judgmentsPage.noteHint')}>
          <textarea id={`${uid}-note`} className="input" rows={2} value={form.note}
                    onChange={e => onChange({ note: e.target.value })} />
        </Field>

        <div>
          <button type="button" className="btn btn-primary" disabled={!canSubmit || creating} onClick={onCreate}>
            {creating ? t('judgmentsPage.saving') : t('judgmentsPage.create')}
          </button>
          {!canSubmit && <span className="jd-section-hint ml-10">{t('judgmentsPage.createBlocked')}</span>}
        </div>
      </div>
    </div>
  )
}

function ChunkColumn({ title, cls, chunks }: {
  title: string; cls: string; chunks: Judgment['relevant']
}) {
  const { t } = useTranslation()
  return (
    <div>
      <div className={`jd-verdict-title ${cls}`}>{title} · {chunks.length}</div>
      {chunks.length === 0
        ? <div className="jd-section-hint flat">{t('judgmentsPage.noneNamed')}</div>
        : chunks.map(c => (
          <div key={c.chunk_id} className="jd-chunk">
            <div className="jd-chunk-path">
              {c.structural_path || c.doc_id || c.chunk_id}
              {c.ref_id
                // Only the document half of the ref id. The other half is the
                // structural path already shown to its left, and printing both
                // made every chunk read as the same text twice.
                ? <span className="jd-chunk-ref"> · {c.ref_id.split(/[#/]/)[0]}</span>
                : <span className="badge badge-warn pin-badge">{t('judgmentsPage.noRefId')}</span>}
            </div>
            {c.text && <div className="jd-chunk-text">{c.text.slice(0, 240)}{c.text.length > 240 ? '…' : ''}</div>}
          </div>
        ))}
    </div>
  )
}

function PromotePanel({ judgment, realmId }: { judgment: Judgment; realmId: string }) {
  const { t } = useTranslation()
  const [datasetId, setDatasetId] = useState('')
  const [answer, setAnswer] = useState('')
  const [result, setResult] = useState<string | null>(null)
  const uid = useId()

  // A real picker rather than a free-text id: the previous cut asked a
  // reviewer to type a dataset identifier they had no way to know.
  const { data: datasets = [] } = useQuery({
    queryKey: ['datasets', realmId],
    queryFn: () => api.datasets.list(realmId),
  })

  const promote = useMutation({
    mutationFn: () => api.judgments.toGoldenQuestion(judgment.id, realmId, judgment.corpus_id, {
      dataset_id: datasetId, reference_answer: answer.trim(),
    }),
    onSuccess: () => { setResult(t('judgmentsPage.promoteDone')); setAnswer('') },
    onError: (e: Error) => setResult(e.message),
  })

  return (
    <div className="jd-section">
      <h3 className="jd-section-title">
        <FlaskConical size={14} aria-hidden="true" /> {t('judgmentsPage.promoteTitle')}
      </h3>
      <p className="jd-section-hint">{t('judgmentsPage.promoteIntro')}</p>
      {!judgment.can_become_test && (
        <div className="guide-callout warn mb-10">
          <div>{t('judgmentsPage.promoteBlocked')}</div>
        </div>
      )}
      <div className="stack-grid">
        <Field id={`${uid}-ds`} label={t('judgmentsPage.datasetLabel')} required>
          <select id={`${uid}-ds`} className="form-select" value={datasetId}
                  onChange={e => setDatasetId(e.target.value)}>
            <option value="">{t('judgmentsPage.datasetPlaceholder')}</option>
            {datasets.map(d => (
              <option key={d.id ?? d.filename} value={d.id ?? ''}>
                {d.name ?? d.filename} {d.count != null ? `(${d.count})` : ''}
              </option>
            ))}
          </select>
        </Field>
        <Field id={`${uid}-ans`} label={t('judgmentsPage.referenceAnswerLabel')} required
               hint={t('judgmentsPage.referenceAnswerHint')}>
          <textarea id={`${uid}-ans`} className="input" rows={2} value={answer}
                    onChange={e => setAnswer(e.target.value)} />
        </Field>
        <div className="flex-row gap-10">
          <button type="button" className="btn"
                  disabled={!judgment.can_become_test || !datasetId || !answer.trim() || promote.isPending}
                  onClick={() => promote.mutate()}>
            {t('judgmentsPage.promote')}
          </button>
          {result && <span className="jd-section-hint flat">{result}</span>}
        </div>
      </div>
    </div>
  )
}

function Detail({ judgment, realmId, onDeleted }: {
  judgment: Judgment; realmId: string; onDeleted: () => void
}) {
  const { confirm, dialog } = useConfirm()
  const { t } = useTranslation()
  const qc = useQueryClient()
  const retired = judgment.status !== 'active'
  const invalidate = () => qc.invalidateQueries({ queryKey: ['judgments', realmId, judgment.corpus_id] })

  const toggle = useMutation({
    mutationFn: () => api.judgments.update(judgment.id, realmId, judgment.corpus_id, {
      status: retired ? 'active' : 'retired',
    }),
    onSuccess: invalidate,
  })
  const remove = useMutation({
    mutationFn: () => api.judgments.delete(judgment.id, realmId, judgment.corpus_id),
    onSuccess: () => { invalidate(); onDeleted() },
  })

  return (
    <div className="card">
      {dialog}
      <div className="jd-detail-head">
        <h2 className="jd-detail-q">{judgment.question}</h2>
        <div className="jd-provenance">
          <span>{judgment.corpus_id}</span>
          <span>{retired ? t('judgmentsPage.statusRetired') : t('judgmentsPage.statusActive')}</span>
          {judgment.created_at && <span>{judgment.created_at.slice(0, 10)}</span>}
          {judgment.author && <span>{judgment.author}</span>}
          {judgment.source_run_id && <span>{t('judgmentsPage.fromRun')} {judgment.source_run_id.slice(0, 8)}</span>}
        </div>
      </div>

      {judgment.note && <p className="jd-note">{judgment.note}</p>}

      <div className="jd-verdict-cols">
        <ChunkColumn title={t('judgmentsPage.relevantLabel')} cls="jd-verdict-relevant" chunks={judgment.relevant} />
        <ChunkColumn title={t('judgmentsPage.irrelevantLabel')} cls="jd-verdict-irrelevant" chunks={judgment.irrelevant} />
      </div>

      <PromotePanel judgment={judgment} realmId={realmId} />

      <div className="jd-actions">
        <button type="button" className="btn" onClick={() => toggle.mutate()}>
          {retired
            ? <><RotateCcw size={14} aria-hidden="true" /> {t('judgmentsPage.reactivate')}</>
            : <><Archive size={14} aria-hidden="true" /> {t('judgmentsPage.retire')}</>}
        </button>
        <button type="button" className="btn btn-danger" onClick={async () => {
          if (await confirm({ title: t('judgmentsPage.deleteConfirm'), danger: true })) remove.mutate()
        }}>
          <Trash2 size={14} aria-hidden="true" /> {t('judgmentsPage.delete')}
        </button>
      </div>
    </div>
  )
}

function BundlePanel({ realmId, corpusId, onClose }: {
  realmId: string; corpusId: string; onClose: () => void
}) {
  const { t } = useTranslation()
  const { data, isLoading, error } = useQuery({
    queryKey: ['judgments-bundle', realmId, corpusId],
    queryFn: () => api.judgments.bundle(realmId, corpusId),
  })

  return (
    <div className="jd-section lead-section">
      <div className="flex-between">
        <h3 className="jd-section-title">
          <Package size={14} aria-hidden="true" /> {t('judgmentsPage.bundleTitle')}
        </h3>
        <button type="button" className="btn-sm" onClick={onClose}>{t('judgmentsPage.cancel')}</button>
      </div>
      <p className="jd-section-hint">{t('judgmentsPage.bundleIntro')}</p>
      {isLoading && <div className="jd-section-hint flat">{t('judgmentsPage.loading')}</div>}
      {error && <div className="guide-callout warn"><div>{String(error)}</div></div>}
      {data && (
        <>
          <div className="stat-band mb-10">
            <div className="stat-cell">
              <div className="metric-val">{data.n_entries}</div>
              <div className="metric-label">{t('judgmentsPage.bundleEntries')}</div>
            </div>
            <div className="stat-cell">
              <div className="metric-val">{data.format_version}</div>
              <div className="metric-label">{t('judgmentsPage.bundleFormat')}</div>
            </div>
            <div className="stat-cell">
              <div className="metric-val">{data.calibration.must_not_match_pairs.length}</div>
              <div className="metric-label">{t('judgmentsPage.bundlePairs')}</div>
            </div>
          </div>
          <button type="button" className="btn"
                  onClick={() => copyToClipboard(JSON.stringify(data, null, 2))}>
            {t('judgmentsPage.bundleCopy')}
          </button>
        </>
      )}
    </div>
  )
}

export default function JudgmentsPage() {
  const { t } = useTranslation()
  const { activeRealmId } = useRealm()
  const qc = useQueryClient()
  const [params, setParams] = useSearchParams()

  const prefillQuestion = params.get('question') ?? ''
  const prefillCorpus = params.get('corpus_id') ?? ''
  // Only a question means "someone sent me here to record a verdict". A
  // corpus alone is just a scoped view of the list, and treating it as a
  // prefill would hide the list behind a form nobody asked for.
  const hasPrefill = !!prefillQuestion

  const [corpusId, setCorpusId] = useState(prefillCorpus)
  useDefaultCorpus(activeRealmId, corpusId, setCorpusId, { skip: Boolean(prefillCorpus) })
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [creating, setCreating] = useState(hasPrefill)
  const [showBundle, setShowBundle] = useState(false)
  const [search, setSearch] = useState('')
  const [status, setStatus] = useState<StatusFilter>('all')
  const [form, setForm] = useState<FormState>({
    ...EMPTY_FORM,
    question: prefillQuestion,
    corpus_id: prefillCorpus,
    source_feedback_id: params.get('source_feedback_id') ?? '',
    source_run_id: params.get('run_id') ?? '',
  })

  const { data: judgments = [], isLoading } = useQuery({
    queryKey: ['judgments', activeRealmId, corpusId],
    queryFn: () => api.judgments.list(activeRealmId!, corpusId),
    enabled: !!activeRealmId && !!corpusId,
  })

  const create = useMutation({
    mutationFn: () => api.judgments.create(toWrite(form, activeRealmId!)),
    onSuccess: created => {
      setCreating(false)
      setForm(EMPTY_FORM)
      setCorpusId(created.corpus_id)
      setSelectedId(created.id)
      setParams(new URLSearchParams())
      qc.invalidateQueries({ queryKey: ['judgments', activeRealmId, created.corpus_id] })
    },
  })

  const visible = useMemo(() => {
    const needle = search.trim().toLowerCase()
    return judgments.filter(j => {
      if (status === 'active' && j.status !== 'active') return false
      if (status === 'retired' && j.status === 'active') return false
      if (!needle) return true
      return j.question.toLowerCase().includes(needle) || j.note.toLowerCase().includes(needle)
    })
  }, [judgments, search, status])

  const stats = useMemo(() => ({
    total: judgments.length,
    active: judgments.filter(j => j.status === 'active').length,
    tests: judgments.filter(j => j.can_become_test).length,
    pairs: judgments.reduce((n, j) => n + j.preference_pairs, 0),
  }), [judgments])

  const canSubmit = !!form.corpus_id.trim() && !!form.question.trim()
    && (splitIds(form.relevant).length > 0 || splitIds(form.irrelevant).length > 0)
  const selected = visible.find(j => j.id === selectedId) ?? null

  // The right column does not sit empty beside a non-empty list: selecting the
  // first is what a reader does with their first click anyway.
  //
  // This must stay above the `activeRealmId` guard. It used to sit below it,
  // so switching to a realm-less state rendered one hook fewer than the render
  // before it, and React threw "rendered fewer hooks than expected". The suite
  // stayed green while doing it, because the throw landed outside any test's
  // assertion; only Vitest's unhandled-error count showed it.
  useEffect(() => {
    if (selected || visible.length === 0) return
    setSelectedId(visible[0].id)
  }, [visible, selected])

  if (!activeRealmId) {
    return <div className="card"><p>{t('judgmentsPage.noRealm')}</p></div>
  }

  return (
    // `page` carries the padding, `page-wide` only raises the width cap. The
    // first cut had `page-wide` alone, so the page ran edge to edge with the
    // heading touching the sidebar and the last summary card cut off by the
    // window — visible in a screenshot rather than in any test.
    <div className="page page-wide">
      <div className="page-head">
        <h1 className="page-title">{t('judgmentsPage.title')} <DraftBadge /></h1>
        <span className="page-act">
          <Link className="btn btn-sm" to="/guide?section=judgments">
            <BookOpen size={13} aria-hidden="true" /> {t('judgmentsPage.guideLink')}
          </Link>
          <button type="button" className="btn btn-sm btn-primary" onClick={() => setCreating(true)}>
            <Plus size={13} aria-hidden="true" /> {t('judgmentsPage.new')}
          </button>
        </span>
      </div>
      <DraftNote />

      <p className="page-lead">{t('judgmentsPage.intro')}</p>

      {/* The corpus picker sits in the toolbar rather than inside the list:
          until one is chosen, no number on this screen means anything. */}
      <div className="table-toolbar">
        <span className="toolbar-label">{t('judgmentsPage.corpusLabel')}</span>
        <CorpusSelect id="jd-corpus-filter" value={corpusId} realmId={activeRealmId} onChange={setCorpusId} />
        <button type="button" className="btn btn-sm push"
                disabled={!corpusId || stats.total === 0}
                onClick={() => setShowBundle(v => !v)}>
          <Package size={13} aria-hidden="true" /> {t('judgmentsPage.bundleAction')}
        </button>
      </div>

      {!corpusId ? (
        <div className="card jd-empty">
          <Scale size={40} className="jd-empty-icon" aria-hidden="true" />
          <div className="jd-empty-title">{t('judgmentsPage.pickCorpusTitle')}</div>
          <div className="jd-empty-body">{t('judgmentsPage.pickCorpusBody')}</div>
        </div>
      ) : creating ? (
        <CreateForm
          form={form}
          onChange={patch => setForm(f => ({ ...f, ...patch }))}
          realmId={activeRealmId}
          canSubmit={canSubmit}
          onCreate={() => create.mutate()}
          creating={create.isPending}
          onCancel={() => setCreating(false)}
        />
      ) : (
        <>
          {showBundle && (
            <BundlePanel realmId={activeRealmId} corpusId={corpusId} onClose={() => setShowBundle(false)} />
          )}

          {/* The number band (`.stat-band`) is the shared template for a row
              of single figures. It used to be a `.stats-row` of cards, and a
              card around one number is a frame around a frame. This row was
              also the third set of classes carrying the same meaning. */}
          {/* The shared summary-row pattern,
              not a second set of classes meaning the same thing. */}
          <div className="stat-band">
            <div className="stat-cell">
              <div className="eyebrow">{t('judgmentsPage.statTotal')}</div>
              <div className="metric-val">{stats.total}</div>
            </div>
            <div className="stat-cell">
              <div className="eyebrow">{t('judgmentsPage.statActive')}</div>
              <div className="metric-val">{stats.active}</div>
            </div>
            <div className="stat-cell">
              <div className="eyebrow">{t('judgmentsPage.statTests')}</div>
              <div className="metric-val">{stats.tests}</div>
            </div>
            <div className="stat-cell">
              <div className="eyebrow">{t('judgmentsPage.statPairs')}</div>
              <div className="metric-val">{stats.pairs}</div>
            </div>
          </div>

          {stats.total === 0 && !isLoading ? (
            <div className="card jd-empty">
              <Scale size={40} className="jd-empty-icon" aria-hidden="true" />
              <div className="jd-empty-title">{t('judgmentsPage.emptyTitle')}</div>
              <div className="jd-empty-body">{t('judgmentsPage.emptyBody')}</div>
              <button type="button" className="btn btn-primary mt-14"
                      onClick={() => { setForm(f => ({ ...f, corpus_id: corpusId })); setCreating(true) }}>
                <Plus size={14} aria-hidden="true" /> {t('judgmentsPage.new')}
              </button>
            </div>
          ) : (
            <div className="jd-split split">
              <div className="split-col">
                <div className="field-search">
                  <Search size={13} aria-hidden="true"
                          className="field-search-icon" />
                  <input className="input" value={search}
                         onChange={e => setSearch(e.target.value)}
                         placeholder={t('judgmentsPage.searchPlaceholder')} />
                </div>
                <div className="chips">
                  {(['all', 'active', 'retired'] as StatusFilter[]).map(s => (
                    <button key={s} type="button" className={`chip${status === s ? ' active' : ''}`}
                            onClick={() => setStatus(s)}>
                      {t(`judgmentsPage.filter_${s}`)}
                    </button>
                  ))}
                </div>
                {isLoading && <p className="jd-section-hint">{t('judgmentsPage.loading')}</p>}
                {!isLoading && visible.length === 0 && (
                  <p className="jd-section-hint">{t('judgmentsPage.noMatches')}</p>
                )}
                <ul className="pick-list">
                  {visible.map(j => (
                    <li key={j.id}>
                      <button type="button"
                              className={`pick-row${j.id === selectedId ? ' active' : ''}${j.status === 'active' ? '' : ' muted'}`}
                              onClick={() => setSelectedId(j.id)}>
                        <div className="pick-row-title">{j.question}</div>
                        <div className="pick-row-meta">
                          <span className="badge badge-success">+{j.relevant.length}</span>
                          <span className="badge badge-danger">−{j.irrelevant.length}</span>
                          {j.can_become_test && (
                            <span className="badge badge-info" title={t('judgmentsPage.badgeTestHint')}>
                              {t('judgmentsPage.badgeTest')}
                            </span>
                          )}
                          {j.status !== 'active' && (
                            <span className="badge badge-warn">{t('judgmentsPage.statusRetired')}</span>
                          )}
                        </div>
                      </button>
                    </li>
                  ))}
                </ul>
              </div>

              {selected
                ? <Detail judgment={selected} realmId={activeRealmId} onDeleted={() => setSelectedId(null)} />
                : (
                  <p className="split-empty split-empty-lg">{t('judgmentsPage.selectHint')}</p>
                )}
            </div>
          )}
        </>
      )}
    </div>
  )
}
