import { useEffect, useState, useMemo } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { useParams, useNavigate, Link } from 'react-router-dom'
import { useTranslation } from 'react-i18next'
import {
  Copy, CircleCheck, CircleX, ThumbsUp, ThumbsDown, Star,
  Rocket, GitCompare, Pin, Database, ListChecks, ChevronRight,
} from 'lucide-react'
import { api, type ExperimentDetail, type Feedback, type FeedbackWrite, type QuestionResult, type SourceRefView } from '../api/client'
import { durationSeconds, formatDuration, formatRunForClipboard, formatFeedbackForClipboard } from '../lib/format'
import { copyToClipboard } from '../lib/clipboard'
import { metricLabel, METRIC_META, classifyDelta } from '../lib/metricMeta'
import RunDiagnostics, { RegressionBadge, LatencyCaveat } from '../components/RunDiagnostics'
import RunPrescription from '../components/RunPrescription'
import RunCharts from '../components/RunCharts'
import PipelineDiagram from '../components/PipelineDiagram'
import FeedbackTriagePanel from '../components/FeedbackTriagePanel'
import { useRealm, useRealmPath } from '../context/RealmContext'
import { useStopExperiment } from '../hooks/useStopExperiment'

// Explains each row of the "Configuration" card — hidden by default behind a
// "?" toggle (same click-to-toggle pattern as CorpusPage.tsx's deep
// diagnostics, not a native title hover, so it works on touch too). Values
// are i18n keys (resolved with t() at the render site), not display text —
// row identity is now a stable key (see the cfgRows array below), not the
// (now-translated) displayed label, so this map's own keys had to move off
// the old Russian label text too.
const CONFIG_FIELD_HINTS: Record<string, string> = {
  name: 'runPage.config.nameHint',
  implementation: 'runPage.config.implementationHint',
  pipeline: 'runPage.config.pipelineHint',
  corpusId: 'runPage.config.corpusIdHint',
  chunkingStrategy: 'runPage.config.chunkingStrategyHint',
  embedder: 'runPage.config.embedderHint',
  generator: 'runPage.config.generatorHint',
  reranker: 'runPage.config.rerankerHint',
  topK: 'runPage.config.topKHint',
  merge: 'runPage.config.mergeHint',
  prompt: 'runPage.config.promptHint',
  controlQuestions: 'runPage.config.controlQuestionsHint',
  questions: 'runPage.config.questionsHint',
  seed: 'runPage.config.seedHint',
  time: 'runPage.config.timeHint',
  duration: 'runPage.config.durationHint',
  configHash: 'runPage.config.configHashHint',
}

// Eval Measurement Trustworthiness, Phase 1 — mirrors core/eval/funnel.py's
// Layer values, no logic duplicated here (verdict computed server-side).
// `label` is an i18n key, resolved with t() at the render site.
const FUNNEL_BADGE: Record<string, { label: string; cls: string }> = {
  not_applicable: { label: 'runPage.funnel.notApplicable', cls: 'badge-info' },
  suspected_ungrounded_answer: { label: 'runPage.funnel.suspectedUngrounded', cls: 'badge-danger' },
  retrieval: { label: 'runPage.funnel.layerRetrieval', cls: 'badge-danger' },
  rerank: { label: 'runPage.funnel.layerRerank', cls: 'badge-danger' },
  generation: { label: 'runPage.funnel.layerGeneration', cls: 'badge-warn' },
  ok: { label: 'runPage.funnel.ok', cls: 'badge-success' },
}

// Mirrors core/eval/root_cause.py's Cause and Lever values. No
// logic here either: both the cause and the lever it points at are decided
// server-side, next to the evidence that produced them. This file only names
// them in the reader's language.
//
// `data_missing` is styled as an error rather than a warning on purpose: it
// means the measurement was taken against a corpus that cannot answer the
// question, which invalidates the number rather than merely worsening it.
const CAUSE_BADGE: Record<string, { label: string; cls: string }> = {
  data_missing: { label: 'runPage.rootCause.dataMissing', cls: 'badge-danger' },
  ranking: { label: 'runPage.rootCause.ranking', cls: 'badge-warn' },
  chunking: { label: 'runPage.rootCause.chunking', cls: 'badge-warn' },
  not_retrievable: { label: 'runPage.rootCause.notRetrievable', cls: 'badge-warn' },
  unknown: { label: 'runPage.rootCause.unknown', cls: 'badge-info' },
}

const LEVER_LABEL: Record<string, string> = {
  ingest: 'runPage.rootCause.leverIngest',
  ranking: 'runPage.rootCause.leverRanking',
  chunking: 'runPage.rootCause.leverChunking',
  vocabulary: 'runPage.rootCause.leverVocabulary',
  // Produced only by runs stored in a narrow window of older versions,
  // when the two above could not be told apart.
  chunking_or_vocabulary: 'runPage.rootCause.leverChunkingOrVocabulary',
  verify_index: 'runPage.rootCause.leverVerifyIndex',
}

// Compact source_refs table, shown on demand — answers the "what did
// retrieval actually find" question without bloating every question row by
// default (143 questions × full chunk text would make the page unusable).
const _RETRIEVAL_TEXT_PREVIEW_LEN = 200

function RetrievalPanel({ refs, preRerankRefs }: { refs: SourceRefView[]; preRerankRefs?: SourceRefView[] }) {
  const { t } = useTranslation()
  const [expandAll, setExpandAll] = useState(false)
  const droppedByRerank = preRerankRefs && preRerankRefs.length > 0
    ? preRerankRefs.filter(p => !refs.some(r => r.chunk_id === p.chunk_id))
    : []
  const anyLong = refs.some(r => (r.chunk_text?.length ?? 0) > _RETRIEVAL_TEXT_PREVIEW_LEN)
  return (
    <div className="gen-block-sm">
      {anyLong && (
        <button className="btn-sm expand-all" onClick={() => setExpandAll(v => !v)}>
          {expandAll ? t('runPage.retrieval.collapseChunks') : t('runPage.retrieval.expandChunks')}
        </button>
      )}
      <table className="table chunk-table-fixed">
        <colgroup>
          <col className="col-8" />
          <col className="col-28" />
          <col className="col-64" />
        </colgroup>
        <thead><tr><th>score</th><th>structural_path</th><th>{t('runPage.retrieval.text')}</th></tr></thead>
        <tbody>
          {refs.map((r, i) => {
            const key = r.chunk_id || String(i)
            const text = r.chunk_text ?? ''
            const isLong = text.length > _RETRIEVAL_TEXT_PREVIEW_LEN
            return (
              <tr key={key}>
                <td>{r.score?.toFixed(3)}</td>
                <td className="wrap-cell">
                  <code>{r.structural_path}</code>
                  {r.pinned && (
                    <span className="badge badge-info pin-badge" title={t('runPage.retrieval.pinnedHint')}>
                      <Pin size={9} className="btn-icon bare" /> {t('runPage.retrieval.pinned')}
                    </span>
                  )}
                </td>
                <td className="wrap-cell">
                  {expandAll || !isLong ? text : `${text.slice(0, _RETRIEVAL_TEXT_PREVIEW_LEN)}…`}
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
      {droppedByRerank.length > 0 && (
        <p className="chunk-note">
          {t('runPage.retrieval.rerankDropped', {
            dropped: droppedByRerank.length, total: preRerankRefs!.length,
            paths: droppedByRerank.map(d => d.structural_path).join('; '),
          })}
        </p>
      )}
    </div>
  )
}

const _MISS_DIAGNOSIS_WIDENED_K = 50

function MissDiagnosisPanel({ runId, questionId }: { runId: string; questionId: string }) {
  const { t } = useTranslation()
  const { data, isFetching, refetch, isFetched } = useQuery({
    queryKey: ['miss-diagnosis', runId, questionId, _MISS_DIAGNOSIS_WIDENED_K],
    queryFn: () => api.experiments.diagnoseMiss(runId, questionId, _MISS_DIAGNOSIS_WIDENED_K),
    enabled: false, // on-demand only — re-querying retrieval per click, not per page load
  })
  return (
    <div className="pad-t6">
      <button className="btn-sm" onClick={() => refetch()} disabled={isFetching}>
        {isFetching ? t('runPage.missDiagnosis.searching') : t('runPage.missDiagnosis.button')}
      </button>
      {isFetched && data && (
        <p className="qrow-sub">
          {data.found
            ? <>{t('runPage.missDiagnosis.foundPrefix')} <strong>{data.rank}</strong> {t('runPage.missDiagnosis.foundSuffix', { widenedK: data.widened_k, score: data.score?.toFixed(3) })}
                {data.structural_path && <> ({data.structural_path})</>}</>
            : <>{t('runPage.missDiagnosis.notFound', { widenedK: data.widened_k })}</>}
        </p>
      )}
    </div>
  )
}

// Human feedback on one answer (binary rating / per-dimension scores / free
// text) — stored separately from the run itself, see feedback.py's module
// docstring for why. One mutation instance per question, shared between the
// hover-reveal binary buttons on the collapsed summary and the full panel
// inside the expanded body, so both reflect the same pending/error state.
function useFeedbackMutation(runId: string, questionId: string) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (body: FeedbackWrite) => api.feedback.upsert(runId, questionId, body),
    onSuccess: (updated) => {
      qc.setQueryData<Record<string, Feedback>>(['feedback', runId], (old) => ({ ...(old ?? {}), [questionId]: updated }))
    },
  })
}

// Two visual variants sharing one component: 'hover' sits inside a native
// <summary> and is invisible until that row is hovered (opacity 0→1 via
// .qrow-feedback-hover in styles.css) — a dimmed-but-visible persistent
// indicator once a rating exists, so a reviewer can tell at a glance which
// questions they've already triaged without hovering every row.
// stopPropagation/preventDefault keep a rating click from also toggling
// the parent <details> open/closed. 'static' sits inside the always-visible
// FeedbackPanel (only reachable once already expanded, so there's no
// hover-reveal case to protect there) — found live: reusing the 'hover'
// variant's opacity-0-by-default styling there made the icons invisible,
// since they were never a descendant of a hovered .qrow-summary.
function BinaryFeedbackButtons({
  rating, onRate, variant = 'hover',
}: { rating?: 'good' | 'bad' | null; onRate: (r: 'good' | 'bad') => void; variant?: 'hover' | 'static' }) {
  const { t } = useTranslation()
  if (variant === 'static') {
    return (
      <span className="feedback-binary-row">
        <button
          type="button" aria-label={t('runPage.feedback.goodAnswer')} title={t('runPage.feedback.goodAnswer')}
          onClick={() => onRate('good')}
          className={`feedback-binary-btn${rating === 'good' ? ' active-good' : ''}`}
        >
          <ThumbsUp size={13} fill={rating === 'good' ? 'currentColor' : 'none'} /> {t('runPage.feedback.good')}
        </button>
        <button
          type="button" aria-label={t('runPage.feedback.badAnswer')} title={t('runPage.feedback.badAnswer')}
          onClick={() => onRate('bad')}
          className={`feedback-binary-btn${rating === 'bad' ? ' active-bad' : ''}`}
        >
          <ThumbsDown size={13} fill={rating === 'bad' ? 'currentColor' : 'none'} /> {t('runPage.feedback.bad')}
        </button>
      </span>
    )
  }
  return (
    <span className={`qrow-feedback-hover${rating ? ' has-rating' : ''}`} onClick={e => e.stopPropagation()}>
      <button
        type="button" aria-label={t('runPage.feedback.goodAnswer')} title={t('runPage.feedback.goodAnswer')}
        onClick={e => { e.preventDefault(); e.stopPropagation(); onRate('good') }}
        className={`rate-btn${rating === 'good' ? ' up' : ''}`}
      >
        <ThumbsUp size={13} fill={rating === 'good' ? 'currentColor' : 'none'} />
      </button>
      <button
        type="button" aria-label={t('runPage.feedback.badAnswer')} title={t('runPage.feedback.badAnswer')}
        onClick={e => { e.preventDefault(); e.stopPropagation(); onRate('bad') }}
        className={`rate-btn${rating === 'bad' ? ' down' : ''}`}
      >
        <ThumbsDown size={13} fill={rating === 'bad' ? 'currentColor' : 'none'} />
      </button>
    </span>
  )
}

// Level 2 — several independent quality dimensions rather than one general
// score (confirmed with the user over one general 1-5 quality score), each
// its own 5-star row so rating one doesn't require touching the others.
// `label` is an i18n key, resolved with t() at the render site.
const SCORE_DIMENSIONS: { key: string; label: string }[] = [
  { key: 'accuracy', label: 'runPage.feedback.dimensionAccuracy' },
  { key: 'completeness', label: 'runPage.feedback.dimensionCompleteness' },
  { key: 'relevance', label: 'runPage.feedback.dimensionRelevance' },
]

function StarRow({ label, value, onChange }: { label: string; value?: number; onChange: (v: number) => void }) {
  const { t } = useTranslation()
  return (
    <div className="star-line">
      <span className="star-label">{label}</span>
      <span className="star-set">
        {[1, 2, 3, 4, 5].map(n => (
          <button
            key={n} type="button" aria-label={t('runPage.feedback.starAriaLabel', { label, n })}
            onClick={() => onChange(n)}
            className={`star-btn${value != null && n <= value ? ' on' : ''}`}
          >
            <Star size={14} fill={value != null && n <= value ? 'currentColor' : 'none'} />
          </button>
        ))}
      </span>
    </div>
  )
}

// Only reachable by expanding the row (already-existing interaction) —
// level 2 (scores) and level 3 (free text) need more attention than a
// quick skim, so they stay behind that same disclosure rather than adding
// their own always-visible chrome. Rendered as a sidebar next to the row's
// main content (.qrow-body in styles.css) rather than stacked below it, so
// it stays visible without scrolling past retrieval/diagnostics. Binary/
// star clicks save immediately; comment/reviewer save on blur (debounce-
// by-blur, not per keystroke) — no separate "Save" button anywhere here.
function FeedbackPanel({ feedback, mutation }: { feedback?: Feedback; mutation: ReturnType<typeof useFeedbackMutation> }) {
  const { t } = useTranslation()
  const [comment, setComment] = useState(feedback?.comment ?? '')
  const [reviewer, setReviewer] = useState(feedback?.reviewer ?? '')

  // Feedback loads via its own query, in parallel with the run itself — it
  // can resolve after this panel already mounted with `feedback` undefined,
  // so the textarea/input need to pick up the value once it arrives (not
  // just capture it once at mount via useState's initial value).
  useEffect(() => { setComment(feedback?.comment ?? '') }, [feedback?.comment])
  useEffect(() => { setReviewer(feedback?.reviewer ?? '') }, [feedback?.reviewer])

  return (
    // Not a card: the two neighbouring columns of an expanded question are
    // just columns with an eyebrow, and a border around the third singles it
    // out for no reason.
    <div className="qrow-panel">
      <div className="eyebrow">
        {t('runPage.feedback.title')} <span className="muted">{t('runPage.feedback.optional')}</span>
      </div>

      <BinaryFeedbackButtons variant="static" rating={feedback?.rating} onRate={r => mutation.mutate({ rating: r })} />

      <div className="stack-6">
        {SCORE_DIMENSIONS.map(dim => (
          <StarRow
            key={dim.key}
            label={t(dim.label)}
            value={feedback?.scores?.[dim.key]}
            onChange={v => mutation.mutate({ scores: { [dim.key]: v } })}
          />
        ))}
      </div>

      <textarea
        rows={3}
        placeholder={t('runPage.feedback.commentPlaceholder')}
        value={comment}
        onChange={e => setComment(e.target.value)}
        onBlur={() => { if (comment !== (feedback?.comment ?? '')) mutation.mutate({ comment }) }}
      />
      <input
        type="text"
        placeholder={t('runPage.feedback.reviewerPlaceholder')}
        value={reviewer}
        onChange={e => setReviewer(e.target.value)}
        onBlur={() => { if (reviewer !== (feedback?.reviewer ?? '')) mutation.mutate({ reviewer }) }}
      />

      {mutation.isPending && <span className="feedback-status saving">{t('runPage.feedback.saving')}</span>}
      {mutation.isError && <span className="feedback-status error">{t('runPage.feedback.saveError')}</span>}
    </div>
  )
}

// "feedback becomes a test", reframed after a fair
// question: the question is already in a dataset (the run was scored
// against one) — so this isn't "add to dataset", it's "confirm or correct
// this question's ground truth", almost always in that same dataset. The
// dataset picker is hidden by default (no choice to make in the common
// case) and only appears behind "Save to a different dataset" for the rare
// cross-dataset case, or automatically if the run's source dataset can't
// be resolved (deleted/renamed since the run). Confirming a question that
// already matches by text updates it in place server-side rather than
// duplicating it (see feedback.py#_find_existing_question) — pre-filling
// the reference answer here, editable, is what makes "correct" possible;
// this never regenerates the answer from the feedback comment on its own.
// No redirect on success — a reviewer works through several questions in
// one sitting, an inline confirmation keeps that flow uninterrupted.
function ConfirmGroundTruthControl({ runId, questionId, referenceAnswer, sourceDatasetName, realmId }: {
  runId: string; questionId: string; referenceAnswer: string; sourceDatasetName: string; realmId: string | null
}) {
  const { t } = useTranslation()
  const [datasetId, setDatasetId] = useState('')
  const [datasetTouched, setDatasetTouched] = useState(false)
  const [advancedOpen, setAdvancedOpen] = useState(false)
  const [refAnswer, setRefAnswer] = useState(referenceAnswer)
  const [articleRefs, setArticleRefs] = useState('')
  const { data: datasets } = useQuery({
    queryKey: ['datasets', realmId],
    queryFn: () => api.datasets.list(realmId),
  })

  const defaultDataset = datasets?.find(d => d.filename === sourceDatasetName)
  // No source dataset resolved (e.g. deleted/renamed since the run) means
  // there's no sensible default — the picker has to show regardless of
  // whether the reviewer asked for it.
  const sourceUnresolved = datasets !== undefined && !defaultDataset
  const showPicker = advancedOpen || sourceUnresolved

  // Default-select the run's own source dataset once the list loads — but
  // only if the reviewer hasn't already picked something else themselves
  // (`datasetTouched` guards against a background refetch silently
  // overriding a manual choice).
  useEffect(() => {
    if (datasetTouched || !defaultDataset) return
    setDatasetId(defaultDataset.id ?? '')
  }, [defaultDataset, datasetTouched])

  const mutation = useMutation({
    mutationFn: () => api.feedback.promote(runId, questionId, {
      target_dataset_id: datasetId,
      reference_answer: refAnswer,
      article_refs: articleRefs.trim()
        ? articleRefs.split(',').map(r => r.trim()).filter(Boolean)
        : undefined,
    }, realmId),
  })

  return (
    <div className="qrow-panel">
      <div className="eyebrow">{t('runPage.confirm.title')}</div>
      {!showPicker && defaultDataset && (
        <div className="field-note">
          {t('runPage.confirm.datasetHint', { filename: defaultDataset.filename })}{' '}
          <button
            type="button" onClick={() => setAdvancedOpen(true)}
            className="link-btn"
          >
            {t('runPage.confirm.saveElsewhere')}
          </button>
        </div>
      )}
      {showPicker && (
        <select
          value={datasetId}
          onChange={e => { setDatasetId(e.target.value); setDatasetTouched(true) }}
        >
          <option value="">{t('runPage.promote.selectDataset')}</option>
          {datasets?.map(d => (
            <option key={d.id ?? d.filename} value={d.id ?? ''}>{d.filename}</option>
          ))}
        </select>
      )}
      <label className="qrow-label">
        {t('runPage.promote.referenceAnswerLabel')}
      </label>
      <textarea
        className="qrow-area"
        value={refAnswer}
        onChange={e => setRefAnswer(e.target.value)}
      />
      <label className="qrow-label">
        {t('runPage.promote.articleRefsLabel')}
      </label>
      <input
        type="text"
        placeholder={t('runPage.promote.articleRefsPlaceholder')}
        value={articleRefs}
        onChange={e => setArticleRefs(e.target.value)}
      />
      <button
        className="btn-sm" disabled={!datasetId || mutation.isPending}
        onClick={() => mutation.mutate()}
      >
        {t('runPage.confirm.button')}
      </button>
      {mutation.isSuccess && (
        <span className="feedback-status ok">
          {mutation.data.created ? t('runPage.promote.success') : t('runPage.promote.updated')}
        </span>
      )}
      {mutation.isError && <span className="feedback-status error">{t('runPage.promote.error')}</span>}
    </div>
  )
}

function QuestionRow({ runId, qr, feedback, sourceDatasetName, corpusId, external }: {
  runId: string; qr: QuestionResult; feedback?: Feedback; sourceDatasetName: string; corpusId: string
  // Only affects how latency is labelled: for a call to somebody else's
  // system the trace measures the handover, not their work.
  external: boolean
}) {
  const { t } = useTranslation()
  const { activeRealmId } = useRealm()
  const funnelBadge = qr.funnel ? FUNNEL_BADGE[qr.funnel.layer] : undefined
  // One number per row, the one the question was judged to have failed on:
  // retrieval recall where it exists, otherwise the first metric computed.
  const leadMetric =
    Object.entries(qr.metrics).find(([k]) => k === 'retrieval_recall_at_k')
    ?? Object.entries(qr.metrics)[0]
  const causeBadge = qr.root_cause ? CAUSE_BADGE[qr.root_cause.cause] : undefined
  const [showRetrieval, setShowRetrieval] = useState(false)
  const refs = qr.source_refs ?? []
  const feedbackMutation = useFeedbackMutation(runId, qr.question_id)
  return (
    <details className="qrow">
      {/* One question row: a layer label, the text, and one number. It used
          to carry every metric in sequence, eight badges to a row, which lost
          the question itself and left neighbouring rows uncomparable because
          the numbers sat at different horizontal positions. The rest of the
          metrics expand along with the question. */}
      <summary className="qrow-summary">
        <ChevronRight size={12} className="qrow-caret" aria-hidden="true" />
        {qr.error && (
          <span className="badge badge-danger" title={qr.error}>
            {t('runPage.questionRow.pipelineError')}
          </span>
        )}
        {funnelBadge && (
          <span className={`badge ${funnelBadge.cls}`} title={qr.funnel?.detail}>
            {t(funnelBadge.label)}
          </span>
        )}
        {causeBadge && (
          <span className={`badge ${causeBadge.cls}`} title={qr.root_cause?.detail}>
            {t(causeBadge.label)}
          </span>
        )}
        <span className="qrow-text">{qr.question}</span>
        {leadMetric && (
          <span className="qrow-score" title={metricLabel(leadMetric[0])}>
            {leadMetric[1].toFixed(2)}
          </span>
        )}
      </summary>
      <div className="qrow-grid">
        <div className="qrow-col">
          <div className="eyebrow">{t('runPage.questionRow.generatedAnswer')}</div>
          {qr.error ? (
            <p className="badge badge-danger block">
              {t('runPage.questionRow.pipelineErrorDetail', { error: qr.error })}
            </p>
          ) : (
            <>
              <pre className="qrow-pre">{qr.generated_answer}</pre>
              {qr.computed_citations && qr.computed_citations.length > 0 && (
                <p className="qrow-cite">
                  <strong>{t('runPage.questionRow.computedCitations')}</strong> {t('runPage.questionRow.computedCitationsHint')}{' '}
                  {qr.computed_citations.join('; ')}
                </p>
              )}
            </>
          )}
        </div>
        <div className="qrow-col">
          <div className="eyebrow">{t('runPage.questionRow.referenceAnswer')}</div>
          <pre className="qrow-pre">{qr.reference_answer}</pre>
          <div className="qrow-metrics">
            {Object.entries(qr.metrics).map(([k, v]) => (
              <span key={k} className="qrow-metric" title={METRIC_META[k]?.hint}>
                <span className="k">{metricLabel(k)}</span>
                <span className="v">{v.toFixed(3)}</span>
              </span>
            ))}
          </div>
        </div>
        <div className="qrow-col">
          <FeedbackPanel feedback={feedback} mutation={feedbackMutation} />
          <FeedbackTriagePanel
            runId={runId} questionId={qr.question_id} feedback={feedback}
            corpusId={corpusId} questionText={qr.question} realmId={activeRealmId}
          />
        </div>
        <div className="qrow-col">
          <ConfirmGroundTruthControl
            runId={runId} questionId={qr.question_id}
            referenceAnswer={qr.reference_answer} sourceDatasetName={sourceDatasetName}
            realmId={activeRealmId}
          />
        </div>
      </div>

      {qr.stage_trace && (
        <div className="qrow-full">
          {external && <div className="caveat-line"><LatencyCaveat /></div>}
          <PipelineDiagram trace={qr.stage_trace} title={t('runPage.questionRow.latencyByStage')} />
        </div>
      )}

      {(qr.funnel || refs.length > 0) && (
        <div className="qrow-full">
          {qr.funnel && (
            <div className="qrow-sub">
              <strong>{t('runPage.questionRow.funnelDiagnosis')}</strong> {qr.funnel.detail}
            </div>
          )}
          {/* The cause, and the lever it points at. Shown right
              under the funnel verdict because the verdict alone names a layer
              and would otherwise send a reader to tune ranking even when the
              expected source is not in the index at all. */}
          {qr.root_cause && (
            <div className="qrow-sub pad-t6">
              <div>
                <strong>{t('runPage.rootCause.heading')}</strong> {qr.root_cause.detail}
              </div>
              <div className="pad-t4">
                <strong>{t('runPage.rootCause.leverHeading')}</strong>{' '}
                {t(LEVER_LABEL[qr.root_cause.lever] ?? LEVER_LABEL.verify_index)}
              </div>
            </div>
          )}
          {/* The on-demand widened re-query stays only for runs stored before
              an older version, whose questions carry no cause. For every newer run the
              same answer is already computed and shown above, so offering the
              button again would invite a second retrieval for nothing. */}
          {qr.funnel?.layer === 'retrieval' && !qr.root_cause && (
            <MissDiagnosisPanel runId={runId} questionId={qr.question_id} />
          )}
          {refs.length > 0 && (
            <div className="pad-t8">
              <button className="btn-sm" onClick={() => setShowRetrieval(v => !v)}>
                {showRetrieval ? '▼' : '▶'} {t('runPage.questionRow.retrieval', { count: refs.length })}
              </button>
              {showRetrieval && <RetrievalPanel refs={refs} preRerankRefs={qr.pre_rerank_source_refs} />}
            </div>
          )}
        </div>
      )}
    </details>
  )
}

// Copies a plain-text dump of the whole run (config, metrics, diagnostics)
// for pasting into an LLM chat (e.g. Claude Code) to diagnose the run — see
// lib/format.ts#formatRunForClipboard for exactly what's included.
function CopyRunButton({ run }: { run: ExperimentDetail }) {
  const { t } = useTranslation()
  const [state, setState] = useState<'idle' | 'copied' | 'error'>('idle')

  const handleCopy = async () => {
    const ok = await copyToClipboard(formatRunForClipboard(run))
    setState(ok ? 'copied' : 'error')
    setTimeout(() => setState('idle'), 2000)
  }

  return (
    <button
      type="button"
      className="btn-sm inline-4"
      onClick={handleCopy}
      title={t('runPage.copyRun.title')}
    >
      {state === 'copied'
        ? <><CircleCheck size={13} /> {t('runPage.copyRun.copied')}</>
        : state === 'error'
          ? <><CircleX size={13} /> {t('runPage.copyRun.failed')}</>
          : <><Copy size={13} /> {t('runPage.copyRun.copy')}</>}
    </button>
  )
}

// Copies only the questions that actually have feedback set — unlike
// CopyRunButton above (deliberately pared down to skip per-question
// content), this button's entire point is per-question data, so it
// includes the question text for traceability. See
// lib/format.ts#formatFeedbackForClipboard.
function CopyFeedbackButton({ run, feedbackMap }: { run: ExperimentDetail; feedbackMap: Record<string, Feedback> }) {
  const { t } = useTranslation()
  const [state, setState] = useState<'idle' | 'copied' | 'error' | 'empty'>('idle')

  const handleCopy = async () => {
    const text = formatFeedbackForClipboard(run, feedbackMap)
    if (!text) {
      setState('empty')
      setTimeout(() => setState('idle'), 2000)
      return
    }
    const ok = await copyToClipboard(text)
    setState(ok ? 'copied' : 'error')
    setTimeout(() => setState('idle'), 2000)
  }

  return (
    <button
      type="button"
      className="btn-sm inline-4"
      onClick={handleCopy}
      title={t('runPage.copyFeedback.title')}
    >
      {state === 'copied'
        ? <><CircleCheck size={13} /> {t('runPage.copyRun.copied')}</>
        : state === 'error'
          ? <><CircleX size={13} /> {t('runPage.copyRun.failed')}</>
          : state === 'empty'
            ? <>{t('runPage.copyFeedback.noneYet')}</>
            : <><Copy size={13} /> {t('runPage.copyFeedback.copy')}</>}
    </button>
  )
}

export default function RunPage() {
  const { activeRealm } = useRealm()
  const [moreOpen, setMoreOpen] = useState(false)
  const { t } = useTranslation()
  const { runId } = useParams<{ runId: string }>()
  const navigate = useNavigate()
  const toRealm = useRealmPath()
  const qc = useQueryClient()
  const [showPromptText, setShowPromptText] = useState(false)
  const [activeTab, setActiveTab] = useState<'overview' | 'diagnostics' | 'prescription' | 'charts'>('overview')
  const [searchQuery, setSearchQuery] = useState('')
  const [layerFilter, setLayerFilter] = useState<string>('attention')
  // Found live: the "Stop" button only ever existed on
  // NewExperimentPage's ProgressWidget (shown right after submitting a new
  // run) — navigating here directly (or reloading) while a run is still
  // "running" hit this page's own polling branch below, which had no stop
  // control at all. useStopExperiment is shared with ProgressWidget; only
  // the "still running" detection differs (this page polls via `status`,
  // ProgressWidget listens on `useProgress`'s WebSocket-driven `done`).
  const { stopping, stopError, stop: stopRun } = useStopExperiment(runId)
  const { data, isLoading, error } = useQuery({
    queryKey: ['experiment', runId],
    queryFn: () => api.experiments.get(runId!),
    enabled: !!runId,
    // Poll every 2 s while the background run is still in flight (the
    // async job-model): the gateway returns status="running" until done.
    refetchInterval: (query) => query.state.data?.status === 'running' ? 2000 : false,
  })
  // Prompt text is fetched lazily — only once the user actually toggles it
  // open, not on every page load (the table cell shows just the id/version
  // by default, see CONFIG_FIELD_HINTS.prompt).
  const { data: promptDetail, isLoading: promptLoading } = useQuery({
    queryKey: ['prompt', data?.prompt_id],
    queryFn: () => api.prompts.get(data!.prompt_id!),
    enabled: showPromptText && !!data?.prompt_id,
  })
  // One fetch for the whole run's feedback, keyed by question_id — passed
  // down to every QuestionRow instead of one query per row.
  const { data: feedbackMap } = useQuery({
    queryKey: ['feedback', runId],
    queryFn: () => api.feedback.get(runId!),
    enabled: !!runId,
  })
  // "Make baseline" — moved here from RunDiagnostics' own quick-actions
  // block (now part of the toolbar row, alongside the other cross-page
  // actions) but the same single global pointer (see
  // services/api_gateway/routers/experiments.py#set_baseline): setting a
  // new baseline silently replaces whichever run held it before.
  const setBaselineMutation = useMutation({
    // Both directions: on a marked run the same button clears the mark.
    mutationFn: () => data?.is_baseline
      ? api.experiments.unsetBaseline()
      : api.experiments.setBaseline(runId!),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['experiment', runId] }),
  })

  // Computed above the early returns: a hook sitting behind
  // `if (isLoading) return` does not run on every render, and React throws
  // "Rendered more hooks than during the previous render" at the exact moment
  // the data arrives. Found by the tests rather than in the browser, where
  // loading finishes before anyone looks.
  /** The loss funnel: how many questions reached each layer.
   *
   *  Derived from the per-question verdicts (each question's `funnel`) and the
   *  server's `root_cause_counts`, the same ones the prescription is built
   *  from. Counting again on the client would give two answers to one
   *  question, and they would diverge silently. */
  const funnel = useMemo(() => {
    const rows = data?.question_results ?? []
    if (rows.length === 0) return null
    const verdict = (q: { funnel?: { layer?: string } }) => q.funnel?.layer
    const total = rows.length
    const notFound = rows.filter(q => verdict(q) === 'retrieval').length
    const dropped = rows.filter(q => verdict(q) === 'rerank').length
    const unsupported = rows.filter(q => verdict(q) === 'generation').length
    const counts = Object.entries(data?.root_cause_counts ?? {})
      .sort((a, b) => b[1] - a[1])
    return {
      total,
      found: total - notFound,
      afterRerank: total - notFound - dropped,
      answered: total - notFound - dropped - unsupported,
      dropped, unsupported,
      topCause: counts[0]?.[0] ?? null,
      topCauseCount: counts[0]?.[1] ?? 0,
    }
  }, [data?.question_results, data?.root_cause_counts])

  /** The tone of a number. One threshold across every metric: all of them are
   *  higher-is-better and live in 0..1, and a per-metric threshold would turn
   *  the colour into a judgement the platform does not make. The run table
   *  uses the same threshold. */
  const metricTone = (_key: string, v: number) =>
    v >= 0.6 ? 'var(--color-success)' : v >= 0.3 ? 'var(--color-warning)' : 'var(--color-danger)'

  // Run metrics: the value, its delta against the baseline, and a verdict.
  //
  // The threshold comes from `classifyDelta`, the same one the server's
  // regression guard applies. Two thresholds over one quantity would mean the
  // page and the guard could disagree about whether something is a regression
  // or noise.
  const metricEntries = useMemo(() => {
    const deltas = new Map(
      (data?.regression?.deltas ?? []).map(d => [d.metric, d]),
    )
    const keyMetrics = new Set(activeRealm?.key_metrics ?? [])
    return Object.entries(data?.aggregate_metrics ?? {})
      .map(([key, value]) => {
        const d = deltas.get(key)
        const delta = d && d.baseline !== 0 ? (d.current - d.baseline) / Math.abs(d.baseline) : null
        return {
          key, value: value as number, delta,
          verdict: d ? classifyDelta(d.baseline, d.current, key) : 'noise' as const,
          isKey: keyMetrics.has(key),
        }
      })
      // The realm's key metrics come first: the band reads left to right, and
      // what the realm called important should not end up at its tail.
      .sort((a, b) => Number(b.isKey) - Number(a.isKey))
  }, [data?.aggregate_metrics, data?.regression, activeRealm?.key_metrics])


  if (isLoading) return <div className="page"><div className="loading">{t('runPage.loading')}</div></div>
  if (error || !data) {
    const msg = error instanceof Error ? error.message : t('runPage.notFound')
    return <div className="page"><div className="empty">{msg}</div></div>
  }

  // the async job-model: the run is backgrounded, so GET may return
  // status="running" with only a progress array — no results yet.
  if (data.status === 'running') {
    const processed = data.progress?.processed ?? 0
    const total = data.progress?.total ?? '?'
    const pct = typeof total === 'number' && total > 0 ? Math.round(processed / total * 100) : null
    return (
      <div className="page">
        <div className="flex-between mb-8">
          <h1>{t('runPage.runHeading', { runId })}</h1>
          <Link to={toRealm('/experiments')} className="btn">{t('runPage.back')}</Link>
        </div>
        <div className="card run-waiting">
          <div className="loading run-waiting-line">
            {stopping ? t('newExperimentPage.progress.stopping') : t('runPage.running.inProgress')}
          </div>
          <p className="run-waiting-sub">
            {t('runPage.running.processed')} <strong>{processed}</strong> {t('runPage.running.of')} <strong>{total}</strong>
            {pct !== null && <> ({pct}%)</>}
          </p>
          {pct !== null && (
            <div className="run-waiting-track">
              <div className="run-waiting-fill" style={{ width: `${pct}%` }} />
            </div>
          )}
          <button
            type="button"
            className="btn-sm btn-danger run-waiting-btn"
            onClick={stopRun}
            disabled={stopping}
          >
            {stopping ? t('newExperimentPage.progress.stopping') : t('newExperimentPage.progress.stopButton')}
          </button>
          {stopError && <div className="badge badge-danger res-sub-btn">{stopError}</div>}
          <p className="run-waiting-note">
            {t('runPage.running.autoRefresh')}
          </p>
        </div>
      </div>
    )
  }

  // diagnostics-depth badge. pipeline_source="http"
  // means an external RAG; whether it returned a trace contract (non-empty
  // source_refs) decides white-box vs black-box. in_process is always
  // white-box — the platform built the pipeline itself.
  const isExternal = data.config?.pipeline_source === 'http'
  // Found live: a run made via external_rag_id (the normal path, not an
  // inline http_endpoint) left http_endpoint null in the stored config —
  // this page could only ever show "External (—)", no way to tell which
  // external service a failure was even against without cross-referencing
  // external_rag_id against GET /external-rags by hand. Backend now
  // backfills both external_rag_name and http_endpoint from the resolved
  // ExternalRag record at run time (see
  // core/experiment/config.py#ExperimentConfig.external_rag_name).
  const externalTarget = data.config?.external_rag_name
    ? `${data.config.external_rag_name} — ${data.config?.http_endpoint ?? '—'}`
    : (data.config?.http_endpoint ?? '—')
  const hasSources = data.question_results.some(qr => (qr.source_refs?.length ?? 0) > 0)
  const diagnosticsDepth = !isExternal
    ? { label: t('runPage.diagnosticsDepth.builtin'), cls: 'flag-ok' }
    : hasSources
      ? { label: t('runPage.diagnosticsDepth.externalTraced'), cls: 'flag-ok' }
      : { label: t('runPage.diagnosticsDepth.externalBlackBox'), cls: 'flag-warn' }

  // Found live: `data.dataset_name` (used for the config table's own
  // display-only "Control questions" cell below) is often extension-
  // stripped ("handbook.v2.full") while `data.config?.dataset_name` carries
  // the real filename ("handbook.v2.full.jsonl") that matches `Dataset.filename`
  // exactly (services/api_gateway/routers/experiments.py's
  // `dataset_name=dataset.name` vs `cfg_data["dataset_name"] = body.dataset_name`).
  // The toolbar's "Manage dataset" link and ConfirmGroundTruthControl's
  // default-dataset lookup both need an exact filename match, not the
  // cosmetic display string, so this prefers config's copy first.
  const runDatasetName = data.config?.dataset_name || data.dataset_name || ''
  // Local, client-side only — the run's own question_results are already
  // fully loaded on this page, so filtering a (at most low hundreds) array
  // in the browser needs no round trip to the backend.
  const trimmedQuery = searchQuery.trim().toLowerCase()
  // Filter by funnel layer first, then search by text. This screen is opened
  // to find the questions that failed rather than to page through a hundred
  // and forty-three, so the default shows those whose verdict is not "ok".
  const layerCounts = data.question_results.reduce<Record<string, number>>((acc, qr) => {
    const layer = qr.funnel?.layer ?? 'unknown'
    acc[layer] = (acc[layer] ?? 0) + 1
    return acc
  }, {})
  const attentionCount = data.question_results.filter(qr => qr.funnel && qr.funnel.layer !== 'ok').length
  // A run may carry no funnel verdicts at all: older runs and external systems
  // without tracing are both like that. Filtering by layer would then leave an
  // empty screen, and "nothing failed" would read as "no questions", so
  // everything is shown.
  const hasVerdicts = data.question_results.some(qr => qr.funnel != null)
  const byLayer = !hasVerdicts ? data.question_results : data.question_results.filter(qr => {
    if (layerFilter === 'attention') return qr.funnel != null && qr.funnel.layer !== 'ok'
    if (layerFilter === 'all') return true
    return qr.funnel?.layer === layerFilter
  })
  const filteredResults = trimmedQuery
    ? byLayer.filter(qr => qr.question.toLowerCase().includes(trimmedQuery))
    : byLayer

  // The configuration fields. The list lives outside the markup because it is
  // long, and editing it inside JSX meant editing layout.
  const CONFIG_ROWS: [string, string, unknown][] = [
    ['name', t('runPage.config.name'), data.config_name],
    ['implementation', t('runPage.config.implementation'), isExternal ? t('runPage.config.implementationExternal', { endpoint: externalTarget }) : t('runPage.config.implementationBuiltin')],
    ['pipeline', 'Pipeline', data.config?.pipeline_id ?? '—'],
    ['corpusId', 'Corpus ID', data.config?.corpus_id ?? 'default'],
    ['chunkingStrategy', t('runPage.config.chunkingStrategy'), data.config?.chunking_strategy?.component_id ?? '—'],
    ['embedder', t('runPage.config.embedder'), data.config?.embedder?.component_id ?? '—'],
    ['generator', t('runPage.config.generator'), data.generator_model || data.config?.generator?.component_id || '—'],
    ['reranker', t('runPage.config.reranker'), data.config?.reranker?.component_id ?? t('runPage.config.rerankerNotUsed')],
    ['topK', 'Top-K', data.config?.top_k ?? '—'],
    ['merge', 'Merge', data.config?.merge_strategy ?? '—'],
    ['prompt', t('runPage.config.prompt'), data.prompt_id ? `${data.prompt_id} (v${data.prompt_version ?? '?'})` : '—'],
    ['controlQuestions', t('runPage.config.controlQuestions'), data.dataset_name || data.config?.dataset_name || '—'],
    ['questions', t('runPage.config.questions'), (() => {
      if (data.n_questions == null) return '—'
      const count = data.stopped ? data.question_results.length : data.n_questions
      return [
        String(count),
        data.stopped ? t('runPage.config.stoppedSuffix', { total: data.n_questions }) : null,
        data.n_errors ? `(${t('runPage.config.pipelineErrorsSuffix', { n: data.n_errors })})` : null,
      ].filter(Boolean).join(' ')
    })()],
    ['seed', 'Seed', data.config?.seed ?? '—'],
    ['time', t('runPage.config.time'), data.started_at ? new Date(data.started_at).toLocaleString('ru-RU') : '—'],
    ['duration', t('runPage.config.duration'), (() => {
      const sec = durationSeconds(data.started_at, data.finished_at)
      return sec != null ? formatDuration(sec) : '—'
    })()],
    ['configHash', 'Config hash', data.config_hash],
  ]

  const runDuration = durationSeconds(data.started_at, data.finished_at)

  return (
    <div className="page page-wide">
      <div className="page-head">
        <h1 className="page-title">{data.config_name || data.run_id.slice(0, 8)}</h1>
        <p className="page-sub">
          {data.run_id.slice(0, 8)}
          {data.started_at && ` · ${new Date(data.started_at).toLocaleString('ru-RU', { day: '2-digit', month: '2-digit', hour: '2-digit', minute: '2-digit' })}`}
          {runDuration != null && ` · ${formatDuration(runDuration)}`}
        </p>
        <span className="run-flags">
          <span className={`flag ${diagnosticsDepth.cls}`} title={t('runPage.diagnosticsDepth.title')}>
            <i className="flag-dot" aria-hidden="true" />{diagnosticsDepth.label}
          </span>
          {data.deepeval_report && (
            <span
              className="badge badge-info"
              title={t('runPage.deepevalTitle', { metrics: Object.entries(data.deepeval_report.metrics).map(([k, v]) => `${k}=${v.toFixed(2)}`).join(', ') })}
            >
              {t('runPage.hasDeepeval')}
            </span>
          )}
          <RegressionBadge run={data} />
          {(data.n_errors ?? 0) > 0 && (
            <span className="flag flag-bad" title={t('runPage.config.pipelineErrorsSuffix', { n: data.n_errors })}>
              <i className="flag-dot" aria-hidden="true" />{t('runPage.errorsFlag', { n: data.n_errors })}
            </span>
          )}
        </span>
      </div>

      {/* Five buttons rather than seven.
          "Manage corpus" and "Manage dataset" navigate to other pages rather
          than acting on the run; both moved under the overflow menu and into
          the command palette. A toolbar that puts an action beside a link
          forces a reader through all seven labels to find one of the two. */}
      <div className="run-toolbar">
        <CopyRunButton run={data} />
        <CopyFeedbackButton run={data} feedbackMap={feedbackMap ?? {}} />
        <button type="button" className="btn-sm" onClick={() => navigate(toRealm(`/new?from=${data.run_id}`))}>
          <Rocket size={13} /> {t('runDiagnostics.newRunBasedOnThis')}
        </button>
        <button type="button" className="btn-sm" onClick={() => navigate(toRealm(`/compare?a=${data.run_id}`))}>
          <GitCompare size={13} /> {t('runDiagnostics.compareWithAnother')}
        </button>
        {/* The button works in both directions: on the baseline it clears the
            mark, on everything else it sets it. The baseline used to have no
            button at all, which left no way to clear the mark. */}
        <button
          type="button" className="btn-sm"
          onClick={() => setBaselineMutation.mutate()} disabled={setBaselineMutation.isPending}
        >
          <Pin size={13} />
          {data.is_baseline ? t('runDiagnostics.unsetBaseline') : t('runDiagnostics.makeBaseline')}
        </button>

        <div className="lang-switcher">
          <button type="button" className="btn-sm" onClick={() => setMoreOpen(o => !o)} aria-label={t('runDiagnostics.more')}>…</button>
          {moreOpen && (
            <>
              <div className="realm-switcher-backdrop" onClick={() => setMoreOpen(false)} />
              <div className="lang-menu view-menu">
                <button
                  type="button" className="lang-item"
                  onClick={() => navigate(toRealm(`/data/upload${data.config?.corpus_id ? `?corpus_id=${encodeURIComponent(data.config.corpus_id)}` : ''}`))}
                >
                  <Database size={13} /><span className="grow">{t('runDiagnostics.manageCorpus')}</span>
                </button>
                <button
                  type="button" className="lang-item"
                  onClick={() => navigate(toRealm(`/data/qa${runDatasetName ? `?dataset=${encodeURIComponent(runDatasetName)}` : ''}`))}
                >
                  <ListChecks size={13} /><span className="grow">{t('runDiagnostics.manageDataset')}</span>
                </button>
              </div>
            </>
          )}
        </div>
      </div>

      <div className="data-tab-bar" role="tablist">
        <button
          type="button" role="tab" aria-selected={activeTab === 'overview'}
          className={`data-tab${activeTab === 'overview' ? ' active' : ''}`}
          onClick={() => setActiveTab('overview')}
        >
          {t('runPage.tabs.overview')}
        </button>
        <button
          type="button" role="tab" aria-selected={activeTab === 'diagnostics'}
          className={`data-tab${activeTab === 'diagnostics' ? ' active' : ''}`}
          onClick={() => setActiveTab('diagnostics')}
        >
          {t('runPage.tabs.diagnostics')}
        </button>
        {/* Phase 4's deliverable. Placed straight after diagnostics because
            it is what a reader does with them: the diagnostics say what is
            wrong, this is the document handed to whoever fixes it. */}
        <button
          type="button" role="tab" aria-selected={activeTab === 'prescription'}
          className={`data-tab${activeTab === 'prescription' ? ' active' : ''}`}
          onClick={() => setActiveTab('prescription')}
        >
          {t('runPage.tabs.prescription')}
        </button>
        <button
          type="button" role="tab" aria-selected={activeTab === 'charts'}
          className={`data-tab${activeTab === 'charts' ? ' active' : ''}`}
          onClick={() => setActiveTab('charts')}
        >
          {t('runPage.tabs.charts')}
        </button>
      </div>

      {activeTab === 'overview' && (
      <>
        <section className="metric-section flush">
          <div className="section-rule flush">
            <h2 className="section-title">{t('runPage.metrics.heading')}</h2>
            <span className="section-meta">
              {metricEntries.length > 0
                ? t('runPage.metrics.shown', { n: metricEntries.length })
                : t('runPage.metrics.none')}
            </span>
          </div>

          {/* A band of numbers rather than a column of cards with scrollbars.
              Each number carries its delta against the baseline and a verdict
              of gain, loss or noise, on the same threshold the regression
              guard uses: one quantity, one threshold, one vocabulary for the
              aggregate and for the question.

              The previous panel coloured its numbers with hex literals
              (#4ade80, #2d3748, #94a3b8), so it looked identical across all
              five palettes and both themes, and lost contrast on the light
              one. The colours now come from tokens. */}
          {metricEntries.length > 0 && (
            <div className="stat-band">
              {metricEntries.map(({ key, value, delta, verdict, isKey }) => (
                <div key={key} className="stat-cell" title={METRIC_META[key]?.hint}>
                  <div className="eyebrow">
                    {metricLabel(key)}{isKey && <span className="key-star" aria-hidden="true"> ★</span>}
                  </div>
                  <div className="metric-val" style={{ color: metricTone(key, value) }}>
                    {value.toFixed(3)}
                  </div>
                  <div className={`stat-sub verdict-${verdict}`}>
                    {delta == null
                      ? t('runPage.metrics.noBaseline')
                      : t(`runPage.metrics.verdict.${verdict}`, { pct: (delta * 100).toFixed(1) })}
                  </div>
                </div>
              ))}
            </div>
          )}

          {/* The loss funnel. It answers what the metric band does not ask:
              at which layer the answer was lost. The numbers come from
              `root_cause_counts`, computed on the server from the same
              per-question verdicts as everything else on the page. Counting
              again on the client would diverge from the prescription. */}
          {funnel && (
            <>
              <div className="section-rule">
                <h2 className="section-title">{t('runPage.funnel.heading')}</h2>
                <span className="section-meta">{t('runPage.funnel.questions', { n: funnel.total })}</span>
              </div>
              <div className="funnel">
                <div className="funnel-cell">
                  <div className="eyebrow">{t('runPage.funnel.search')}</div>
                  <div className="metric-val">{funnel.found}</div>
                  <div className="stat-sub">{t('runPage.funnel.foundOf', { total: funnel.total })}</div>
                </div>
                <div className={`funnel-cell${funnel.dropped > 0 ? ' hot' : ''}`}>
                  <div className="eyebrow">{t('runPage.funnel.rerank')}</div>
                  <div className="metric-val">{funnel.afterRerank}</div>
                  <div className="stat-sub">{t('runPage.funnel.droppedBelow', { n: funnel.dropped })}</div>
                </div>
                <div className="funnel-cell">
                  <div className="eyebrow">{t('runPage.funnel.generation')}</div>
                  <div className="metric-val">{funnel.answered}</div>
                  <div className="stat-sub">{t('runPage.funnel.noSupport', { n: funnel.unsupported })}</div>
                </div>
                <div className="funnel-cell hot">
                  <div className="eyebrow">{t('runPage.funnel.cause')}</div>
                  <div className="metric-val cause">{funnel.topCause ?? t('runPage.funnel.none')}</div>
                  <div className="stat-sub">{t('runPage.funnel.questions', { n: funnel.topCauseCount })}</div>
                </div>
              </div>
            </>
          )}

          {!('correct_refusal' in data.aggregate_metrics) && Object.keys(data.aggregate_metrics).length > 0 && (
            <p className="hint-line">{t('runPage.metrics.legacySchema')}</p>
          )}
        </section>
        <div className="section run-config">
          <div className="section-rule">
            <h2 className="section-title">{t('runPage.config.heading')}</h2>
          </div>
          {/* Key and value pairs on rules, rather than a table.
              A two-column table with no headers and no sorting is a list of
              pairs typeset as though somebody were going to compare down a
              column. What it gets compared against is another run, and that
              has a screen of its own. */}
          <dl className="kv-list">
            {CONFIG_ROWS.map(([key, label, value]) => (
              <div key={key} className="kv-row">
                <dt className="kv-key">
                  {label}
                  {CONFIG_FIELD_HINTS[key] && (
                    <span
                      className="kv-hint" tabIndex={0}
                      title={t(CONFIG_FIELD_HINTS[key])}
                      aria-label={t('runPage.config.whatDoesThisMean', { label })}
                    >?</span>
                  )}
                </dt>
                <dd className="kv-val">
                  {String(value)}
                  {/* The prompt text loads on demand: it is long, and on a run
                      page it is wanted rarely, when the result is explained by
                      the wording rather than by the pipeline. */}
                  {key === 'prompt' && data.prompt_id && (
                    <button type="button" className="link-btn kv-toggle" onClick={() => setShowPromptText(v => !v)}>
                      {showPromptText ? t('runPage.config.hidePrompt') : t('runPage.config.showPrompt')}
                    </button>
                  )}
                  {key === 'prompt' && showPromptText && (
                    promptLoading
                      ? <p className="loading">{t('runPage.loading')}</p>
                      : <pre className="tpl-block">{promptDetail?.template ?? '—'}</pre>
                  )}
                </dd>
              </div>
            ))}
          </dl>
        </div>
      </>
      )}

      {activeTab === 'diagnostics' && <RunDiagnostics run={data} />}

      {activeTab === 'prescription' && <RunPrescription run={data} />}

      {activeTab === 'charts' && <RunCharts run={data} />}

      <div className="section">
        <div className="section-rule">
          <h2 className="section-title">{t('runPage.attentionHeading')}</h2>
          <span className="section-meta">
            {t('runPage.attentionMeta', { n: attentionCount, total: data.question_results.length })}
          </span>
        </div>

        <div className="table-toolbar">
          <input
            type="search" className="toolbar-search"
            placeholder={t('runPage.results.searchPlaceholder')}
            aria-label={t('runPage.results.searchPlaceholder')}
            value={searchQuery}
            onChange={e => setSearchQuery(e.target.value)}
          />
          {/* Chips by failure layer. "Where was the answer lost" is the same
              question the funnel above asks, and filtering by it leads from a
              number to the questions behind it. */}
          <div className="chips">
            <button type="button" className={`chip${layerFilter === 'attention' ? ' active' : ''}`}
                    onClick={() => setLayerFilter('attention')}>
              {t('runPage.attentionChip')}<span className="chip-count">{attentionCount}</span>
            </button>
            {Object.entries(layerCounts)
              .filter(([layer]) => layer !== 'ok' && layer !== 'unknown')
              .sort((a, b) => b[1] - a[1])
              .map(([layer, n]) => (
                <button key={layer} type="button"
                        className={`chip${layerFilter === layer ? ' active' : ''}`}
                        onClick={() => setLayerFilter(layer)}>
                  {t(FUNNEL_BADGE[layer]?.label ?? layer)}<span className="chip-count">{n}</span>
                </button>
              ))}
            <button type="button" className={`chip${layerFilter === 'all' ? ' active' : ''}`}
                    onClick={() => setLayerFilter('all')}>
              {t('runPage.allChip')}<span className="chip-count">{data.question_results.length}</span>
            </button>
          </div>
        </div>
        {filteredResults.length === 0 ? (
          <div className="empty">{t('runPage.results.searchNoMatches', { query: searchQuery })}</div>
        ) : (
          filteredResults.map(qr => (
            <QuestionRow
              key={qr.question_id} runId={data.run_id} qr={qr} feedback={feedbackMap?.[qr.question_id]}
              sourceDatasetName={runDatasetName} corpusId={data.config?.corpus_id || ''}
              external={isExternal}
            />
          ))
        )}
      </div>
    </div>
  )
}
