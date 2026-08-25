import type { ExperimentDetail, Feedback } from '../api/client'
import { diagnoseMetrics, diagnoseRetrieval, diagnoseLatency } from './diagnostics'
import i18n from '../i18n'

/** Formats a duration in seconds as a human-readable string with minutes —
 * "971 s" tells you nothing at a glance; "16 min 11 s" does. Switches units
 * by magnitude so short runs (smoke tests) still show seconds precisely. */
export function formatDuration(totalSeconds: number): string {
  const s = Math.round(totalSeconds)
  if (s < 60) return i18n.t('format.duration.secondsOnly', { s })
  const hours = Math.floor(s / 3600)
  const minutes = Math.floor((s % 3600) / 60)
  const seconds = s % 60
  if (hours > 0) return i18n.t('format.duration.hoursMinutes', { hours, minutes })
  return i18n.t('format.duration.minutesSeconds', { minutes, seconds })
}

export function durationSeconds(startedAt?: string | null, finishedAt?: string | null): number | null {
  if (!startedAt || !finishedAt) return null
  return (new Date(finishedAt).getTime() - new Date(startedAt).getTime()) / 1000
}

const SEVERITY_ICON: Record<string, string> = { ok: '🟢', warn: '🟡', error: '🔴', info: '🔵' }

/** Plain-text dump of a run for pasting into an LLM chat for diagnosis —
 * config, aggregate metrics, and the same precomputed diagnostics shown in
 * the diagnostics card. Deliberately excludes the per-question breakdown
 * (question text, answers, per-question metrics) — with ~150 questions that
 * dump made the copied text unwieldy to paste, and the aggregate metrics
 * plus diagnostics already cover the "why" a run needs diagnosing. */
export function formatRunForClipboard(run: ExperimentDetail): string {
  const lines: string[] = []
  const push = (s = '') => lines.push(s)

  push(i18n.t('format.clipboard.run', { runId: run.run_id }))
  push()
  push(i18n.t('format.clipboard.configuration'))
  // Found live: this stayed name-less even after RunPage.tsx's own render
  // was fixed to show the resolved external RAG's name alongside its URL
  // (see core/experiment/config.py#ExperimentConfig.external_rag_name) — a
  // second, independent formatter for the same data, missed the first time.
  const externalTarget = run.config?.external_rag_name
    ? `${run.config.external_rag_name} — ${run.config?.http_endpoint ?? '—'}`
    : (run.config?.http_endpoint ?? '—')
  const cfgRows: [string, unknown][] = [
    [i18n.t('format.clipboard.name'), run.config_name],
    [i18n.t('format.clipboard.implementation'), run.config?.pipeline_source === 'http'
      ? i18n.t('format.clipboard.externalImpl', { endpoint: externalTarget })
      : i18n.t('format.clipboard.builtinImpl')],
    ['Pipeline', run.config?.pipeline_id ?? '—'],
    ['Corpus ID', run.config?.corpus_id ?? 'default'],
    [i18n.t('format.clipboard.chunkingStrategy'), run.config?.chunking_strategy?.component_id ?? '—'],
    [i18n.t('format.clipboard.embedder'), run.config?.embedder?.component_id ?? '—'],
    [i18n.t('format.clipboard.generator'), run.config?.generator?.component_id ?? '—'],
    [i18n.t('format.clipboard.reranker'), run.config?.reranker?.component_id ?? i18n.t('format.clipboard.rerankerNotUsed')],
    ['Top-K', run.config?.top_k ?? '—'],
    ['Merge', run.config?.merge_strategy ?? '—'],
    [i18n.t('format.clipboard.prompt'), run.prompt_id ? `${run.prompt_id} (v${run.prompt_version ?? '?'})` : '—'],
    [i18n.t('format.clipboard.controlQuestions'), run.dataset_name || run.config?.dataset_name || '—'],
    [i18n.t('format.clipboard.questions'), run.n_questions != null
      ? `${run.n_questions}${run.n_errors ? ` (${i18n.t('format.clipboard.pipelineErrors', { n: run.n_errors })})` : ''}`
      : '—'],
    ['Seed', run.config?.seed ?? '—'],
    [i18n.t('format.clipboard.time'), run.started_at ? new Date(run.started_at).toLocaleString('ru-RU') : '—'],
    [i18n.t('format.clipboard.duration'), (() => {
      const sec = durationSeconds(run.started_at, run.finished_at)
      return sec != null ? formatDuration(sec) : '—'
    })()],
    ['Config hash', run.config_hash],
  ]
  for (const [label, val] of cfgRows) push(`  ${label}: ${val}`)

  push()
  push(i18n.t('format.clipboard.aggregateMetrics'))
  const metricEntries = Object.entries(run.aggregate_metrics || {})
  if (metricEntries.length === 0) push(`  ${i18n.t('format.clipboard.none')}`)
  for (const [k, v] of metricEntries) push(`  ${k}: ${v.toFixed(4)}`)

  if (run.deepeval_report) {
    push()
    push(i18n.t('format.clipboard.deepeval'))
    for (const [k, v] of Object.entries(run.deepeval_report.metrics)) push(`  ${k}: ${v.toFixed(4)}`)
  }

  if (run.is_baseline || run.regression) {
    push()
    if (run.is_baseline) push(i18n.t('format.clipboard.baselineYes'))
    if (run.regression) {
      const status = run.regression.passed
        ? i18n.t('format.clipboard.no')
        : i18n.t('format.clipboard.yesWithCount', { n: run.regression.violations.length })
      push(i18n.t('format.clipboard.regression', { status }))
      for (const v of run.regression.violations) push(`  - ${v}`)
    }
  }

  const detectorItems = run.diagnostics ?? []
  const autoItems = [...diagnoseMetrics(run), ...diagnoseRetrieval(run), ...diagnoseLatency(run.avg_stage_trace)]
  if (detectorItems.length || autoItems.length) {
    push()
    push(i18n.t('format.clipboard.diagnostics'))
    for (const d of detectorItems) push(`  ${SEVERITY_ICON[d.severity] ?? ''} ${d.title} — ${d.detail}`)
    for (const d of autoItems) push(`  ${SEVERITY_ICON[d.severity] ?? ''} ${d.title} — ${d.detail}`)
  }

  return lines.join('\n')
}

const RATING_LABEL: Record<string, () => string> = {
  good: () => i18n.t('format.clipboard.ratingGood'),
  bad: () => i18n.t('format.clipboard.ratingBad'),
}

/** Plain-text dump of the human feedback set on a run's answers — unlike
 * formatRunForClipboard above (deliberately pared down to skip per-question
 * content), this button's entire point *is* per-question data, so every
 * annotated question's text is included for traceability. Questions with no
 * feedback at all are skipped entirely. Returns '' when nothing has been
 * annotated, so the caller can show an honest "nothing to copy yet" state
 * instead of copying an empty report. */
export function formatFeedbackForClipboard(run: ExperimentDetail, feedbackMap: Record<string, Feedback>): string {
  const entries = Object.values(feedbackMap).filter(fb =>
    fb.rating != null || Object.keys(fb.scores || {}).length > 0 || !!fb.comment?.trim(),
  )
  if (entries.length === 0) return ''

  const lines: string[] = []
  const push = (s = '') => lines.push(s)

  push(i18n.t('format.clipboard.feedbackForRun', { runId: run.run_id }))
  push(i18n.t('format.clipboard.annotatedCount', { count: entries.length, total: run.question_results.length }))

  for (const fb of entries) {
    const qr = run.question_results.find(q => q.question_id === fb.question_id)
    push()
    push(`[${fb.question_id}] ${qr?.question ?? i18n.t('format.clipboard.questionNotFound')}`)
    if (fb.rating) push(`  ${i18n.t('format.clipboard.rating', { value: RATING_LABEL[fb.rating]?.() ?? fb.rating })}`)
    const scoreEntries = Object.entries(fb.scores || {})
    if (scoreEntries.length > 0) push(`  ${i18n.t('format.clipboard.metrics', { value: scoreEntries.map(([k, v]) => `${k}=${v}`).join(', ') })}`)
    if (fb.comment) push(`  ${i18n.t('format.clipboard.comment', { value: fb.comment })}`)
    if (fb.reviewer) push(`  ${i18n.t('format.clipboard.reviewedBy', { value: fb.reviewer })}`)
  }

  return lines.join('\n')
}
