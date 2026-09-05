import type { ExperimentDetail, StageTrace } from '../api/client'
import i18n from '../i18n'

export type Severity = 'ok' | 'warn' | 'error' | 'info'

// Every judgement this module can make. The interface computes findings of its
// own, with thresholds of its own and no counterpart on the server, and a
// catalogue that knew only the server's half would be describing half the
// platform.
export const SIGNAL_IDS = [
  'legacy_metric_schema',
  'refusal_band',
  'recall_band',
  'similarity_band',
  'not_found_count',
  'sources_absent',
  'dense_score_low',
  'latency_unavailable',
  'latency_total',
  'latency_generate',
  'latency_embed',
] as const

export type SignalId = typeof SIGNAL_IDS[number]

export interface DiagnosticItem {
  // A stable name for the judgement, independent of the sentence that renders
  // it. The catalogue in core/eval/atlas.py points at these, and a Python
  // module cannot resolve a TypeScript function, so SIGNAL_IDS below is the
  // declaration it reads instead. `atlasSignals.test.ts` keeps that
  // declaration honest against what these functions actually produce.
  id: SignalId
  title: string
  detail: string
  severity: Severity
  action?: { label: string; href: string }
}

// Whether an answer refuses is decided once, on the server, and arrives on each
// question as `is_refusal`. A list of phrases used to live here as well, and
// the two disagreed: eleven fixed substrings against the server's fifteen
// alternatives, several of which match on a word stem. This side did not know
// `не указан`, `нет информац`, the stem of `отсутству`, `не содержит` in any
// form but one, `could not find`, `does not say/include/mention`, `cannot
// find` or `no relevant information`, so the same answer counted as a refusal
// in one panel and not in the one beside it.
//
// A run stored before the field existed carries none, and the count then
// reports what it can see and does not guess with a weaker rule.

// Eval Measurement Trustworthiness, Phase 0 — replaces the old Jaccard
// token-overlap metrics (faithfulness/answer_relevancy/reference_overlap),
// which were structurally incapable of producing a meaningful score (see
// core/eval/answerability.py and the Phase 0 plan for the full audit).
// Mirrors core/eval/detectors.py's detect_incorrect_refusals + the
// retrieval/answer metric ranges from _CompositeEvaluator.
export function diagnoseMetrics(run: ExperimentDetail): DiagnosticItem[] {
  const items: DiagnosticItem[] = []
  const m = run.aggregate_metrics || {}

  // Legacy runs (pinned before Phase 0) only have the old keys — detect
  // that explicitly rather than silently showing misleading zeros.
  const hasNewSchema = 'correct_refusal' in m || 'retrieval_recall_at_k' in m
  if (!hasNewSchema) {
    items.push({
      id: 'legacy_metric_schema',
      title: i18n.t('diagnostics.metrics.legacySchemaTitle'),
      detail: i18n.t('diagnostics.metrics.legacySchemaDetail'),
      severity: 'info',
    })
    return items
  }

  const correctRefusal = m.correct_refusal
  if (correctRefusal !== undefined) {
    const title = i18n.t('diagnostics.metrics.correctRefusalTitle', { pct: (correctRefusal * 100).toFixed(0) })
    if (correctRefusal >= 0.8) {
      items.push({ id: 'refusal_band', title, detail: i18n.t('diagnostics.metrics.correctRefusalOk'), severity: 'ok' })
    } else if (correctRefusal >= 0.6) {
      items.push({ id: 'refusal_band', title, detail: i18n.t('diagnostics.metrics.correctRefusalWarn'), severity: 'warn' })
    } else {
      items.push({ id: 'refusal_band', title, detail: i18n.t('diagnostics.metrics.correctRefusalError'), severity: 'error' })
    }
  }

  const recall = m.retrieval_recall_at_k
  if (recall !== undefined) {
    const title = i18n.t('diagnostics.metrics.recallTitle', { value: recall.toFixed(3) })
    if (recall >= 0.6) {
      items.push({ id: 'recall_band', title, detail: i18n.t('diagnostics.metrics.recallOk'), severity: 'ok' })
    } else if (recall >= 0.3) {
      items.push({ id: 'recall_band', title, detail: i18n.t('diagnostics.metrics.recallWarn'), severity: 'warn' })
    } else {
      items.push({ id: 'recall_band', title, detail: i18n.t('diagnostics.metrics.recallError'), severity: 'error' })
    }
  }

  const answerSim = m.answer_similarity
  if (answerSim !== undefined) {
    const title = i18n.t('diagnostics.metrics.answerSimilarityTitle', { value: answerSim.toFixed(3) })
    if (answerSim >= 0.7) {
      items.push({ id: 'similarity_band', title, detail: i18n.t('diagnostics.metrics.answerSimilarityOk'), severity: 'ok' })
    } else if (answerSim >= 0.4) {
      items.push({ id: 'similarity_band', title, detail: i18n.t('diagnostics.metrics.answerSimilarityWarn'), severity: 'warn' })
    } else {
      items.push({ id: 'similarity_band', title, detail: i18n.t('diagnostics.metrics.answerSimilarityWeak'), severity: 'warn' })
    }
  }

  const notFoundCount = (run.question_results || []).filter(q => q.is_refusal).length
  const total = run.question_results?.length || 1
  if (notFoundCount > 0) {
    items.push({
      id: 'not_found_count',
      title: i18n.t('diagnostics.metrics.notFoundTitle', { count: notFoundCount, total }),
      detail: i18n.t('diagnostics.metrics.notFoundDetail'),
      severity: 'info',
    })
  }

  return items
}

export function diagnoseRetrieval(run: ExperimentDetail): DiagnosticItem[] {
  const items: DiagnosticItem[] = []
  const refs = run.question_results?.flatMap(q => (q as any).source_refs || []) || []

  if (refs.length === 0) {
    items.push({ id: 'sources_absent', title: 'Source refs', detail: i18n.t('diagnostics.retrieval.noData'), severity: 'info' })
    return items
  }

  // Check for low dense scores
  const denseScores = refs.map((r: any) => r.dense_score || 0).filter((s: number) => s > 0)
  if (denseScores.length > 0) {
    const avgDense = denseScores.reduce((a: number, b: number) => a + b, 0) / denseScores.length
    const title = i18n.t('diagnostics.retrieval.avgDenseScoreTitle', { value: avgDense.toFixed(2) })
    if (avgDense < 0.4) {
      items.push({
        id: 'dense_score_low',
        title,
        detail: i18n.t('diagnostics.retrieval.lowDenseScore'),
        severity: 'error',
        action: { label: i18n.t('diagnostics.retrieval.reindexAction'), href: '/corpus' },
      })
    } else {
      items.push({ id: 'dense_score_low', title, detail: i18n.t('diagnostics.retrieval.normalDenseScore'), severity: 'ok' })
    }
  }

  return items
}

export function diagnoseLatency(stageTrace: StageTrace | null | undefined): DiagnosticItem[] {
  if (!stageTrace || stageTrace.total_ms === 0) {
    return [{ id: 'latency_unavailable', title: 'Latency', detail: i18n.t('diagnostics.latency.noData'), severity: 'info' }]
  }

  const items: DiagnosticItem[] = []
  const t = stageTrace

  const totalTitle = i18n.t('diagnostics.latency.totalTimeTitle', { seconds: (t.total_ms / 1000).toFixed(1) })
  if (t.total_ms < 3000) {
    items.push({ id: 'latency_total', title: totalTitle, detail: i18n.t('diagnostics.latency.excellent'), severity: 'ok' })
  } else if (t.total_ms < 10000) {
    items.push({ id: 'latency_total', title: totalTitle, detail: i18n.t('diagnostics.latency.acceptable'), severity: 'warn' })
  } else {
    items.push({ id: 'latency_total', title: totalTitle, detail: i18n.t('diagnostics.latency.slow'), severity: 'error' })
  }

  if (t.total_ms > 0) {
    const genPct = Math.round(t.generate_ms / t.total_ms * 100)
    if (genPct > 80) {
      items.push({
        id: 'latency_generate',
        title: i18n.t('diagnostics.latency.generateTitle', { pct: genPct }),
        detail: i18n.t('diagnostics.latency.generateBottleneck'),
        severity: 'warn',
        action: { label: i18n.t('diagnostics.latency.changeModelAction'), href: '/chat' },
      })
    }
    if (t.embed_ms > 800) {
      items.push({
        id: 'latency_embed',
        title: i18n.t('diagnostics.latency.embedTitle', { ms: Math.round(t.embed_ms) }),
        detail: i18n.t('diagnostics.latency.slowEmbedder'),
        severity: 'warn',
      })
    }
  }

  return items
}
