import type { ExperimentDetail, StageTrace } from '../api/client'
import i18n from '../i18n'

export type Severity = 'ok' | 'warn' | 'error' | 'info'

export interface DiagnosticItem {
  title: string
  detail: string
  severity: Severity
  action?: { label: string; href: string }
}

// Both languages, because this decides a diagnostic on the run screen and the
// platform runs against corpora in either. Russian-only, it counted zero
// refusals on the English demo realm, whose two out-of-scope questions exist
// precisely so `correct_refusal` has something to measure: the diagnostic
// reported "no refusals" for a run that had refused twice.
//
// Substrings and not whole answers: a refusal is a sentence inside an answer,
// not the answer itself.
const NOT_FOUND_PHRASES = [
  'информация по данному вопросу отсутствует',
  'не найдено',
  'не содержится',
  'отсутствует в предоставленных',
  'does not cover',
  'does not contain',
  'is not covered',
  'not found in the provided',
  'no information',
  'cannot answer',
  'unable to answer',
]

function isNotFound(text: string): boolean {
  const lower = text.toLowerCase()
  return NOT_FOUND_PHRASES.some(p => lower.includes(p))
}

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
      items.push({ title, detail: i18n.t('diagnostics.metrics.correctRefusalOk'), severity: 'ok' })
    } else if (correctRefusal >= 0.6) {
      items.push({ title, detail: i18n.t('diagnostics.metrics.correctRefusalWarn'), severity: 'warn' })
    } else {
      items.push({ title, detail: i18n.t('diagnostics.metrics.correctRefusalError'), severity: 'error' })
    }
  }

  const recall = m.retrieval_recall_at_k
  if (recall !== undefined) {
    const title = i18n.t('diagnostics.metrics.recallTitle', { value: recall.toFixed(3) })
    if (recall >= 0.6) {
      items.push({ title, detail: i18n.t('diagnostics.metrics.recallOk'), severity: 'ok' })
    } else if (recall >= 0.3) {
      items.push({ title, detail: i18n.t('diagnostics.metrics.recallWarn'), severity: 'warn' })
    } else {
      items.push({ title, detail: i18n.t('diagnostics.metrics.recallError'), severity: 'error' })
    }
  }

  const answerSim = m.answer_similarity
  if (answerSim !== undefined) {
    const title = i18n.t('diagnostics.metrics.answerSimilarityTitle', { value: answerSim.toFixed(3) })
    if (answerSim >= 0.7) {
      items.push({ title, detail: i18n.t('diagnostics.metrics.answerSimilarityOk'), severity: 'ok' })
    } else if (answerSim >= 0.4) {
      items.push({ title, detail: i18n.t('diagnostics.metrics.answerSimilarityWarn'), severity: 'warn' })
    } else {
      items.push({ title, detail: i18n.t('diagnostics.metrics.answerSimilarityWeak'), severity: 'warn' })
    }
  }

  const notFoundCount = (run.question_results || []).filter(q => isNotFound(q.generated_answer)).length
  const total = run.question_results?.length || 1
  if (notFoundCount > 0) {
    items.push({
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
    items.push({ title: 'Source refs', detail: i18n.t('diagnostics.retrieval.noData'), severity: 'info' })
    return items
  }

  // Check for low dense scores
  const denseScores = refs.map((r: any) => r.dense_score || 0).filter((s: number) => s > 0)
  if (denseScores.length > 0) {
    const avgDense = denseScores.reduce((a: number, b: number) => a + b, 0) / denseScores.length
    const title = i18n.t('diagnostics.retrieval.avgDenseScoreTitle', { value: avgDense.toFixed(2) })
    if (avgDense < 0.4) {
      items.push({
        title,
        detail: i18n.t('diagnostics.retrieval.lowDenseScore'),
        severity: 'error',
        action: { label: i18n.t('diagnostics.retrieval.reindexAction'), href: '/corpus' },
      })
    } else {
      items.push({ title, detail: i18n.t('diagnostics.retrieval.normalDenseScore'), severity: 'ok' })
    }
  }

  return items
}

export function diagnoseLatency(stageTrace: StageTrace | null | undefined): DiagnosticItem[] {
  if (!stageTrace || stageTrace.total_ms === 0) {
    return [{ title: 'Latency', detail: i18n.t('diagnostics.latency.noData'), severity: 'info' }]
  }

  const items: DiagnosticItem[] = []
  const t = stageTrace

  const totalTitle = i18n.t('diagnostics.latency.totalTimeTitle', { seconds: (t.total_ms / 1000).toFixed(1) })
  if (t.total_ms < 3000) {
    items.push({ title: totalTitle, detail: i18n.t('diagnostics.latency.excellent'), severity: 'ok' })
  } else if (t.total_ms < 10000) {
    items.push({ title: totalTitle, detail: i18n.t('diagnostics.latency.acceptable'), severity: 'warn' })
  } else {
    items.push({ title: totalTitle, detail: i18n.t('diagnostics.latency.slow'), severity: 'error' })
  }

  if (t.total_ms > 0) {
    const genPct = Math.round(t.generate_ms / t.total_ms * 100)
    if (genPct > 80) {
      items.push({
        title: i18n.t('diagnostics.latency.generateTitle', { pct: genPct }),
        detail: i18n.t('diagnostics.latency.generateBottleneck'),
        severity: 'warn',
        action: { label: i18n.t('diagnostics.latency.changeModelAction'), href: '/chat' },
      })
    }
    if (t.embed_ms > 800) {
      items.push({
        title: i18n.t('diagnostics.latency.embedTitle', { ms: Math.round(t.embed_ms) }),
        detail: i18n.t('diagnostics.latency.slowEmbedder'),
        severity: 'warn',
      })
    }
  }

  return items
}
