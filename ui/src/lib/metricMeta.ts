import i18n from '../i18n'

// Shared metric metadata — single source of truth for labels/hints/noise
// threshold so RunPage, RunCharts and ComparisonPage describe the same
// metric the same way. Mirrors:
//   - services/api_gateway/routers/experiments.py:_CompositeEvaluator (keys)
//   - core/eval/regression.py:DEFAULT_THRESHOLDS (noise threshold, 5%)
export interface MetricMeta {
  label: string
  hint: string
  /** Relative-change threshold below which a delta is noise, not signal —
   * matches core/eval/regression.py's regression-guard threshold (5%). */
  noiseThreshold: number
}

// `hint` is a getter (not a plain string) so it re-resolves through i18next
// on every access — a plain string baked in at module-load time would freeze
// whatever language was active on first import and never update when the
// user switches languages later.
export const METRIC_META: Record<string, MetricMeta> = {
  correct_refusal: {
    label: 'Correct Refusal',
    get hint() { return i18n.t('metricMeta.correctRefusal.hint') },
    noiseThreshold: 0.05,
  },
  retrieval_recall_at_k: {
    label: 'Retrieval Recall@K',
    get hint() { return i18n.t('metricMeta.retrievalRecallAtK.hint') },
    noiseThreshold: 0.05,
  },
  retrieval_precision_at_k: {
    label: 'Retrieval Precision@K',
    get hint() { return i18n.t('metricMeta.retrievalPrecisionAtK.hint') },
    noiseThreshold: 0.05,
  },
  answer_similarity: {
    label: 'Answer Similarity',
    get hint() { return i18n.t('metricMeta.answerSimilarity.hint') },
    noiseThreshold: 0.05,
  },
  context_support: {
    label: 'Context Support',
    get hint() { return i18n.t('metricMeta.contextSupport.hint') },
    noiseThreshold: 0.05,
  },
  grounded_in_correct_source: {
    label: 'Grounded in Correct Source',
    get hint() { return i18n.t('metricMeta.groundedInCorrectSource.hint') },
    noiseThreshold: 0.05,
  },
  citation_number_coverage: {
    label: 'Citation Number Coverage',
    get hint() { return i18n.t('metricMeta.citationNumberCoverage.hint') },
    noiseThreshold: 0.05,
  },
  pre_rerank_recall_at_k: {
    // The metric name itself mixes an English abbreviation with a Russian
    // qualifier ("before rerank") — unlike every other label here, so it
    // needs its own live-translated getter too.
    get label() { return i18n.t('metricMeta.preRerankRecallAtK.label') },
    get hint() { return i18n.t('metricMeta.preRerankRecallAtK.hint') },
    noiseThreshold: 0.05,
  },
  retrieval_average_precision: {
    label: 'Average Precision',
    get hint() { return i18n.t('metricMeta.retrievalAveragePrecision.hint') },
    noiseThreshold: 0.05,
  },
  answer_relevance: {
    label: 'Answer Relevance',
    get hint() { return i18n.t('metricMeta.answerRelevance.hint') },
    noiseThreshold: 0.05,
  },
  // legacy (pre-Phase-0 token-overlap evaluator)
  faithfulness: { label: 'Faithfulness (legacy)', get hint() { return i18n.t('metricMeta.legacyJaccard.hint') }, noiseThreshold: 0.05 },
  answer_relevancy: { label: 'Answer Relevancy (legacy)', get hint() { return i18n.t('metricMeta.legacyJaccard.hint') }, noiseThreshold: 0.05 },
  reference_overlap: { label: 'Reference Overlap (legacy)', get hint() { return i18n.t('metricMeta.legacyJaccard.hint') }, noiseThreshold: 0.05 },
}

export function metricLabel(key: string): string {
  return METRIC_META[key]?.label ?? key
}

export type DeltaVerdict = 'improved' | 'regressed' | 'noise'

/** Classifies a before→after change the same way the backend regression
 * guard does: a relative move smaller than noiseThreshold is noise, not a
 * real signal — regardless of which direction it points. All current
 * metrics are "higher is better", so sign alone decides improved/regressed
 * once it clears the noise band. */
export function classifyDelta(before: number, after: number, key: string): DeltaVerdict {
  const threshold = METRIC_META[key]?.noiseThreshold ?? 0.05
  if (before === 0) return after === 0 ? 'noise' : 'improved'
  const relChange = (after - before) / Math.abs(before)
  if (Math.abs(relChange) < threshold) return 'noise'
  return relChange > 0 ? 'improved' : 'regressed'
}
