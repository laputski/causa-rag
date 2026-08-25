import {
  ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, Cell,
  ComposedChart, Line,
} from 'recharts'
import { useEffect, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { tokens } from '../lib/themeTokens'
import type { ExperimentDetail } from '../api/client'
import i18n from '../i18n'

interface Props { run: ExperimentDetail }

// The colours are read off the document root, from the same tokens as the rest
// of the interface. Recharts puts a colour into an SVG attribute, where `var()`
// does not resolve, so the string has to be computed here. The literals that
// used to stand in this place gave the charts a dark palette of their own: they
// looked identical under all five palettes, and on the light theme the grid and
// the labels all but disappeared. The fallbacks live in one place,
// ui/src/lib/themeTokens.ts.
const TOKEN_NAMES = {
  bg: '--color-bg',
  surface: '--color-surface',
  border: '--color-border',
  text: '--color-text',
  textMuted: '--color-text-muted',
  primary: '--color-primary',
  primaryH: '--color-primary-h',
  success: '--color-success',
  warning: '--color-warning',
  danger: '--color-danger',
  // An axis label's size is the same step off the scale as everywhere else,
  // read out as a string for the same reason: it is an SVG attribute too.
  tickFs: '--fs-2xs',
} as const

function readTokens() {
  return tokens(TOKEN_NAMES)
}

/** The values used on the first render. The component recomputes them through
 *  `useChartColors` whenever theme or palette changes; this object exists for
 *  module-level constants evaluated before mount. */
const COLOR = readTokens()

/** Recomputes the colours when theme or palette changes: both stamp an
 *  attribute on the document root, and the observer watches exactly that rather
 *  than subscribing to a context — a chart has no need to know who changed the
 *  appearance or why. */
function useChartColors() {
  const [colors, setColors] = useState(COLOR)
  useEffect(() => {
    const update = () => setColors(readTokens())
    update()
    const mo = new MutationObserver(update)
    mo.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme', 'data-palette'] })
    return () => mo.disconnect()
  }, [])
  return colors
}

// Eval Measurement Trustworthiness, Phase 0 — current metric schema
// (services/api_gateway/routers/experiments.py:_CompositeEvaluator). Legacy
// keys kept so charts on historical pre-Phase-0 runs still render.
const METRIC_META: Record<string, { label: string; color: string }> = {
  correct_refusal: { label: 'Correct Refusal', color: COLOR.primary },
  retrieval_recall_at_k: { label: 'Retrieval Recall@K', color: COLOR.success },
  retrieval_precision_at_k: { label: 'Retrieval Precision@K', color: COLOR.primaryH },
  answer_similarity: { label: 'Answer Similarity', color: COLOR.warning },
  context_support: { label: 'Context Support', color: COLOR.danger },
  // Eval Measurement Trustworthiness, Phase 1.
  grounded_in_correct_source: { label: 'Grounded in Correct Source', color: COLOR.primaryH },
  // Mixes an English abbreviation with a Russian qualifier ("before rerank")
  // unlike every other label here — a live getter (see lib/metricMeta.ts's
  // METRIC_META for the same pattern) so it re-resolves through i18next on
  // every access instead of freezing whatever language was active at
  // module-load time.
  get pre_rerank_recall_at_k() { return { label: i18n.t('runCharts.preRerankRecallAtKLabel'), color: COLOR.warning } },
  retrieval_average_precision: { label: 'Average Precision', color: COLOR.success },
  answer_relevance: { label: 'Answer Relevance', color: COLOR.primary },
  // legacy (pre-Phase-0 runs)
  faithfulness: { label: 'Faithfulness (legacy)', color: COLOR.primary },
  answer_relevancy: { label: 'Answer Relevancy (legacy)', color: COLOR.success },
  reference_overlap: { label: 'Reference Overlap (legacy)', color: COLOR.warning },
}

/** A titled chart. It used to be a card, a border around the heading and the
 *  content together, which is exactly what the visual language replaced with a
 *  rule: a border encloses a self-contained object, and a heading with what
 *  sits under it is a section. */
function ChartCard({ title, hint, children }: { title: string; hint?: string; children: React.ReactNode }) {
  return (
    <section className="section chart-section">
      <div className="section-rule">
        <h2 className="section-title">{title}</h2>
      </div>
      {hint && <p className="chart-hint">{hint}</p>}
      {children}
    </section>
  )
}

function ChartTooltip({ active, payload, label }: any) {
  if (!active || !payload?.length) return null
  return (
    <div className="chart-tip">
      <div className="chart-tip-label">{label}</div>
      {payload.map((p: any) => (
        <div key={p.dataKey} style={{ color: p.color }}>
          {p.name}: <strong>{typeof p.value === 'number' ? p.value.toFixed(p.value < 1 ? 3 : 0) : p.value}</strong>
        </div>
      ))}
    </div>
  )
}

const axisProps = {
  stroke: COLOR.border,
  // A number rather than a token: `tick` ends up in an SVG attribute, where
  // `var()` does not resolve, leaving the axis label with no font size.
  tick: { fill: COLOR.textMuted, fontSize: COLOR.tickFs },
}

/** Caps rendered x-axis tick labels regardless of how many questions a run
 * has — confirmed at n=100 the default "show every tick" packs labels
 * ~8px apart while each rotated label spans 60-80px, i.e. solid overlap.
 * Bars/lines still render for every data point; only label density drops.
 * Below maxLabels nothing changes (interval=0 = old behaviour). */
function tickInterval(n: number, maxLabels = 12): number {
  return n <= maxLabels ? 0 : Math.ceil(n / maxLabels) - 1
}

/** Steeper angle + smaller font for dense per-question axes — keeps each
 * label's horizontal footprint well under the spacing tickInterval leaves
 * between displayed labels. */
type ChartColors = ReturnType<typeof readTokens>

function questionAxisProps(n: number, colors: ChartColors) {
  const dense = n > 12
  return {
    interval: tickInterval(n),
    angle: dense ? -60 : -35,
    textAnchor: 'end' as const,
    height: dense ? 70 : 60,
    tick: { fill: colors.textMuted, fontSize: dense ? 10 : 11 },
  }
}

/** Minimum horizontal pixels given to each question's bar group — wide
 * enough for a steep-angle rotated id label not to collide with its
 * neighbors, so a dense run scrolls instead of squeezing/skipping labels. */

/** 1. The distribution of per-question scores.
 *
 * A chart of a hundred and forty-three questions across eight metrics stood
 * here, more than a thousand bars on one axis. It announced that it showed
 * "which questions pull the average down" and that is precisely what it could
 * not show: a single question cannot be picked out of a thousand bars, the
 * labels overlapped, and scrolling sideways made comparing neighbours
 * impossible.
 *
 * The question somebody opens a chart like this for is whether the average is
 * representative at all, or whether two clusters sit under it. A distribution
 * answers that: one histogram per metric, ten buckets from zero to one, with
 * the mean marked by a line. Which questions actually failed is answered by the
 * "Needs attention" list on the overview tab, and duplicating it as a chart
 * gains nothing.
 */
function ScoreDistributionChart({ run }: Props) {
  const { t } = useTranslation()
  const presentMetrics = Object.keys(run.aggregate_metrics).filter(k => METRIC_META[k])

  const dists = presentMetrics.map(key => {
    const values = run.question_results
      .map(qr => qr.metrics?.[key])
      .filter((v): v is number => typeof v === 'number')
    const buckets = new Array(10).fill(0)
    for (const v of values) buckets[Math.min(9, Math.max(0, Math.floor(v * 10)))] += 1
    const mean = values.length ? values.reduce((a, b) => a + b, 0) / values.length : 0
    return { key, buckets, mean, n: values.length, peak: Math.max(1, ...buckets) }
  }).filter(d => d.n > 0)

  if (dists.length === 0) return null

  return (
    <ChartCard
      title={t('runCharts.distribution.title')}
      hint={t('runCharts.distribution.hint')}
    >
      <div className="dist-grid">
        {dists.map(d => (
          <div key={d.key} className="dist-cell">
            <div className="dist-head">
              <span className="dist-name">{METRIC_META[d.key].label}</span>
              <span className="dist-mean">{d.mean.toFixed(3)}</span>
            </div>
            <div className="dist-bars">
              {d.buckets.map((n, i) => (
                <div
                  key={i}
                  className={`dist-bar${i < 3 ? ' low' : ''}${Math.floor(d.mean * 10) === i ? ' at-mean' : ''}`}
                  style={{ height: `${Math.max(2, (n / d.peak) * 100)}%` }}
                  title={t('runCharts.distribution.bucket', { n, from: (i / 10).toFixed(1), to: ((i + 1) / 10).toFixed(1) })}
                />
              ))}
            </div>
            <div className="dist-axis"><span>0</span><span>1</span></div>
          </div>
        ))}
      </div>
    </ChartCard>
  )
}

/** 2. Baseline vs current — turns regression.deltas (currently plain text like
 * "0.0339 < 0.0385") into an at-a-glance paired comparison. */
function BaselineVsCurrentChart({ run }: Props) {
  const colors = useChartColors()
  const { t } = useTranslation()
  if (!run.regression) return null
  const data = run.regression.deltas.map(d => ({
    metric: METRIC_META[d.metric]?.label ?? d.metric,
    baseline: d.baseline,
    current: d.current,
    regressed: d.regressed,
  }))

  return (
    <ChartCard
      title={t('runCharts.baselineVsCurrent.title')}
      hint={t('runCharts.baselineVsCurrent.hint', { baselineRunId: run.regression.baseline_run_id })}
    >
      <ResponsiveContainer width="100%" height={Math.max(320, data.length * 46)}>
        <BarChart data={data} layout="vertical" margin={{ top: 8, right: 24, left: 8, bottom: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={colors.border} horizontal={false} />
          <XAxis type="number" {...axisProps} />
          <YAxis type="category" dataKey="metric" {...axisProps} width={140} />
          <Tooltip content={<ChartTooltip />} />
          <Legend wrapperStyle={{ fontSize: 'var(--fs-xs)', color: colors.textMuted }} />
          <Bar dataKey="baseline" name="Baseline" fill={colors.textMuted} radius={[0, 3, 3, 0]} />
          <Bar dataKey="current" name={t('runCharts.baselineVsCurrent.current')} radius={[0, 3, 3, 0]}>
            {data.map((d, i) => (
              <Cell key={i} fill={d.regressed ? colors.danger : colors.success} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </ChartCard>
  )
}

/** 3. Answer length + retrieved-source count per question — surfaces clusters
 * of identical-length refusal boilerplate (the exact pattern found while
 * debugging runs 1f45055e/d296f848/3e496a9f) without grepping JSON by hand. */
function AnswerShapeChart({ run }: Props) {
  const colors = useChartColors()
  const { t } = useTranslation()
  const data = run.question_results.map(qr => ({
    id: qr.question_id.length > 14 ? qr.question_id.slice(0, 13) + '…' : qr.question_id,
    fullId: qr.question_id,
    answerLength: qr.generated_answer.length,
    sources: qr.source_refs?.length ?? 0,
  }))
  const avgLen = data.reduce((s, d) => s + d.answerLength, 0) / (data.length || 1)
  const suspiciouslyUniform = data.filter(d => Math.abs(d.answerLength - avgLen) < 1 && d.answerLength < avgLen * 1.2).length
  const refusalClusterHint = suspiciouslyUniform >= 3
    ? ` ${t('runCharts.answerShape.refusalClusterHint')}`
    : ''

  return (
    <ChartCard
      title={t('runCharts.answerShape.title')}
      hint={`${t('runCharts.answerShape.hint')}${refusalClusterHint}`}
    >
      <ResponsiveContainer width="100%" height={260}>
        <ComposedChart data={data} margin={{ top: 8, right: 8, left: -8, bottom: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={colors.border} vertical={false} />
          <XAxis dataKey="id" stroke={colors.border} {...questionAxisProps(data.length, colors)} />
          <YAxis yAxisId="len" {...axisProps} />
          <YAxis yAxisId="src" orientation="right" {...axisProps} allowDecimals={false} />
          <Tooltip content={<ChartTooltip />} labelFormatter={(_l, p) => p?.[0]?.payload?.fullId ?? ''} />
          <Legend wrapperStyle={{ fontSize: 'var(--fs-xs)', color: colors.textMuted }} />
          <Bar yAxisId="len" dataKey="answerLength" name={t('runCharts.answerShape.answerLength')} fill={colors.primaryH} radius={[3, 3, 0, 0]} />
          <Line yAxisId="src" dataKey="sources" name={t('runCharts.answerShape.sources')} stroke={colors.warning} strokeWidth={2} dot={{ r: 3 }} />
        </ComposedChart>
      </ResponsiveContainer>
    </ChartCard>
  )
}

export default function RunCharts({ run }: Props) {
  if (!run.question_results.length) return null
  return (
    <>
      <ScoreDistributionChart run={run} />
      <BaselineVsCurrentChart run={run} />
      <AnswerShapeChart run={run} />
    </>
  )
}
