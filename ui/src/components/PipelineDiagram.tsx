import { useTranslation } from 'react-i18next'
import { StageTrace } from '../api/client'

/**
 * Horizontal RAG-pipeline diagram: one segment per stage, segment width ∝ its
 * share of total latency, colour by status. Stages that didn't run (0 ms) are
 * omitted. Works for a single question's trace or a dataset-averaged trace.
 *
 * Data comes from core/models.py:StageTrace, persisted per question and
 * averaged in core/experiment/runner.py:avg_stage_trace().
 */

interface StageDef {
  key: keyof StageTrace
  // i18n key, resolved at the render site. The stages used to be labelled with
  // English strings in the code, the one place in the interface that did not
  // get translated along with everything else.
  label: string
  // Optional companion counter shown under the latency (e.g. n_reranked).
  // countLabel is an i18n key (resolved with t() at the render site), not
  // display text.
  countKey?: keyof StageTrace
  countLabel?: string
}

// Order follows the actual pipeline flow (core/pipeline.py NaivePipeline.run).
const STAGES: StageDef[] = [
  { key: 'embed_ms', label: 'pipelineDiagram.stages.embed', countKey: 'input_tokens', countLabel: 'pipelineDiagram.units.tokens' },
  { key: 'dense_retrieve_ms', label: 'pipelineDiagram.stages.dense', countKey: 'n_dense', countLabel: 'pipelineDiagram.units.chunks' },
  { key: 'sparse_retrieve_ms', label: 'pipelineDiagram.stages.sparse', countKey: 'n_sparse', countLabel: 'pipelineDiagram.units.chunks' },
  { key: 'graph_ms', label: 'pipelineDiagram.stages.graph', countKey: 'n_graph', countLabel: 'pipelineDiagram.units.nodes' },
  { key: 'merge_ms', label: 'pipelineDiagram.stages.merge', countKey: 'n_merged', countLabel: 'pipelineDiagram.units.chunks' },
  { key: 'rerank_ms', label: 'pipelineDiagram.stages.rerank', countKey: 'n_reranked', countLabel: 'pipelineDiagram.units.chunks' },
  { key: 'generate_ms', label: 'pipelineDiagram.stages.generate', countKey: 'output_tokens', countLabel: 'pipelineDiagram.units.tokens' },
  { key: 'grounding_ms', label: 'pipelineDiagram.stages.ground', countKey: 'n_unsupported', countLabel: 'pipelineDiagram.units.unsupported' },
]

function ms(v: number | undefined): number {
  return typeof v === 'number' && isFinite(v) ? v : 0
}

// The stage that stands out is the one eating half the time, plus grounding
// when it found unsupported statements. The rest stay neutral: a strip where
// every cell is coloured green also colours what there is no reason to look at,
// and then the colour stops meaning anything.
function stageClass(stage: StageDef, latency: number, share: number, trace: StageTrace): string {
  if (stage.key === 'grounding_ms' && ms(trace.n_unsupported) > 0) return 'pipeline-seg-hot'
  if (share >= 0.5 && latency > 0) return 'pipeline-seg-hot'
  return ''
}

export default function PipelineDiagram({
  trace,
  title,
}: {
  trace: StageTrace | null | undefined
  title?: string
}) {
  const { t } = useTranslation()
  if (!trace) {
    return (
      <div className="pipeline-diagram-empty">
        {t('pipelineDiagram.unavailable')}
      </div>
    )
  }

  const active = STAGES.map(s => ({ def: s, latency: ms(trace[s.key] as number) }))
    .filter(s => s.latency > 0)

  const totalMs = ms(trace.total_ms)
  // Sum of measured stages; fall back to it when total_ms is missing/0 so the
  // width proportions still add up.
  const sumStages = active.reduce((acc, s) => acc + s.latency, 0)
  const denom = sumStages > 0 ? sumStages : 1

  if (active.length === 0) {
    return (
      <div className="pipeline-diagram-empty">
        {t('pipelineDiagram.allZero')}
      </div>
    )
  }

  return (
    <div className="pipeline-diagram">
      {title && (
        <div className="pipeline-diagram-head">
          <span className="pipeline-diagram-title">{title}</span>
          <span className="pipeline-diagram-total">{t('pipelineDiagram.totalMs', { ms: totalMs.toFixed(1) })}</span>
        </div>
      )}
      <div className="pipeline-bar">
        {active.map(({ def, latency }) => {
          const share = latency / denom
          const cls = stageClass(def, latency, share, trace)
          const count = def.countKey ? ms(trace[def.countKey] as number) : undefined
          return (
            <div
              key={def.key as string}
              className={`pipeline-seg${cls ? ` ${cls}` : ''}`}
              style={{ flexGrow: Math.max(latency, 0.5) }}
              title={t('pipelineDiagram.segmentTitle', { label: t(def.label), ms: latency.toFixed(1), pct: (share * 100).toFixed(0) })}
            >
              <span className="pipeline-seg-label">{t(def.label)}</span>
              <span className="pipeline-seg-ms">{t('pipelineDiagram.segmentMs', { ms: latency.toFixed(1) })}</span>
              {count !== undefined && def.countLabel && (
                <span className="pipeline-seg-count">{count} {t(def.countLabel)}</span>
              )}
            </div>
          )
        })}
      </div>
    </div>
  )
}
