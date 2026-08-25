import { useTranslation } from 'react-i18next'
import type { StageTrace } from '../api/client'

interface Props {
  trace: StageTrace
  promptPreview?: string
}

function ms(v: number) {
  if (v >= 1000) return `${(v / 1000).toFixed(1)}s`
  return `${Math.round(v)}ms`
}

function color(v: number, warn = 500, bad = 2000) {
  if (v === 0) return 'trace-zero'
  if (v < warn) return 'trace-good'
  if (v < bad) return 'trace-warn'
  return 'trace-bad'
}

interface StageBlock {
  label: string
  time: number
  warnMs: number
  badMs: number
  detail: string
}

export default function PipelineTrace({ trace, promptPreview }: Props) {
  const { t } = useTranslation()
  const stages: StageBlock[] = [
    {
      label: 'Embed',
      time: trace.embed_ms,
      warnMs: 200, badMs: 1000,
      detail: `${trace.context_chars > 0 ? `${trace.context_chars} chars context` : 'query vector'}`,
    },
    {
      label: 'Dense',
      time: trace.dense_retrieve_ms,
      warnMs: 300, badMs: 1500,
      detail: `${trace.n_dense || trace.n_merged} chunks`,
    },
    ...(trace.sparse_retrieve_ms > 0 ? [{
      label: 'Sparse',
      time: trace.sparse_retrieve_ms,
      warnMs: 300, badMs: 1500,
      detail: `${trace.n_sparse} chunks`,
    }] : []),
    ...(trace.merge_ms > 0 ? [{
      label: 'Merge',
      time: trace.merge_ms,
      warnMs: 50, badMs: 200,
      detail: `→ ${trace.n_merged} unique`,
    }] : []),
    {
      label: 'Generate',
      time: trace.generate_ms,
      warnMs: 3000, badMs: 10000,
      detail: trace.output_tokens > 0 ? `${trace.output_tokens} tok` : '',
    },
  ]

  return (
    <div className="pipeline-trace">
      <div className="trace-stages">
        {stages.map((s, i) => (
          <div key={i} className="trace-stage-wrap">
            <div className={`trace-stage ${color(s.time, s.warnMs, s.badMs)}`}>
              <span className="trace-label">{s.label}</span>
              <span className="trace-time">{s.time > 0 ? ms(s.time) : '—'}</span>
              {s.detail && <span className="trace-detail">{s.detail}</span>}
            </div>
            {i < stages.length - 1 && <span className="trace-arrow">→</span>}
          </div>
        ))}
        <div className="trace-total">
          {t('pipelineTrace.total')} <strong>{ms(trace.total_ms)}</strong>
          {trace.n_deduped > 0 && trace.n_deduped < trace.n_merged &&
            <span className="trace-dedup"> · dedup {trace.n_merged}→{trace.n_deduped}</span>}
          {trace.input_tokens > 0 &&
            <span className="trace-tokens"> · {trace.input_tokens}+{trace.output_tokens} tok</span>}
        </div>
      </div>
      {promptPreview && (
        <details className="prompt-preview">
          <summary>{t('pipelineTrace.promptPreview')}</summary>
          <pre>{promptPreview}</pre>
        </details>
      )}
    </div>
  )
}
