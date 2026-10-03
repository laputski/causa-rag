import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { useTranslation } from 'react-i18next'
import { Download } from 'lucide-react'
import { api, type ExperimentDetail, type ServedPart } from '../api/client'

// What ran a run: each service that answered for it and where that service
// is. The run recorded this itself; nothing here is read from the Realm as it
// is configured today, which may no longer be how the run was made.

const STORE_LABEL: Record<string, string> = {
  qdrant: 'runPage.servedBy.store.qdrant',
  opensearch: 'runPage.servedBy.store.opensearch',
  neo4j: 'runPage.servedBy.store.neo4j',
}

interface Row {
  key: string
  label: string
  what: string
  where: string
  // Undefined when the run did not say, which is not the same as local.
  remote?: boolean
}

function useRows(run: ExperimentDetail): Row[] {
  const { t } = useTranslation()
  const served = run.served_by ?? {}
  const part = (key: string, label: string, p?: ServedPart): Row[] => p ? [{
    key, label,
    what: p.model || '—',
    where: p.provider === 'in_process' ? t('runPage.servedBy.insidePlatform') : (p.address || '—'),
    remote: p.remote,
  }] : []
  const rows: Row[] = [
    ...(served.external_system ? [{
      key: 'external', label: t('runPage.servedBy.externalSystem'), what: '—',
      where: served.external_system.address, remote: served.external_system.remote,
    }] : []),
    ...part('generator', t('runPage.servedBy.generator'), served.generator),
    ...part('embedder', t('runPage.servedBy.embedder'), served.embedder),
    ...Object.entries(served.stores ?? {}).map(([kind, store]) => ({
      key: kind, label: STORE_LABEL[kind] ? t(STORE_LABEL[kind]) : kind, what: kind,
      where: store.address, remote: store.remote,
    })),
  ]
  const judge = run.judge
  if (judge && (judge.model || judge.base_url)) {
    rows.push({ key: 'judge', label: t('runPage.servedBy.judge'), what: judge.model || '—', where: judge.base_url || '—' })
  }
  return rows
}

// Decimal units, because that is what the labels say: a kilobyte is a
// thousand bytes, and the number then agrees with what a file manager shows.
function formatBytes(bytes: number, t: (key: string, options?: Record<string, unknown>) => string): string {
  if (bytes >= 1_000_000) return t('runPage.export.megabytes', { n: (bytes / 1_000_000).toFixed(1) })
  return t('runPage.export.kilobytes', { n: Math.max(1, Math.round(bytes / 1000)) })
}

/** Two sizes, asked only when somebody opens the control: the answer costs
 *  the server a read of every fragment text the run refers to. */
function RunExport({ runId }: { runId: string }) {
  const { t } = useTranslation()
  const [open, setOpen] = useState(false)
  const [withTexts, setWithTexts] = useState(false)
  const { data: size, isError } = useQuery({
    queryKey: ['experiment-export-size', runId],
    queryFn: () => api.experiments.exportSize(runId),
    enabled: open,
  })
  if (!open) {
    return (
      <button type="button" className="btn-sm inline-4" onClick={() => setOpen(true)}>
        <Download size={13} /> {t('runPage.export.open')}
      </button>
    )
  }
  return (
    <div data-testid="run-export">
      <div className="eyebrow">{t('runPage.export.heading')}</div>
      <div className="export-row" role="radiogroup" aria-label={t('runPage.export.heading')}>
        <label className="export-choice">
          <input type="radio" name={`export-${runId}`} checked={!withTexts} onChange={() => setWithTexts(false)} />
          {t('runPage.export.withoutTexts')}
          {size && <span className="size">{formatBytes(size.bytes_without_texts, t)}</span>}
        </label>
        <label className="export-choice">
          <input type="radio" name={`export-${runId}`} checked={withTexts} onChange={() => setWithTexts(true)} />
          {t('runPage.export.withTexts')}
          {size && <span className="size">{formatBytes(size.bytes_with_texts, t)}</span>}
        </label>
        <a className="btn btn-primary" href={api.experiments.exportUrl(runId, withTexts)} download>
          {t('runPage.export.download')}
        </a>
      </div>
      {isError && <p className="field-note">{t('runPage.export.sizeUnknown')}</p>}
    </div>
  )
}

export default function RunServedBy({ run }: { run: ExperimentDetail }) {
  const { t } = useTranslation()
  const rows = useRows(run)
  const lastCall = run.served_by?.generator?.last_call
  const judge = run.judge
  const cleared = run.debug_cleared_at
  return (
    <div className="section" data-testid="run-served-by">
      <div className="section-rule">
        <h2 className="section-title">{t('runPage.servedBy.heading')}</h2>
      </div>
      {/* Said whenever the run did not record it, and not only when the table
          would be empty: a run with a judge and nothing else showed one row,
          which read as a run made by a judge alone (found by opening a run
          stored before services were recorded). */}
      {!run.served_by && <p className="field-note">{t('runPage.servedBy.notRecorded')}</p>}
      {rows.length > 0 && (
        <table className="served-table">
          <thead>
            <tr>
              <th>{t('runPage.servedBy.col.part')}</th>
              <th>{t('runPage.servedBy.col.what')}</th>
              <th>{t('runPage.servedBy.col.where')}</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {rows.map(row => (
              <tr key={row.key}>
                <td>{row.label}</td>
                <td className="mono">{row.what}</td>
                <td className="mono where">{row.where}</td>
                <td className="place">
                  {row.remote === true && <span className="badge badge-warn">{t('runPage.servedBy.external')}</span>}
                  {row.remote === false && <span className="badge badge-success">{t('runPage.servedBy.local')}</span>}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      {lastCall && (
        <p className="field-note">
          {lastCall.seed_sent ? t('runPage.servedBy.seedSent') : t('runPage.servedBy.seedNotSent')}
          {lastCall.served_model && <> {t('runPage.servedBy.servedModel', { model: lastCall.served_model })}</>}
        </p>
      )}
      {judge?.gap && (
        <p className="plain-note warn">{t('runPage.servedBy.noJudge')}{judge.note ? ` ${judge.note}` : ''}</p>
      )}
      {judge && !judge.gap && (judge.calls ?? 0) > 0 && (
        <p className="field-note">
          {t('runPage.servedBy.judgeWork', { count: judge.calls, seconds: Math.round(judge.seconds ?? 0) })}
          {' '}{t('runPage.servedBy.judgeCaveat')}
        </p>
      )}
      {cleared && (
        <p className="plain-note">
          {t('runPage.servedBy.cleared', { date: new Date(cleared).toLocaleDateString('ru-RU') })}
        </p>
      )}
      <RunExport runId={run.run_id} />
    </div>
  )
}
