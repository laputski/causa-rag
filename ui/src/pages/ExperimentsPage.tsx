import { useMemo, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { Link } from 'react-router-dom'
import {
  ChevronUp, ChevronDown, ChevronsUpDown, Columns3, Trash2, Search, X, Pin,
} from 'lucide-react'
import { api, type ExperimentItem } from '../api/client'
import { durationSeconds, formatDuration } from '../lib/format'
import { useRealm, useRealmPath } from '../context/RealmContext'
import { useColumnWidths } from '../hooks/useColumnWidths'
import { useConfirm } from '../components/ConfirmDialog'
import { METRIC_META, metricLabel } from '../lib/metricMeta'

// The run table is one of the main screens, and until this change it showed
// three columns that could never fill.
//
// `faithfulness`, `answer_relevancy` and `reference_overlap` are the names of a
// token-overlap evaluator retired in Phase 0. The current `_CompositeEvaluator`
// writes ten different keys.
//
// Checked against a live install: 74 runs carrying metrics across two realms,
// and not one of them held any of the three old names. The dash stood in a
// hundred percent of rows, and it read as "the runs are empty" rather than as
// "these columns name something nobody computes".
//
// Which metrics appear is decided by the realm's key-metric list, the same one
// that governs the number band on the run screen. The old three appear only on
// runs that genuinely carry them, labelled legacy.

type SortKey = string
type SortDir = 'asc' | 'desc'

/** Default widths. The utility columns are narrow and fixed, the metric
 *  columns identical: their numbers are of the same order, and differing widths
 *  would read as differing importance. */
const BASE_WIDTHS: Record<string, number> = {
  select: 34, run_id: 88, name: 300, started_at: 132, dataset_name: 150, n_questions: 64,
}
const METRIC_WIDTH = 118

function SortIcon({ col, active, dir }: { col: SortKey; active: SortKey; dir: SortDir }) {
  if (col !== active) return <ChevronsUpDown size={11} className="sort-icon" />
  return dir === 'asc'
    ? <ChevronUp size={11} className="sort-icon on" />
    : <ChevronDown size={11} className="sort-icon on" />
}

function getVal(e: ExperimentItem, key: SortKey): string | number {
  switch (key) {
    case 'run_id':       return e.run_id ?? ''
    case 'name':         return e.name ?? ''
    case 'started_at':   return e.started_at ?? ''
    case 'dataset_name': return e.dataset_name ?? ''
    case 'n_questions':  return e.n_questions ?? 0
    default:             return e.aggregate_metrics?.[key] ?? -1
  }
}

/** The threshold below which a number is coloured as a concern. One across
 *  every metric: all of them are higher-is-better and live in 0..1, and a
 *  per-column threshold would turn the colour into a judgement the platform
 *  does not make. */
function metricColor(v: number): string {
  return v >= 0.6 ? 'var(--color-success)' : v >= 0.3 ? 'var(--color-warning)' : 'var(--color-danger)'
}

export default function ExperimentsPage() {
  const { t } = useTranslation()
  const { activeRealm, activeRealmId } = useRealm()
  const toRealm = useRealmPath()
  const queryClient = useQueryClient()
  const { confirm, dialog } = useConfirm()

  const [sortKey, setSortKey] = useState<SortKey>('started_at')
  const [sortDir, setSortDir] = useState<SortDir>('desc')
  const [selected, setSelected] = useState<string[]>([])
  const [deleting, setDeleting] = useState<string | null>(null)
  const [bulkDeleting, setBulkDeleting] = useState(false)
  const [query, setQuery] = useState('')
  const [statusFilter, setStatusFilter] = useState<'all' | 'running' | 'baseline'>('all')
  const [datasetFilter, setDatasetFilter] = useState<string>('')
  const [columnsOpen, setColumnsOpen] = useState(false)

  const { data: experiments = [], isLoading } = useQuery({
    queryKey: ['experiments', activeRealmId],
    queryFn: () => api.experiments.list({ realmId: activeRealmId }),
    refetchInterval: q => (q.state.data?.some(e => e.status === 'running') ? 3000 : false),
  })

  // The realm's key metrics. Where a realm declares none, the live keys of the
  // runs themselves are used, minus the old three: filling in the names of a
  // retired evaluator would reproduce exactly the defect this exists to fix.
  const keyMetrics = useMemo(() => {
    const declared = activeRealm?.key_metrics
    if (declared && declared.length > 0) return declared
    const seen = new Set<string>()
    for (const e of experiments) for (const k of Object.keys(e.aggregate_metrics ?? {})) seen.add(k)
    const live = [...seen].filter(k => !k.endsWith('_legacy') && !LEGACY.has(k))
    return live.length > 0 ? live.slice(0, 5) : DEFAULT_METRICS
  }, [activeRealm?.key_metrics, experiments])

  // A legacy metric becomes a column only when at least one run genuinely
  // carries it: an empty column asserts that the quantity exists and equals a
  // dash.
  const legacyPresent = useMemo(
    () => [...LEGACY].filter(k => experiments.some(e => e.aggregate_metrics?.[k] != null)),
    [experiments],
  )

  const columns = useMemo(() => [...keyMetrics, ...legacyPresent], [keyMetrics, legacyPresent])

  const defaults = useMemo(
    () => ({ ...BASE_WIDTHS, ...Object.fromEntries(columns.map(c => [c, METRIC_WIDTH])) }),
    [columns],
  )
  const { widths, onPointerDown, onPointerMove, stop, reset } =
    useColumnWidths(`rag-platform-cols:${activeRealmId ?? 'none'}`, defaults)

  // The band's numbers, computed from the list already loaded. A separate
  // request for four numbers would be one more source that can disagree with
  // the table beneath it.
  const baseline = useMemo(() => experiments.find(e => e.is_baseline), [experiments])
  const last = useMemo(
    () => [...experiments].sort((a, b) => (b.started_at ?? '').localeCompare(a.started_at ?? ''))[0],
    [experiments],
  )
  const best = useMemo(() => {
    const key = keyMetrics[0]
    if (!key) return null
    let top: { value: number; run: ExperimentItem } | null = null
    for (const run of experiments) {
      const v = run.aggregate_metrics?.[key]
      if (typeof v === 'number' && (top == null || v > top.value)) top = { value: v, run }
    }
    return top
  }, [experiments, keyMetrics])

  const datasets = useMemo(
    () => [...new Set(experiments.map(e => e.dataset_name).filter(Boolean))].sort(),
    [experiments],
  )

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase()
    return experiments.filter(e => {
      if (statusFilter === 'running' && e.status !== 'running') return false
      if (statusFilter === 'baseline' && !e.is_baseline) return false
      if (datasetFilter && e.dataset_name !== datasetFilter) return false
      if (!q) return true
      return `${e.name} ${e.run_id} ${e.dataset_name}`.toLowerCase().includes(q)
    })
  }, [experiments, query, statusFilter, datasetFilter])

  const sorted = useMemo(() => {
    return [...filtered].sort((a, b) => {
      const va = getVal(a, sortKey)
      const vb = getVal(b, sortKey)
      const cmp = va < vb ? -1 : va > vb ? 1 : 0
      return sortDir === 'asc' ? cmp : -cmp
    })
  }, [filtered, sortKey, sortDir])

  const toggleSort = (key: SortKey) => {
    if (sortKey === key) setSortDir(d => (d === 'asc' ? 'desc' : 'asc'))
    else { setSortKey(key); setSortDir('desc') }
  }

  const saveKeyMetrics = async (next: string[]) => {
    if (!activeRealmId) return
    await fetch(`/api/realms/${activeRealmId}/key-metrics`, {
      method: 'PUT', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ key_metrics: next }),
    })
    await queryClient.invalidateQueries({ queryKey: ['realms'] })
    window.location.reload()
  }

  const handleDelete = async (id: string) => {
    const ok = await confirm({
      title: t('experimentsPage.confirmDelete', { id }), danger: true,
      confirmLabel: t('experimentsPage.delete'),
    })
    if (!ok) return
    setDeleting(id)
    try {
      await api.experiments.delete(id)
      setSelected(prev => prev.filter(x => x !== id))
      await queryClient.invalidateQueries({ queryKey: ['experiments'] })
    } finally {
      setDeleting(null)
    }
  }

  const handleBulkDelete = async () => {
    if (selected.length === 0) return
    const ok = await confirm({
      title: t('experimentsPage.confirmBulkDelete', { count: selected.length }), danger: true,
      confirmLabel: t('experimentsPage.delete'),
    })
    if (!ok) return
    setBulkDeleting(true)
    try {
      await Promise.all(selected.map(id => api.experiments.delete(id)))
      setSelected([])
      await queryClient.invalidateQueries({ queryKey: ['experiments'] })
    } finally {
      setBulkDeleting(false)
    }
  }

  const toggleSelect = (id: string) =>
    setSelected(prev => (prev.includes(id) ? prev.filter(x => x !== id) : [...prev, id]))
  const allSelected = sorted.length > 0 && sorted.every(e => selected.includes(e.run_id))

  const Th = ({ col, children, align }: { col: SortKey; children: React.ReactNode; align?: 'right' }) => (
    <th
      className={`sortable-th resizable-th${align === 'right' ? ' num' : ''}`}
      style={{ width: widths[col] ?? defaults[col] }}
      onClick={() => toggleSort(col)}
    >
      <span className="inline-center">
        {children}
        <SortIcon col={col} active={sortKey} dir={sortDir} />
      </span>
      {/* The resize edge. Its own pointer handler, because clicking it must
          not also sort the column. */}
      <span
        className="col-grip" role="separator" aria-orientation="vertical"
        onPointerDown={onPointerDown(col)} onPointerMove={onPointerMove}
        onPointerUp={stop} onPointerCancel={stop} onClick={e => e.stopPropagation()}
      />
    </th>
  )

  return (
    <div className="page page-wide">
      {dialog}
      <div className="flex-between mb-8 align-end">
        <div>
          <h1 className="page-title tighter">{t('experimentsPage.title')}</h1>
          <p className="text-muted sm">
            {t('experimentsPage.subtitle', { shown: sorted.length, total: experiments.length })}
          </p>
        </div>
        <div className="flex-row res-actions">
          {selected.length === 2 && (
            <Link to={toRealm(`/compare?a=${selected[0]}&b=${selected[1]}`)} className="btn btn-primary">
              {t('experimentsPage.compareSelected')}
            </Link>
          )}
          {selected.length > 0 && (
            <button onClick={handleBulkDelete} disabled={bulkDeleting} className="btn btn-danger">
              <Trash2 size={13} />
              {bulkDeleting ? t('experimentsPage.deleting') : t('experimentsPage.deleteSelected', { count: selected.length })}
            </button>
          )}
          <Link to={toRealm('/new')} className="btn btn-primary">{t('experimentsPage.newRun')}</Link>
        </div>
      </div>

      {/* The number band above the table. The page used to open straight into
          the list, so "how many are there and which is best" had to be
          extracted by sorting. */}
      <div className="stat-band">
        <div className="stat-cell">
          <div className="eyebrow">{t('experimentsPage.stats.totalRuns')}</div>
          <div className="metric-val">{experiments.length || '—'}</div>
          <div className="stat-sub">{t('experimentsPage.stats.shown', { n: sorted.length })}</div>
        </div>
        <div className="stat-cell">
          <div className="eyebrow">{t('experimentsPage.baseline')}</div>
          <div className="metric-val mono-val">{baseline ? baseline.run_id.slice(0, 8) : '—'}</div>
          <div className="stat-sub">
            {baseline?.started_at
              ? new Date(baseline.started_at).toLocaleDateString('ru-RU', { day: '2-digit', month: '2-digit' })
              : t('experimentsPage.stats.noBaseline')}
          </div>
        </div>
        <div className="stat-cell">
          <div className="eyebrow">{t('experimentsPage.stats.best', { metric: metricLabel(keyMetrics[0] ?? '') })}</div>
          <div className="metric-val" style={{ color: best != null ? metricColor(best.value) : undefined }}>
            {best != null ? best.value.toFixed(3) : '—'}
          </div>
          <div className="stat-sub">{best?.run.run_id.slice(0, 8) ?? '—'}</div>
        </div>
        <div className="stat-cell">
          <div className="eyebrow">{t('experimentsPage.stats.lastRun')}</div>
          <div className="metric-val lg">
            {last?.started_at
              ? new Date(last.started_at).toLocaleString('ru-RU', { day: '2-digit', month: '2-digit', hour: '2-digit', minute: '2-digit' })
              : '—'}
          </div>
          <div className="stat-sub">{last?.status === 'running' ? t('experimentsPage.running') : last?.run_id.slice(0, 8) ?? '—'}</div>
        </div>
      </div>

      <div className="table-toolbar">
        <label className="field-inline">
          <Search size={13} aria-hidden="true" />
          <input
            data-page-search className="field-inline-input" value={query}
            onChange={e => setQuery(e.target.value)}
            placeholder={t('experimentsPage.searchPlaceholder')}
            aria-label={t('experimentsPage.searchPlaceholder')}
          />
          {query && (
            <button type="button" className="icon-btn" onClick={() => setQuery('')} aria-label={t('experimentsPage.clearSearch')}>
              <X size={12} />
            </button>
          )}
        </label>

        {(['all', 'running', 'baseline'] as const).map(f => (
          <button
            key={f} type="button"
            className={`chip${statusFilter === f ? ' active' : ''}`}
            onClick={() => setStatusFilter(f)}
          >
            {t(`experimentsPage.filter.${f}`)}
          </button>
        ))}

        {datasets.length > 1 && (
          <select
            className="chip-select" value={datasetFilter}
            onChange={e => setDatasetFilter(e.target.value)}
            aria-label={t('experimentsPage.columns.dataset')}
          >
            <option value="">{t('experimentsPage.allDatasets')}</option>
            {datasets.map(d => <option key={d} value={d}>{d}</option>)}
          </select>
        )}

        <div className="lang-switcher push">
          <button type="button" className="chip" onClick={() => setColumnsOpen(o => !o)}>
            <Columns3 size={12} />{t('experimentsPage.columns.pick')}
          </button>
          {columnsOpen && (
            <>
              <div className="realm-switcher-backdrop" onClick={() => setColumnsOpen(false)} />
              <div className="lang-menu cols-menu">
                <div className="eyebrow view-group">{t('experimentsPage.columns.keyMetrics')}</div>
                {Object.keys(METRIC_META).filter(k => !LEGACY.has(k)).map(k => (
                  <button
                    key={k} type="button" className={`lang-item ${keyMetrics.includes(k) ? 'active' : ''}`}
                    onClick={() => saveKeyMetrics(
                      keyMetrics.includes(k) ? keyMetrics.filter(m => m !== k) : [...keyMetrics, k],
                    )}
                  >
                    <span className="grow">{metricLabel(k)}</span>
                    <code className="mono-sm">{k}</code>
                  </button>
                ))}
                <div className="cmdk-group">{t('experimentsPage.columns.widthsNote')}</div>
                <button type="button" className="lang-item" onClick={() => { reset(); setColumnsOpen(false) }}>
                  {t('experimentsPage.columns.resetWidths')}
                </button>
              </div>
            </>
          )}
        </div>
      </div>

      <div className="table-wrap">
        {isLoading && <div className="loading pad-20">{t('experimentsPage.loading')}</div>}
        {!isLoading && experiments.length === 0 && (
          <div className="empty">
            {t('experimentsPage.empty')} <Link to={toRealm('/new')}>{t('experimentsPage.emptyCta')}</Link>
          </div>
        )}
        {!isLoading && experiments.length > 0 && sorted.length === 0 && (
          <div className="empty">{t('experimentsPage.noMatches')}</div>
        )}
        {sorted.length > 0 && (
          <table className="run-table">
            <thead>
              <tr>
                <th className="pick-cell" style={{ width: BASE_WIDTHS.select }}>
                  <input
                    type="checkbox" checked={allSelected}
                    onChange={() => setSelected(allSelected ? [] : sorted.map(e => e.run_id))}
                    aria-label={t('experimentsPage.selectAll')}
                  />
                </th>
                <Th col="run_id">Run</Th>
                <Th col="name">{t('experimentsPage.columns.name')}</Th>
                <Th col="started_at">{t('experimentsPage.columns.time')}</Th>
                <Th col="dataset_name">{t('experimentsPage.columns.dataset')}</Th>
                <Th col="n_questions" align="right">{t('experimentsPage.columns.questions')}</Th>
                {columns.map(m => (
                  <Th key={m} col={m} align="right">
                    {metricLabel(m)}{LEGACY.has(m) && <span className="legacy-tag">legacy</span>}
                  </Th>
                ))}
                <th className="col-34" />
              </tr>
            </thead>
            <tbody>
              {sorted.map(e => {
                const running = e.status === 'running'
                const secs = durationSeconds(e.started_at, e.finished_at)
                return (
                  <tr key={e.run_id} className={running ? 'row-running' : undefined}>
                    <td className="pick-cell">
                      {!running && (
                        <input
                          type="checkbox" checked={selected.includes(e.run_id)}
                          onChange={() => toggleSelect(e.run_id)}
                          aria-label={t('experimentsPage.selectRow', { runId: e.run_id })}
                        />
                      )}
                    </td>
                    <td>
                      <Link to={toRealm(`/experiments/${e.run_id}`)} className="mono-sm">
                        {e.run_id.slice(0, 8)}
                      </Link>
                    </td>
                    <td>
                      <span className="run-name">{e.name}</span>
                      {e.is_baseline && (
                        <span className="badge badge-success ml-7">
                          <Pin size={10} />{t('experimentsPage.baseline')}
                        </span>
                      )}
                      {running && (
                        <span className="badge badge-info ml-7">
                          {t('experimentsPage.running')}
                          {e.progress_total ? ` ${e.progress_processed}/${e.progress_total}` : ''}
                        </span>
                      )}
                      {!running && e.stopped && (
                        <span className="badge badge-warn ml-7">{t('experimentsPage.stopped')}</span>
                      )}
                    </td>
                    <td className="mono-sm text-muted nowrap">
                      {e.started_at ? new Date(e.started_at).toLocaleString('ru-RU', { day: '2-digit', month: '2-digit', hour: '2-digit', minute: '2-digit' }) : '—'}
                      {!running && secs != null && <span className="ml-5">{formatDuration(secs)}</span>}
                    </td>
                    <td className="mono-sm">{e.dataset_name || '—'}</td>
                    <td className="num">{running ? '—' : (e.n_questions || '—')}</td>
                    {columns.map(m => {
                      const v = e.aggregate_metrics?.[m]
                      if (running || v == null) return <td key={m} className="num text-muted">—</td>
                      return (
                        <td key={m} className="num">
                          <span className="bar" aria-hidden="true">
                            <b style={{ width: `${Math.min(100, Math.max(0, v * 100))}%`, background: metricColor(v) }} />
                          </span>
                          <span style={{ color: metricColor(v) }}>{v.toFixed(3)}</span>
                        </td>
                      )
                    })}
                    <td>
                      {!running && (
                        <button
                          type="button" className="icon-btn" onClick={() => handleDelete(e.run_id)}
                          disabled={deleting === e.run_id} title={t('experimentsPage.delete')}
                        >
                          {deleting === e.run_id ? '…' : <Trash2 size={12} />}
                        </button>
                      )}
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        )}
      </div>
    </div>
  )
}

/** The names of a token-overlap evaluator retired in Phase 0. No live run
 *  produces them; the list exists so they are never offered as a column, and so
 *  they can be labelled legacy where old runs still carry them. */
const LEGACY = new Set(['faithfulness', 'answer_relevancy', 'reference_overlap'])

/** The fallback set, for a realm with neither declared metrics nor runs.
 *  Matches DEFAULT_KEY_METRICS in services/api_gateway/routers/realms.py. */
const DEFAULT_METRICS = [
  'retrieval_recall_at_k', 'answer_similarity', 'context_support',
  'correct_refusal', 'retrieval_precision_at_k',
]
