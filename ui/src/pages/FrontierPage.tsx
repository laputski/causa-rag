import { useMemo, useState } from 'react'
import { useTranslation } from 'react-i18next'
import GuideLink from '../components/GuideLink'
import { DraftBadge, DraftNote } from '../components/DraftNotice'
import { useQuery } from '@tanstack/react-query'
import { Link } from 'react-router-dom'
import { ChartScatter, TriangleAlert, ChevronDown, ChevronRight } from 'lucide-react'
import SelectBox from '../components/SelectBox'
import FrontierPlot from '../components/FrontierPlot'
import { api, type ConfigPoint } from '../api/client'
import { useRealm, useRealmPath } from '../context/RealmContext'

// The choice between configurations, made over
// runs that already happened.
//
// This page starts nothing. A configuration search that launched runs would
// hide its own cost at the call site, and every run here was paid for once
// already; what was missing was only the comparison.
//
// The grouping by pipeline source is the whole point of the layout, not a
// detail of it: the platform's stage trace measures the platform's own work,
// so for a call to an external system it records the handover and not the
// remote system's cost. One measured pair differed by a factor of 160 000 for
// that reason alone. Two groups with a stated caveat is the only honest way to
// show both.

const SOURCE_LABEL: Record<string, string> = {
  in_process: 'frontierPage.sourceInProcess',
  http: 'frontierPage.sourceHttp',
}

// The config fields worth seeing beside a point: these are the levers a reader
// would actually change, and a full config dump per row would bury them.
const SHOWN_CONFIG = ['top_k', 'fetch_k', 'merge_strategy', 'merge_alpha', 'reranker', 'pipeline_id'] as const

function configSummary(config: Record<string, unknown>): string {
  return SHOWN_CONFIG
    .map(k => {
      const v = config[k]
      if (v == null || v === '') return null
      const shown = typeof v === 'object' ? (v as { component_id?: string }).component_id : v
      return shown == null || shown === '' ? null : `${k}=${shown}`
    })
    .filter(Boolean)
    .join('  ')
}

function PointTable({ points, external, tokensIncluded }: {
  points: ConfigPoint[]; external: boolean
  // The token column is always shown, but it only counts as an
  // axis when it actually took part in the comparison for this group.
  tokensIncluded: boolean
}) {
  const { t } = useTranslation()
  const toRealm = useRealmPath()
  return (
    <table>
      <thead>
        <tr>
          <th>{t('frontierPage.colRun')}</th>
          <th>{t('frontierPage.colQuality')}</th>
          <th className="nowrap">
            {t('frontierPage.colLatency')}
            {external && (
              <span className="latency-caveat ml-6" title={t('frontierPage.latencyCaveatFull')}>
                <TriangleAlert size={10} aria-hidden="true" /> {t('frontierPage.latencyCaveatShort')}
              </span>
            )}
          </th>
          <th className="nowrap">
            {t('frontierPage.colTokens')}
            {!tokensIncluded && (
              <span className="latency-caveat ml-6" title={t('frontierPage.tokensExcludedFull')}>
                <TriangleAlert size={10} aria-hidden="true" /> {t('frontierPage.tokensExcludedShort')}
              </span>
            )}
          </th>
          <th>{t('frontierPage.colConfig')}</th>
        </tr>
      </thead>
      <tbody>
        {points.map(p => (
          <tr key={p.run_id}>
            <td>
              <Link to={toRealm(`/experiments/${p.run_id}`)}>{p.label || p.run_id.slice(0, 8)}</Link>
            </td>
            <td className="mono-cell">{p.quality.toFixed(3)}</td>
            <td className="mono-cell nowrap">
              {p.latency_ms >= 1000
                ? t('frontierPage.seconds', { value: (p.latency_ms / 1000).toFixed(1) })
                : t('frontierPage.milliseconds', { value: p.latency_ms.toFixed(1) })}
            </td>
            <td className="mono-cell nowrap">
              {p.tokens == null ? '—' : Math.round(p.tokens).toLocaleString('ru-RU')}
            </td>
            <td className="frontier-cfg">{configSummary(p.config) || '—'}</td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}

export default function FrontierPage() {
  const { t } = useTranslation()
  const { activeRealmId } = useRealm()
  const [metric, setMetric] = useState('retrieval_recall_at_k')
  const [showDominated, setShowDominated] = useState(false)

  // Metric options come from the runs themselves rather than from a fixed
  // list, so the selector can never offer a metric this Realm has never
  // measured — which would return an empty frontier and read as "no runs".
  const { data: runs = [] } = useQuery({
    queryKey: ['experiments', activeRealmId],
    queryFn: () => api.experiments.list({ realmId: activeRealmId }),
  })
  const metricOptions = useMemo(() => {
    const names = new Set<string>(['retrieval_recall_at_k'])
    for (const r of runs) for (const k of Object.keys(r.aggregate_metrics ?? {})) names.add(k)
    return [...names].sort()
  }, [runs])

  const { data, isLoading, error } = useQuery({
    queryKey: ['frontier', activeRealmId, metric],
    queryFn: () => api.experiments.frontier(activeRealmId!, metric),
    enabled: !!activeRealmId,
  })

  if (!activeRealmId) {
    return <div className="page"><div className="card"><p>{t('frontierPage.noRealm')}</p></div></div>
  }

  const groups = Object.entries(data?.frontier_by_source ?? {})
  const onFrontier = groups.reduce((n, [, points]) => n + points.length, 0)
  const baselineRunId = runs.find(r => r.is_baseline)?.run_id
  // One group per plot: the platform measures latency differently for the
  // built-in pipeline and for an external system, so points from different
  // groups on one axis would compare things that are not comparable. Exactly
  // what the caveat further down the page warns about.
  const plotGroup = groups.slice().sort((a, b) => b[1].length - a[1].length)[0]

  return (
    <div className="page page-wide">
      <div className="page-head">
        <h1 className="page-title">{t('frontierPage.title')} <DraftBadge /></h1>
        <span className="page-act"><GuideLink section="frontier" /></span>
      </div>
      <DraftNote />

      <p className="page-lead">{t('frontierPage.intro')}</p>

      <div className="table-toolbar">
        <span className="toolbar-label">{t('frontierPage.metricLabel')}</span>
        <SelectBox value={metric} onChange={e => setMetric(e.target.value)}
                   aria-label={t('frontierPage.metricLabel')}>
          {metricOptions.map(m => <option key={m} value={m}>{m}</option>)}
        </SelectBox>
      </div>

      {isLoading && <p className="text-muted">{t('frontierPage.loading')}</p>}
      {error && <p className="conn-status conn-status-err">{String(error)}</p>}

      {data && (
        <>
          <div className="stat-band">
            <div className="stat-cell">
              <div className="eyebrow">{t('frontierPage.statConsidered')}</div>
              <div className="metric-val">{data.considered}</div>
            </div>
            <div className="stat-cell">
              <div className="eyebrow">{t('frontierPage.statOnFrontier')}</div>
              <div className="metric-val">{onFrontier}</div>
            </div>
            <div className="stat-cell">
              <div className="eyebrow">{t('frontierPage.statDominated')}</div>
              <div className="metric-val">{data.dominated.length}</div>
            </div>
          </div>

          {/* The frontier is a claim about the shape of a set: nothing here
              gets cheaper without costing quality. That shape does not read
              out of two columns of numbers, and reads instantly out of where
              the points sit. */}
          {data.considered > 0 && (
            <FrontierPlot
              frontier={plotGroup?.[1] ?? []}
              dominated={data.dominated.filter(p => p.pipeline_source === plotGroup?.[0])}
              baselineRunId={baselineRunId}
              metric={metric}
              source={plotGroup ? t(SOURCE_LABEL[plotGroup[0]] ?? 'frontierPage.sourceOther', { source: plotGroup[0] }) : ''}
            />
          )}

          {data.considered === 0 ? (
            <div className="card jd-empty">
              <ChartScatter size={40} className="jd-empty-icon" aria-hidden="true" />
              <div className="jd-empty-title">{t('frontierPage.emptyTitle')}</div>
              <div className="jd-empty-body">{t('frontierPage.emptyBody', { metric })}</div>
            </div>
          ) : (
            <div>
              {groups.map(([source, points]) => (
                <div key={source} className="section">
                  <div className="section-rule">
                    <h2 className="section-title">
                      {t(SOURCE_LABEL[source] ?? 'frontierPage.sourceOther', { source })}
                    </h2>
                    <span className="section-meta">
                      {t('frontierPage.groupCount', { count: points.length })}
                    </span>
                  </div>
                  <PointTable
                    points={points} external={source !== 'in_process'}
                    tokensIncluded={data.tokens_included?.[source] !== false}
                  />
                </div>
              ))}

              {/* Both lists, because "these are your choices" reads very
                  differently depending on how many configurations were beaten
                  outright — collapsed, because the beaten ones are context
                  rather than a decision. */}
              {data.dominated.length > 0 && (
                <>
                  <button type="button" className="btn-sm" onClick={() => setShowDominated(v => !v)}>
                    {showDominated ? <ChevronDown size={12} aria-hidden="true" /> : <ChevronRight size={12} aria-hidden="true" />}
                    {' '}{t('frontierPage.dominatedToggle', { count: data.dominated.length })}
                  </button>
                  {showDominated && (
                    <div className="mt-10">
                      <p className="jd-section-hint">{t('frontierPage.dominatedIntro')}</p>
                      <PointTable points={data.dominated} external={false} tokensIncluded />
                    </div>
                  )}
                </>
              )}
            </div>
          )}

          <div className="guide-callout mt-4">
            <div>{t('frontierPage.latencyCaveatFull')}</div>
          </div>
          {/* Tokens, not money: money is tokens times a price the platform has
              no source for, and a locally hosted generator costs no money and
              plenty of compute. */}
          <div className="guide-callout mt-4">
            <div>{t('frontierPage.tokensExplain')}</div>
          </div>
        </>
      )}
    </div>
  )
}
