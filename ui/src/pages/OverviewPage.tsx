import { useMemo } from 'react'
import { Link } from 'react-router-dom'
import { useTranslation } from 'react-i18next'
import { useQuery } from '@tanstack/react-query'
import {
  Database, HelpCircle, FileEdit, LayoutTemplate, FlaskConical,
  Check, ArrowRight, Download, Play,
} from 'lucide-react'
import { api } from '../api/client'
import { useRealm, useRealmPath } from '../context/RealmContext'
import { useRealmHealth, type HealthState } from '../hooks/useRealmHealth'
import { formatDuration, durationSeconds } from '../lib/format'
import { metricLabel } from '../lib/metricMeta'

// The realm's overview, and the entry point.
//
// `/` used to land on the run list, so the first thing anyone saw was a
// history of what had already been done. That answers neither "does everything
// work" nor "what do I do next", and those are the two questions asked most
// often: the first at every local start, the second until the realm is fully
// configured.
//
// The overview and the run list stay separate pages. The overview shows four
// recent entries, but that is a link to the list rather than the list itself:
// a screen answering two questions at once finishes neither.

/** One setup step. Derived from whether its own list is non-empty and stored
 *  nowhere: a stored "step complete" flag disagrees with reality the moment
 *  somebody deletes the only corpus. */
interface Step {
  id: string
  label: string
  done: boolean
  to: string
  icon: React.ElementType
  /** What the step was completed with: a corpus name, a question count, the
   *  active prompt. */
  detail: string | null
}

function healthDotClass(state: HealthState): string {
  return `health-dot health-${state}`
}

export default function OverviewPage() {
  const { t } = useTranslation()
  const { activeRealm, activeRealmId } = useRealm()
  const toRealm = useRealmPath()
  const health = useRealmHealth()

  const corpora = useQuery({
    queryKey: ['corpus-collections', activeRealmId],
    queryFn: () => api.corpus.collections(activeRealmId),
    enabled: Boolean(activeRealmId),
  })
  const datasets = useQuery({
    queryKey: ['datasets', activeRealmId],
    queryFn: () => api.datasets.list(activeRealmId),
    enabled: Boolean(activeRealmId),
  })
  const prompts = useQuery({
    queryKey: ['prompts', activeRealmId],
    queryFn: () => api.prompts.list(activeRealmId),
    enabled: Boolean(activeRealmId),
  })
  const presets = useQuery({
    queryKey: ['generation-presets', activeRealmId],
    queryFn: () => api.generationPresets.list(activeRealmId),
    enabled: Boolean(activeRealmId),
  })
  const experiments = useQuery({
    queryKey: ['experiments', activeRealmId],
    queryFn: () => api.experiments.list({ realmId: activeRealmId }),
    enabled: Boolean(activeRealmId),
    refetchInterval: q => (q.state.data?.some(e => e.status === 'running') ? 3000 : false),
  })

  // A completed step has content: not "done" but what it was done with. A bare
  // tick answers "is the realm configured" and not "with what", and somebody
  // opening a realm they did not build still had to walk the sections.
  const steps: Step[] = useMemo(() => {
    const corporaList = corpora.data ?? []
    const datasetList = datasets.data ?? []
    const promptList = prompts.data ?? []
    const presetList = presets.data ?? []
    const runList = experiments.data ?? []
    const activePrompt = promptList.find(p => p.is_active)
    const questionTotal = datasetList.reduce((n, d) => n + (d.count ?? 0), 0)
    const baseline = runList.find(r => r.is_baseline)
    return [
      { id: 'corpus', label: t('overview.steps.corpus'), done: corporaList.length > 0, to: '/data/upload', icon: Database,
        detail: corporaList.length > 0
          ? corporaList.slice(0, 2).map(c => c.corpus_id).join(', ') + (corporaList.length > 2 ? ` +${corporaList.length - 2}` : '')
          : null },
      { id: 'dataset', label: t('overview.steps.dataset'), done: datasetList.length > 0, to: '/data/qa', icon: HelpCircle,
        detail: datasetList.length > 0
          ? t('overview.stepDetail.datasets', { sets: datasetList.length, questions: questionTotal })
          : null },
      { id: 'prompt', label: t('overview.steps.prompt'), done: promptList.length > 0, to: '/prompts', icon: FileEdit,
        detail: activePrompt
          ? t('overview.stepDetail.promptActive', { name: activePrompt.name })
          : promptList.length > 0 ? t('overview.stepDetail.promptNoneActive', { count: promptList.length }) : null },
      { id: 'preset', label: t('overview.steps.preset'), done: presetList.length > 0, to: '/data/presets', icon: LayoutTemplate,
        detail: presetList.length > 0 ? t('overview.stepDetail.presets', { count: presetList.length }) : null },
      { id: 'firstRun', label: t('overview.steps.firstRun'), done: runList.length > 0, to: '/new', icon: FlaskConical,
        detail: runList.length > 0
          ? (baseline
              ? t('overview.stepDetail.runsBaseline', { count: runList.length, id: baseline.run_id.slice(0, 6) })
              : t('overview.stepDetail.runsNoBaseline', { count: runList.length }))
          : null },
    ]
  }, [t, corpora.data, datasets.data, prompts.data, presets.data, experiments.data])

  const doneCount = steps.filter(s => s.done).length
  const nextStep = steps.find(s => !s.done)
  const allDone = doneCount === steps.length

  const runs = experiments.data ?? []
  const recent = useMemo(
    () => [...runs].sort((a, b) => (b.started_at ?? '').localeCompare(a.started_at ?? '')).slice(0, 4),
    [runs],
  )

  // The realm's key metrics, the same list that governs the run table's
  // columns. The first three are taken here: an overview shows that runs exist
  // and roughly how they went, rather than analysing them.
  const keyMetrics = (activeRealm?.key_metrics ?? []).slice(0, 3)
  // The corpus registry does not store a fragment count, so corpora themselves
  // are counted. An overview should not show "24,108" fetched by a separate
  // health request per corpus: a number band is read at a glance, and that
  // request takes seconds.
  const corpusCount = corpora.data?.length ?? 0
  const questionCount = (datasets.data ?? []).reduce((n, d) => n + (d.count ?? 0), 0)
  const baselineRun = runs.find(r => r.is_baseline)

  return (
    <div className="page page-wide">
      <div className="page-head">
        <h1 className="page-title">{activeRealm?.name ?? '—'}</h1>
        <p className="page-sub">{activeRealm?.description || t('overview.noDescription')}</p>
        <span className="page-act">
          <Link to={toRealm('/realms')} className="btn btn-sm"><Download size={13} />{t('overview.exportRealm')}</Link>
          <Link to={toRealm('/new')} className="btn btn-primary"><Play size={13} />{t('overview.newRun')}</Link>
        </span>
      </div>

      <div className="stat-band">
        <div className="stat-cell">
          <div className="eyebrow">{t('overview.band.health')}</div>
          <div className={`metric-val tone-${health.worst === 'down' ? 'bad' : health.worst === 'warn' ? 'warn' : 'ok'}`}>
            {health.loading ? '…' : `${health.ok}/${health.total}`}
          </div>
          <div className="stat-sub">{t('overview.band.healthSub')}</div>
        </div>
        <div className="stat-cell">
          <div className="eyebrow">{t('overview.band.corpus')}</div>
          <div className="metric-val">{corpusCount || '—'}</div>
          <div className="stat-sub">{t('overview.band.corpusSub')}</div>
        </div>
        <div className="stat-cell">
          {/* The question count rather than the dataset count: questions are
              what a run measures, and "5" beside "143" gives two different
              answers to "how many". */}
          <div className="eyebrow">{t('overview.band.questions')}</div>
          <div className="metric-val">{questionCount || '—'}</div>
          <div className="stat-sub">{t('overview.band.questionsSub', { count: datasets.data?.length ?? 0 })}</div>
        </div>
        <div className="stat-cell">
          <div className="eyebrow">{t('overview.band.prompts')}</div>
          <div className="metric-val">{prompts.data?.length ?? '—'}</div>
          <div className="stat-sub">{t('overview.band.promptsSub')}</div>
        </div>
        <div className="stat-cell">
          <div className="eyebrow">{t('overview.band.runs')}</div>
          <div className="metric-val">{runs.length || '—'}</div>
          <div className="stat-sub">
            {baselineRun ? t('overview.band.runsBaseline', { id: baselineRun.run_id.slice(0, 6) }) : t('overview.band.runsSub')}
          </div>
        </div>
      </div>

      <div className="ov-split">
        {/* The setup band does not vanish at five of five; it collapses to a
            line. The first cut removed it entirely, and then somebody opening a
            realm they did not build could not tell whether it was configured at
            all: an absent band read as absent setup. */}
        <section>
            <div className="section-rule">
              <h2 className="section-title">{t('overview.setup')}</h2>
              <span className="section-meta">{t('overview.setupCount', { done: doneCount, total: steps.length })}</span>
            </div>
            {allDone && (
              <p className="setup-done"><Check size={13} /> {t('overview.setupDone')}</p>
            )}
            <ol className="step-list">
              {steps.map(step => {
                const isNext = step.id === nextStep?.id
                return (
                  <li key={step.id} className={`step-row${step.done ? ' done' : ''}`}>
                    <span className={`step-mark${step.done ? ' ok' : isNext ? ' now' : ''}`}>
                      {step.done ? <Check size={11} /> : steps.indexOf(step) + 1}
                    </span>
                    <span className="step-label">
                      {step.label}
                      {step.detail && <span className="step-detail">{step.detail}</span>}
                    </span>
                    <Link to={toRealm(step.to)} className="step-go">
                      {step.done ? t('overview.stepOpen') : isNext ? t('overview.stepGo') : t('overview.stepLater')}
                      {(isNext || step.done) && <ArrowRight size={12} />}
                    </Link>
                  </li>
                )
              })}
            </ol>
        </section>

        <section>
          <div className="section-rule">
            <h2 className="section-title">{t('overview.health')}</h2>
            {/* The button runs the checks and navigates to the status screen.
                It used to do the first silently: no progress, no navigation,
                and whoever pressed it could not tell "checked" from "did
                nothing". */}
            <Link
              to={toRealm('/settings/resources')} className="section-meta link-btn"
              onClick={() => health.refetch()}
            >
              {health.loading ? t('overview.checking') : t('overview.checkAll')} <ArrowRight size={11} />
            </Link>
          </div>
          {health.items.length === 0 && <p className="empty">{t('overview.noResources')}</p>}
          {health.items.map(item => (
            <div key={item.id} className="health-row">
              <span className={healthDotClass(item.state)} aria-hidden="true" />
              <span className="health-name">{item.label}</span>
              {/* A tool and a store can share a name: qdrant is both the
                  backend the platform talks to and its own web interface.
                  Without a kind label the two rows read as a duplicate. */}
              {item.kind !== 'resource' && (
                <span className="health-kind">{t(`overview.kind.${item.kind}`)}</span>
              )}
              <span className="health-state">{item.detail || t(`overview.state.${item.state}`)}</span>
            </div>
          ))}
          {health.items.length > 0 && (
            <Link to={toRealm('/settings/resources')} className="link-muted inline-block mt-10">
              {t('overview.openStatus')}
            </Link>
          )}
        </section>
      </div>

      <div className="section-rule">
        <h2 className="section-title">{t('overview.recentRuns')}</h2>
        <Link to={toRealm('/experiments')} className="section-meta">
          {t('overview.allRuns', { count: runs.length })}
        </Link>
      </div>
      {recent.length === 0 ? (
        <p className="empty">{t('overview.noRuns')}</p>
      ) : (
        <table>
          <thead>
            <tr>
              <th className="col-82">Run</th>
              <th>{t('experimentsPage.columns.name')}</th>
              <th className="col-118 nowrap">{t('experimentsPage.columns.time')}</th>
              {keyMetrics.map((m: string) => <th key={m} className="num col-108">{metricLabel(m)}</th>)}
            </tr>
          </thead>
          <tbody>
            {recent.map(run => {
              const running = run.status === 'running'
              const secs = durationSeconds(run.started_at, run.finished_at)
              return (
                <tr key={run.run_id}>
                  <td><Link to={toRealm(`/experiments/${run.run_id}`)} className="mono-sm">{run.run_id.slice(0, 8)}</Link></td>
                  <td>
                    {run.name}
                    {running && (
                      <span className="badge badge-info ml-8">
                        {t('experimentsPage.running')}
                        {run.progress_total ? ` ${run.progress_processed}/${run.progress_total}` : ''}
                      </span>
                    )}
                  </td>
                  <td className="mono-sm text-muted nowrap">
                    {run.started_at ? new Date(run.started_at).toLocaleString('ru-RU', { day: '2-digit', month: '2-digit', hour: '2-digit', minute: '2-digit' }) : '—'}
                    {!running && secs != null && <span className="ml-5">{formatDuration(secs)}</span>}
                  </td>
                  {keyMetrics.map((m: string) => {
                    const v = run.aggregate_metrics[m]
                    return (
                      <td key={m} className="num mono-sm">
                        {running || v == null ? '—' : v.toFixed(3)}
                      </td>
                    )
                  })}
                </tr>
              )
            })}
          </tbody>
        </table>
      )}
    </div>
  )
}
