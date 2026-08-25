import { useMemo, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { useSearchParams } from 'react-router-dom'
import { Download } from 'lucide-react'
import SelectBox from '../components/SelectBox'
import { useTranslation, Trans } from 'react-i18next'
import GuideLink from '../components/GuideLink'
import type { TFunction } from 'i18next'
import { api, type ExperimentItem, type PairedDiffResult } from '../api/client'
import { metricLabel, classifyDelta, type DeltaVerdict } from '../lib/metricMeta'
import { useRealm } from '../context/RealmContext'
import { useProgress } from '../hooks/useProgress'

/** A short configuration line under a run's name: the id, plus the fields that
 *  differ between the pair. Showing all of them would hide the difference among
 *  the matches, and a configuration holds about thirty. */
function runSummary(
  run: ExperimentItem | undefined,
  diff: Record<string, { before: unknown; after: unknown }> | undefined,
  side: 'before' | 'after',
): string {
  if (!run) return ''
  const parts = [run.run_id.slice(0, 8)]
  for (const [key, v] of Object.entries(diff ?? {})) {
    if (key === 'name') continue
    const value = v[side]
    if (value === null || value === undefined) continue
    const shown = typeof value === 'object' ? '…' : String(value)
    if (shown.length > 28) continue
    parts.push(`${key}=${shown}`)
    if (parts.length > 4) break
  }
  return parts.join(' · ')
}

function runOptionLabel(e: ExperimentItem, t: TFunction): string {
  const date = e.started_at ? new Date(e.started_at).toLocaleString('ru-RU', { day: '2-digit', month: '2-digit', hour: '2-digit', minute: '2-digit' }) : '—'
  return `${e.run_id} · ${e.name || t('comparisonPage.unnamed')} · ${date} · ${t('comparisonPage.questionsCount', { count: e.n_questions ?? '?' })}`
}

// Values are i18n keys, resolved with t() at the render site.
const VERDICT_META: Record<DeltaVerdict, { label: string; cls: string }> = {
  improved: { label: 'comparisonPage.verdict.improved', cls: 'badge-success' },
  regressed: { label: 'comparisonPage.verdict.regressed', cls: 'badge-danger' },
  noise: { label: 'comparisonPage.verdict.noise', cls: 'badge-info' },
}

// Same funnel-layer labels as RunPage.tsx#FUNNEL_BADGE (core/eval/funnel.py's
// Layer values) — duplicated rather than shared because the two pages only
// need the label text here, not RunPage's CSS badge class.
const FUNNEL_LAYER_LABEL: Record<string, string> = {
  not_applicable: 'runPage.funnel.notApplicable',
  suspected_ungrounded_answer: 'runPage.funnel.suspectedUngrounded',
  retrieval: 'runPage.funnel.layerRetrieval',
  rerank: 'runPage.funnel.layerRerank',
  generation: 'runPage.funnel.layerGeneration',
  ok: 'runPage.funnel.ok',
}

function MetricRow({ metric, before, after, delta, delta_pct }: {
  metric: string; before: number; after: number; delta: number; delta_pct: number | null
}) {
  const { t } = useTranslation()
  const verdict = classifyDelta(before, after, metric)
  const meta = VERDICT_META[verdict]
  const cls = verdict === 'improved' ? 'num-ok' : verdict === 'regressed' ? 'num-bad' : 'num-dim'

  return (
    <tr>
      <td>{metricLabel(metric)}</td>
      <td className="num">{before.toFixed(3)}</td>
      <td className="num">{after.toFixed(3)}</td>
      <td className={`num ${cls}`}>
        {delta_pct != null
          ? `${delta_pct >= 0 ? '+' : '\u2212'}${Math.abs(delta_pct).toFixed(1)} %`
          : `${delta >= 0 ? '+' : '\u2212'}${Math.abs(delta).toFixed(3)}`}
      </td>
      <td><span className={`badge ${meta.cls}`}>{t(meta.label)}</span></td>
      {/* A column that did not exist: a number says what changed rather than
          what it means, and every reader drew that conclusion themselves. */}
      <td className="cmp-means">{t(`comparisonPage.means.${verdict}`, { metric: metricLabel(metric) })}</td>
    </tr>
  )
}

// per-question paired diff: the "blast radius" of
// whatever changed between the two runs. The aggregate MetricRow cards
// above answer "did the average move"; this answers "did any question
// that used to pass now fail" by naming it, which an average can hide.
function PairedDiffRow({ questionId, diff, kind, t }: {
  questionId: string; diff: PairedDiffResult; kind: 'fixed' | 'flips' | 'unchanged'; t: TFunction
}) {
  const info = diff.questions[questionId]
  const before = info?.funnel_before
  const after = info?.funnel_after
  const label = (layer?: string) => (layer ? t(FUNNEL_LAYER_LABEL[layer] ?? layer) : '—')
  // The cause in B is the layer the question failed at after the change. A
  // fixed question has no failing layer, so the column says where it left.
  const reasonCls = kind === 'flips' ? 'badge-danger' : kind === 'fixed' ? 'badge-success' : 'badge-info'
  return (
    <tr>
      <td className="q-id"><code>{questionId.slice(0, 8)}</code></td>
      <td className="cmp-qtext">{info?.question || <code>{questionId}</code>}</td>
      <td className="num">{label(before)}</td>
      <td className="num">{label(after)}</td>
      <td><span className={`badge ${reasonCls}`}>{kind === 'fixed' ? label(before) : label(after)}</span></td>
    </tr>
  )
}

function PairedDiffCard({ diff }: { diff: PairedDiffResult }) {
  const { t } = useTranslation()
  // Broken first: if the change broke something, that is where reading starts.
  const [kind, setKind] = useState<'fixed' | 'flips' | 'unchanged'>(
    diff.flips.length > 0 ? 'flips' : diff.fixed.length > 0 ? 'fixed' : 'unchanged',
  )
  // The server does not assemble question texts for the unchanged group: there
  // are dozens and a count is enough. The table carries the ones there is
  // something to show about.
  const shown = diff[kind].filter(qid => diff.questions[qid] || kind !== 'unchanged').slice(0, 60)

  return (
    <div className="section">
      <div className="section-rule flush">
        <h2 className="section-title">{t('comparisonPage.pairedDiff.heading')}</h2>
        <span className="section-meta">
          {t('comparisonPage.pairedDiff.matched', { count: diff.fixed.length + diff.flips.length + diff.unchanged.length })}
        </span>
      </div>

      {/* Four numbers in a band, rather than two warnings and a count in prose.
          Run averages hide that a change helped some questions and broke
          others; the band puts both numbers side by side, and "it got better"
          stops being the only available reading. */}
      <div className="stat-band">
        <div className="stat-cell">
          <div className="eyebrow">{t('comparisonPage.pairedDiff.bandFixed')}</div>
          <div className={`metric-val${diff.fixed.length ? ' tone-ok' : ''}`}>
            {diff.fixed.length}
          </div>
          <div className="stat-sub">{t('comparisonPage.pairedDiff.subFixed')}</div>
        </div>
        <div className="stat-cell">
          <div className="eyebrow">{t('comparisonPage.pairedDiff.bandFlips')}</div>
          <div className={`metric-val${diff.flips.length ? ' tone-bad' : ''}`}>
            {diff.flips.length}
          </div>
          <div className="stat-sub">{t('comparisonPage.pairedDiff.subFlips')}</div>
        </div>
        <div className="stat-cell">
          <div className="eyebrow">{t('comparisonPage.pairedDiff.bandUnchanged')}</div>
          <div className="metric-val">{diff.unchanged.length}</div>
          <div className="stat-sub">{t('comparisonPage.pairedDiff.subUnchanged')}</div>
        </div>
        <div className="stat-cell">
          <div className="eyebrow">{t('comparisonPage.pairedDiff.bandNoise')}</div>
          <div className="metric-val">{diff.noise_filtered?.length ?? 0}</div>
          <div className="stat-sub">{t('comparisonPage.pairedDiff.noiseHint')}</div>
        </div>
      </div>

      {/* The note stays under the band: a number says how many, not why they
          ended up there, and being filtered out as noise is not the same as not
          having changed. The first cut dropped this line along with its panel,
          and the column's only explanation went with it. */}
      {!!diff.noise_filtered?.length && (
        <p className="hint-line">
          {t('comparisonPage.pairedDiff.noiseFilteredNote', { count: diff.noise_filtered.length })}
        </p>
      )}
      {diff.flips.length > 0 && (
        <p className="hint-line bad">
          {t('comparisonPage.pairedDiff.flipsWarning', { count: diff.flips.length })}
        </p>
      )}
      {/* Chips rather than two lists side by side: questions get read one kind
          at a time, and two columns force a comparison between things that do
          not compare. */}
      <div className="chips">
        {(['flips', 'fixed', 'unchanged'] as const).map(k => (
          <button
            key={k} type="button"
            className={`chip${kind === k ? ' active' : ''}`}
            onClick={() => setKind(k)}
          >
            {t(`comparisonPage.pairedDiff.band${k === 'flips' ? 'Flips' : k === 'fixed' ? 'Fixed' : 'Unchanged'}`)}
            <span className="chip-count">{diff[k].length}</span>
          </button>
        ))}
      </div>

      {shown.length === 0 ? (
        <p className="empty tight">{t('comparisonPage.pairedDiff.noChanges')}</p>
      ) : (
        <table className="cmp-table">
          <thead>
            <tr>
              <th>{t('comparisonPage.pairedDiff.colQuestion')}</th>
              <th>{t('comparisonPage.pairedDiff.colText')}</th>
              <th className="num">A</th>
              <th className="num">B</th>
              <th>{t('comparisonPage.pairedDiff.colReason')}</th>
            </tr>
          </thead>
          <tbody>
            {shown.map(qid => <PairedDiffRow key={qid} questionId={qid} diff={diff} kind={kind} t={t} />)}
          </tbody>
        </table>
      )}
    </div>
  )
}

export default function ComparisonPage() {
  const { t } = useTranslation()
  const [searchParams] = useSearchParams()
  const initA = searchParams.get('a') ?? ''
  const initB = searchParams.get('b') ?? ''

  const [idA, setIdA] = useState(initA)
  const [idB, setIdB] = useState(initB)
  const [compareIds, setCompareIds] = useState<[string, string] | null>(
    initA && initB ? [initA, initB] : null,
  )

  const { activeRealmId } = useRealm()
  const { data: experiments } = useQuery({
    queryKey: ['experiments', activeRealmId],
    queryFn: () => api.experiments.list({ realmId: activeRealmId }),
  })
  // A comparison can take minutes: contested questions are asked again, up to
  // forty pipeline runs. The client invents the identifier, because a socket
  // under that name is needed before the request's response arrives.
  const progressId = useMemo(
    () => (compareIds ? `cmp-${compareIds[0]}-${compareIds[1]}-${Date.now()}` : null),
    [compareIds],
  )
  const { data: result, isLoading, error } = useQuery({
    queryKey: ['compare', compareIds],
    queryFn: () => api.experiments.compare(compareIds!, progressId ?? undefined),
    enabled: !!compareIds,
  })
  const { latest: progress } = useProgress(isLoading ? progressId : null)

  const changedFieldsCount = result ? Object.keys(result.config_diff).length : 0
  const runA = experiments?.find(e => e.run_id === compareIds?.[0])
  const runB = experiments?.find(e => e.run_id === compareIds?.[1])

  // The export is a text summary the server has already composed. It gets
  // pasted into messages and tickets, and rebuilding it on the client would
  // mean two versions of one conclusion.
  const exportSummary = () => {
    if (!result) return
    const blob = new Blob([result.summary], { type: 'text/plain;charset=utf-8' })
    const a = document.createElement('a')
    a.href = URL.createObjectURL(blob)
    a.download = `compare-${compareIds?.[0]?.slice(0, 8)}-${compareIds?.[1]?.slice(0, 8)}.txt`
    a.click()
    URL.revokeObjectURL(a.href)
  }

  return (
    <div className="page">
      <div className="page-head">
        <h1 className="page-title">{t('comparisonPage.title')}</h1>
        {result && (
          <p className="page-sub">
            {compareIds?.[0]?.slice(0, 8)} ↔ {compareIds?.[1]?.slice(0, 8)}
            {runA?.dataset_name && ` · ${runA.dataset_name}`}
            {runA?.n_questions != null && ` · ${t('comparisonPage.questionsCount', { count: runA.n_questions })}`}
          </p>
        )}
        {result && (
          <span className="page-act">
            <GuideLink section="regression" />
            <button className="btn btn-sm" onClick={exportSummary}>
              <Download size={13} />{t('comparisonPage.export')}
            </button>
          </span>
        )}
      </div>

      {/* The pair of runs: what is being compared reads immediately, as a
          role, a name, the baseline mark and a short configuration. Choosing a
          different pair is a disclosure below rather than the screen's main
          content. */}
      {result && (
      <div className="pair">
        <div className="pair-side">
          <div className="eyebrow">{t('comparisonPage.runA')}</div>
          <div className="pair-name">
            {runA?.name ?? '—'}
            {runA?.is_baseline && <span className="badge badge-success">{t('comparisonPage.baselineTag')}</span>}
          </div>
          <div className="pair-cfg">{runSummary(runA, result?.config_diff, 'before')}</div>
        </div>
        <div className="pair-side">
          <div className="eyebrow">{t('comparisonPage.runB')}</div>
          <div className="pair-name">
            {runB?.name ?? '—'}
            {runB?.is_baseline && <span className="badge badge-success">{t('comparisonPage.baselineTag')}</span>}
          </div>
          <div className="pair-cfg">{runSummary(runB, result?.config_diff, 'after')}</div>
        </div>
      </div>
      )}

      {/* Open until a pair is chosen: on an empty screen the choice is the
          only thing there is to do, and hiding it behind a row means making
          somebody hunt for the reason they opened the screen. Once a
          comparison is on screen the picker collapses, because it is not
          needed again straight away. */}
      <details className="cmp-pick" open={!result}>
        <summary>{result ? t('comparisonPage.changePair') : t('comparisonPage.pickPair')}</summary>
        <div className="table-toolbar">
          <span className="toolbar-label">{t('comparisonPage.runA')}</span>
          <SelectBox value={idA} onChange={e => setIdA(e.target.value)} aria-label={t('comparisonPage.runA')}>
            <option value="">{t('comparisonPage.selectPrompt')}</option>
            {experiments?.map(e => <option key={e.run_id} value={e.run_id}>{runOptionLabel(e, t)}</option>)}
          </SelectBox>
          <span className="toolbar-label">{t('comparisonPage.runB')}</span>
          <SelectBox value={idB} onChange={e => setIdB(e.target.value)} aria-label={t('comparisonPage.runB')}>
            <option value="">{t('comparisonPage.selectPrompt')}</option>
            {experiments?.map(e => <option key={e.run_id} value={e.run_id}>{runOptionLabel(e, t)}</option>)}
          </SelectBox>
          <button
            className="btn btn-sm btn-primary push"
            disabled={!idA || !idB || idA === idB}
            onClick={() => setCompareIds([idA, idB])}
          >
            {t('comparisonPage.compare')}
          </button>
        </div>
        {/* The explanation lives here rather than above the comparison: it is
            needed once, and it pushed the page down every time. */}
        <p className="prose-p">
          <Trans i18nKey="comparisonPage.whyBody" t={t} components={{ 0: <strong />, 1: <strong /> }} />
        </p>
      </details>

      {/* Until the first event arrives there is no "how many of how many" to
          state: only the server knows how many questions have to be asked
          again, and only after the pairwise comparison. */}
      {isLoading && (
        <div className="loading">
          {progress?.total
            ? t('comparisonPage.loadingResample', {
                processed: progress.processed ?? 0, total: progress.total,
              })
            : t('comparisonPage.loadingComparison')}
        </div>
      )}
      {error && <div className="empty">{t('comparisonPage.comparisonError')}</div>}

      {result && (
        <>
          <div className="section">
            <div className="section-rule flush">
              <h2 className="section-title">{t('comparisonPage.metrics')}</h2>
              <span className="section-meta">{t('comparisonPage.noiseThresholdHint')}</span>
            </div>
            {result.metric_deltas.length === 0 ? (
              <p className="empty tight">{t('comparisonPage.noMetrics')}</p>
            ) : (
              <table className="cmp-table">
                <thead>
                  <tr>
                    <th>{t('comparisonPage.metric')}</th>
                    <th className="num">A</th>
                    <th className="num">B</th>
                    <th className="num">Δ</th>
                    <th>{t('comparisonPage.verdictCol')}</th>
                    <th>{t('comparisonPage.meansCol')}</th>
                  </tr>
                </thead>
                <tbody>
                  {result.metric_deltas
                    .slice()
                    .sort((a, b) => metricLabel(a.metric).localeCompare(metricLabel(b.metric)))
                    .map(d => <MetricRow key={d.metric} {...d} />)}
                </tbody>
              </table>
            )}
            {/* The configuration diff sits under the metrics, collapsed: the
                pair above already named the fields that differ, and the full
                table is wanted when the answer to "why" did not match what was
                expected. */}
            {changedFieldsCount > 0 && (
              <details className="cmp-cfg">
                <summary>{t('comparisonPage.configChanges')} · {changedFieldsCount}</summary>
                {changedFieldsCount > 1 && (
                  <p className="conn-status conn-status-warn">
                    {t('comparisonPage.multipleFieldsWarning', { count: changedFieldsCount })}
                  </p>
                )}
                <table className="cmp-table">
                  <thead><tr><th>{t('comparisonPage.parameter')}</th><th>{t('comparisonPage.before')}</th><th>{t('comparisonPage.after')}</th></tr></thead>
                  <tbody>
                    {Object.entries(result.config_diff).map(([key, { before, after }]) => (
                      <tr key={key}>
                        <td><code className="inline-code">{key}</code></td>
                        <td className="cmp-val"><code>{JSON.stringify(before)}</code></td>
                        <td className="cmp-val"><code>{JSON.stringify(after)}</code></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </details>
            )}
          </div>

          <PairedDiffCard diff={result.paired_diff} />

          <details className="card">
            <summary className="plain-summary">{t('comparisonPage.textVersion')}</summary>
            <pre className="qrow-pre mt-10">{result.summary}</pre>
          </details>
        </>
      )}
    </div>
  )
}
