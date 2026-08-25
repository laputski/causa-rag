import { useEffect, useId, useMemo, useState } from 'react'
import { useTranslation } from 'react-i18next'
import GuideLink from '../components/GuideLink'
import { DraftBadge, DraftNote } from '../components/DraftNotice'
import { useDefaultCorpus } from '../hooks/useDefaultCorpus'
import SelectBox from '../components/SelectBox'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import {
  SatelliteDish, Download, FlaskConical, Search, CircleCheck,
} from 'lucide-react'
import { api, type ProductionTrace } from '../api/client'
import { useRealm } from '../context/RealmContext'

// The questions nobody thought of in advance.
//
// Every failure the rest of the platform diagnoses comes from a run over a
// prepared set of questions, so it only ever sees what somebody anticipated. A
// question nobody anticipated cannot fail a test that does not exist, which
// makes it exactly the gap no metric moves for.
//
// **Pull, never push.** The platform reaches out to a served system's own
// trace endpoint when an operator asks it to. A system that pushed would have
// to know the platform's address and be up when it is, which is the dependency
// this platform refuses to create.

function Field({ id, label, required, hint, children }: {
  id: string; label: string; required?: boolean; hint?: string; children: React.ReactNode
}) {
  return (
    <div className="form-group">
      <label className="jd-toolbar-label block-label" htmlFor={id}>
        {label} {required && <span className="req">*</span>}
      </label>
      {children}
      {hint && <div className="jd-section-hint tight-top">{hint}</div>}
    </div>
  )
}

function CorpusSelect({ id, value, realmId, onChange }: {
  id: string; value: string; realmId: string | null; onChange: (v: string) => void
}) {
  const { t } = useTranslation()
  const { data: collections = [], isLoading, isError } = useQuery({
    queryKey: ['corpus-collections', realmId],
    queryFn: () => api.corpus.collections(realmId),
  })
  const ids = Array.from(new Set(collections.map(c => c.corpus_id)))
  // Three states, not two. "The list has not arrived", "the request failed"
  // and "this Realm has no corpora" used to render as the same bare text
  // input, so a reader typing into it could not tell whether the corpus they
  // wanted was missing or merely unfetched. Only the failure keeps the input,
  // because that is the case where typing an id is still useful.
  if (isLoading) {
    return <input id={id} className="input" value={value} disabled placeholder={t('productionPage.loading')} />
  }
  if (isError) {
    return (
      <>
        <input id={id} className="input" value={value} onChange={e => onChange(e.target.value)} />
        <div className="jd-section-hint tight-top">{t('productionPage.corpusListFailed')}</div>
      </>
    )
  }
  if (ids.length === 0) {
    return (
      <>
        <input id={id} className="input" value={value} onChange={e => onChange(e.target.value)} />
        <div className="jd-section-hint tight-top">{t('productionPage.corpusListEmpty')}</div>
      </>
    )
  }
  return (
    <SelectBox id={id} value={value} onChange={e => onChange(e.target.value)}
               aria-label={t('productionPage.corpusLabel')}>
      <option value="">{t('productionPage.corpusPlaceholder')}</option>
      {ids.map(cid => <option key={cid} value={cid}>{cid}</option>)}
    </SelectBox>
  )
}

// Collecting reaches out to an address the operator types. It is deliberately
// a button and not a schedule the page keeps for itself: the platform has no
// business polling somebody's production system on its own initiative.
function CollectPanel({ realmId, corpusId, onDone }: {
  realmId: string; corpusId: string; onDone: () => void
}) {
  const { t } = useTranslation()
  const [url, setUrl] = useState('')
  const [result, setResult] = useState<string | null>(null)
  const uid = useId()

  const collect = useMutation({
    mutationFn: () => api.production.collect({ realm_id: realmId, corpus_id: corpusId, url: url.trim() }),
    onSuccess: r => {
      // A deployment with recording switched off is said so rather than
      // reported as an empty success: without this it looks identical to a
      // system with no traffic, and an operator must tell those apart.
      setResult(r.recording_enabled
        ? t('productionPage.collectDone', { collected: r.collected, stored: r.stored })
        : t('productionPage.collectDisabled'))
      onDone()
    },
    onError: (e: Error) => setResult(e.message),
  })

  return (
    <div className="jd-section lead-section">
      <h3 className="jd-section-title">
        <Download size={14} aria-hidden="true" /> {t('productionPage.collectTitle')}
      </h3>
      <p className="jd-section-hint">{t('productionPage.collectIntro')}</p>
      <div className="stack-grid narrow">
        <Field id={`${uid}-url`} label={t('productionPage.urlLabel')} required hint={t('productionPage.urlHint')}>
          <input id={`${uid}-url`} className="input" value={url} placeholder="http://localhost:8020/traces"
                 onChange={e => setUrl(e.target.value)} />
        </Field>
        <div className="flex-row gap-10">
          <button type="button" className="btn btn-primary" disabled={!url.trim() || collect.isPending}
                  onClick={() => collect.mutate()}>
            {collect.isPending ? t('productionPage.collecting') : t('productionPage.collect')}
          </button>
          {result && <span className="jd-section-hint flat">{result}</span>}
        </div>
      </div>
    </div>
  )
}

function PromotePanel({ trace, realmId, onPromoted }: {
  trace: ProductionTrace; realmId: string; onPromoted: () => void
}) {
  const { t } = useTranslation()
  const [datasetId, setDatasetId] = useState('')
  const [answer, setAnswer] = useState('')
  const [refs, setRefs] = useState('')
  const [result, setResult] = useState<string | null>(null)
  const uid = useId()

  const { data: datasets = [] } = useQuery({
    queryKey: ['datasets', realmId],
    queryFn: () => api.datasets.list(realmId),
  })

  const promote = useMutation({
    mutationFn: () => api.production.promote({
      trace_id: trace.trace_id, dataset_id: datasetId, realm_id: realmId,
      reference_answer: answer.trim(),
      article_refs: refs.split(/[,\n]/).map(s => s.trim()).filter(Boolean),
    }),
    onSuccess: r => { setResult(t('productionPage.promoteDone', { id: r.question_id })); onPromoted() },
    onError: (e: Error) => setResult(e.message),
  })

  // Only a *live* promotion closes the form. A trace whose question was
  // deleted afterwards keeps its mark — that is history, and history is worth
  // keeping — but it can be promoted again, because otherwise deleting one
  // question silently made a production failure untestable forever.
  if (trace.promoted_question_id && trace.promotion_live !== false) {
    return (
      <div className="jd-section">
        <h3 className="jd-section-title">
          <CircleCheck size={14} aria-hidden="true" /> {t('productionPage.alreadyPromotedTitle')}
        </h3>
        <p className="jd-section-hint flat">
          {t('productionPage.alreadyPromoted', { id: trace.promoted_question_id })}
        </p>
      </div>
    )
  }

  return (
    <div className="jd-section">
      <h3 className="jd-section-title">
        <FlaskConical size={14} aria-hidden="true" /> {t('productionPage.promoteTitle')}
      </h3>
      <p className="jd-section-hint">{t('productionPage.promoteIntro')}</p>
      {trace.promoted_question_id && (
        <div className="guide-callout warn mb-10">
          <div>{t('productionPage.promotionGone', { id: trace.promoted_question_id })}</div>
        </div>
      )}
      <div className="stack-grid">
        <Field id={`${uid}-ds`} label={t('productionPage.datasetLabel')} required>
          <select id={`${uid}-ds`} className="form-select" value={datasetId}
                  onChange={e => setDatasetId(e.target.value)}>
            <option value="">{t('productionPage.datasetPlaceholder')}</option>
            {datasets.map(d => (
              <option key={d.id ?? d.filename} value={d.id ?? ''}>
                {d.name ?? d.filename} {d.count != null ? `(${d.count})` : ''}
              </option>
            ))}
          </select>
        </Field>
        <Field id={`${uid}-ans`} label={t('productionPage.referenceAnswerLabel')} required
               hint={t('productionPage.referenceAnswerHint')}>
          <textarea id={`${uid}-ans`} className="input" rows={3} value={answer}
                    onChange={e => setAnswer(e.target.value)} />
        </Field>
        <Field id={`${uid}-refs`} label={t('productionPage.refsLabel')} hint={t('productionPage.refsHint')}>
          <input id={`${uid}-refs`} className="input" value={refs} onChange={e => setRefs(e.target.value)} />
        </Field>
        <div className="flex-row gap-10">
          <button type="button" className="btn" disabled={!datasetId || !answer.trim() || promote.isPending}
                  onClick={() => promote.mutate()}>
            {t('productionPage.promote')}
          </button>
          {result && <span className="jd-section-hint flat">{result}</span>}
        </div>
      </div>
    </div>
  )
}

// What the prepared set of questions does not speak to.
function CoveragePanel({ realmId, corpusId }: { realmId: string; corpusId: string }) {
  const { t } = useTranslation()
  const [datasetName, setDatasetName] = useState('')
  const [threshold, setThreshold] = useState(0.6)
  const [asked, setAsked] = useState<{ dataset: string; threshold: number } | null>(null)
  const uid = useId()

  const { data: datasets = [] } = useQuery({
    queryKey: ['datasets', realmId],
    queryFn: () => api.datasets.list(realmId),
  })

  const { data, isLoading, error } = useQuery({
    queryKey: ['coverage', realmId, corpusId, asked?.dataset, asked?.threshold],
    queryFn: () => api.production.coverage(realmId, corpusId, asked!.dataset, asked!.threshold),
    enabled: !!asked,
  })

  // The dataset picks itself and the measure computes immediately: a section
  // showing two fields until a button is pressed reads as a form rather than
  // an answer, and rarely got as far as the answer. The largest dataset wins,
  // because that is the one people measure against.
  useEffect(() => {
    if (datasetName || datasets.length === 0) return
    const biggest = [...datasets].sort((a, b) => (b.count ?? 0) - (a.count ?? 0))[0]
    setDatasetName(biggest.filename)
    setAsked({ dataset: biggest.filename, threshold })
  }, [datasets, datasetName, threshold])

  return (
    <div className="section">
      <div className="section-rule">
        <h2 className="section-title">{t('productionPage.coverageTitle')}</h2>
        <span className="section-meta">
          {data ? t('productionPage.coverageMeta', { golden: data.n_golden, prod: data.n_production }) : ''}
        </span>
      </div>
      <p className="hint-line bare">
        {t('productionPage.coverageIntro')}
      </p>

      <div className="table-toolbar">
        <span className="toolbar-label">{t('productionPage.coverageDatasetLabel')}</span>
        <SelectBox id={`${uid}-ds`} value={datasetName}
                   onChange={e => { setDatasetName(e.target.value); setAsked({ dataset: e.target.value, threshold }) }}
                   aria-label={t('productionPage.coverageDatasetLabel')}>
          <option value="">{t('productionPage.datasetPlaceholder')}</option>
          {/* The filename rather than the display name: `_load_dataset` looks
              it up by that first, and only then falls back to a weaker search
              by title. */}
          {datasets.map(d => (
            <option key={d.filename} value={d.filename}>
              {d.name ?? d.filename} {d.count != null ? `(${d.count})` : ''}
            </option>
          ))}
        </SelectBox>
        <span className="toolbar-label">{t('productionPage.thresholdLabel')}</span>
        <input id={`${uid}-thr`} className="toolbar-num" type="number" step={0.05} min={0} max={1}
               value={threshold}
               onChange={e => { const v = Number(e.target.value); setThreshold(v); if (datasetName) setAsked({ dataset: datasetName, threshold: v }) }} />
      </div>

      {isLoading && <p className="text-muted">{t('productionPage.measuring')}</p>}
      {error && <p className="conn-status conn-status-err">{String(error)}</p>}

      {data && (
        <>
          <div className="stat-band">
            <div className="stat-cell">
              <div className="eyebrow">{t('productionPage.statUncoveredShare')}</div>
              <div className="metric-val" style={data.uncovered_share > 0.2 ? { color: 'var(--color-warning)' } : undefined}>
                {Math.round(data.uncovered_share * 100)}%
              </div>
            </div>
            <div className="stat-cell">
              <div className="eyebrow">{t('productionPage.statUncoveredCount')}</div>
              <div className="metric-val">{data.uncovered_total ?? data.uncovered.length}</div>
            </div>
            <div className="stat-cell">
              <div className="eyebrow">{t('productionPage.statGolden')}</div>
              <div className="metric-val">{data.n_golden}</div>
            </div>
          </div>

          {/* The distribution on the left, what fell outside it on the right:
              one answers "how far off", the other "which ones". */}
          <div className="ov-split">
            <section>
              {data.deciles.length > 0 && (
                <>
                  <p className="eyebrow">{t('productionPage.decilesTitle')}</p>
                  <div className="decile-bars">
                    {data.deciles.map((d, i) => (
                      <div key={i} className={`decile-bar${d < data.threshold ? ' below' : ''}`}
                           style={{ height: `${Math.max(2, Math.min(1, d) * 100)}%` }}
                           title={t('productionPage.decileTitle', { n: i + 1, value: d.toFixed(2) })} />
                    ))}
                  </div>
                  <div className="decile-axis">
                    <span>{t('productionPage.decileFirst')}</span>
                    <span>{t('productionPage.decileLast')}</span>
                  </div>
                  <p className="hint-line">{t('productionPage.decilesHint')}</p>
                </>
              )}
            </section>
            <section>
              <p className="eyebrow">{t('productionPage.uncoveredTitle')}</p>
              {data.uncovered.length === 0 ? (
                <p className="text-muted all-covered">{t('productionPage.allCovered')}</p>
              ) : (
                <ul className="uncovered-list">
                  {data.uncovered.map((q, i) => <li key={i}>{q}</li>)}
                </ul>
              )}
              <p className="hint-line">{t('productionPage.uncoveredHint')}</p>
            </section>
          </div>
        </>
      )}
    </div>
  )
}

function Detail({ trace, realmId, onPromoted }: {
  trace: ProductionTrace; realmId: string; onPromoted: () => void
}) {
  const { t } = useTranslation()
  return (
    <div className="card">
      <div className="jd-detail-head">
        <h2 className="jd-detail-q">{trace.query}</h2>
        <div className="jd-provenance">
          <span>{trace.corpus_id}</span>
          {trace.created_at && <span>{trace.created_at.slice(0, 19).replace('T', ' ')}</span>}
          <span className="jd-chunk-ref">{trace.trace_id.slice(0, 12)}</span>
        </div>
      </div>

      <div className="jd-verdict-cols">
        <div>
          <div className="jd-verdict-title">{t('productionPage.answerPreview')}</div>
          {trace.answer_preview
            ? <div className="jd-chunk-text">{trace.answer_preview}</div>
            : <div className="jd-section-hint flat">{t('productionPage.noAnswer')}</div>}
        </div>
        <div>
          <div className="jd-verdict-title">{t('productionPage.sources', { count: trace.sources.length })}</div>
          {trace.sources.length === 0
            // Recorded rather than smoothed over: a served answer with no
            // retrieved source is itself the finding.
            ? <div className="jd-section-hint flat">{t('productionPage.noSources')}</div>
            : trace.sources.map((s, i) => {
              const hasPath = s.structural_path && s.structural_path !== 'root'
              return (
                <div key={i} className="jd-chunk">
                  <div className="jd-chunk-path">
                    {hasPath
                      ? s.structural_path
                      : <><code className="inline-code">{(s.chunk_id ?? '—').slice(0, 8)}</code>
                          <span className="q-noref"> {t('productionPage.noPath')}</span></>}
                  </div>
                </div>
              )
            })}
        </div>
      </div>

      <PromotePanel trace={trace} realmId={realmId} onPromoted={onPromoted} />
    </div>
  )
}

export default function ProductionPage() {
  const { t } = useTranslation()
  const { activeRealmId } = useRealm()
  const qc = useQueryClient()
  const [corpusId, setCorpusId] = useState('')
  useDefaultCorpus(activeRealmId, corpusId, setCorpusId)
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [search, setSearch] = useState('')
  const [showCollect, setShowCollect] = useState(false)

  const { data: traces = [], isLoading } = useQuery({
    queryKey: ['production-traces', activeRealmId, corpusId],
    queryFn: () => api.production.list(activeRealmId!, corpusId),
    enabled: !!activeRealmId && !!corpusId,
  })

  const visible = useMemo(() => {
    const needle = search.trim().toLowerCase()
    if (!needle) return traces
    return traces.filter(tr => tr.query.toLowerCase().includes(needle))
  }, [traces, search])

  const stats = useMemo(() => ({
    total: traces.length,
    promoted: traces.filter(tr => tr.promoted_question_id && tr.promotion_live !== false).length,
    noSources: traces.filter(tr => tr.sources.length === 0).length,
  }), [traces])

  if (!activeRealmId) {
    return <div className="page"><div className="card"><p>{t('productionPage.noRealm')}</p></div></div>
  }

  const invalidate = () => qc.invalidateQueries({ queryKey: ['production-traces', activeRealmId, corpusId] })
  const selected = visible.find(tr => tr.trace_id === selectedId) ?? null
  useEffect(() => {
    if (selected || visible.length === 0) return
    setSelectedId(visible[0].trace_id)
  }, [visible, selected])

  return (
    <div className="page page-wide">
      <div className="page-head">
        <h1 className="page-title">{t('productionPage.title')} <DraftBadge /></h1>
        <span className="page-act"><GuideLink section="production" /></span>
      </div>
      <DraftNote />

      <p className="page-lead">{t('productionPage.intro')}</p>

      <div className="table-toolbar">
        <span className="toolbar-label">{t('productionPage.corpusLabel')}</span>
        <CorpusSelect id="prod-corpus" value={corpusId} realmId={activeRealmId} onChange={setCorpusId} />
        <button type="button" className="btn btn-sm push"
                disabled={!corpusId} onClick={() => setShowCollect(v => !v)}>
          <Download size={13} aria-hidden="true" /> {t('productionPage.collect')}
        </button>
      </div>

      {!corpusId ? (
        <div className="card jd-empty">
          <SatelliteDish size={40} className="jd-empty-icon" aria-hidden="true" />
          <div className="jd-empty-title">{t('productionPage.pickCorpusTitle')}</div>
          <div className="jd-empty-body">{t('productionPage.pickCorpusBody')}</div>
        </div>
      ) : (
        <>
          {showCollect && (
            <CollectPanel realmId={activeRealmId} corpusId={corpusId} onDone={invalidate} />
          )}

          <div className="stat-band">
            <div className="stat-cell">
              <div className="eyebrow">{t('productionPage.statTotal')}</div>
              <div className="metric-val">{stats.total}</div>
            </div>
            <div className="stat-cell">
              <div className="eyebrow">{t('productionPage.statPromoted')}</div>
              <div className="metric-val">{stats.promoted}</div>
            </div>
            <div className="stat-cell">
              <div className="eyebrow">{t('productionPage.statNoSources')}</div>
              <div className="metric-val">{stats.noSources}</div>
            </div>
          </div>

          <CoveragePanel realmId={activeRealmId} corpusId={corpusId} />

          {stats.total === 0 && !isLoading ? (
            <div className="card jd-empty">
              <SatelliteDish size={40} className="jd-empty-icon" aria-hidden="true" />
              <div className="jd-empty-title">{t('productionPage.emptyTitle')}</div>
              <div className="jd-empty-body">{t('productionPage.emptyBody')}</div>
              <button type="button" className="btn btn-primary mt-14"
                      onClick={() => setShowCollect(true)}>
                <Download size={14} aria-hidden="true" /> {t('productionPage.collect')}
              </button>
            </div>
          ) : (
            <div className="jd-split split">
              <div className="split-col">
                <div className="section-rule flush">
                  <h2 className="section-title">{t('productionPage.logHeading')}</h2>
                  <span className="section-meta">{t('productionPage.logCount', { count: visible.length })}</span>
                </div>
                <div className="field-search">
                  <Search size={13} aria-hidden="true"
                          className="field-search-icon" />
                  <input className="input" value={search}
                         onChange={e => setSearch(e.target.value)}
                         placeholder={t('productionPage.searchPlaceholder')} />
                </div>
                {isLoading && <p className="jd-section-hint mt-10">{t('productionPage.loading')}</p>}
                {!isLoading && visible.length === 0 && (
                  <p className="jd-section-hint mt-10">{t('productionPage.noMatches')}</p>
                )}
                <ul className="pick-list">
                  {visible.map(tr => (
                    <li key={tr.trace_id}>
                      <button type="button"
                              className={`pick-row${tr.trace_id === selectedId ? ' active' : ''}`}
                              onClick={() => setSelectedId(tr.trace_id)}>
                        <div className="pick-row-title">{tr.query}</div>
                        <div className="pick-row-meta">
                          <span className="badge badge-info">{t('productionPage.badgeSources', { count: tr.sources.length })}</span>
                          {tr.promoted_question_id && tr.promotion_live !== false && (
                            <span className="badge badge-success">{t('productionPage.badgePromoted')}</span>
                          )}
                          {tr.promoted_question_id && tr.promotion_live === false && (
                            <span className="badge badge-warn">{t('productionPage.badgePromotionGone')}</span>
                          )}
                        </div>
                      </button>
                    </li>
                  ))}
                </ul>
              </div>

              {selected
                ? <Detail trace={selected} realmId={activeRealmId} onPromoted={invalidate} />
                : (
                  <div className="card jd-empty">
                    <SatelliteDish size={32} className="jd-empty-icon" aria-hidden="true" />
                    <div className="jd-empty-body">{t('productionPage.selectHint')}</div>
                  </div>
                )}
            </div>
          )}

        </>
      )}
    </div>
  )
}
