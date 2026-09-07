import { Link, useNavigate } from 'react-router-dom'
import { useTranslation } from 'react-i18next'
import { TriangleAlert, Pin } from 'lucide-react'
import type { DetectorItem, ExperimentDetail } from '../api/client'
import { diagnoseMetrics, diagnoseRetrieval, diagnoseLatency, type DiagnosticItem } from '../lib/diagnostics'
import { useRealmPath } from '../context/RealmContext'
import PipelineDiagram from './PipelineDiagram'

interface Props { run: ExperimentDetail }

// Same severity vocabulary as the old inline-emoji version, now as
// lucide-react icons matching the main nav's icon language (App.tsx) rather
// than unicode characters — icon color still carries the severity, no
// separate legend needed.

// i18n keys for core/eval/root_cause.py's Cause values. Shared
// wording with RunPage's per-question badge, so the aggregate and the row a
// reader clicks into never name the same cause differently.
const LEVER_LABEL: Record<string, string> = {
  ingest: 'runPage.rootCause.leverIngest',
  ranking: 'runPage.rootCause.leverRanking',
  chunking: 'runPage.rootCause.leverChunking',
  vocabulary: 'runPage.rootCause.leverVocabulary',
  chunking_or_vocabulary: 'runPage.rootCause.leverChunkingOrVocabulary',
  verify_index: 'runPage.rootCause.leverVerifyIndex',
}

const CAUSE_LABEL: Record<string, string> = {
  data_missing: 'runPage.rootCause.dataMissing',
  ranking: 'runPage.rootCause.ranking',
  chunking: 'runPage.rootCause.chunking',
  not_retrievable: 'runPage.rootCause.notRetrievable',
  unknown: 'runPage.rootCause.unknown',
}

function DiagItem({ item }: { item: DiagnosticItem }) {
  const navigate = useNavigate()
  const toRealm = useRealmPath()
  return (
    // The same finding block as corpus health uses: a dot in its own colour, a
    // heading, an explanation and an action. The content here is of the same
    // kind, findings carrying a severity and what to do about them, and one
    // thing should not have two different sets of classes.
    <div className="find-row">
      <span className={`find-dot find-${item.severity}`} aria-hidden="true" />
      <div>
        <div className="find-title">{item.title}</div>
        <p className="find-detail">{item.detail}</p>
        {item.action && (
          <button className="btn btn-sm mt-6"
                  onClick={() => navigate(toRealm(item.action!.href))}>
            {item.action.label}
          </button>
        )}
      </div>
    </div>
  )
}

// Backend-precomputed detectors (core/eval/detectors.py) carry a
// plain-text `action` hint, not a {label, href} navigable link like
// DiagnosticItem's frontend-computed items below — rendered as inline text
// rather than a button for that reason, same as the pre-restructure
// DetectorSection.
// Params whose value is a code the platform uses for a thing, and not a number
// or a name: a funnel layer, a half of a merge. The server sends the code and
// never a label, because a label composed there is an English word arriving
// inside a Russian sentence.
const A_WORD_IN_ITS_OWN_RIGHT = new Set(['layer', 'half', 'other'])

function DetectorPanelItem({ item }: { item: DetectorItem }) {
  const { t } = useTranslation()
  const toRealm = useRealmPath()
  const params = Object.fromEntries(
    Object.entries(item.params ?? {}).map(([name, value]) => [
      name,
      A_WORD_IN_ITS_OWN_RIGHT.has(name)
        ? t(`runDiagnostics.findingWord.${value}`, { defaultValue: String(value) })
        : value,
    ]),
  )
  // The title is looked up by the finding's id, and the server's English
  // sentence stays as the fallback for an id this file does not know yet.
  // Corpus health has done it this way for a while; this panel rendered raw
  // server prose, so the Russian interface showed an English finding beside a
  // translated one.
  return (
    <div className="find-row">
      <span className={`find-dot find-${item.severity}`} aria-hidden="true" />
      <div>
        <div className="find-title">
          {t(`runDiagnostics.finding.${item.id}`, { defaultValue: item.title, ...params })}
          {(item.failure_ids ?? []).map(id => (
            <Link key={id} className="link-btn mono-sm ml-8" to={toRealm(`/atlas?entry=${id}`)}>
              {id}
            </Link>
          ))}
        </div>
        {/* The detail carries what the detector measured. The numbers and names
            now arrive beside it, so the sentence is composed here and the
            server's English is the fallback, the same as the title and the
            action above. Four of the values are themselves sentences the
            server composed (a reason, a list of unmet promises, the metrics
            that disagreed, the metrics without grounds); those stay English
            inside a translated frame, and a test says which they are. */}
        <p className="find-detail">
          {t(`runDiagnostics.findingDetail.${item.detail_key ?? item.id}`, {
            defaultValue: item.detail, ...params,
          })}
        </p>
        {item.action && (
          <p className="find-detail find-action">
            {t(`runDiagnostics.findingAction.${item.id}`, { defaultValue: item.action })}
          </p>
        )}
      </div>
    </div>
  )
}


/** What this run made impossible to check.
 *
 *  It used to live on the prescription tab, away from the findings. A reader
 *  saw what was found and had no sign that a whole class of findings could not
 *  have been produced at all, which reads as a clean bill of health for a check
 *  that never ran. */
function TraceGaps({ run }: Props) {
  const { t } = useTranslation()
  const gaps = run.trace_gaps ?? []
  return (
    <>
      {run.diagnosis_depth && (
        <p className="find-detail mb-8">
          {t('runDiagnostics.diagnosisDepth')}:{' '}
          <b>{t(`prescription.depth_${run.diagnosis_depth}`, { defaultValue: run.diagnosis_depth })}</b>
        </p>
      )}
      {gaps.length === 0
        ? <p className="diag-detail diag-panel-empty">{t('runDiagnostics.noTraceGaps')}</p>
        : gaps.map(gap => (
            <div key={gap.field} className="find-row">
              <span className="find-dot find-info" aria-hidden="true" />
              <div>
                {/* Named for a reader, with the field it stands for beside
                    it. It used to show the field alone: a person reading a run
                    met `candidate_source_refs` and two English sentences. */}
                <div className="find-title">
                  {t(`runDiagnostics.traceGap.${gap.field}.label`, { defaultValue: gap.field })}
                  <span className="mono-sm ml-8 text-muted">{gap.field}</span>
                </div>
                <p className="find-detail">
                  {t(`runDiagnostics.traceGap.${gap.field}.unavailable`, { defaultValue: gap.unavailable })}
                </p>
                <p className="find-detail find-action">
                  {t(`runDiagnostics.traceGap.${gap.field}.remedy`, { defaultValue: gap.remedy })}
                </p>
              </div>
            </div>
          ))}
    </>
  )
}

// A panel always renders (fixed slot in the 2-col grid below) — an empty
// items list used to just leave the heading floating with nothing under it;
// now it shows an explicit "clean" state instead of blank space.
function Panel({
  title, items, emptyLabel, note, children,
}: {
  title: string; items?: DiagnosticItem[]; emptyLabel?: string
  // Rendered between the heading and the items, for a qualification that
  // applies to every item below it rather than to one of them.
  note?: React.ReactNode
  children?: React.ReactNode
}) {
  return (
    <section className="section">
      <div className="section-rule flush">
        <h2 className="section-title">{title}</h2>
      </div>
      {note && <div className="mb-8">{note}</div>}
      {items && (items.length > 0
        ? items.map((item, i) => <DiagItem key={i} item={item} />)
        : <p className="find-detail">{emptyLabel}</p>)}
      {children}
    </section>
  )
}

// Regression vs the pinned baseline. Exported: RunPage.tsx's
// header shows the same badge next to the diagnostics-depth/DeepEval badges
// (a "main read-only status", not something that should require switching
// to the Diagnostics tab to see at all).
/**
 * A run's status against the baseline.
 *
 * These used to be emoji circles (📌 🟢 🔴). An emoji is drawn by the system
 * font: it follows neither palette nor theme, loses contrast on the light one,
 * and beside a typeset dot on a neighbouring badge reads as a foreign picture.
 * The dot uses the same tokens as the status dots on the overview.
 */
export function RegressionBadge({ run }: { run: ExperimentDetail }) {
  const { t } = useTranslation()
  const r = run.regression
  if (run.is_baseline) {
    return (
      <span className="flag flag-base" title={t('runDiagnostics.baselineHint')}>
        <Pin size={11} aria-hidden="true" />{t('runDiagnostics.baseline')}
      </span>
    )
  }
  if (!r) return null
  return (
    <span className={`flag ${r.passed ? 'flag-ok' : 'flag-bad'}`} title={r.violations.join('\n')}>
      <i className="flag-dot" aria-hidden="true" />
      {r.passed ? t('runDiagnostics.noRegression') : t('runDiagnostics.regression', { count: r.violations.length })}
    </span>
  )
}

// Exported so RunPage can put the same wording next to a per-question stage
// trace. One component rather than a repeated literal: the rule is that no
// external run's latency is ever shown without it, and a rule with one
// implementation is a rule that holds.
export function LatencyCaveat() {
  const { t } = useTranslation()
  return (
    <span className="latency-caveat" title={t('frontierPage.latencyCaveatFull')}>
      <TriangleAlert size={10} aria-hidden="true" /> {t('frontierPage.latencyCaveatShort')}
    </span>
  )
}

export default function RunDiagnostics({ run }: Props) {
  const { t } = useTranslation()
  const stageTrace = (run as any).stage_trace ?? null
  // "http" means the platform called somebody else's system over the network.
  const isExternal = run.config?.pipeline_source === 'http'
  const metricItems = diagnoseMetrics(run)
  const retrievalItems = diagnoseRetrieval(run)
  const latencyItems = diagnoseLatency(stageTrace)
  const detectorItems = run.diagnostics ?? []
  // Already ordered most-reaching first by core/eval/prioritization.py —
  // deliberately not re-sorted here, so the interface cannot disagree with
  // the ordering the rest of the platform reasons about.
  const fixTasks = run.fix_tasks ?? []

  const all = [...metricItems, ...retrievalItems, ...latencyItems]
  const hasError = all.some(i => i.severity === 'error') ||
    detectorItems.some(d => d.severity === 'error') || run.regression?.passed === false
  const hasWarn = all.some(i => i.severity === 'warn') || detectorItems.some(d => d.severity === 'warn')
  // The same vocabulary as the badge beside it: a dot from the palette inside
  // an outlined pill. Emoji circles stood here, drawn by the system font,
  // following neither the theme nor the palette, and beside the typeset dot of
  // the neighbouring badge they read as somebody else's picture. RegressionBadge
  // went through the same replacement earlier.
  const health = hasError
    ? { tone: 'flag-bad', label: t('runDiagnostics.needsAttention') }
    : hasWarn
      ? { tone: 'flag-warn', label: t('runDiagnostics.hasNotes') }
      : { tone: 'flag-ok', label: t('runDiagnostics.allGood') }

  return (
    <>
      <p className="diag-lead">
        <span className={`flag ${health.tone}`}>
          <span className="flag-dot" />{health.label}
        </span>
        <RegressionBadge run={run} />
      </p>
      <div className="diag-grid">

      {/* Phase 2 — the ordered list of work. This is the panel that answers
          "what should I fix first", which no other panel here does: the rest
          report what is wrong, this one says which fix buys the most. It is
          therefore first after the health summary; it previously sat third of
          seven, below panels a reader has no action to take from.

          It replaced a plain per-cause tally, which ranked kinds of work but
          still left the reader to find out *which* document each concerned —
          and a document is the thing someone actually goes and fixes. */}
      <Panel title={t('runDiagnostics.fixTasks')}>
        {fixTasks.length > 0 ? (
          <>
            {fixTasks.map((task, i) => (
              <div key={`${task.cause}-${task.entity}-${i}`} className="diag-item diag-info">
                <div className="diag-item-header">
                  <span className="diag-title">
                    {task.entity || t('runDiagnostics.taskNoEntity')}
                  </span>
                  <span className="badge badge-info ml-8">
                    {t('runDiagnostics.causeQuestions', { count: task.questions })}
                  </span>
                </div>
                <p className="diag-detail">
                  {t(CAUSE_LABEL[task.cause] ?? 'runDiagnostics.causeUnknown')}
                  {' → '}
                  {t(LEVER_LABEL[task.lever] ?? 'runPage.rootCause.leverVerifyIndex')}
                </p>
              </div>
            ))}
            {/* Stated next to the list it describes, because the
                list looks equally actionable whether the failures generalise
                or every one of them is its own isolated case. */}
            {/* Phase 3 — what a different context size would have bought,
                answered from what the run already recorded rather than by
                running it again. Shown inside the task list because it is
                itself a proposed task, with its payoff already measured. */}
            {run.context_size_advice && (
              <div className="diag-item diag-info mt-6">
                <div className="diag-item-header">
                  <span className="diag-title">{t('runDiagnostics.contextSizeTitle')}</span>
                  <span className="badge badge-info ml-8">
                    {t('runDiagnostics.causeQuestions', { count: run.context_size_advice.questions_gained })}
                  </span>
                </div>
                <p className="diag-detail">
                  {t('runDiagnostics.contextSizeAdvice', {
                    current: run.context_size_advice.current_k,
                    recommended: run.context_size_advice.recommended_k,
                  })}
                  {run.context_size_advice.unreachable > 0 && ' ' + t('runDiagnostics.contextSizeUnreachable', {
                    count: run.context_size_advice.unreachable,
                  })}
                </p>
              </div>
            )}
            <p className="diag-detail dim mt-6">
              {t('runDiagnostics.clusterableShare', {
                percent: Math.round((run.clusterable_share ?? 0) * 100),
              })}
            </p>
          </>
        ) : <p className="diag-detail diag-panel-empty">{t('runDiagnostics.noRetrievalFailures')}</p>}
      </Panel>

      <Panel title={t('runDiagnostics.silentDegradation')}>
        {detectorItems.length > 0
          ? detectorItems.map((d, i) => <DetectorPanelItem key={i} item={d} />)
          : <p className="diag-detail diag-panel-empty">{t('runDiagnostics.allGood')}</p>}
      </Panel>
      <Panel title={t('runDiagnostics.traceGaps')}>
        <TraceGaps run={run} />
      </Panel>
      <Panel title={t('runDiagnostics.metrics')} items={metricItems} emptyLabel={t('runDiagnostics.allGood')} />
      <Panel title={t('runDiagnostics.retrieval')} items={retrievalItems} emptyLabel={t('runDiagnostics.allGood')} />

      {/* Found live: for a run against an external system these
          numbers are the platform's own handover and not the remote system's
          cost, which in one measured pair differed by a factor of 160 000. The
          caveat stands next to every latency number rather than in a footnote,
          because a number without it reads as the system's speed. */}
      <Panel title={t('runDiagnostics.performance')} items={latencyItems}
        emptyLabel={t('runDiagnostics.allGood')} note={isExternal ? <LatencyCaveat /> : undefined}
      />

      <Panel title={t('runDiagnostics.pipelineHeading')}>
        {run.avg_stage_trace ? (
          <>
            <p className="diag-detail mb-6">{t('runPage.avgLatencyHeading')}</p>
            {isExternal && <LatencyCaveat />}
            <PipelineDiagram trace={run.avg_stage_trace} />
          </>
        ) : (
          <p className="diag-detail diag-panel-empty">{t('runDiagnostics.noPipelineTrace')}</p>
        )}
      </Panel>
      </div>
    </>
  )
}
