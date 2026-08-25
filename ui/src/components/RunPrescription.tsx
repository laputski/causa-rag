import { useState } from 'react'
import { useTranslation } from 'react-i18next'
import { useQuery } from '@tanstack/react-query'
import { Copy, Check, Wrench, EyeOff, Ruler } from 'lucide-react'
import { api, type ExperimentDetail } from '../api/client'
import { copyToClipboard } from '../lib/clipboard'

// The deliverable was stated as "the owner of an
// external system receives a document from the platform they can work from".
// Everything needed for it existed on the server and nothing showed it, so the
// phase's own output was the one thing a reader could not reach.
//
// Two renderings of one document, deliberately: the panels are for reading here
// and the text is for handing over. They come from the same response, so they
// cannot drift apart — the markdown is rendered server-side from the same
// object the panels read.

const CAUSE_LABEL: Record<string, string> = {
  data_missing: 'runPage.rootCause.dataMissing',
  ranking: 'runPage.rootCause.ranking',
  chunking: 'runPage.rootCause.chunking',
  not_retrievable: 'runPage.rootCause.notRetrievable',
  unknown: 'runPage.rootCause.unknown',
}

const LEVER_LABEL: Record<string, string> = {
  ingest: 'runPage.rootCause.leverIngest',
  ranking: 'runPage.rootCause.leverRanking',
  chunking: 'runPage.rootCause.leverChunking',
  vocabulary: 'runPage.rootCause.leverVocabulary',
  chunking_or_vocabulary: 'runPage.rootCause.leverChunkingOrVocabulary',
  verify_index: 'runPage.rootCause.leverVerifyIndex',
}

function CopyButton({ text }: { text: string }) {
  const { t } = useTranslation()
  const [done, setDone] = useState(false)
  return (
    <button
      type="button" className="btn-sm"
      onClick={async () => { await copyToClipboard(text); setDone(true); setTimeout(() => setDone(false), 2000) }}
    >
      {done
        ? <><Check size={13} aria-hidden="true" /> {t('prescription.copied')}</>
        : <><Copy size={13} aria-hidden="true" /> {t('prescription.copy')}</>}
    </button>
  )
}

export default function RunPrescription({ run }: { run: ExperimentDetail }) {
  const { t } = useTranslation()
  const { data, isLoading, error } = useQuery({
    queryKey: ['prescription', run.run_id],
    queryFn: () => api.experiments.prescription(run.run_id),
  })

  if (isLoading) return <div className="card"><div className="loading">{t('prescription.loading')}</div></div>
  if (error) return <div className="card"><div className="guide-callout warn"><div>{String(error)}</div></div></div>
  if (!data) return null

  return (
    <section className="section">
      <div className="section-rule flush">
        <h2 className="section-title">{t('prescription.title')}</h2>
        <CopyButton text={data.markdown} />
      </div>
      <p className="jd-section-hint">{t('prescription.intro')}</p>

      <div className="stat-band">
        <div className="stat-cell">
          <div className="metric-val">{data.fixes.length}</div>
          <div className="metric-label">{t('prescription.statFixes')}</div>
        </div>
        <div className="stat-cell">
          <div className="metric-val">{data.fixes.reduce((n, f) => n + f.questions, 0)}</div>
          <div className="metric-label">{t('prescription.statQuestions')}</div>
        </div>
        <div className="stat-cell">
          <div className="metric-val">{data.gaps.length}</div>
          <div className="metric-label">{t('prescription.statGaps')}</div>
        </div>
        <div className="stat-cell">
          <div className="metric-val sm">
            {t(`prescription.depth_${data.diagnosis_depth}`, { defaultValue: data.diagnosis_depth })}
          </div>
          <div className="metric-label">{t('prescription.statDepth')}</div>
        </div>
      </div>

      <h3 className="icon-heading">
        <Wrench size={14} aria-hidden="true" /> {t('prescription.fixesHeading')}
      </h3>
      {data.fixes.length === 0 ? (
        <p className="jd-section-hint">{t('prescription.noFixes')}</p>
      ) : (
        data.fixes.map((fix, i) => (
          <div key={`${fix.cause}-${fix.entity}-${i}`} className="fix-row">
            <div className="fix-row-head">
              <span className="fix-row-entity">{fix.entity || t('runDiagnostics.taskNoEntity')}</span>
              <span className="badge badge-info">{t('runDiagnostics.causeQuestions', { count: fix.questions })}</span>
              {/* The acceptance set, fixed when the document was written. A
                  criterion the recipient could still widen afterwards would
                  not be a criterion at all. */}
              <span className="badge" title={fix.verification_question_ids.join(', ')}>
                {t('prescription.verifiedOn', { count: fix.verification_question_ids.length })}
              </span>
            </div>
            <p className="fix-row-lever">
              {t(CAUSE_LABEL[fix.cause] ?? 'runDiagnostics.causeUnknown')}
              {' → '}
              <strong>{t(LEVER_LABEL[fix.lever] ?? 'runPage.rootCause.leverVerifyIndex')}</strong>
            </p>
            {fix.examples.length > 0 && (
              <ul className="fix-row-examples">
                {fix.examples.map((ex, j) => <li key={j}>{ex}</li>)}
              </ul>
            )}
          </div>
        ))
      )}

      {data.context_size_advice && (
        <div className="fix-row">
          <div className="fix-row-head">
            <Ruler size={14} aria-hidden="true" />
            <span className="fix-row-entity">{t('runDiagnostics.contextSizeTitle')}</span>
            <span className="badge badge-info">
              {t('runDiagnostics.causeQuestions', { count: data.context_size_advice.questions_gained })}
            </span>
          </div>
          <p className="fix-row-lever">
            {t('runDiagnostics.contextSizeAdvice', {
              current: data.context_size_advice.current_k,
              recommended: data.context_size_advice.recommended_k,
            })}
          </p>
        </div>
      )}

      {/* Not a complaint about the system under test but a statement of which
          verdicts are structurally unreachable without the missing field, plus
          what its owner would change to make them reachable. */}
      <h3 className="icon-heading mt-18">
        <EyeOff size={14} aria-hidden="true" /> {t('prescription.gapsHeading')}
      </h3>
      {data.gaps.length === 0 ? (
        <p className="jd-section-hint">{t('prescription.noGaps')}</p>
      ) : (
        data.gaps.map(gap => (
          <div key={gap.field} className="fix-row">
            <div className="fix-row-head">
              <span className="fix-row-entity">{gap.field}</span>
            </div>
            <p className="fix-row-lever">{gap.unavailable}</p>
            <p className="jd-section-hint flat">→ {gap.remedy}</p>
          </div>
        ))
      )}

      <h3 className="mt-18">{t('prescription.documentHeading')}</h3>
      <p className="jd-section-hint">{t('prescription.documentHint')}</p>
      <pre className="doc-md">{data.markdown}</pre>
    </section>
  )
}
