import { useState } from 'react'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import { useNavigate } from 'react-router-dom'
import { useTranslation } from 'react-i18next'
import { Sparkles, Check, X, Scale } from 'lucide-react'
import { api, type Feedback } from '../api/client'
import { useRealmPath } from '../context/RealmContext'

// Feedback triage, read-only recommendation only
// (the improvement plan's first stage: the agent
// proposes, a human decides). Sits inside the Feedback column of
// QuestionRow (ui/src/pages/RunPage.tsx), directly below FeedbackPanel —
// not a fifth column of its own, so it doesn't disturb the four-equal-
// column layout that redesign deliberately settled on. Only ever shown
// once a comment exists to triage; this component never records a verdict
// or changes anything about the run itself — the action only deep-links to
// JudgmentsPage's own create flow, pre-filled, for the reviewer to finish.
export default function FeedbackTriagePanel({
  runId, questionId, feedback, corpusId, questionText, realmId,
}: {
  runId: string
  questionId: string
  feedback?: Feedback
  corpusId: string
  questionText: string
  realmId: string | null
}) {
  const { t } = useTranslation()
  const qc = useQueryClient()
  const navigate = useNavigate()
  const toRealm = useRealmPath()
  const [error, setError] = useState<string | null>(null)

  const triageMut = useMutation({
    mutationFn: () => api.feedback.triage(runId, questionId, realmId),
    onSuccess: (updated) => {
      setError(null)
      qc.setQueryData<Record<string, Feedback>>(['feedback', runId], (old) => ({ ...(old ?? {}), [questionId]: updated }))
    },
    onError: (e) => setError(String(e)),
  })

  const decideMut = useMutation({
    mutationFn: (action: 'confirm' | 'reject') => api.feedback.triageDecide(runId, questionId, { action }, realmId),
    onSuccess: (updated) => {
      qc.setQueryData<Record<string, Feedback>>(['feedback', runId], (old) => ({ ...(old ?? {}), [questionId]: updated }))
    },
  })

  if (!feedback?.comment) return null

  const triage = feedback.triage_result

  // Navigates to the judgments page with the question pre-filled. It was
  // labelled "create a pin" long after pins were removed, which named an
  // action the platform no longer has — and named it on the button that
  // performs the action which replaced it.
  const goToRecordJudgment = () => {
    const qs = new URLSearchParams()
    if (corpusId) qs.set('corpus_id', corpusId)
    qs.set('question', questionText)
    if (feedback.id) qs.set('source_feedback_id', feedback.id)
    navigate(toRealm(`/data/judgments?${qs.toString()}`))
  }

  return (
    <div className="feedback-card mt-10">
      <h4 className="feedback-card-header">
        <Sparkles size={13} /> {t('feedbackTriage.title')}
      </h4>

      {!triage ? (
        <button className="btn-sm" onClick={() => triageMut.mutate()} disabled={triageMut.isPending}>
          {triageMut.isPending ? t('feedbackTriage.analyzing') : t('feedbackTriage.analyze')}
        </button>
      ) : (
        <div className="triage-body">
          {triage.needs_manual_review && (
            <p className="badge badge-warn">
              {t('feedbackTriage.needsManualReview')}{triage.parse_error ? `: ${triage.parse_error}` : ''}
            </p>
          )}

          {triage.error_classes.length > 0 && (
            <div className="triage-chips">
              {triage.error_classes.map(cls => (
                <span key={cls} className="badge badge-info">{cls}</span>
              ))}
              <span className="text-muted xs">
                {t('feedbackTriage.confidence', { pct: Math.round(triage.confidence * 100) })}
              </span>
            </div>
          )}

          {triage.expected_refs.length > 0 && (
            <div>
              <strong className="text-muted xs">{t('feedbackTriage.expectedRefs')}</strong>
              <ul className="triage-refs">
                {triage.expected_refs.map(ref => (
                  <li key={ref}>
                    {ref}{' '}
                    <span className={`ml-4 badge ${triage.verified_refs[ref] ? 'badge-success' : 'badge-warn'}`}>
                      {triage.verified_refs[ref] ? t('feedbackTriage.verified') : t('feedbackTriage.unverified')}
                    </span>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {triage.proposal.primary_lever && (
            <div className="triage-lever">
              <strong className="xs">{t('feedbackTriage.proposedLever')}</strong> {triage.proposal.primary_lever}
              {triage.proposal.justification && (
                <p className="triage-lever-note">
                  {triage.proposal.justification}
                </p>
              )}
            </div>
          )}

          <div className="triage-actions">
            <button
              className="btn-sm" onClick={() => decideMut.mutate('confirm')} disabled={decideMut.isPending}
              style={feedback.triage_status === 'confirmed' ? { borderColor: 'var(--color-success)', color: 'var(--color-success)' } : undefined}
            >
              <Check size={12} /> {t('feedbackTriage.confirm')}
            </button>
            <button
              className="btn-sm" onClick={() => decideMut.mutate('reject')} disabled={decideMut.isPending}
              style={feedback.triage_status === 'rejected' ? { borderColor: 'var(--color-danger)', color: 'var(--color-danger)' } : undefined}
            >
              <X size={12} /> {t('feedbackTriage.reject')}
            </button>
            <button className="btn-sm" onClick={goToRecordJudgment}>
              <Scale size={12} /> {t('feedbackTriage.recordJudgment')}
            </button>
          </div>
          {feedback.triage_status && (
            <span className="text-muted xs">
              {t('feedbackTriage.status', { status: t(`feedbackTriage.statusValues.${feedback.triage_status}`) })}
            </span>
          )}
        </div>
      )}
      {error && <span className="feedback-status error">{error}</span>}
    </div>
  )
}
