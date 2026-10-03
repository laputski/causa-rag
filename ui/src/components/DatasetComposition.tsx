import { useQuery } from '@tanstack/react-query'
import { useTranslation } from 'react-i18next'
import { api } from '../api/client'

// What a set is made of, in one line above its questions.
//
// A set of one kind measures one thing, and a run over it says nothing about
// the other; the line says so before the run, since afterwards the only trace
// is a metric that is simply absent.

export default function DatasetComposition({ datasetId, realmId, signature }: {
  // The set's id, or its file name when it has none; the Realm then says
  // which set of that name is meant.
  datasetId: string
  realmId?: string | null
  // What the line depends on in the rows, so an edit asks for it again.
  signature: string
}) {
  const { t } = useTranslation()
  const { data } = useQuery({
    queryKey: ['dataset-profile', datasetId, realmId, signature],
    queryFn: () => api.datasets.profile(datasetId, realmId),
  })
  if (!data || data.questions === 0) return null
  const whole = data.by_scope.global ?? 0
  const ordinary = data.questions - whole
  const statements = data.with_assertions === data.questions
    ? t('datasetsPage.composition.statementsAll')
    : data.with_assertions === 0
      ? t('datasetsPage.composition.statementsNone')
      : t('datasetsPage.composition.statementsSome', { n: data.with_assertions, total: data.questions })
  return (
    <div className="set-line" data-testid="dataset-composition">
      <span>
        {t('datasetsPage.questionsCount', { count: data.questions })}
        {' · '}{t('datasetsPage.composition.sources', { count: data.distinct_references })}
        {' · '}{statements}
      </span>
      <span className="badge badge-info">{t('datasetsPage.composition.whole', { n: whole })}</span>
      <span className="badge badge-info">{t('datasetsPage.composition.ordinary', { n: ordinary })}</span>
      {(whole === 0 || ordinary === 0) && (
        <span className="set-warn">{t('datasetsPage.composition.oneKind')}</span>
      )}
    </div>
  )
}
