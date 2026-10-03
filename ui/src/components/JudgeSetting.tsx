import { useEffect, useId, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useTranslation } from 'react-i18next'
import { api, type JudgeSetting as Judge } from '../api/client'

// Which of the Realm's model servers checks answers against their statements,
// and with which model. The judge belongs to the measurement and not to the
// system under test, so it is set on the Realm and recorded on each run
// instead of standing in a run's configuration.

const SERVER_LABEL: Record<string, string> = {
  ollama: 'realmResourcesPage.judge.server.ollama',
  openai_compatible: 'realmResourcesPage.judge.server.openaiCompatible',
}

export default function JudgeSetting({ realmId, servers }: {
  realmId: string
  // The model servers this Realm declares, by resource type.
  servers: string[]
}) {
  const { t } = useTranslation()
  const uid = useId()
  const queryClient = useQueryClient()
  const { data } = useQuery({
    queryKey: ['judge-setting', realmId],
    queryFn: () => api.settings.judge(realmId),
  })
  const [provider, setProvider] = useState<Judge['provider']>('')
  const [model, setModel] = useState('')
  useEffect(() => {
    if (!data) return
    setProvider(data.provider ?? '')
    setModel(data.model ?? '')
  }, [data])
  const save = useMutation({
    mutationFn: () => api.settings.setJudge(realmId, { provider, model: model.trim() }),
    onSuccess: saved => queryClient.setQueryData(['judge-setting', realmId], { ...data, ...saved }),
  })

  if (!data || typeof data.provider !== 'string') return null
  // A server named earlier and removed from the Realm since stays in the list:
  // dropping it would show a choice nobody made.
  const offered = [...new Set([...servers, provider].filter(s => s in SERVER_LABEL))]
  const changed = provider !== data.provider || model.trim() !== data.model

  return (
    <section data-testid="judge-setting">
      <div className="section-rule">
        <h2 className="section-title">{t('realmResourcesPage.judge.heading')}</h2>
        <span className="section-meta">{t('realmResourcesPage.judge.hint')}</span>
      </div>
      <div className="inline-form">
        <div className="form-group">
          <label htmlFor={`${uid}-server`}>{t('realmResourcesPage.judge.serverLabel')}</label>
          <select
            id={`${uid}-server`} className="input" value={provider}
            onChange={e => setProvider(e.target.value as Judge['provider'])}
          >
            <option value="">{t('realmResourcesPage.judge.server.default')}</option>
            {offered.map(s => <option key={s} value={s}>{t(SERVER_LABEL[s])}</option>)}
          </select>
        </div>
        <div className="form-group">
          <label htmlFor={`${uid}-model`}>{t('realmResourcesPage.judge.modelLabel')}</label>
          <input
            id={`${uid}-model`} className="input" value={model}
            placeholder={data.default_model} onChange={e => setModel(e.target.value)}
          />
        </div>
        <button type="button" className="btn btn-primary" disabled={!changed || save.isPending} onClick={() => save.mutate()}>
          {t('realmResourcesPage.save')}
        </button>
      </div>
      {provider === 'openai_compatible' && !model.trim() && (
        <p className="plain-note warn">{t('realmResourcesPage.judge.modelNeeded')}</p>
      )}
      {save.isError && <p className="hint-line bad">{(save.error as Error).message}</p>}
    </section>
  )
}
