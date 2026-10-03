import { useEffect, useId, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useTranslation } from 'react-i18next'
import { api, type DebugRetention } from '../api/client'

// The rule by which old runs lose their intermediate retrieval lists.
//
// One rule for the whole installation, shown on the Realms page because it
// counts "the last runs" per Realm. Beside the rule stands what it would do
// now, asked of the server without doing it: a rule whose effect is only
// visible after it ran is a rule nobody can set.
//
// The lower bounds are the server's (routers/settings.py): a rule that cleared
// a run the moment it was saved would leave nothing to debug.

const FIELDS: { key: keyof Omit<DebugRetention, 'automatic'>; label: string; min: number }[] = [
  { key: 'keep_last_per_realm', label: 'realmsPage.retention.keepLast', min: 1 },
  { key: 'max_age_days', label: 'realmsPage.retention.maxAge', min: 1 },
  { key: 'max_mib', label: 'realmsPage.retention.maxVolume', min: 16 },
]

export default function RunRetention() {
  const { t } = useTranslation()
  const uid = useId()
  const queryClient = useQueryClient()
  const { data } = useQuery({ queryKey: ['debug-retention'], queryFn: () => api.settings.retention() })
  const { data: preview } = useQuery({
    queryKey: ['debug-retention-preview', data?.keep_last_per_realm, data?.max_age_days, data?.max_mib],
    queryFn: () => api.settings.retentionPreview(),
    enabled: !!data,
  })
  const [rule, setRule] = useState<DebugRetention | null>(null)
  useEffect(() => { if (data) setRule(data) }, [data])
  const save = useMutation({
    mutationFn: (next: DebugRetention) => api.settings.setRetention({
      keep_last_per_realm: next.keep_last_per_realm, max_age_days: next.max_age_days, max_mib: next.max_mib,
    }),
    onSuccess: saved => queryClient.setQueryData(['debug-retention'], { ...data, ...saved }),
  })

  if (!data || !rule || typeof data.max_age_days !== 'number') return null
  const changed = FIELDS.some(f => rule[f.key] !== data[f.key])
  const valid = FIELDS.every(f => Number.isInteger(rule[f.key]) && rule[f.key] >= f.min)

  return (
    <section data-testid="run-retention">
      <div className="section-rule">
        <h2 className="section-title">{t('realmsPage.retention.heading')}</h2>
      </div>
      <p className="field-note">{t('realmsPage.retention.intro')}</p>
      <div className="inline-form">
        {FIELDS.map(f => (
          <div key={f.key} className="form-group">
            <label htmlFor={`${uid}-${f.key}`}>{t(f.label)}</label>
            <input
              id={`${uid}-${f.key}`} className="input" type="number" min={f.min} value={rule[f.key]}
              onChange={e => setRule({ ...rule, [f.key]: Number(e.target.value) })}
            />
          </div>
        ))}
        <button type="button" className="btn btn-primary" disabled={!changed || !valid || save.isPending} onClick={() => save.mutate(rule)}>
          {t('realmsPage.save')}
        </button>
      </div>
      <div className="inline-form-foot">
        {preview && (
          <span className="field-note">{t('realmsPage.retention.preview', { count: preview.cleared.length })}</span>
        )}
        <span className="field-note">{t('realmsPage.retention.baseline')}</span>
      </div>
      {data.automatic === false && <p className="plain-note">{t('realmsPage.retention.automaticOff')}</p>}
      {save.isError && <p className="hint-line bad">{(save.error as Error).message}</p>}
    </section>
  )
}
