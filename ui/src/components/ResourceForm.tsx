import { useId, useState } from 'react'
import { useTranslation } from 'react-i18next'

// The form of one Realm resource, for adding it and for changing it.
//
// A secret is never shown and never sent back unless somebody typed a new
// one. The server answers a stored secret with a mask and reads the mask on a
// save as "keep what is stored" (core/secrets.py), so the three things a
// reader may want of a stored secret are three different values here: the
// mask keeps it, a typed value replaces it, an empty string removes it.

export interface FieldSchema { type: string; default?: unknown; secret?: boolean }
export type ResourceValues = Record<string, unknown>

/** What the server shows in place of a stored secret (core/secrets.py#MASK). */
export const STORED_SECRET = '••••••••'

// A field's name as a reader meets it. A field with no entry is shown by its
// own name, which is what a connector type added later gets until it is named.
const FIELD_LABEL: Record<string, string> = {
  host: 'realmResourcesPage.field.host',
  port: 'realmResourcesPage.field.port',
  url: 'realmResourcesPage.field.url',
  uri: 'realmResourcesPage.field.uri',
  https: 'realmResourcesPage.field.https',
  use_ssl: 'realmResourcesPage.field.useSsl',
  verify_certs: 'realmResourcesPage.field.verifyCerts',
  user: 'realmResourcesPage.field.user',
  password: 'realmResourcesPage.field.password',
  api_key: 'realmResourcesPage.field.apiKey',
  database: 'realmResourcesPage.field.database',
  base_url: 'realmResourcesPage.field.baseUrl',
  model: 'realmResourcesPage.field.model',
  embedding_model: 'realmResourcesPage.field.embeddingModel',
}

const isTrue = (value: unknown) => value === true || ['1', 'true', 'yes', 'on'].includes(String(value).toLowerCase())

function SecretField({ id, stored, onChange }: {
  id: string
  stored: boolean
  // Told what a save should send: the mask, a new value, or an empty string.
  onChange: (v: string) => void
}) {
  const { t } = useTranslation()
  const [mode, setMode] = useState<'kept' | 'typing' | 'removed'>(stored ? 'kept' : 'typing')
  const [typed, setTyped] = useState('')
  const keep = () => { setMode('kept'); setTyped(''); onChange(STORED_SECRET) }

  if (mode === 'kept') {
    return (
      <span className="secret-state">
        <span className="secret-saved">{t('realmResourcesPage.secret.stored')}</span>
        <button type="button" className="btn-sm" onClick={() => setMode('typing')}>
          {t('realmResourcesPage.secret.replace')}
        </button>
        <button type="button" className="btn-sm" onClick={() => { setMode('removed'); onChange('') }}>
          {t('realmResourcesPage.secret.remove')}
        </button>
      </span>
    )
  }
  if (mode === 'removed') {
    return (
      <span className="secret-state">
        <span className="secret-saved">{t('realmResourcesPage.secret.willBeRemoved')}</span>
        <button type="button" className="btn-sm" onClick={keep}>{t('realmResourcesPage.secret.keep')}</button>
      </span>
    )
  }
  return (
    <span className="secret-state">
      <input
        id={id} className="form-input" type="password" autoComplete="new-password" value={typed}
        onChange={e => {
          setTyped(e.target.value)
          // A stored secret is replaced only by something typed: an input left
          // empty keeps it, and removing it is a button of its own.
          onChange(e.target.value === '' && stored ? STORED_SECRET : e.target.value)
        }}
      />
      {stored && <button type="button" className="btn-sm" onClick={keep}>{t('realmResourcesPage.secret.keep')}</button>}
    </span>
  )
}

export default function ResourceForm({ schema, initial, saving, note, onSave, onCancel }: {
  schema: Record<string, FieldSchema>
  // The resource as the server shows it, when one is being changed.
  initial?: ResourceValues
  saving: boolean
  note?: string
  onSave: (values: ResourceValues) => void
  onCancel: () => void
}) {
  const { t } = useTranslation()
  const uid = useId()
  const [values, setValues] = useState<ResourceValues>(() => {
    const start: ResourceValues = {}
    for (const [key, field] of Object.entries(schema)) {
      const given = initial?.[key]
      if (field.type === 'boolean') start[key] = isTrue(given ?? field.default ?? false)
      else start[key] = given !== undefined && given !== null ? String(given) : String(field.default ?? '')
    }
    return start
  })
  const set = (key: string, value: unknown) => setValues(prev => ({ ...prev, [key]: value }))

  return (
    <div data-testid="resource-form">
      {Object.entries(schema).map(([key, field]) => {
        const label = FIELD_LABEL[key] ? t(FIELD_LABEL[key]) : key
        const id = `${uid}-${key}`
        if (field.type === 'boolean') {
          return (
            <div key={key} className="form-group">
              <label className="check-line" htmlFor={id}>
                <input id={id} type="checkbox" checked={Boolean(values[key])} onChange={e => set(key, e.target.checked)} />
                {label}
              </label>
            </div>
          )
        }
        return (
          <div key={key} className="form-group">
            <label htmlFor={id}>{label}</label>
            {field.secret ? (
              <SecretField id={id} stored={initial?.[key] === STORED_SECRET} onChange={v => set(key, v)} />
            ) : (
              <input
                id={id} className="form-input"
                type={field.type === 'integer' ? 'number' : 'text'}
                value={String(values[key] ?? '')} onChange={e => set(key, e.target.value)}
              />
            )}
          </div>
        )
      })}
      {note && <p className="field-note">{note}</p>}
      <div className="flex-row res-actions">
        <button className="btn-primary" onClick={() => onSave(values)} disabled={saving}>
          {t('realmResourcesPage.save')}
        </button>
        <button className="btn" onClick={onCancel}>{t('realmResourcesPage.cancel')}</button>
      </div>
    </div>
  )
}
