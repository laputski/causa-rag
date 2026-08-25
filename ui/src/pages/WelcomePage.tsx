import { useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { Plus, Upload, Sparkles, ArrowRight } from 'lucide-react'
import { api, type RealmBundle } from '../api/client'
import { useRealm } from '../context/RealmContext'

// First start: there are no realms.
//
// The shell used to answer every path with the realm management page in this
// state, and the whole menu went dim with a single tooltip. Somebody opening
// the platform for the first time saw a greyed-out interface and could not tell
// whether it was broken or deliberate.
//
// Three doors rather than one, because there genuinely are three ways in: start
// your own, carry one over from another installation, look at a finished
// example. The third is not a separate mechanism; it is an export file loaded
// through the same route as any other, which makes the demo realm a live check
// that import works.

type Mode = 'choose' | 'create'

export default function WelcomePage() {
  const { t } = useTranslation()
  const { reload, setActiveRealmId } = useRealm()
  const [mode, setMode] = useState<Mode>('choose')
  const [name, setName] = useState('')
  const [description, setDescription] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const fileRef = useRef<HTMLInputElement>(null)

  const create = async () => {
    if (!name.trim()) return
    setBusy(true); setError('')
    try {
      const res = await fetch('/api/realms', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: name.trim(), description: description.trim() }),
      })
      if (!res.ok) {
        const d = await res.json().catch(() => ({}))
        setError(d.detail ?? t('welcome.createError'))
        return
      }
      const created = await res.json()
      await reload()
      setActiveRealmId(created.id)
    } catch (e) {
      setError(String(e))
    } finally {
      setBusy(false)
    }
  }

  const importFile = async (file: File) => {
    setBusy(true); setError('')
    try {
      const bundle = JSON.parse(await file.text()) as RealmBundle
      const report = await api.realms.importRealm(bundle, { onConflict: 'rename' })
      await reload()
      setActiveRealmId(report.realm_id)
    } catch (e) {
      setError(t('welcome.importError', { detail: String(e) }))
    } finally {
      setBusy(false)
    }
  }

  const openDemo = async () => {
    setBusy(true); setError('')
    try {
      // The demo realm ships beside the build as an ordinary export file and
      // loads through the same route. There is no separate "create example":
      // that would let the example and the import path diverge, and the example
      // would stop being a check.
      const res = await fetch('/demo.realm.json')
      if (!res.ok) throw new Error(t('welcome.demoMissing'))
      const report = await api.realms.importRealm(await res.json(), { onConflict: 'rename' })
      await reload()
      setActiveRealmId(report.realm_id)
    } catch (e) {
      setError(t('welcome.importError', { detail: String(e) }))
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="page welcome-page">
      <h1 className="page-title">{t('welcome.title')}</h1>
      <p className="guide-lead">{t('welcome.lead')}</p>

      {mode === 'create' ? (
        <div className="section">
          <div className="section-rule flush">
            <h2 className="section-title">{t('welcome.createTitle')}</h2>
          </div>
          <div className="form-group">
            <label htmlFor="realm-name">{t('realmsPage.nameLabel')}</label>
            <input
              id="realm-name" className="input" value={name} autoFocus
              onChange={e => setName(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && create()}
              placeholder={t('realmsPage.namePlaceholder')}
            />
          </div>
          <div className="form-group">
            <label htmlFor="realm-desc">{t('realmsPage.descriptionLabel')}</label>
            <textarea
              id="realm-desc" className="input" rows={2} value={description}
              onChange={e => setDescription(e.target.value)}
              placeholder={t('realmsPage.descriptionPlaceholder')}
            />
          </div>
          <div className="flex-row welcome-actions">
            <button type="button" className="btn btn-primary" onClick={create} disabled={busy || !name.trim()}>
              {busy ? t('welcome.creating') : t('realmsPage.create')}
            </button>
            <button type="button" className="btn" onClick={() => setMode('choose')} disabled={busy}>
              {t('realmsPage.cancel')}
            </button>
          </div>
          {error && <p className="conn-status conn-status-err welcome-err">{error}</p>}
        </div>
      ) : (
        <div className="choice-list">
          <button type="button" className="choice choice-primary" onClick={() => setMode('create')} disabled={busy}>
            <span className="choice-icon"><Plus size={16} /></span>
            <span className="choice-body">
              <span className="choice-title">{t('welcome.create')}</span>
              <span className="choice-desc">{t('welcome.createDesc')}</span>
            </span>
            <ArrowRight size={14} className="choice-arrow" aria-hidden="true" />
          </button>

          <button type="button" className="choice" onClick={() => fileRef.current?.click()} disabled={busy}>
            <span className="choice-icon"><Upload size={16} /></span>
            <span className="choice-body">
              <span className="choice-title">{t('welcome.import')}</span>
              <span className="choice-desc">{t('welcome.importDesc')}</span>
            </span>
            <ArrowRight size={14} className="choice-arrow" aria-hidden="true" />
          </button>

          <button type="button" className="choice" onClick={openDemo} disabled={busy}>
            <span className="choice-icon"><Sparkles size={16} /></span>
            <span className="choice-body">
              <span className="choice-title">{t('welcome.demo')}</span>
              <span className="choice-desc">{t('welcome.demoDesc')}</span>
            </span>
            <ArrowRight size={14} className="choice-arrow" aria-hidden="true" />
          </button>

          <input
            ref={fileRef} type="file" accept="application/json" hidden
            onChange={e => { const f = e.target.files?.[0]; if (f) importFile(f); e.target.value = '' }}
          />
          {error && <p className="conn-status conn-status-err">{error}</p>}
        </div>
      )}

      <p className="welcome-foot">{t('welcome.foot')}</p>
    </div>
  )
}
