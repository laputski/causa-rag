import { useEffect, useRef, useState } from 'react'
import { Plus, RotateCcw, Flame, Pencil, Check, X, Download, Upload } from 'lucide-react'
import { useTranslation } from 'react-i18next'
import { useQuery } from '@tanstack/react-query'
import { useRealm } from '../context/RealmContext'
import { api, type RealmImportReport } from '../api/client'
import { platform } from '../lib/platform'
import { useConfirm } from '../components/ConfirmDialog'

interface Realm {
  id: string
  name: string
  description?: string
  resources: unknown[]
  created_at: string
  deleted_at?: string | null
}

export default function RealmsPage() {
  const { t } = useTranslation()
  const { realms, reload } = useRealm()
  // The create form starts collapsed: a realm is created once and the list is
  // read on every visit, and an expanded form pushed it down the page.
  // The run count per realm: the run list already carries `realm_id`, so this
  // is one request rather than one per row.
  const { data: allRuns = [] } = useQuery({
    queryKey: ['experiments', 'all-realms'],
    queryFn: () => api.experiments.list({}),
  })
  const runCounts = allRuns.reduce<Record<string, number>>((acc, r) => {
    const id = r.realm_id
    if (id) acc[id] = (acc[id] ?? 0) + 1
    return acc
  }, {})

  const [showCreate, setShowCreate] = useState(false)
  const [name, setName] = useState('')
  const [description, setDescription] = useState('')
  const [creating, setCreating] = useState(false)
  const [error, setError] = useState('')
  const [deletedRealms, setDeletedRealms] = useState<Realm[]>([])
  const [showDeleted, setShowDeleted] = useState(false)
  const [editingId, setEditingId] = useState<string | null>(null)
  const [editName, setEditName] = useState('')
  const [editDescription, setEditDescription] = useState('')
  const [renaming, setRenaming] = useState(false)
  const { confirm, dialog } = useConfirm()
  const [error2, setError2] = useState('')
  const [preview, setPreview] = useState<RealmImportReport | null>(null)
  const [pendingBundle, setPendingBundle] = useState<unknown>(null)
  const [busy, setBusy] = useState(false)
  const fileRef = useRef<HTMLInputElement>(null)

  // Export goes through the platform layer rather than a `createObjectURL`
  // here: in the desktop build that becomes a native save dialog, and swapping
  // one function costs less than editing pages.
  const exportRealm = async (r: Realm) => {
    const bundle = await api.realms.exportRealm(r.id)
    await platform.saveFile(`${r.id}.realm.json`, JSON.stringify(bundle, null, 2))
  }

  // A reconciliation before writing: how many records will appear, what does
  // not carry over, which fields have to be entered again. An import you run to
  // see what happens is not an import.
  const previewImport = async (file: File) => {
    setBusy(true)
    try {
      const bundle = JSON.parse(await file.text())
      const report = await api.realms.importRealm(bundle, { onConflict: 'rename', dryRun: true })
      setPendingBundle(bundle)
      setPreview(report)
    } catch (e) {
      setError2(String(e))
    } finally {
      setBusy(false)
    }
  }

  const confirmImport = async () => {
    if (!pendingBundle) return
    setBusy(true)
    try {
      await api.realms.importRealm(pendingBundle as never, { onConflict: 'rename' })
      setPreview(null); setPendingBundle(null)
      await reload()
    } finally {
      setBusy(false)
    }
  }

  const loadDeleted = async () => {
    try {
      const res = await fetch('/api/realms?include_deleted=true')
      if (res.ok) {
        const all: Realm[] = await res.json()
        setDeletedRealms(all.filter(r => r.deleted_at))
      }
    } catch { /* non-fatal */ }
  }

  useEffect(() => { reload(); loadDeleted() }, [])  // eslint-disable-line

  const create = async () => {
    if (!name.trim()) return
    setCreating(true)
    setError('')
    try {
      const res = await fetch('/api/realms', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: name.trim(), description: description.trim() }),
      })
      if (!res.ok) {
        const d = await res.json()
        setError(d.detail ?? t('realmsPage.createError'))
      } else {
        setName('')
        setDescription('')
        await reload()
      }
    } finally {
      setCreating(false)
    }
  }

  // Soft delete by default — a Realm carries resources/RAG endpoints/runs/
  // settings that take a while to re-enter, so a misclick shouldn't destroy
  // them. Hides the Realm from the switcher and lists; the data stays in the
  // DB and is reachable via "Restore" below.
  const remove = async (r: Realm) => {
    const ok = await confirm({
      title: t('realmsPage.confirmHide', { name: r.name }),
      confirmLabel: t('realmsPage.hideTitle'),
    })
    if (!ok) return
    const res = await fetch(`/api/realms/${r.id}`, { method: 'DELETE' })
    if (!res.ok && res.status !== 204) {
      const d = await res.json()
      setError2(d.detail ?? t('realmsPage.deleteError'))
    } else {
      await reload()
      await loadDeleted()
    }
  }

  const restore = async (id: string) => {
    const res = await fetch(`/api/realms/${id}/restore`, { method: 'POST' })
    if (!res.ok) {
      const d = await res.json()
      setError2(d.detail ?? t('realmsPage.restoreError'))
    } else {
      await reload()
      await loadDeleted()
    }
  }

  const startEdit = (r: Realm) => {
    setEditingId(r.id)
    setEditName(r.name)
    setEditDescription(r.description ?? '')
  }

  const saveRename = async (id: string) => {
    if (!editName.trim()) return
    setRenaming(true)
    try {
      const res = await fetch(`/api/realms/${id}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: editName.trim(), description: editDescription.trim() }),
      })
      if (!res.ok) {
        const d = await res.json()
        setError2(d.detail ?? t('realmsPage.renameError'))
      } else {
        setEditingId(null)
        await reload()
      }
    } finally {
      setRenaming(false)
    }
  }

  const hardDelete = async (r: Realm) => {
    // Typing the id is not ceremony: it is the only thing separating a
    // deliberate irreversible delete from a miss on the neighbouring button.
    const ok = await confirm({
      title: t('realmsPage.hardDeleteTitle'),
      body: t('realmsPage.hardDeletePrompt', { name: r.name, id: r.id }),
      confirmLabel: t('realmsPage.hardDeleteTitle'), danger: true,
      requireTyped: r.id, typedHint: t('realmsPage.idMismatch'),
    })
    if (!ok) return
    const res = await fetch(`/api/realms/${r.id}?hard=true`, { method: 'DELETE' })
    if (!res.ok && res.status !== 204) {
      const d = await res.json()
      setError2(d.detail ?? t('realmsPage.deleteError'))
    } else {
      await loadDeleted()
    }
  }

  return (
    <div className="page page-wide">
      {dialog}
      {error2 && <p className="conn-status conn-status-err mb-12">{error2}</p>}

      <div className="page-head">
        <h1 className="page-title">{t('realmsPage.title')}</h1>
        <p className="page-sub">
          {t('realmsPage.countActive', { count: realms.length })}
          {deletedRealms.length > 0 && ` · ${t('realmsPage.countHidden', { count: deletedRealms.length })}`}
        </p>
        <span className="page-act">
          <button type="button" className="btn btn-sm" onClick={() => fileRef.current?.click()} disabled={busy}>
            <Upload size={13} />{t('realmsPage.importRealm')}
          </button>
          <button type="button" className="btn btn-sm btn-primary" onClick={() => setShowCreate(v => !v)}>
            <Plus size={13} />{t('realmsPage.createHeading')}
          </button>
        </span>
      </div>

      <input
        ref={fileRef} type="file" accept="application/json" hidden
        onChange={e => { const f = e.target.files?.[0]; if (f) previewImport(f); e.target.value = '' }}
      />

      {showCreate && (
      <div className="edit-form">
        <div className="edit-form-head">
          <h3 className="edit-form-title">{t('realmsPage.createHeading')}</h3>
          <button className="btn btn-sm" onClick={() => setShowCreate(false)}>{t('realmsPage.cancel')}</button>
        </div>
        <div className="form-group">
          <label>{t('realmsPage.nameLabel')}</label>
          <input
            className="form-input"
            value={name}
            onChange={e => setName(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && create()}
            placeholder={t('realmsPage.namePlaceholder')}
          />
        </div>
        <div className="form-group">
          <label>{t('realmsPage.descriptionLabel')}</label>
          <textarea
            className="form-input"
            value={description}
            onChange={e => setDescription(e.target.value)}
            placeholder={t('realmsPage.descriptionPlaceholder')}
            rows={2}
          />
        </div>
        <button className="btn-primary" onClick={create} disabled={creating || !name.trim()}>
          <Plus size={14} className="btn-icon" />{t('realmsPage.create')}
        </button>
        {error && <p className="conn-status conn-status-err mt-10">{error}</p>}
      </div>
      )}

      {realms.length === 0 ? (
        <div className="card">
          <p className="text-muted">{t('realmsPage.empty')}</p>
        </div>
      ) : (
        <div className="card frame-card">
          <table className="table full-table">
            <thead>
              <tr>
                <th>{t('realmsPage.name')}</th>
                <th>{t('realmsPage.idCol')}</th>
                <th className="num">{t('realmsPage.resources')}</th>
                <th className="num">{t('realmsPage.runs')}</th>
                <th className="col-actions" />
              </tr>
            </thead>
            <tbody>
              {realms.map((r: Realm) => (
                <tr key={r.id}>
                  <td className="strong-cell">
                    {editingId === r.id ? (
                      <div className="realm-edit">
                        <input
                          className="form-input realm-edit-name"
                          value={editName}
                          autoFocus
                          onChange={e => setEditName(e.target.value)}
                          onKeyDown={e => e.key === 'Escape' && setEditingId(null)}
                        />
                        <textarea
                          className="form-input realm-edit-desc"
                          value={editDescription}
                          placeholder={t('realmsPage.descriptionLabel')}
                          rows={2}
                          onChange={e => setEditDescription(e.target.value)}
                          onKeyDown={e => e.key === 'Escape' && setEditingId(null)}
                        />
                        <div className="flex-row gap-6">
                          <button className="btn-sm" onClick={() => saveRename(r.id)} disabled={renaming || !editName.trim()} title={t('realmsPage.save')}>
                            <Check size={13} />
                          </button>
                          <button className="btn-sm" onClick={() => setEditingId(null)} title={t('realmsPage.cancel')}>
                            <X size={13} />
                          </button>
                        </div>
                      </div>
                    ) : (
                      <div>
                        <span className="flex-row gap-6">
                          {r.name}
                          <button className="btn-sm bare-btn" onClick={() => startEdit(r)} title={t('realmsPage.editTitle')}>
                            <Pencil size={12} />
                          </button>
                        </span>
                        {r.description && (
                          <p className="text-muted realm-desc">{r.description}</p>
                        )}
                      </div>
                    )}
                  </td>
                  <td><code className="inline-code">{r.id}</code></td>
                  <td className="num">{r.resources.length}</td>
                  {/* A run count column, which did not exist: a realm with no
                      runs and a realm with fifty differ in exactly that, rather
                      than in their resource count. */}
                  <td className="num">{runCounts[r.id] ?? '—'}</td>
                  <td className="q-actions">
                    <button className="btn btn-sm" onClick={() => exportRealm(r)}>
                      <Download size={13} />{t('realmsPage.exportAction')}
                    </button>
                    <button className="btn btn-sm btn-danger" onClick={() => remove(r)}>
                      {t('realmsPage.hideAction')}
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {preview && (
        <div className="overlay" onMouseDown={() => setPreview(null)} role="presentation">
          <div className="sheet sheet-sm" role="dialog" aria-modal="true"
               onMouseDown={e => e.stopPropagation()}>
            <div className="sheet-head">
              <h2 className="sheet-title">{t('realmsPage.importPreview')}</h2>
              <button type="button" className="icon-btn" onClick={() => setPreview(null)}><X size={15} /></button>
            </div>
            <p className="text-muted sheet-sub">
              {t('realmsPage.importPreviewLead', { id: preview.realm_id })}
            </p>
            {preview.entries.map(entry => (
              <div key={entry.kind} className="sheet-row">
                <span>{t(`realmsPage.entry.${entry.kind}`, { defaultValue: entry.kind })}</span>
                <span className="mono-sm">
                  {entry.created > 0 ? `+${entry.created}` : t('realmsPage.notTransferred', { count: entry.skipped })}
                </span>
              </div>
            ))}
            {preview.warnings.map(w => (
              <p key={w} className="tool-card-note mt-10">{w}</p>
            ))}
            <div className="flex-row res-actions mt-16">
              <button type="button" className="btn btn-primary" onClick={confirmImport} disabled={busy}>
                {busy ? t('welcome.creating') : t('realmsPage.importConfirm')}
              </button>
              <button type="button" className="btn" onClick={() => setPreview(null)}>{t('realmsPage.cancel')}</button>
            </div>
          </div>
        </div>
      )}

      {deletedRealms.length > 0 && (
        <div className="mt-20">
          <button className="btn-sm" onClick={() => setShowDeleted(s => !s)}>
            {showDeleted ? t('realmsPage.hide') : t('realmsPage.show')} {t('realmsPage.deletedCount', { count: deletedRealms.length })}
          </button>
          {showDeleted && (
            <div className="card frame-card mt-8">
              <table className="table full-table">
                <thead>
                  <tr>
                    <th>{t('realmsPage.name')}</th>
                    <th>ID</th>
                    <th className="col-actions-sm"></th>
                  </tr>
                </thead>
                <tbody>
                  {deletedRealms.map(r => (
                    <tr key={r.id}>
                      <td className="text-muted strong-cell">{r.name}</td>
                      <td><code className="id-code">{r.id}</code></td>
                      <td className="right">
                        <div className="flex-row gap-6 end">
                          <button className="btn-sm" onClick={() => restore(r.id)} title={t('realmsPage.restoreTitle')}>
                            <RotateCcw size={13} />
                          </button>
                          <button className="btn-sm btn-danger" onClick={() => hardDelete(r)} title={t('realmsPage.hardDeleteTitle')}>
                            <Flame size={13} />
                          </button>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
