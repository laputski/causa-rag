import { useEffect, useState } from 'react'
import { useNavigate, useLocation } from 'react-router-dom'
import { useTranslation, Trans } from 'react-i18next'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { Sparkles, FileEdit, LayoutTemplate } from 'lucide-react'
import { api, PromptTemplate } from '../api/client'
import { useConfirm } from '../components/ConfirmDialog'
import { useRealm, useRealmPath } from '../context/RealmContext'

import PresetsPage from './PresetsPage'

// Prompts and presets are one screen with two tabs.
//
// Both answer "how does the system phrase things": a prompt governs how it
// answers, a preset how it invents questions about a corpus. They used to be
// two menu items and two screens built identically — a list on the left, detail
// on the right, the same form with placeholders. What differs between them is
// smaller than what they share, and keeping them apart meant looking for "the
// template" in two places.
//
// Each tab keeps its own URL (`/prompts` and `/data/presets`), so existing
// links and bookmarks still work, and the menu item highlights on both through
// `matchPaths`.
export default function PromptsPage() {
  const { t } = useTranslation()
  const navigate = useNavigate()
  const location = useLocation()
  const toRealm = useRealmPath()
  const onPresets = location.pathname === '/data/presets'

  return (
    <div className="page-tabbed">
      <div className="data-tab-bar" role="tablist">
        <button
          type="button" role="tab" aria-selected={!onPresets}
          className={`data-tab${!onPresets ? ' active' : ''}`}
          onClick={() => navigate(toRealm('/prompts'))}
        >
          <FileEdit size={14} aria-hidden="true" />{t('promptsPage.tabPrompts')}
        </button>
        <button
          type="button" role="tab" aria-selected={onPresets}
          className={`data-tab${onPresets ? ' active' : ''}`}
          onClick={() => navigate(toRealm('/data/presets'))}
        >
          <LayoutTemplate size={14} aria-hidden="true" />{t('promptsPage.tabPresets')}
        </button>
      </div>
      {onPresets ? <PresetsPage /> : <PromptsTab />}
    </div>
  )
}

function PromptsTab() {
  const { confirm, dialog } = useConfirm()
  const { t } = useTranslation()
  const { activeRealmId } = useRealm()
  const qc = useQueryClient()
  const { data: prompts = [], isLoading } = useQuery({
    queryKey: ['prompts', activeRealmId],
    queryFn: () => api.prompts.list(activeRealmId),
  })

  const [selected, setSelected] = useState<PromptTemplate | null>(null)
  const [creating, setCreating] = useState(false)

  // The active prompt opens rather than an empty right column: this list is
  // never empty, and "choose a prompt" is one extra click for a message that
  // says nothing.
  useEffect(() => {
    if (selected || creating || prompts.length === 0) return
    setSelected(prompts.find(x => x.is_active) ?? prompts[0])
  }, [prompts, selected, creating])
  const [form, setForm] = useState({ name: '', description: '', template: '' })

  const activateMut = useMutation({
    mutationFn: (id: string) => api.prompts.activate(id),
    onSuccess: (updated) => {
      qc.invalidateQueries({ queryKey: ['prompts'] })
      setSelected(updated)
    },
  })

  const deleteMut = useMutation({
    mutationFn: (id: string) => api.prompts.delete(id),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['prompts'] })
      setSelected(null)
    },
  })

  const createMut = useMutation({
    mutationFn: (body: { name: string; description: string; template: string }) =>
      api.prompts.create({ ...body, realm_id: activeRealmId }),
    onSuccess: (t) => {
      qc.invalidateQueries({ queryKey: ['prompts'] })
      setCreating(false)
      setSelected(t)
    },
  })

  const handleNewVersion = () => {
    if (!selected) return
    setForm({ name: selected.name, description: '', template: selected.template })
    setCreating(true)
    setSelected(null)
  }

  const handleCreate = () => {
    setForm({ name: '', description: '', template: '' })
    setCreating(true)
    setSelected(null)
  }

  return (
    <div className="page-flush split">
      {dialog}
      <aside className="split-aside">
        <div className="split-aside-head">
          <h1 className="split-title">{t('promptsPage.title')}</h1>
          <button className="btn btn-sm btn-primary" onClick={handleCreate}>
            {t('promptsPage.newButton')}
          </button>
        </div>

        {isLoading ? (
          <p className="split-empty">{t('promptsPage.loading')}</p>
        ) : prompts.length === 0 ? (
          <p className="split-empty">{t('promptsPage.empty')}</p>
        ) : (
          <ul className="pick-list">
            {prompts.map(item => {
              const isSelected = selected?.id === item.id && !creating
              return (
                <li key={item.id}>
                  <button
                    type="button"
                    className={`pick-row${isSelected ? ' active' : ''}`}
                    onClick={() => { setSelected(item); setCreating(false) }}
                  >
                    <div className="pick-row-title">{item.name}</div>
                    <div className="pick-row-meta">
                      <span className="pick-ver">v{item.version}</span>
                      {item.is_active && (
                        <span className="badge badge-success">{t('promptsPage.activeTag')}</span>
                      )}
                      {item.description && <span className="pick-row-sub">{item.description}</span>}
                    </div>
                  </button>
                </li>
              )
            })}
          </ul>
        )}
      </aside>

      {/* Main content */}
      <main className="split-main">
        {creating ? (
          <CreateForm
            form={form}
            setForm={setForm}
            onSave={() => createMut.mutate(form)}
            onCancel={() => setCreating(false)}
            saving={createMut.isPending}
          />
        ) : selected ? (
          <PromptDetail
            prompt={selected}
            onActivate={() => activateMut.mutate(selected.id)}
            onDelete={async () => {
              if (await confirm({ title: t('promptsPage.confirmDelete', { name: selected.name }), danger: true })) deleteMut.mutate(selected.id)
            }}
            onNewVersion={handleNewVersion}
            activating={activateMut.isPending}
            deleting={deleteMut.isPending}
          />
        ) : (
          <p className="split-empty split-empty-lg">{t('promptsPage.selectPrompt')}</p>
        )}
      </main>
    </div>
  )
}

function PromptDetail({ prompt, onActivate, onDelete, onNewVersion, activating, deleting }: {
  prompt: PromptTemplate
  onActivate: () => void
  onDelete: () => void
  onNewVersion: () => void
  activating: boolean
  deleting: boolean
}) {
  const { t } = useTranslation()
  return (
    <div className="split-body">
      <div className="jd-detail-head">
        <div className="flex-between align-start gap-16">
          <div>
            <h1 className="page-title tight">{prompt.name}</h1>
            <div className="jd-provenance">
              <code className="inline-code">{prompt.id}</code>
              <span>{t('promptsPage.version', { version: prompt.version })}</span>
              {prompt.created_at && (
                <span>{new Date(prompt.created_at).toLocaleDateString('ru-RU', { day: 'numeric', month: 'long', year: 'numeric' })}</span>
              )}
              {/* A label rather than a full-width green banner: a prompt being
                  active is one word about it rather than a message. */}
              {prompt.is_active && <span className="badge badge-success">{t('promptsPage.activeTag')}</span>}
            </div>
          </div>

          <div className="detail-actions">
            <button className="btn btn-sm" onClick={onNewVersion}>{t('promptsPage.newVersion')}</button>
            {!prompt.is_active && (
              <button className="btn btn-sm btn-primary" onClick={onActivate} disabled={activating}>
                {activating ? '...' : t('promptsPage.activate')}
              </button>
            )}
            {!prompt.is_active && (
              <button className="btn btn-sm btn-danger" onClick={onDelete} disabled={deleting}>
                {deleting ? '...' : t('promptsPage.delete')}
              </button>
            )}
          </div>
        </div>

        {prompt.description && <p className="prose-p flat">{prompt.description}</p>}
        {/* What "active" means, as a caption rather than a full-width banner:
            it explains the label above rather than announcing anything. */}
        {prompt.is_active && <p className="hint-line">{t('promptsPage.activeBadge')}</p>}
      </div>

      <div className="section">
        <div className="section-rule flush">
          <h2 className="section-title">{t('promptsPage.templateHeading')}</h2>
          <span className="section-meta">
            <PlaceholderBadge label="{context}" />{' '}<PlaceholderBadge label="{query}" />
          </span>
        </div>
        <pre className="tpl-block tpl-block-tall">{prompt.template}</pre>
      </div>
    </div>
  )
}

function PlaceholderBadge({ label }: { label: string }) {
  return <span className="ph-badge">{label}</span>
}

/** Found live: prompts were only ever written by hand — no way to get a
 * first draft tailored to a specific corpus's actual content and a
 * specific model's own quirks. Samples real chunks from the chosen corpus,
 * asks the chosen model to draft {name, description, template}, and
 * pre-fills the surrounding CreateForm — the user still reviews/edits
 * before saving, this never saves anything itself (matches the "draft
 * only" posture DatasetsPage's own question generator already has). */
function AiDraftPanel({ onDraft }: {
  onDraft: (draft: { name: string; description: string; template: string }) => void
}) {
  const { t } = useTranslation()
  const { activeRealmId } = useRealm()
  const [open, setOpen] = useState(false)
  const [model, setModel] = useState('')
  const [corpusId, setCorpusId] = useState('')

  const { data: models = [] } = useQuery({
    queryKey: ['models', 'completion'],
    queryFn: () => api.models(true),
    enabled: open,
  })
  const { data: corpusCollections = [] } = useQuery({
    queryKey: ['corpus-collections', activeRealmId],
    queryFn: () => api.corpus.collections(activeRealmId),
    enabled: open,
  })

  const generateMut = useMutation({
    mutationFn: () => api.prompts.generate({ realm_id: activeRealmId || '', corpus_id: corpusId, model }),
    onSuccess: (draft) => {
      onDraft(draft)
      setOpen(false)
    },
  })

  if (!open) {
    return (
      <button className="btn btn-sm mb-18" onClick={() => setOpen(true)}>
        <Sparkles size={13} /> {t('promptsPage.aiDraft.openButton')}
      </button>
    )
  }

  const canGenerate = !!model && !!corpusId && !generateMut.isPending

  return (
    <div className="section">
      <div className="section-rule flush">
        <h2 className="section-title">{t('promptsPage.aiDraft.title')}</h2>
        <button className="btn btn-sm push" onClick={() => setOpen(false)}>
          {t('promptsPage.cancel')}
        </button>
      </div>
      <p className="hint-line flush">{t('promptsPage.aiDraft.subtitle')}</p>

      <div className="form-grid">
        <div className="form-group">
          <label htmlFor="ai-draft-model">{t('promptsPage.aiDraft.modelLabel')}</label>
          <select id="ai-draft-model" value={model} onChange={e => setModel(e.target.value)} disabled={generateMut.isPending}>
            <option value="">{t('promptsPage.aiDraft.selectPrompt')}</option>
            {models.map(m => <option key={m.name} value={m.name}>{m.name}</option>)}
          </select>
        </div>
        <div className="form-group">
          <label htmlFor="ai-draft-corpus">{t('promptsPage.aiDraft.corpusLabel')}</label>
          <select id="ai-draft-corpus" value={corpusId} onChange={e => setCorpusId(e.target.value)} disabled={generateMut.isPending}>
            <option value="">{t('promptsPage.aiDraft.selectPrompt')}</option>
            {corpusCollections.map(c => (
              <option key={c.corpus_id} value={c.corpus_id}>
                {c.corpus_id}{c.description ? ` — ${c.description}` : ''}
              </option>
            ))}
          </select>
        </div>
      </div>

      {generateMut.isError && (
        <p className="conn-status conn-status-err mt-10">{String(generateMut.error)}</p>
      )}
      <button className="btn btn-primary mt-12" disabled={!canGenerate} onClick={() => generateMut.mutate()}>
        <Sparkles size={13} /> {generateMut.isPending ? t('promptsPage.aiDraft.generating') : t('promptsPage.aiDraft.generate')}
      </button>
    </div>
  )
}

function CreateForm({ form, setForm, onSave, onCancel, saving }: {
  form: { name: string; description: string; template: string }
  setForm: (f: { name: string; description: string; template: string }) => void
  onSave: () => void
  onCancel: () => void
  saving: boolean
}) {
  const { t } = useTranslation()
  const valid = form.name.trim() && form.template.trim()

  return (
    <div className="split-body">
      <div className="edit-form">
        <div className="edit-form-head">
          <h3 className="edit-form-title">{t('promptsPage.newPromptTitle')}</h3>
          <div className="detail-actions">
            <button className="btn btn-sm" onClick={onCancel}>{t('promptsPage.cancel')}</button>
            <button className="btn btn-sm btn-primary" onClick={onSave} disabled={saving || !valid}>
              {saving ? t('promptsPage.creating') : t('promptsPage.create')}
            </button>
          </div>
        </div>

        <AiDraftPanel onDraft={draft => setForm({ ...form, ...draft })} />

        <div className="edit-grid">
          <div className="form-group">
            <label htmlFor="prompt-name">
              {t('promptsPage.nameLabel')} <span className="req-mark">*</span>
            </label>
            <input
              id="prompt-name" value={form.name}
              onChange={e => setForm({ ...form, name: e.target.value })}
              placeholder={t('promptsPage.namePlaceholder')}
            />
          </div>
          <div className="form-group">
            <label htmlFor="prompt-desc">{t('promptsPage.descriptionLabel')}</label>
            <input
              id="prompt-desc" value={form.description}
              onChange={e => setForm({ ...form, description: e.target.value })}
              placeholder={t('promptsPage.descriptionPlaceholder')}
            />
          </div>

          <div className="form-group form-span">
            <label htmlFor="prompt-tpl">
              {t('promptsPage.templateLabel')} <span className="req-mark">*</span>
              <span className="ph-row inline">
                <PlaceholderBadge label="{context}" />{' '}<PlaceholderBadge label="{query}" />
              </span>
            </label>
            <textarea
              id="prompt-tpl" className="tpl-input"
              value={form.template}
              onChange={e => setForm({ ...form, template: e.target.value })}
              placeholder={t('promptsPage.templatePlaceholder')}
            />
        <p className="hint-line">
          <Trans i18nKey="promptsPage.placeholdersHint" t={t}>
            The <code className="inline-code">{'{context}'}</code> and <code className="inline-code">{'{query}'}</code> placeholders are required: both are substituted on every request
          </Trans>
        </p>
          </div>
        </div>
      </div>
    </div>
  )
}
