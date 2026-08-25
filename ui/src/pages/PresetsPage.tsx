import { useEffect, useId, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { api, GenerationPreset } from '../api/client'
import { useConfirm } from '../components/ConfirmDialog'
import { useRealm } from '../context/RealmContext'

type PresetForm = { name: string; description: string; template: string }

const EMPTY_FORM: PresetForm = { name: '', description: '', template: '' }

/** The placeholders, grouped by sampling form: a preset is written for one of
 *  the three, and mixed together they do not say which. */
const SHAPE_PLACEHOLDERS = {
  single: ['{chunk_text}'],
  pair: ['{chunk_text_a}', '{chunk_text_b}'],
  range: ['{chunks_text_joined}'],
} as const
/** Available in any form. */
const COMMON_PLACEHOLDERS = ['{question_type}', '{question_type_hint}']

export default function PresetsPage() {
  const { confirm, dialog } = useConfirm()
  const { t } = useTranslation()
  const { activeRealmId } = useRealm()
  const qc = useQueryClient()
  const { data: presets = [], isLoading } = useQuery({
    queryKey: ['generation-presets', activeRealmId],
    queryFn: () => api.generationPresets.list(activeRealmId),
  })

  const [selected, setSelected] = useState<GenerationPreset | null>(null)
  const [creating, setCreating] = useState(false)

  // The first preset opens straight away, for the same reason as on the
  // prompts tab: there is always something to choose from.
  useEffect(() => {
    if (selected || creating || presets.length === 0) return
    setSelected(presets[0])
  }, [presets, selected, creating])
  const [form, setForm] = useState<PresetForm>(EMPTY_FORM)

  const createMut = useMutation({
    mutationFn: (body: PresetForm) => api.generationPresets.create({ ...body, realm_id: activeRealmId }),
    onSuccess: (p) => {
      qc.invalidateQueries({ queryKey: ['generation-presets'] })
      setCreating(false)
      setSelected(p)
    },
  })

  const updateMut = useMutation({
    mutationFn: (body: PresetForm) => api.generationPresets.update(selected!.id, body, activeRealmId),
    onSuccess: (p) => {
      qc.invalidateQueries({ queryKey: ['generation-presets'] })
      setSelected(p)
    },
  })

  const deleteMut = useMutation({
    mutationFn: (id: string) => api.generationPresets.delete(id, activeRealmId),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['generation-presets'] })
      setSelected(null)
    },
  })

  const handleCreate = () => {
    setForm(EMPTY_FORM)
    setCreating(true)
    setSelected(null)
  }

  const handleSelect = (p: GenerationPreset) => {
    setSelected(p)
    setCreating(false)
  }

  return (
    <div className="page-flush split">
      {dialog}
      <aside className="split-aside">
        <div className="split-aside-head">
          <h1 className="split-title">{t('presetsPage.title')}</h1>
          <button className="btn btn-sm btn-primary" onClick={handleCreate}>
            {t('presetsPage.newButton')}
          </button>
        </div>
        <p className="split-note">{t('presetsPage.optionalHint')}</p>

        {isLoading ? (
          <p className="split-empty">{t('presetsPage.loading')}</p>
        ) : presets.length === 0 ? (
          <p className="split-empty">{t('presetsPage.empty')}</p>
        ) : (
          <ul className="pick-list">
            {presets.map(p => {
              const isSelected = selected?.id === p.id && !creating
              return (
                <li key={p.id}>
                  <button
                    type="button"
                    className={`pick-row${isSelected ? ' active' : ''}`}
                    onClick={() => handleSelect(p)}
                  >
                    <div className="pick-row-title">{p.name}</div>
                    {p.description && (
                      <div className="pick-row-meta">
                        <span className="pick-row-sub">{p.description}</span>
                      </div>
                    )}
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
          <PresetForm_
            form={form}
            setForm={setForm}
            title={t('presetsPage.newPresetTitle')}
            onSave={() => createMut.mutate(form)}
            onCancel={() => setCreating(false)}
            saving={createMut.isPending}
            saveLabel={t('presetsPage.create')}
            savingLabel={t('presetsPage.creating')}
          />
        ) : selected ? (
          <PresetForm_
            form={{ name: selected.name, description: selected.description, template: selected.template }}
            setForm={f => setSelected({ ...selected, ...f })}
            title={selected.name}
            onSave={() => updateMut.mutate({ name: selected.name, description: selected.description, template: selected.template })}
            onCancel={null}
            saving={updateMut.isPending}
            saveLabel={t('presetsPage.save')}
            savingLabel={t('presetsPage.saving')}
            onDelete={async () => {
              if (await confirm({ title: t('presetsPage.confirmDelete', { name: selected.name }), danger: true })) deleteMut.mutate(selected.id)
            }}
            deleting={deleteMut.isPending}
          />
        ) : (
          <div className="split-empty split-empty-lg">
            <p>{t('presetsPage.selectPreset')}</p>
            <p className="hint-line">{t('presetsPage.optionalHint')}</p>
          </div>
        )}
      </main>
    </div>
  )
}

function PlaceholderBadge({ label }: { label: string }) {
  return <span className="ph-badge">{label}</span>
}

function PresetForm_({ form, setForm, title, onSave, onCancel, saving, saveLabel, savingLabel, onDelete, deleting }: {
  form: PresetForm
  setForm: (f: PresetForm) => void
  title: string
  onSave: () => void
  onCancel: (() => void) | null
  saving: boolean
  saveLabel: string
  savingLabel: string
  onDelete?: () => void
  deleting?: boolean
}) {
  const { t } = useTranslation()
  const uid = useId()
  const valid = form.name.trim() && form.template.trim()

  return (
    <div className="split-body">
      <div className="edit-form">
        <div className="edit-form-head">
          <h3 className="edit-form-title">{title}</h3>
          <div className="detail-actions">
          {onDelete && (
            <button className="btn btn-sm btn-danger" onClick={onDelete} disabled={deleting}>
              {deleting ? '...' : t('presetsPage.delete')}
            </button>
          )}
          {onCancel && <button className="btn btn-sm" onClick={onCancel}>{t('presetsPage.cancel')}</button>}
            <button className="btn btn-sm btn-primary" onClick={onSave} disabled={saving || !valid}>
              {saving ? savingLabel : saveLabel}
            </button>
          </div>
        </div>

        <div className="edit-grid">
          <div className="form-group">
            <label htmlFor={`${uid}-name`}>
              {t('presetsPage.nameLabel')} <span className="req-mark">*</span>
            </label>
            <input
              id={`${uid}-name`} value={form.name}
              onChange={e => setForm({ ...form, name: e.target.value })}
              placeholder={t('presetsPage.namePlaceholder')}
            />
          </div>
          <div className="form-group">
            <label htmlFor={`${uid}-description`}>{t('presetsPage.descriptionLabel')}</label>
            <input
              id={`${uid}-description`} value={form.description}
              onChange={e => setForm({ ...form, description: e.target.value })}
              placeholder={t('presetsPage.descriptionPlaceholder')}
            />
          </div>

          <div className="form-group form-span">
            <label htmlFor={`${uid}-template`}>
              {t('presetsPage.templateLabel')} <span className="req-mark">*</span>
            </label>
            <textarea
              id={`${uid}-template`} className="tpl-input tpl-input-sm"
              value={form.template} onChange={e => setForm({ ...form, template: e.target.value })}
              placeholder={t('presetsPage.templatePlaceholder')}
            />
            {/* The placeholders as a list of forms, and not as a paragraph or a
                row of badges on the label. A preset answers to one sampling
                form, and that is what gets chosen here: its name, what it
                means, and what it works with. */}
            <div className="ph-shapes">
              {(['single', 'pair', 'range'] as const).map(shape => (
                <div key={shape} className="ph-shape">
                  <span className="ph-shape-names">
                    {SHAPE_PLACEHOLDERS[shape].map(ph => <PlaceholderBadge key={ph} label={ph} />)}
                  </span>
                  <span className="ph-shape-desc">{t(`presetsPage.shape.${shape}`)}</span>
                </div>
              ))}
            </div>
            <p className="hint-line">
              {COMMON_PLACEHOLDERS.map(ph => <PlaceholderBadge key={ph} label={ph} />)}
              {' '}{t('presetsPage.shapeCommon')}
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}
