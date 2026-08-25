import { useState } from 'react'
import { useTranslation, Trans } from 'react-i18next'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { api, type DomainPack } from '../api/client'
import { useRealm } from '../context/RealmContext'

/** What a pack brings the platform, by kind. The manifest stores the kind's
 *  name alone (`structure_parser`), and a label for it is the only thing that
 *  makes the list readable: "parser", and not a registry identifier. An
 *  unfamiliar kind is shown as it is, because a pack may export a kind this
 *  page does not know yet, and staying silent about it is worse than naming it
 *  by its raw id. */
const KIND_LABEL: Record<string, string> = {
  structure_parser: 'domainPacksPage.kinds.structureParser',
  mask_engine: 'domainPacksPage.kinds.maskEngine',
  scorer: 'domainPacksPage.kinds.scorer',
  refusal: 'domainPacksPage.kinds.refusal',
  route_policy: 'domainPacksPage.kinds.routePolicy',
  actuality_policy: 'domainPacksPage.kinds.actualityPolicy',
}

function PackTile({ pack, pending, onToggle }: {
  pack: DomainPack
  pending: boolean
  onToggle: (id: string, next: boolean) => void
}) {
  const { t } = useTranslation()
  return (
    <div className={`res-tile pack-tile${pack.active ? ' on' : ''}`}>
      <div className="pack-head">
        <strong>{pack.display_name || pack.id}</strong>
        <span className={`flag${pack.active ? ' flag-ok' : ''}`}>
          <span className="flag-dot" />
          {pack.active ? t('domainPacksPage.active') : t('domainPacksPage.inactive')}
        </span>
        {/* A real checkbox under the toggle, and not a div with an onClick: a
            screen reader announces the state, space flips it, and a test finds
            it by role. ::before/::after in styles.css draw the visible
            part. */}
        <label className="switch" title={pack.active ? t('domainPacksPage.inactive') : t('domainPacksPage.active')}>
          <input
            type="checkbox" checked={pack.active} disabled={pending}
            aria-label={pack.display_name || pack.id}
            onChange={e => onToggle(pack.id, e.target.checked)}
          />
          <i />
        </label>
      </div>
      {pack.description && <p className="pack-desc">{pack.description}</p>}
      <div className="pack-kinds">
        {pack.exported_kinds.map(kind => (
          <span key={kind} className="pack-kind">
            <span className="k">{KIND_LABEL[kind] ? t(KIND_LABEL[kind]) : kind}</span>
            <span className="v">{kind}</span>
          </span>
        ))}
      </div>
      <span className="pack-id">{pack.id}{pack.version ? ` · v${pack.version}` : ''}</span>
    </div>
  )
}

export default function DomainPacksPage() {
  const { t } = useTranslation()
  const qc = useQueryClient()
  const { activeRealmId } = useRealm()
  const { data: packs = [], isLoading } = useQuery({
    queryKey: ['domain-packs', activeRealmId],
    queryFn: () => api.domainPacks.list(activeRealmId),
  })
  const [pendingNotice, setPendingNotice] = useState(false)

  const setActiveMut = useMutation({
    mutationFn: (active_packs: string[]) => api.domainPacks.setActive(active_packs, activeRealmId),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['domain-packs'] })
      setPendingNotice(true)
    },
  })

  const handleToggle = (id: string, next: boolean) => {
    const currentActive = packs.filter(p => p.active).map(p => p.id)
    const updated = next
      ? [...currentActive, id]
      : currentActive.filter(p => p !== id)
    setActiveMut.mutate(updated)
  }

  const activeCount = packs.filter(p => p.active).length

  return (
    <div className="page">
      {/* The caption reads "one of two enabled" and not "two discovered":
          the page exists for what is enabled, and that is also the first
          question anybody brings to it. */}
      <div className="page-head">
        <h1 className="page-title">{t('domainPacksPage.title')}</h1>
        <p className="page-sub">
          {t('domainPacksPage.activeOfTotal', { active: activeCount, total: packs.length })}
        </p>
      </div>

      {pendingNotice && (
        <p className="conn-status conn-status-warn">{t('domainPacksPage.pendingNotice')}</p>
      )}

      {isLoading && <p className="text-muted">{t('domainPacksPage.loading')}</p>}
      {!isLoading && packs.length === 0 && <p className="text-muted">{t('domainPacksPage.noneFound')}</p>}
      <div className="res-grid">
        {packs.map(pack => (
          <PackTile
            key={pack.id}
            pack={pack}
            pending={setActiveMut.isPending}
            onToggle={handleToggle}
          />
        ))}
      </div>

      <p className="hint-line">{t('domainPacksPage.catalogHint')}</p>

      {/* Four paragraphs of explanation stood above the tiles and pushed the
          one thing anybody does on this page off the screen. Collapsed: they
          are read once, and packs are enabled more than once. */}
      <details className="prose-details">
        <summary>{t('domainPacksPage.whatAndWhyTitle')}</summary>
        <p className="prose-p">{t('domainPacksPage.whatAndWhyBody1')}</p>
        <p className="prose-p">
          <Trans i18nKey="domainPacksPage.whatAndWhyBody2" t={t}>
            Activating a domain for chat needs a <strong>gateway restart</strong>:
            packs load once at startup and live reloading is not supported.
            Running an experiment does not depend on activation, since the
            components can be chosen directly on the New run form regardless of
            this global switch.
          </Trans>
        </p>
        <p className="prose-p">{t('domainPacksPage.whatAndWhyBody3')}</p>
        <p className="prose-p">{t('domainPacksPage.whatAndWhyBody4')}</p>
        <p className="prose-p text-muted">{t('domainPacksPage.whatAndWhyBody5')}</p>
      </details>
    </div>
  )
}
