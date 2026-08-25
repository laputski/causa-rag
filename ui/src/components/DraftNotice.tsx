import { useTranslation } from 'react-i18next'

/**
 * The mark on a section that works but has not been checked all the way through.
 *
 * Two parts, and both are needed. The badge beside the heading is visible from
 * the same second as the heading itself and explains nothing. The line under
 * the header explains, and is read once. Apart, the first says nothing and the
 * second arrives too late.
 *
 * It neither blocks nor discourages: the section opens and works. It states
 * exactly what is known, which is that test coverage is incomplete and the
 * numbers deserve a second look.
 */
export function DraftBadge() {
  const { t } = useTranslation()
  return (
    <span className="draft-badge" title={t('draft.tooltip')}>{t('draft.short')}</span>
  )
}

export function DraftNote() {
  const { t } = useTranslation()
  return <p className="draft-note">{t('draft.note')}</p>
}
