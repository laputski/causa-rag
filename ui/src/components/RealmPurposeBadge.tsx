import { useTranslation } from 'react-i18next'

/**
 * The mark on a realm whose data carries defects on purpose.
 *
 * Without it, the red diagnostics of the proving ground read as a broken
 * installation, and the installation check would be reporting on a realm built
 * to fail. Warning-coloured and not danger-coloured: nothing there is wrong,
 * and something there is deliberate.
 *
 * One component and not three copies of a condition, so that the realm
 * switcher, its menu and the overview cannot come to disagree about which
 * realms are marked or about what the mark says.
 */
export const PROVING_GROUND = 'proving_ground'

export function RealmPurposeBadge({ purpose }: { purpose?: string }) {
  const { t } = useTranslation()
  if (purpose !== PROVING_GROUND) return null
  return (
    <span className="badge badge-warn realm-purpose" title={t('realm.provingGround.hint')}>
      {t('realm.provingGround.label')}
    </span>
  )
}
