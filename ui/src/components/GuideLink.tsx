import { Link } from 'react-router-dom'
import { BookOpen } from 'lucide-react'
import { useTranslation } from 'react-i18next'
import { useRealmPath } from '../context/RealmContext'

/**
 * A link from a screen into the guide section about that screen.
 *
 * The guide answers "what is this and what is it for"; a screen answers "what
 * is happening now". Readers arrive at a screen and not at the guide, and the
 * only path between them was to remember the guide exists, open it, and find
 * the right one of twenty-four sections.
 *
 * `section` is an identifier from the guide page's READING_ORDER.
 * GuideLinks.test.tsx keeps the two in agreement: given an unfamiliar name the
 * guide silently opens on its first section.
 */
export default function GuideLink({ section }: { section: string }) {
  const { t } = useTranslation()
  const toRealm = useRealmPath()
  return (
    <Link className="btn btn-sm" to={toRealm(`/guide?section=${section}`)} title={t('guideLink.title')}>
      <BookOpen size={13} aria-hidden="true" /> {t('guideLink.label')}
    </Link>
  )
}
