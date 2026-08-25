import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { useTranslation } from 'react-i18next'
import { Bell, Check, CircleCheck, CircleX, TriangleAlert, Database, X } from 'lucide-react'
import { useNotifications, type Notice, type NoticeKind } from '../context/NotificationContext'

const ICONS: Record<NoticeKind, React.ElementType> = {
  success: CircleCheck, warning: TriangleAlert, error: CircleX, progress: Database,
}

/** Auto-dismiss. Errors never fade on their own: a message nobody managed to
 *  read is indistinguishable from one that never appeared. */
const TOAST_MS = 8000

function relativeTime(at: number, t: ReturnType<typeof useTranslation>['t']): string {
  const s = Math.round((Date.now() - at) / 1000)
  if (s < 60) return t('notifications.justNow')
  if (s < 3600) return t('notifications.minutesAgo', { n: Math.round(s / 60) })
  return new Date(at).toLocaleTimeString('ru-RU', { hour: '2-digit', minute: '2-digit' })
}

function Toast({ notice }: { notice: Notice }) {
  const { dismiss } = useNotifications()
  const { t } = useTranslation()
  const Icon = ICONS[notice.kind]

  useEffect(() => {
    if (notice.kind === 'error' || notice.kind === 'progress') return
    const id = window.setTimeout(() => dismiss(notice.id), TOAST_MS)
    return () => window.clearTimeout(id)
  }, [notice.id, notice.kind, dismiss])

  return (
    <div className={`toast toast-${notice.kind}`} role="status">
      <div className="toast-head">
        <Icon size={14} className="toast-icon" aria-hidden="true" />
        <span className="toast-title">{notice.title}</span>
        <button type="button" className="icon-btn" onClick={() => dismiss(notice.id)}
                aria-label={t('notifications.dismiss')}>
          <X size={12} />
        </button>
      </div>
      {notice.body && <p className="toast-body">{notice.body}</p>}
      {notice.progress != null && (
        <div className="progress-bar"><div className="progress-fill" style={{ width: `${Math.round(notice.progress * 100)}%` }} /></div>
      )}
      {notice.href && (
        <Link to={notice.href} className="toast-action" onClick={() => dismiss(notice.id)}>
          {notice.hrefLabel ?? t('notifications.open')}
        </Link>
      )}
    </div>
  )
}

/** The toast stack. One node for the whole shell rather than a `position:
 *  fixed` inside a page, which would give every page its own screen corner. */
export function ToastStack() {
  const { notices } = useNotifications()
  const shown = notices.filter(n => n.toast).slice(0, 3)
  if (shown.length === 0) return null
  return (
    <div className="toast-stack" aria-live="polite">
      {shown.map(n => <Toast key={n.id} notice={n} />)}
    </div>
  )
}

export function NotificationBell() {
  const { notices, unread, markAllRead, clear } = useNotifications()
  const { t } = useTranslation()
  const [open, setOpen] = useState(false)

  return (
    <div className="lang-switcher">
      <button
        className="icon-btn bell" onClick={() => { setOpen(o => !o); if (!open) markAllRead() }}
        title={t('notifications.title')} aria-label={t('notifications.title')} aria-expanded={open}
      >
        <Bell size={15} />
        {unread > 0 && <span className="bell-badge">{unread > 9 ? '9+' : unread}</span>}
      </button>
      {open && (
        <>
          <div className="realm-switcher-backdrop" onClick={() => setOpen(false)} />
          <div className="notice-panel">
            <div className="notice-head">
              <span className="notice-head-title">{t('notifications.title')}</span>
              {notices.length > 0 && (
                <button type="button" className="link-btn" onClick={clear}>{t('notifications.clear')}</button>
              )}
            </div>
            {notices.length === 0 ? (
              <p className="notice-empty">{t('notifications.empty')}</p>
            ) : (
              <div className="notice-list">
                {notices.slice(0, 20).map(n => {
                  const Icon = ICONS[n.kind]
                  const row = (
                    <>
                      <Icon size={13} className={`notice-icon notice-${n.kind}`} aria-hidden="true" />
                      <span className="notice-body">
                        <span className="notice-title">{n.title}</span>
                        {n.body && <span className="notice-sub">{n.body}</span>}
                      </span>
                      <span className="notice-time">{relativeTime(n.at, t)}</span>
                    </>
                  )
                  return n.href ? (
                    <Link key={n.id} to={n.href} className="notice-row" onClick={() => setOpen(false)}>{row}</Link>
                  ) : (
                    <div key={n.id} className="notice-row">{row}</div>
                  )
                })}
              </div>
            )}
            <div className="notice-foot">
              <Check size={12} aria-hidden="true" />{t('notifications.foot')}
            </div>
          </div>
        </>
      )}
    </div>
  )
}
