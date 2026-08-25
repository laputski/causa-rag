import { createContext, useCallback, useContext, useEffect, useMemo, useState, ReactNode } from 'react'

// Notifications.
//
// There is no server half, and that is not a simplification. The platform has
// no accounts, so a server-side notification has no addressee: nobody to
// address it to and nowhere to record "read". Every event is derived in the
// browser from data already being polled, and the last fifty live in
// localStorage so a page reload does not erase the history.
//
// The rule about staying quiet: only something that was waited for and has
// arrived gets to interrupt. The rest accumulates in the centre.

export type NoticeKind = 'success' | 'warning' | 'error' | 'progress'

export interface Notice {
  id: string
  kind: NoticeKind
  title: string
  body?: string
  /** A link into the platform: "Open", "Compare with baseline". */
  href?: string
  hrefLabel?: string
  /** 0..1, drawn as a progress bar. Only for `progress`. */
  progress?: number
  at: number
  read: boolean
  /** Whether to surface it. Progress accumulates quietly and surfaces once it
   *  completes. */
  toast: boolean
}

const STORAGE_KEY = 'rag-platform-notices'
const KEEP = 50

interface NotificationContextValue {
  notices: Notice[]
  unread: number
  /** Returns an id, so the same event can be updated later. */
  notify: (n: Omit<Notice, 'id' | 'at' | 'read'> & { id?: string }) => string
  dismiss: (id: string) => void
  markAllRead: () => void
  clear: () => void
}

const NotificationContext = createContext<NotificationContextValue>({
  notices: [], unread: 0,
  notify: () => '', dismiss: () => {}, markAllRead: () => {}, clear: () => {},
})

function load(): Notice[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (!raw) return []
    const parsed = JSON.parse(raw)
    // Progress entries deliberately do not survive a reload: by then the
    // process behind them has either finished or has nobody left to report to,
    // and a bar frozen at 76% is a lie.
    return Array.isArray(parsed) ? parsed.filter((n: Notice) => n.kind !== 'progress') : []
  } catch {
    return []
  }
}

export function NotificationProvider({ children }: { children: ReactNode }) {
  const [notices, setNotices] = useState<Notice[]>(load)

  useEffect(() => {
    const keep = notices.filter(n => n.kind !== 'progress').slice(0, KEEP)
    try { localStorage.setItem(STORAGE_KEY, JSON.stringify(keep)) } catch { /* quota */ }
  }, [notices])

  const notify: NotificationContextValue['notify'] = useCallback(input => {
    const id = input.id ?? `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`
    setNotices(prev => {
      const existing = prev.findIndex(n => n.id === id)
      const next: Notice = { ...input, id, at: Date.now(), read: false }
      // Update the existing entry rather than adding a second: a progress bar
      // arrives dozens of times, and each frame must not be a notification.
      if (existing >= 0) {
        const copy = [...prev]
        copy[existing] = { ...copy[existing], ...next, at: copy[existing].at }
        return copy
      }
      return [next, ...prev].slice(0, KEEP + 10)
    })
    return id
  }, [])

  const dismiss = useCallback((id: string) => {
    setNotices(prev => prev.map(n => (n.id === id ? { ...n, toast: false, read: true } : n)))
  }, [])

  const markAllRead = useCallback(() => {
    setNotices(prev => prev.map(n => ({ ...n, read: true })))
  }, [])

  const clear = useCallback(() => setNotices([]), [])

  const unread = useMemo(() => notices.filter(n => !n.read).length, [notices])

  return (
    <NotificationContext.Provider value={{ notices, unread, notify, dismiss, markAllRead, clear }}>
      {children}
    </NotificationContext.Provider>
  )
}

export function useNotifications() {
  return useContext(NotificationContext)
}
