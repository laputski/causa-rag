import { describe, it, expect, beforeEach } from 'vitest'
import { renderHook, act } from '@testing-library/react'
import { NotificationProvider, useNotifications } from '../context/NotificationContext'

// Notifications are derived from data polled every three seconds. There is one
// likely failure here: an event born from every response rather than from a
// state transition, and it surfaces not as an error but as forty identical
// messages.

const wrapper = ({ children }: { children: React.ReactNode }) => (
  <NotificationProvider>{children}</NotificationProvider>
)

describe('notifications', () => {
  beforeEach(() => localStorage.clear())

  it('a repeated event with the same id updates the entry rather than adding a second', () => {
    const { result } = renderHook(() => useNotifications(), { wrapper })

    act(() => { result.current.notify({ id: 'run:abc:done', kind: 'progress', title: '18 %', toast: true }) })
    act(() => { result.current.notify({ id: 'run:abc:done', kind: 'progress', title: '76 %', toast: true }) })
    act(() => { result.current.notify({ id: 'run:abc:done', kind: 'success', title: 'Done', toast: true }) })

    expect(result.current.notices).toHaveLength(1)
    expect(result.current.notices[0].title).toBe('Done')
  })

  it('different events live separately', () => {
    const { result } = renderHook(() => useNotifications(), { wrapper })
    act(() => {
      result.current.notify({ id: 'a', kind: 'success', title: 'Run', toast: true })
      result.current.notify({ id: 'b', kind: 'warning', title: 'Regression', toast: true })
    })
    // Completion and regression answer different questions, and somebody who has
    // seen the first should not have to read on to learn about the second.
    expect(result.current.notices).toHaveLength(2)
    expect(result.current.unread).toBe(2)
  })

  it('dismissing a toast does not remove the entry from the centre', () => {
    const { result } = renderHook(() => useNotifications(), { wrapper })
    act(() => { result.current.notify({ id: 'a', kind: 'success', title: 'Run', toast: true }) })
    act(() => { result.current.dismiss('a') })

    expect(result.current.notices).toHaveLength(1)
    expect(result.current.notices[0].toast).toBe(false)
    expect(result.current.unread).toBe(0)
  })

  it('progress does not survive a reload', () => {
    const first = renderHook(() => useNotifications(), { wrapper })
    act(() => {
      first.result.current.notify({ id: 'p', kind: 'progress', title: '40 %', toast: true })
      first.result.current.notify({ id: 'd', kind: 'success', title: 'Done', toast: true })
    })
    first.unmount()

    // A bar frozen at forty percent after a reload is a lie: the process that
    // moved it has by then either finished or has nobody left to report to.
    const second = renderHook(() => useNotifications(), { wrapper })
    expect(second.result.current.notices.map(n => n.id)).toEqual(['d'])
  })
})
