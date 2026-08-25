import { describe, it, expect, beforeEach } from 'vitest'
import { renderHook, act } from '@testing-library/react'
import {
  ShellProvider, useShell, SIDEBAR_MIN, SIDEBAR_MAX, SIDEBAR_COLLAPSED,
} from '../context/ShellContext'

// The menu's width and its collapsed state survive a reload. A setting you have
// to set again on every visit is not a setting.

const wrapper = ({ children }: { children: React.ReactNode }) => (
  <ShellProvider>{children}</ShellProvider>
)

describe('the shell', () => {
  beforeEach(() => localStorage.clear())

  it('the width is clamped to its bounds on assignment', () => {
    const { result } = renderHook(() => useShell(), { wrapper })
    act(() => result.current.setWidth(20))
    expect(result.current.width).toBe(SIDEBAR_MIN)
    act(() => result.current.setWidth(9000))
    expect(result.current.width).toBe(SIDEBAR_MAX)
  })

  it('a width corrupted in localStorage does not break the menu', () => {
    // The field gets edited by hand and survives a change to the bounds in code,
    // so the only reliable place to clamp is the one every assignment passes
    // through, including the read at startup.
    localStorage.setItem('rag-platform-sidebar-width', '99999')
    const { result } = renderHook(() => useShell(), { wrapper })
    expect(result.current.width).toBe(SIDEBAR_MAX)
  })

  it('the collapsed state survives a reload', () => {
    const first = renderHook(() => useShell(), { wrapper })
    act(() => first.result.current.toggleCollapsed())
    expect(first.result.current.collapsed).toBe(true)
    first.unmount()

    const second = renderHook(() => useShell(), { wrapper })
    expect(second.result.current.collapsed).toBe(true)
  })

  it('a collapsed menu sets the width variable on the root', () => {
    const { result } = renderHook(() => useShell(), { wrapper })
    act(() => result.current.toggleCollapsed())
    // The variable sits on the root rather than on the menu: two things depend
    // on it, the menu and the content, and two elements cannot read a variable
    // declared on one of them.
    expect(document.documentElement.style.getPropertyValue('--sidebar-w'))
      .toBe(`${SIDEBAR_COLLAPSED}px`)
  })
})
