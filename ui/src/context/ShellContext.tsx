import { createContext, useContext, useState, useEffect, useCallback, ReactNode } from 'react'

// Shell state: the sidebar's width and whether it is collapsed.
//
// Its own context rather than state inside `Sidebar`, because three things read
// it: `Sidebar` itself (drawing itself), `App` (setting `--sidebar-w` on the
// root) and the command palette (the "Collapse menu" command). State edited
// from three places lives above all three.

const WIDTH_KEY = 'rag-platform-sidebar-width'
const COLLAPSED_KEY = 'rag-platform-sidebar-collapsed'

/** The drag bounds. Below 180 an item's label breaks mid-word; above 420 the
 *  menu starts competing with the content for attention. */
export const SIDEBAR_MIN = 180
export const SIDEBAR_MAX = 420
export const SIDEBAR_DEFAULT = 236
/** The collapsed width: a 15px icon plus enough margin for a finger. */
export const SIDEBAR_COLLAPSED = 52

interface ShellContextValue {
  width: number
  collapsed: boolean
  setWidth: (w: number) => void
  toggleCollapsed: () => void
  setCollapsed: (v: boolean) => void
}

function initialWidth(): number {
  const stored = Number(localStorage.getItem(WIDTH_KEY))
  if (!Number.isFinite(stored) || stored <= 0) return SIDEBAR_DEFAULT
  return Math.min(SIDEBAR_MAX, Math.max(SIDEBAR_MIN, stored))
}

const ShellContext = createContext<ShellContextValue>({
  width: SIDEBAR_DEFAULT,
  collapsed: false,
  setWidth: () => {},
  toggleCollapsed: () => {},
  setCollapsed: () => {},
})

export function ShellProvider({ children }: { children: ReactNode }) {
  const [width, setWidthState] = useState<number>(initialWidth)
  const [collapsed, setCollapsedState] = useState<boolean>(
    () => localStorage.getItem(COLLAPSED_KEY) === '1',
  )

  // The width is clamped here rather than in the drag handler: the localStorage
  // field gets edited by hand and survives a change to the bounds in code, so
  // the only reliable place to clamp is the one every assignment passes
  // through.
  const setWidth = useCallback((w: number) => {
    setWidthState(Math.min(SIDEBAR_MAX, Math.max(SIDEBAR_MIN, Math.round(w))))
  }, [])

  useEffect(() => { localStorage.setItem(WIDTH_KEY, String(width)) }, [width])
  useEffect(() => { localStorage.setItem(COLLAPSED_KEY, collapsed ? '1' : '0') }, [collapsed])

  // The variable is set on the root rather than on `.sidebar` itself: both the
  // menu's width and the content's offset depend on it, and two elements cannot
  // read a variable declared on one of them.
  useEffect(() => {
    document.documentElement.style.setProperty(
      '--sidebar-w', `${collapsed ? SIDEBAR_COLLAPSED : width}px`,
    )
  }, [width, collapsed])

  const toggleCollapsed = useCallback(() => setCollapsedState(c => !c), [])

  return (
    <ShellContext.Provider
      value={{ width, collapsed, setWidth, toggleCollapsed, setCollapsed: setCollapsedState }}
    >
      {children}
    </ShellContext.Provider>
  )
}

export function useShell() {
  return useContext(ShellContext)
}
