import { useCallback, useEffect, useRef, useState } from 'react'

// Column widths, dragged with the mouse.
//
// Stored per realm: one corpus produces longer run names than another, and one
// width across both is a width that suits neither.
//
// Pointer events rather than mouse events: a pen and a trackpad send the same
// thing, and `setPointerCapture` carries the gesture through if the cursor
// leaves the window.

const MIN_WIDTH = 60

export function useColumnWidths(storageKey: string, defaults: Record<string, number>) {
  const [widths, setWidths] = useState<Record<string, number>>(() => {
    try {
      const raw = localStorage.getItem(storageKey)
      return raw ? { ...defaults, ...JSON.parse(raw) } : defaults
    } catch {
      return defaults
    }
  })

  // The set of columns changes when somebody edits the key-metric list. A new
  // column must get a default width rather than none, or it collapses to zero
  // and looks like broken layout.
  useEffect(() => {
    setWidths(prev => {
      const missing = Object.keys(defaults).filter(k => !(k in prev))
      if (missing.length === 0) return prev
      return { ...prev, ...Object.fromEntries(missing.map(k => [k, defaults[k]])) }
    })
  }, [Object.keys(defaults).join('|')])  // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    try { localStorage.setItem(storageKey, JSON.stringify(widths)) } catch { /* quota */ }
  }, [storageKey, widths])

  const drag = useRef<{ key: string; startX: number; startW: number } | null>(null)

  const onPointerDown = useCallback((key: string) => (e: React.PointerEvent) => {
    e.preventDefault()
    e.stopPropagation()
    drag.current = { key, startX: e.clientX, startW: widths[key] ?? defaults[key] ?? 120 }
    ;(e.target as HTMLElement).setPointerCapture(e.pointerId)
    document.body.style.cursor = 'col-resize'
    document.body.style.userSelect = 'none'
  }, [widths, defaults])

  const onPointerMove = useCallback((e: React.PointerEvent) => {
    const d = drag.current
    if (!d) return
    const next = Math.max(MIN_WIDTH, d.startW + (e.clientX - d.startX))
    setWidths(prev => ({ ...prev, [d.key]: next }))
  }, [])

  const stop = useCallback(() => {
    drag.current = null
    document.body.style.cursor = ''
    document.body.style.userSelect = ''
  }, [])

  const reset = useCallback(() => setWidths(defaults), [defaults])

  return { widths, onPointerDown, onPointerMove, stop, reset }
}
