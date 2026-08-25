import { useEffect } from 'react'

/** Escape closes an overlay.
 *
 *  The window is listened to rather than the element. A handler on the node
 *  requires focus to be inside it: the command palette has that (focus sits in
 *  its search field), but the shortcut list has no focusable content at all, so
 *  that dialog opened and could be closed by nothing but a click outside. Found
 *  live. */
export function useEscape(active: boolean, onEscape: () => void) {
  useEffect(() => {
    if (!active) return
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key !== 'Escape') return
      e.preventDefault()
      e.stopPropagation()
      onEscape()
    }
    // Captured on the capture phase: otherwise Escape reaches the page beneath
    // the overlay first, and it collapses its own expanded block.
    window.addEventListener('keydown', onKeyDown, true)
    return () => window.removeEventListener('keydown', onKeyDown, true)
  }, [active, onEscape])
}
