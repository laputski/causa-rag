import { useEffect, useRef } from 'react'

// The shell's keyboard shortcuts.
//
// **`event.code` is what gets listened to, never `event.key`.** The interface
// speaks several languages, and a good half of the work happens on a Cyrillic
// layout. There, `event.key` for the same physical key returns `п` instead of
// `g` and `к` instead of `r`, so any single-letter shortcut written against
// `key` silently stops working for exactly the people the interface was built
// for. `event.code` names the key rather than the letter and does not depend on
// the layout.
//
// A defect of this kind is not caught by a test written against `key`; it
// arrives as a complaint. So the tests dispatch events with a Cyrillic `key`
// and a Latin `code`.

/** The key from `event.code`: `KeyG` → `g`, `Slash` → `/`, `Backslash` → `\`. */
function codeToToken(code: string): string | null {
  if (code.startsWith('Key')) return code.slice(3).toLowerCase()
  if (code.startsWith('Digit')) return code.slice(5)
  switch (code) {
    case 'Slash': return '/'
    case 'Backslash': return '\\'
    case 'Escape': return 'escape'
    case 'Enter': return 'enter'
    case 'Period': return '.'
    case 'Comma': return ','
    default: return null
  }
}

/** A text field where single-letter shortcuts must stay silent: otherwise the
 *  `n` in a word being typed starts a run. `⌘K` still works here, because it is
 *  a way out of any state rather than an action on the field's contents. */
function isTypingTarget(el: EventTarget | null): boolean {
  if (!(el instanceof HTMLElement)) return false
  const tag = el.tagName
  return tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || el.isContentEditable
}

export interface Hotkey {
  /** `mod+k`, `mod+shift+t`, `?`, `n`, `j`, or a chord such as `g r`. */
  combo: string
  run: () => void
  /** Allow firing while a text field has focus. For `mod+…` only. */
  allowInInput?: boolean
}

/** How long the second key of a chord is waited for. A second is how long `g`,
 *  a pause and `r` take for somebody who has not memorised it yet. */
const CHORD_MS = 1000

export function useHotkeys(hotkeys: Hotkey[], enabled = true) {
  // A ref rather than a dependency: the list is rebuilt on every render (it
  // closes over navigate), and the subscription would otherwise be torn down
  // and re-established ten times a second, losing any half-typed chord.
  const ref = useRef(hotkeys)
  ref.current = hotkeys

  useEffect(() => {
    if (!enabled) return
    let pending: string | null = null
    let timer: number | undefined

    const clearChord = () => {
      pending = null
      if (timer) window.clearTimeout(timer)
    }

    const onKeyDown = (e: KeyboardEvent) => {
      const token = codeToToken(e.code)
      if (!token) {
        // Escape clears a half-typed chord even when bound to nothing itself.
        if (e.key === 'Escape') clearChord()
        return
      }

      const mod = e.metaKey || e.ctrlKey
      const typing = isTypingTarget(e.target)

      if (mod) {
        clearChord()
        const combo = `mod+${e.shiftKey ? 'shift+' : ''}${token}`
        const hit = ref.current.find(h => h.combo === combo)
        if (hit) { e.preventDefault(); hit.run() }
        return
      }

      if (e.altKey) return
      if (typing) { clearChord(); return }

      // `?` is Shift+/, and by `code` it arrives as `Slash` with shiftKey.
      const solo = e.shiftKey && token === '/' ? '?' : e.shiftKey ? null : token
      if (!solo) return

      if (pending) {
        const combo = `${pending} ${solo}`
        clearChord()
        const hit = ref.current.find(h => h.combo === combo)
        if (hit) { e.preventDefault(); hit.run() }
        return
      }

      // A key that starts any chord cannot itself be an action: otherwise `g`
      // fires before the second key is pressed.
      const startsChord = ref.current.some(h => h.combo.startsWith(`${solo} `))
      if (startsChord) {
        e.preventDefault()
        pending = solo
        timer = window.setTimeout(clearChord, CHORD_MS)
        return
      }

      const hit = ref.current.find(h => h.combo === solo)
      if (hit) { e.preventDefault(); hit.run() }
    }

    window.addEventListener('keydown', onKeyDown)
    return () => { window.removeEventListener('keydown', onKeyDown); clearChord() }
  }, [enabled])
}

/** Rendering a shortcut in a hint: `mod` is ⌘ on a Mac and Ctrl everywhere
 *  else. */
export function formatCombo(combo: string): string {
  const mac = typeof navigator !== 'undefined' && /Mac|iPhone|iPad/.test(navigator.platform)
  return combo
    .split(' ')
    .map(part =>
      part
        .split('+')
        .map(k => (k === 'mod' ? (mac ? '⌘' : 'Ctrl') : k === 'shift' ? '⇧' : k.toUpperCase()))
        .join(mac ? '' : '+'),
    )
    .join(' ')
}
