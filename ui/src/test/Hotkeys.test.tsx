import { describe, it, expect, vi, beforeEach } from 'vitest'
import { renderHook, act } from '@testing-library/react'
import { useHotkeys, formatCombo } from '../hooks/useHotkeys'

// The keyboard is the one part of the shell that breaks silently: a shortcut
// written against `event.key` works for whoever wrote it and fails for whoever
// the interface was built for.

/** A key pressed on a CYRILLIC layout: `code` names the physical key and does
 *  not change, while `key` arrives as a Cyrillic letter. This is exactly what
 *  the events look like for half of this platform's users. */
function pressCyrillic(code: string, key: string, opts: Partial<KeyboardEventInit> = {}) {
  act(() => {
    window.dispatchEvent(new KeyboardEvent('keydown', {
      code, key, bubbles: true, cancelable: true, ...opts,
    }))
  })
}

describe('keyboard shortcuts', () => {
  beforeEach(() => { document.body.innerHTML = '' })

  it('the "G then R" chord fires on a Cyrillic layout', () => {
    const run = vi.fn()
    renderHook(() => useHotkeys([{ combo: 'g r', run }]))

    // On a Russian layout these keys produce "п" and "к". An implementation
    // reading `event.key` would see nothing here.
    pressCyrillic('KeyG', 'п')
    pressCyrillic('KeyR', 'к')

    expect(run).toHaveBeenCalledTimes(1)
  })

  it('the first key of a chord starts nothing on its own', () => {
    const chord = vi.fn()
    const solo = vi.fn()
    renderHook(() => useHotkeys([{ combo: 'g r', run: chord }, { combo: 'g', run: solo }]))

    pressCyrillic('KeyG', 'п')

    // Otherwise "g" fires before the second key can be pressed.
    expect(solo).not.toHaveBeenCalled()
    expect(chord).not.toHaveBeenCalled()
  })

  it('a single-letter shortcut stays silent while a text field has focus', () => {
    const run = vi.fn()
    const input = document.createElement('input')
    document.body.appendChild(input)
    input.focus()
    renderHook(() => useHotkeys([{ combo: 'n', run }]))

    act(() => {
      input.dispatchEvent(new KeyboardEvent('keydown', {
        code: 'KeyN', key: 'т', bubbles: true, cancelable: true,
      }))
    })

    // The "н" in a word being typed must not start a run.
    expect(run).not.toHaveBeenCalled()
  })

  it('a modifier shortcut works inside a text field too', () => {
    const run = vi.fn()
    const input = document.createElement('input')
    document.body.appendChild(input)
    input.focus()
    renderHook(() => useHotkeys([{ combo: 'mod+k', run }]))

    act(() => {
      input.dispatchEvent(new KeyboardEvent('keydown', {
        code: 'KeyK', key: 'л', metaKey: true, bubbles: true, cancelable: true,
      }))
    })

    // ⌘K is a way out of any state rather than an action on the field's contents.
    expect(run).toHaveBeenCalledTimes(1)
  })

  it('"?" is recognised as Shift plus Slash', () => {
    const run = vi.fn()
    renderHook(() => useHotkeys([{ combo: '?', run }]))
    pressCyrillic('Slash', '?', { shiftKey: true })
    expect(run).toHaveBeenCalledTimes(1)
  })

  it('disabled bindings do not fire', () => {
    const run = vi.fn()
    renderHook(() => useHotkeys([{ combo: 'n', run }], false))
    pressCyrillic('KeyN', 'т')
    expect(run).not.toHaveBeenCalled()
  })

  it('a shortcut label reads as something a person can say', () => {
    expect(formatCombo('g r')).toBe('G R')
    expect(formatCombo('?')).toBe('?')
  })
})
