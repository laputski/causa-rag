/** Copies text to the clipboard, falling back to the legacy execCommand
 * path when the async Clipboard API is unavailable or denied (e.g. no
 * secure context, or an embedding iframe without clipboard-write in its
 * Permissions-Policy) — `navigator.clipboard.writeText` alone silently
 * leaves the user with nothing copied in those cases. Returns whether the
 * copy actually succeeded so callers can show an honest error state. */
export async function copyToClipboard(text: string): Promise<boolean> {
  if (navigator.clipboard?.writeText) {
    try {
      await navigator.clipboard.writeText(text)
      return true
    } catch {
      // fall through to the legacy path below
    }
  }
  try {
    const ta = document.createElement('textarea')
    ta.value = text
    ta.style.position = 'fixed'
    ta.style.opacity = '0'
    document.body.appendChild(ta)
    ta.focus()
    ta.select()
    const ok = document.execCommand('copy')
    document.body.removeChild(ta)
    return ok
  } catch {
    return false
  }
}
