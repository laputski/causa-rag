// A seam over the two things a browser and a desktop build do differently.
//
// Exactly two: saving a file and showing a notification. In a browser those are
// a `download` link and an in-page dialog; in a Tauri or Electron build they are
// a native save dialog and a system notification.
//
// It exists now, before any desktop build, for one reason: swapping one function
// later costs less than editing pages, and every page calling
// `URL.createObjectURL` directly would have to be edited.

export interface Platform {
  kind: 'web' | 'desktop'
  /** Hands a file to the user: a download on the web, a "Save as" dialog on the
   *  desktop. */
  saveFile: (name: string, content: string, mime?: string) => Promise<void>
  /** A system notification. A no-op on the web: the in-page notification centre
   *  already shows it, and asking for a browser permission to repeat the same
   *  text buys nothing. */
  notify: (title: string, body?: string) => void
}

const web: Platform = {
  kind: 'web',
  saveFile: async (name, content, mime = 'application/json') => {
    const blob = new Blob([content], { type: mime })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = name
    document.body.appendChild(a)
    a.click()
    a.remove()
    // Revoking immediately is wrong: in some browsers the download has not
    // started yet, and a revoked URL yields an empty file.
    setTimeout(() => URL.revokeObjectURL(url), 10_000)
  },
  notify: () => {},
}

export const platform: Platform = web
