import { useEffect, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { useEscape } from '../hooks/useEscape'

// Confirmations and error messages use an in-page dialog rather than
// `confirm()`, `prompt()` and `alert()`.
//
// In a browser those look foreign, and in a desktop build they block the whole
// process: while one is open the renderer stops responding, and a running run's
// progress bar freezes. This is the kind of desktop preparation that pays for
// itself without a desktop build.
//
// A destructive action can require typing the target's name. That is not
// ceremony: it is the only thing separating a deliberate realm delete from a
// miss on the neighbouring button.

export interface ConfirmRequest {
  title: string
  body?: string
  confirmLabel?: string
  danger?: boolean
  /** When set, the confirm button unlocks only once this has been typed. */
  requireTyped?: string
  typedHint?: string
}

export function ConfirmDialog({ request, onResolve }: {
  request: ConfirmRequest | null
  onResolve: (ok: boolean) => void
}) {
  const { t } = useTranslation()
  const [typed, setTyped] = useState('')
  const inputRef = useRef<HTMLInputElement>(null)
  useEscape(Boolean(request), () => onResolve(false))

  useEffect(() => {
    setTyped('')
    if (request?.requireTyped) requestAnimationFrame(() => inputRef.current?.focus())
  }, [request])

  if (!request) return null
  const ready = !request.requireTyped || typed === request.requireTyped

  return (
    <div className="overlay" onMouseDown={() => onResolve(false)} role="presentation">
      <div
        className="sheet sheet-xs" role="alertdialog" aria-modal="true"
        aria-label={request.title} onMouseDown={e => e.stopPropagation()}
      >
        <h2 className="sheet-title mb-8">{request.title}</h2>
        {request.body && (
          <p className="text-muted sheet-body">{request.body}</p>
        )}
        {request.requireTyped && (
          <div className="form-group mt-12">
            <label htmlFor="confirm-typed">{request.typedHint}</label>
            <input
              id="confirm-typed" ref={inputRef} className="input" value={typed}
              onChange={e => setTyped(e.target.value)}
              onKeyDown={e => { if (e.key === 'Enter' && ready) onResolve(true) }}
              placeholder={request.requireTyped} autoComplete="off" spellCheck={false}
            />
          </div>
        )}
        <div className="flex-row res-actions mt-16">
          <button
            type="button" autoFocus={!request.requireTyped}
            className={`btn ${request.danger ? 'btn-danger' : 'btn-primary'}`}
            onClick={() => onResolve(true)} disabled={!ready}
          >
            {request.confirmLabel ?? t('confirm.yes')}
          </button>
          <button type="button" className="btn" onClick={() => onResolve(false)}>{t('confirm.no')}</button>
        </div>
      </div>
    </div>
  )
}

/** A promise rather than a callback, so calling code reads the way it did with
 *  `confirm()` and the switch needs no handler rewrites. */
export function useConfirm() {
  const [request, setRequest] = useState<ConfirmRequest | null>(null)
  const resolver = useRef<((ok: boolean) => void) | null>(null)

  const confirm = (req: ConfirmRequest) =>
    new Promise<boolean>(resolve => { resolver.current = resolve; setRequest(req) })

  const onResolve = (ok: boolean) => {
    setRequest(null)
    resolver.current?.(ok)
    resolver.current = null
  }

  return { confirm, dialog: <ConfirmDialog request={request} onResolve={onResolve} /> }
}
