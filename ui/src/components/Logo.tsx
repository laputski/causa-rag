/* The mark.
 *
 * Three variants were designed and one is in use. They are all here because the
 * choice between them is a matter of taste that ought to be inspectable rather
 * than argued from description, and because a mark that only exists as one
 * committed file cannot be compared with the alternatives it beat.
 *
 * Every variant obeys the same three constraints, which is what makes them
 * comparable at all:
 *   — legible at 16 px, where the sidebar actually renders it;
 *   — one colour, taken from `currentColor`, so it follows the palette;
 *   — no text inside the mark, so the wordmark can be set in any typeface.
 */

export type LogoVariant = 'cascade' | 'notch' | 'fan' | 'aperture'

/** The variant in use. Named rather than inlined so switching it is one edit. */
export const LOGO_VARIANT: LogoVariant = 'cascade'

/* ── cascade — three bars narrowing downward ──
 * The chosen mark. A funnel drawn as three steps: everything that was
 * retrieved, what survived reranking, what reached the answer. It is the
 * same shape the run page draws as `.funnel`, so the mark states what the
 * platform measures rather than alluding to it. The narrowing, not the
 * colour, carries the meaning — it survives being one colour at 16 px.
 */
function Cascade({ size }: { size: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <rect x="2.5" y="4"  width="19" height="4" rx="1.4" fill="currentColor" />
      <rect x="5.5" y="10" width="13" height="4" rx="1.4" fill="currentColor" opacity=".62" />
      <rect x="9"   y="16" width="6"  height="4" rx="1.4" fill="currentColor" opacity=".34" />
    </svg>
  )
}

/* ── notch — a disc with a wedge missing ──
 * The cause is the piece that is not there. A full disc says "everything is
 * accounted for"; the missing wedge is what the platform exists to name. Reads
 * as a single silhouette at 16 px, which the other two only just manage. */
function Notch({ size }: { size: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" aria-hidden="true">
      {/* Disc with a 60° wedge cut from the upper right, drawn as one path so
          the counter stays crisp when the icon is scaled down. */}
      <path
        d="M12 2a10 10 0 1 0 0 20 10 10 0 0 0 10-10h-10z"
        fill="currentColor"
      />
      <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="1.6" opacity="0.45" />
      <circle cx="12" cy="12" r="2.4" fill="currentColor" opacity="0.35" />
    </svg>
  )
}

/* ── fan — one point, three diverging rays ──
 * The shape of the root-cause diagram: one verdict splits into several causes.
 * The most literal of the three about what the platform does. */
function Fan({ size }: { size: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <circle cx="4.5" cy="12" r="2.6" fill="currentColor" />
      <path d="M7.4 11.2 19 4.6M7.4 12H19M7.4 12.8 19 19.4"
            stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" />
      <circle cx="20.4" cy="4" r="1.5" fill="currentColor" opacity="0.55" />
      <circle cx="20.4" cy="12" r="1.5" fill="currentColor" />
      <circle cx="20.4" cy="20" r="1.5" fill="currentColor" opacity="0.55" />
    </svg>
  )
}

/* ── aperture — a C that closes on a point ──
 * A monogram for Causa that also reads as a lens stopping down on one spot.
 * The most abstract, and the most typographic of the three. */
function Aperture({ size }: { size: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <path
        d="M19.5 6.2A9 9 0 1 0 19.5 17.8"
        stroke="currentColor" strokeWidth="2.4" strokeLinecap="round"
      />
      <circle cx="12" cy="12" r="2.8" fill="currentColor" />
    </svg>
  )
}

const VARIANTS: Record<LogoVariant, (p: { size: number }) => JSX.Element> = {
  cascade: Cascade, notch: Notch, fan: Fan, aperture: Aperture,
}

export function LogoMark({ size = 18, variant = LOGO_VARIANT }: {
  size?: number; variant?: LogoVariant
}) {
  const Mark = VARIANTS[variant]
  return <Mark size={size} />
}

/** Mark plus wordmark, as the sidebar header uses it. */
export function Logo({ variant = LOGO_VARIANT }: { variant?: LogoVariant }) {
  return (
    <span className="brand">
      <span className="brand-mark"><LogoMark size={20} variant={variant} /></span>
      <span className="brand-word">
        Causa<span className="brand-word-accent">RAG</span>
      </span>
    </span>
  )
}
