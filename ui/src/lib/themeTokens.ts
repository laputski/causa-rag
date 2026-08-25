/**
 * Colour for the places where it has to be a string and not a CSS property.
 *
 * Recharts puts a colour into an SVG attribute and sigma puts one into a WebGL
 * buffer; `var(--color-primary)` resolves in neither. So the value has to be
 * read off the document in advance, and letting every drawing component read it
 * its own way means keeping as many palettes as there are components.
 *
 * This is the one interface file where a colour is written as a hex literal,
 * and it is written exactly once: as the fallback for when there is no document
 * at all (tests, server rendering), which is to say when there is nothing to
 * draw anyway. Held by designLanguage.test.ts.
 */

/** The dark theme's values from styles.css, unreachable in a browser. */
const FALLBACK: Record<string, string> = {
  '--color-bg': '#0e1013',
  '--color-surface': '#15181d',
  '--color-border': '#262a31',
  '--color-text': '#e8eaed',
  '--color-text-muted': '#9ba1ac',
  '--color-primary': '#6fa8dc',
  '--color-primary-h': '#93c1e8',
  '--color-success': '#3dbf97',
  '--color-warning': '#f0b33c',
  '--color-danger': '#f07b7b',
  '--diag-green': '#3dbf97',
  '--diag-cyan': '#7fc8f0',
  '--diag-yellow': '#e3c14b',
  '--diag-orange': '#f07b2e',
  '--diag-line': '#3b4149',
  '--fs-2xs': '11px',
}

/** A token's current value, read off the document root. */
export function token(name: string): string {
  if (typeof document === 'undefined') return FALLBACK[name] ?? ''
  const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim()
  return v || FALLBACK[name] || ''
}

/** Several tokens at once: one style read instead of N. */
export function tokens<K extends string>(names: Record<K, string>): Record<K, string> {
  const cs = typeof document === 'undefined' ? null : getComputedStyle(document.documentElement)
  const out = {} as Record<K, string>
  for (const key of Object.keys(names) as K[]) {
    const name = names[key]
    const v = cs?.getPropertyValue(name).trim()
    out[key] = v || FALLBACK[name] || ''
  }
  return out
}
