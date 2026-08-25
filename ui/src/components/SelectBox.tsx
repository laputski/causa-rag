import { Children, isValidElement, type ReactNode } from 'react'
import { ChevronDown } from 'lucide-react'

/**
 * A select whose caret sits right after the text, and which opens on a click
 * anywhere across the field.
 *
 * A native `<select>` draws its caret at the right edge whatever the value's
 * length, so a form of ten fields produces a column of carets divorced from the
 * values they belong to. Shrinking the select to its content moves the caret,
 * and takes the click target with it: half the field's width stops working.
 *
 * So the `<select>` here is stretched across the whole wrapper and made
 * transparent: it is the click target, the focus target and the keyboard
 * target. The visible value and caret are drawn as a separate line beneath it,
 * where the caret can sit tight against the text.
 *
 * The control stays a native `<select>`: a screen reader reads that rather than
 * the painted label, and the option list opens as the system's own.
 */
export default function SelectBox({ id, value, onChange, disabled, children, title, 'aria-label': ariaLabel }: {
  id?: string
  value: string
  onChange: (e: React.ChangeEvent<HTMLSelectElement>) => void
  disabled?: boolean
  children: ReactNode
  title?: string
  'aria-label'?: string
}) {
  return (
    <span className={`sel${disabled ? ' disabled' : ''}`} title={title}>
      <select
        id={id} value={value} onChange={onChange} disabled={disabled}
        aria-label={ariaLabel} className="sel-native"
      >
        {children}
      </select>
      {/* Kept out of the accessibility tree: this label duplicates what the
          screen reader has already read from the select itself. */}
      <span className="sel-face" aria-hidden="true">
        <span className="sel-text">{labelOf(children, value)}</span>
        <ChevronDown size={12} className="sel-caret" />
      </span>
    </span>
  )
}

/** The selected `<option>`'s children as they are, without coercing to a
 *  string: options are sometimes assembled from several pieces and numbers.
 *
 *  With no match — the value has not arrived yet, or belongs to something else
 *  — the first option is used, because that is what the `<select>` itself shows
 *  in that case, and a painted line disagreeing with the open list would be
 *  worse than an empty one. */
function labelOf(children: ReactNode, value: string): ReactNode {
  let found: ReactNode = null
  let first: ReactNode = null
  const walk = (nodes: ReactNode) => {
    Children.forEach(nodes, node => {
      if (found !== null || !isValidElement(node)) return
      const props = node.props as { value?: string | number; children?: ReactNode }
      // Option groups (`<optgroup>`) are walked through: they carry no `value`.
      if (node.type === 'optgroup') { walk(props.children); return }
      if (first === null) first = props.children ?? ''
      if (String(props.value ?? '') === value) found = props.children ?? ''
    })
  }
  walk(children)
  return found ?? first
}
