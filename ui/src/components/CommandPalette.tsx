import { useEffect, useMemo, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { Search } from 'lucide-react'
import { type Command, type CommandGroup, filterCommands } from '../hooks/useCommands'
import { formatCombo } from '../hooks/useHotkeys'
import { useEscape } from '../hooks/useEscape'

// The ⌘K command palette. Written here rather than taken as a dependency:
// thirty rows with substring filtering is about a hundred and fifty lines,
// whereas `cmdk` brings its own focus model and its own styles to override.
//
// It doubles as the main way shortcuts are learned: each row carries its
// shortcut on the right, so people pick them up while working rather than from
// a separate list.

const GROUP_ORDER: CommandGroup[] = ['nav', 'action', 'view']

export function CommandPalette({ commands, open, onClose }: {
  commands: Command[]
  open: boolean
  onClose: () => void
}) {
  const { t } = useTranslation()
  const [query, setQuery] = useState('')
  const [cursor, setCursor] = useState(0)
  const inputRef = useRef<HTMLInputElement>(null)
  const listRef = useRef<HTMLDivElement>(null)

  useEscape(open, onClose)

  const matched = useMemo(() => filterCommands(commands, query), [commands, query])

  // The groups are built from the filtered list rather than the other way
  // round: otherwise an empty group would remain as a heading over nothing.
  const grouped = useMemo(() => {
    const by = new Map<CommandGroup, Command[]>()
    for (const c of matched) {
      const arr = by.get(c.group)
      if (arr) arr.push(c); else by.set(c.group, [c])
    }
    return GROUP_ORDER.filter(g => by.has(g)).map(g => ({ group: g, items: by.get(g)! }))
  }, [matched])

  const flat = useMemo(() => grouped.flatMap(g => g.items), [grouped])

  useEffect(() => {
    if (!open) return
    setQuery('')
    setCursor(0)
    // Focus the field after the render: before it the element is not in the
    // document yet.
    const id = requestAnimationFrame(() => inputRef.current?.focus())
    return () => cancelAnimationFrame(id)
  }, [open])

  useEffect(() => { setCursor(0) }, [query])

  // The highlighted row is scrolled into view: without this, arrowing down to
  // the thirtieth command moves the highlight past the bottom edge and the list
  // never follows.
  useEffect(() => {
    listRef.current?.querySelector('[data-active="true"]')?.scrollIntoView({ block: 'nearest' })
  }, [cursor])

  if (!open) return null

  const runAt = (i: number) => {
    const cmd = flat[i]
    if (!cmd) return
    onClose()
    cmd.run()
  }

  const onKeyDown = (e: React.KeyboardEvent) => {
    // Handled here rather than by a global binding: while the palette is open,
    // the arrows and Enter belong to it rather than to the page beneath.
    if (e.key === 'ArrowDown') { e.preventDefault(); setCursor(c => Math.min(flat.length - 1, c + 1)) }
    else if (e.key === 'ArrowUp') { e.preventDefault(); setCursor(c => Math.max(0, c - 1)) }
    else if (e.key === 'Enter') { e.preventDefault(); runAt(cursor) }
    else if (e.key === 'Escape') { e.preventDefault(); onClose() }
  }

  return (
    <div className="overlay" onMouseDown={onClose} role="presentation">
      <div
        className="cmdk" role="dialog" aria-modal="true" aria-label={t('commands.title')}
        onMouseDown={e => e.stopPropagation()} onKeyDown={onKeyDown}
      >
        <div className="cmdk-field">
          <Search size={15} aria-hidden="true" />
          <input
            ref={inputRef} className="cmdk-input" value={query}
            onChange={e => setQuery(e.target.value)}
            placeholder={t('commands.placeholder')}
            aria-label={t('commands.placeholder')}
            autoComplete="off" spellCheck={false}
          />
          <kbd className="kbd">esc</kbd>
        </div>

        <div className="cmdk-list" ref={listRef} role="listbox">
          {flat.length === 0 && <div className="cmdk-empty">{t('commands.nothing', { query })}</div>}
          {grouped.map(({ group, items }) => (
            <div key={group}>
              <div className="cmdk-group">{t(`commands.groups.${group}`)}</div>
              {items.map(cmd => {
                const i = flat.indexOf(cmd)
                const Icon = cmd.icon
                return (
                  <button
                    key={cmd.id} type="button" role="option" aria-selected={i === cursor}
                    data-active={i === cursor}
                    className={`cmdk-row${i === cursor ? ' active' : ''}`}
                    onMouseMove={() => setCursor(i)}
                    onClick={() => runAt(i)}
                  >
                    <Icon size={14} className="cmdk-icon" aria-hidden="true" />
                    <span className="cmdk-label">{cmd.label}</span>
                    {cmd.combo && <kbd className="kbd">{formatCombo(cmd.combo)}</kbd>}
                  </button>
                )
              })}
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
