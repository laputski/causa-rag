import { useTranslation } from 'react-i18next'
import { X } from 'lucide-react'
import { type Command } from '../hooks/useCommands'
import { formatCombo } from '../hooks/useHotkeys'
import { useEscape } from '../hooks/useEscape'

// The shortcut list behind "?". Built from the same command registry as the
// palette, so it cannot disagree with it: a shortcut written in two places
// diverges on the second edit.
//
// The rows not in the registry are list behaviour rather than commands (j/k,
// Enter, Esc). They are listed separately because they belong to whatever is
// selected rather than to the screen.

const LIST_KEYS: [string, string][] = [
  ['j k', 'shortcuts.listMove'],
  ['enter', 'shortcuts.listOpen'],
  ['escape', 'shortcuts.close'],
]

export function ShortcutsDialog({ commands, open, onClose }: {
  commands: Command[]
  open: boolean
  onClose: () => void
}) {
  const { t } = useTranslation()
  useEscape(open, onClose)
  if (!open) return null

  const withCombo = commands.filter(c => c.combo)
  const chords = withCombo.filter(c => c.combo!.includes(' '))
  const singles = withCombo.filter(c => !c.combo!.includes(' '))

  return (
    <div className="overlay" onMouseDown={onClose} role="presentation">
      <div
        className="sheet" role="dialog" aria-modal="true" aria-label={t('shortcuts.title')}
        onMouseDown={e => e.stopPropagation()}
      >
        <div className="sheet-head">
          <h2 className="sheet-title">{t('shortcuts.title')}</h2>
          <button type="button" className="icon-btn" onClick={onClose} aria-label={t('shortcuts.close')}>
            <X size={15} />
          </button>
        </div>

        <div className="sheet-cols">
          <div>
            <div className="eyebrow sheet-group">{t('shortcuts.everywhere')}</div>
            {singles.map(c => (
              <div key={c.id} className="sheet-row">
                <span>{c.label}</span>
                <kbd className="kbd">{formatCombo(c.combo!)}</kbd>
              </div>
            ))}
            <div className="eyebrow sheet-group">{t('shortcuts.inLists')}</div>
            {LIST_KEYS.map(([combo, key]) => (
              <div key={combo} className="sheet-row">
                <span>{t(key)}</span>
                <kbd className="kbd">{formatCombo(combo)}</kbd>
              </div>
            ))}
          </div>
          <div>
            <div className="eyebrow sheet-group">{t('shortcuts.goTo')}</div>
            {chords.map(c => (
              <div key={c.id} className="sheet-row">
                <span>{c.label}</span>
                <kbd className="kbd">{formatCombo(c.combo!)}</kbd>
              </div>
            ))}
          </div>
        </div>

        <p className="sheet-foot">{t('shortcuts.foot')}</p>
      </div>
    </div>
  )
}
