import { useState } from 'react'
import { useTranslation } from 'react-i18next'
import { Sun, Moon, Check, Globe, Palette as PaletteIcon } from 'lucide-react'
import { useTheme } from '../context/ThemeContext'
import { usePalette, PALETTES } from '../context/PaletteContext'
import { SUPPORTED_LANGUAGES } from '../i18n'

// The appearance and language controls belong to the shell rather than to a
// page.
//
// Lifted out of App.tsx for the same reason the command palette and the
// notifications live there: App.tsx assembles the shell rather than drawing its
// details. A side effect is that they became testable — importing from App.tsx
// pulls in the whole page graph, including sigma, which needs WebGL, and jsdom
// has none.

// The "View" menu holds theme, palette and language in one list.
//
// These used to be three separate icons in the header plus a link to the guide.
// A fourth does not fit beside the notification bell, and with the menu
// collapsed to 52px not even one does: header icons are the most expensive
// space in the interface, and three of them went to settings edited once a
// session.
//
// Every one of them is also a command in the palette (`useCommands`), so nothing
// is lost while the menu is collapsed.
export function ViewMenu() {
  const { theme, setTheme } = useTheme()
  const { palette, setPalette } = usePalette()
  const { t } = useTranslation()
  const [open, setOpen] = useState(false)

  return (
    <div className="lang-switcher">
      <button
        className="icon-btn" onClick={() => setOpen(o => !o)}
        title={t('view.label')} aria-label={t('view.label')} aria-expanded={open}
      >
        <PaletteIcon size={15} />
      </button>
      {open && (
        <>
          <div className="realm-switcher-backdrop" onClick={() => setOpen(false)} />
          <div className="lang-menu view-menu">
            <div className="eyebrow view-group">{t('view.theme')}</div>
            <div className="view-seg">
              {(['light', 'dark'] as const).map(mode => (
                <button
                  key={mode} type="button"
                  className={`view-seg-btn${theme === mode ? ' active' : ''}`}
                  onClick={() => setTheme(mode)}
                >
                  {mode === 'light' ? <Sun size={12} /> : <Moon size={12} />}
                  {t(`view.${mode}`)}
                </button>
              ))}
            </div>

            <div className="eyebrow view-group">{t('view.palette')}</div>
            {PALETTES.map(p => (
              <button
                key={p} type="button"
                className={`lang-item ${palette === p ? 'active' : ''}`}
                onClick={() => setPalette(p)}
              >
                <span className={`palette-dot palette-dot-${p}`} />
                <span className="grow">{t(`palette.names.${p}`)}</span>
                {palette === p && <Check size={12} />}
              </button>
            ))}
          </div>
        </>
      )}
    </div>
  )
}

// Language gets its own control beside the palette rather than a section
// inside it.
//
// Changing language and changing palette answer different questions: the first
// is which language to read in, the second how it looks. Folded into one menu,
// they force somebody to look for language under a heading that says "view" —
// and in a multilingual interface that is the one item a person looks for
// first, while still unable to read the rest.
export function LanguageMenu() {
  const { t, i18n } = useTranslation()
  const [open, setOpen] = useState(false)

  return (
    <div className="lang-switcher">
      <button
        className="icon-btn" onClick={() => setOpen(o => !o)}
        title={t('language.label')} aria-label={t('language.label')} aria-expanded={open}
      >
        <Globe size={15} />
        <span className="lang-code" aria-hidden="true">{i18n.language.slice(0, 2).toUpperCase()}</span>
      </button>
      {open && (
        <>
          <div className="realm-switcher-backdrop" onClick={() => setOpen(false)} />
          <div className="lang-menu view-menu">
            {SUPPORTED_LANGUAGES.map(l => (
              <button
                key={l.code} type="button"
                className={`lang-item ${i18n.language === l.code ? 'active' : ''}`}
                onClick={() => { i18n.changeLanguage(l.code); setOpen(false) }}
              >
                <span className="grow">{l.label}</span>
                {i18n.language === l.code && <Check size={12} />}
              </button>
            ))}
          </div>
        </>
      )}
    </div>
  )
}
