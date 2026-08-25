import { useMemo } from 'react'
import { useNavigate } from 'react-router-dom'
import { useTranslation } from 'react-i18next'
import {
  MessageSquare, Database, FileEdit, FlaskConical, Plus, GitCompare,
  LayoutDashboard, BookOpen, Puzzle, HelpCircle, Settings, Cpu,
  LayoutTemplate, Scale, ChartScatter, SatelliteDish, Sun, Moon,
  Palette as PaletteIcon, Languages, PanelLeft, Keyboard, Search, Download,
} from 'lucide-react'
import { useRealmPath } from '../context/RealmContext'
import { useTheme } from '../context/ThemeContext'
import { usePalette, PALETTES } from '../context/PaletteContext'
import { useShell } from '../context/ShellContext'
import { SUPPORTED_LANGUAGES } from '../i18n'

// The command registry: one source for three consumers, the ⌘K palette, the
// hotkey bindings and the shortcut list behind "?". The desktop build's native
// menu is built from it too: a shortcut described twice diverges on the second
// edit.
//
// The hints come from here as well. There is no separate place that teaches the
// keys, and none is needed: the palette shows each shortcut on the right of its
// row, and people learn them by using it.

export type CommandGroup = 'nav' | 'action' | 'view'

export interface Command {
  id: string
  group: CommandGroup
  label: string
  /** Extra search terms beyond the label: an alternative name, a synonym. */
  keywords?: string
  /** `mod+k`, `?`, `n`, or a chord such as `g r`. Absent means palette-only. */
  combo?: string
  icon: React.ElementType
  run: () => void
}

/** Focus the current page's search field. A page declares it with an attribute
 *  rather than a class name: a class is presentation, and what is needed here is
 *  a role. */
function focusPageSearch() {
  const el = document.querySelector<HTMLInputElement>('[data-page-search]')
  if (el) { el.focus(); el.select() }
}

export function useCommands(openShortcuts: () => void): Command[] {
  const navigate = useNavigate()
  const toRealm = useRealmPath()
  const { t, i18n } = useTranslation()
  const { theme, setTheme } = useTheme()
  const { palette, setPalette } = usePalette()
  const { toggleCollapsed } = useShell()

  return useMemo(() => {
    const go = (path: string) => () => navigate(toRealm(path))

    const nav: Command[] = [
      { id: 'go.overview',   group: 'nav', label: t('nav.overview'),      keywords: 'overview обзор главная', combo: 'g o', icon: LayoutDashboard, run: go('/overview') },
      { id: 'go.chat',       group: 'nav', label: t('nav.chat'),          keywords: 'chat',                   combo: 'g c', icon: MessageSquare,  run: go('/chat') },
      { id: 'go.runs',       group: 'nav', label: t('nav.experiments'),   keywords: 'runs experiments прогоны', combo: 'g r', icon: FlaskConical, run: go('/experiments') },
      { id: 'go.compare',    group: 'nav', label: t('nav.comparison'),    keywords: 'compare diff',           combo: 'g k', icon: GitCompare,     run: go('/compare') },
      { id: 'go.frontier',   group: 'nav', label: t('nav.frontier'),      keywords: 'frontier pareto',        icon: ChartScatter,  run: go('/frontier') },
      { id: 'go.corpus',     group: 'nav', label: t('nav.dataCorpus'),    keywords: 'corpus корпус',          combo: 'g d', icon: Database,       run: go('/data/upload') },
      { id: 'go.questions',  group: 'nav', label: t('nav.dataQa'),        keywords: 'questions dataset golden', combo: 'g q', icon: HelpCircle,   run: go('/data/qa') },
      { id: 'go.judgments',  group: 'nav', label: t('nav.dataJudgments'), keywords: 'judgments суждения',     icon: Scale,         run: go('/data/judgments') },
      { id: 'go.production', group: 'nav', label: t('nav.production'),    keywords: 'production запросы',     icon: SatelliteDish, run: go('/production') },
      { id: 'go.presets',    group: 'nav', label: t('nav.dataPresets'),   keywords: 'presets пресеты',        icon: LayoutTemplate, run: go('/data/presets') },
      { id: 'go.prompts',    group: 'nav', label: t('nav.prompts'),       keywords: 'prompts промпты',        icon: FileEdit,      run: go('/prompts') },
      { id: 'go.status',     group: 'nav', label: t('nav.resources'),     keywords: 'status resources panels состояние ресурсы панели', combo: 'g s', icon: Cpu, run: go('/settings/resources') },
      { id: 'go.packs',      group: 'nav', label: t('nav.domainPacks'),   keywords: 'domain packs',           icon: Puzzle,        run: go('/domain-packs') },
      { id: 'go.realms',     group: 'nav', label: t('nav.realms'),        keywords: 'realms реалмы',          icon: Settings,      run: go('/realms') },
      { id: 'go.guide',      group: 'nav', label: t('nav.guide'),         keywords: 'guide help руководство', combo: 'g h', icon: BookOpen, run: go('/guide') },
    ]

    const actions: Command[] = [
      { id: 'act.newRun',    group: 'action', label: t('commands.newRun'),        combo: 'n',       icon: Plus,     run: go('/new') },
      { id: 'act.search',    group: 'action', label: t('commands.searchOnPage'),  combo: '/',       icon: Search,   run: focusPageSearch },
      { id: 'act.sidebar',   group: 'action', label: t('commands.toggleSidebar'), combo: 'mod+\\',  icon: PanelLeft, run: toggleCollapsed },
      { id: 'act.shortcuts', group: 'action', label: t('commands.shortcuts'),     combo: '?',       icon: Keyboard, run: openShortcuts },
      { id: 'act.exportRealm', group: 'action', label: t('commands.exportRealm'), icon: Download,   run: go('/realms') },
    ]

    const view: Command[] = [
      {
        id: 'view.theme', group: 'view',
        label: theme === 'dark' ? t('theme.toLight') : t('theme.toDark'),
        keywords: 'theme dark light тема',
        combo: 'mod+shift+t', icon: theme === 'dark' ? Sun : Moon,
        run: () => setTheme(theme === 'dark' ? 'light' : 'dark'),
      },
      // Palette and language expand into one command per option rather than a
      // single "change palette": a command that opens another list is a menu
      // item rather than a command, and is useless inside a palette.
      ...PALETTES.filter(p => p !== palette).map(p => ({
        id: `view.palette.${p}`, group: 'view' as const,
        label: t('commands.palette', { name: t(`palette.names.${p}`) }),
        keywords: `palette ${p} палитра`,
        icon: PaletteIcon, run: () => setPalette(p),
      })),
      ...SUPPORTED_LANGUAGES.filter(l => l.code !== i18n.language).map(l => ({
        id: `view.lang.${l.code}`, group: 'view' as const,
        label: t('commands.language', { name: l.label }),
        keywords: `language ${l.code} ${l.label} язык`,
        icon: Languages, run: () => i18n.changeLanguage(l.code),
      })),
    ]

    return [...nav, ...actions, ...view]
  }, [navigate, toRealm, t, i18n, theme, setTheme, palette, setPalette, toggleCollapsed, openShortcuts])
}

/** Substring matching over the label and keywords, with no fuzzy search: there
 *  are about thirty commands, and at that size a fuzzy match misleads more often
 *  than it helps. */
export function filterCommands(commands: Command[], query: string): Command[] {
  const q = query.trim().toLowerCase()
  if (!q) return commands
  return commands.filter(c =>
    c.label.toLowerCase().includes(q) || (c.keywords ?? '').toLowerCase().includes(q),
  )
}
