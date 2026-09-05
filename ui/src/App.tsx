import { Routes, Route, NavLink, Navigate, useLocation } from 'react-router-dom'
import {
  MessageSquare, Database, FileEdit, FlaskConical, Plus, GitCompare,
  LayoutDashboard, BookOpen, Puzzle,
  HelpCircle, Settings, Cpu, ChevronDown, Settings2,
  Scale, ChartScatter, SatelliteDish,
  Keyboard, Search as SearchIcon,
} from 'lucide-react'
import { useCallback, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { RealmProvider, useRealm, useRealmPath } from './context/RealmContext'
import { PaletteProvider } from './context/PaletteContext'
import { ShellProvider, useShell, SIDEBAR_MIN, SIDEBAR_MAX } from './context/ShellContext'
import { useHotkeys, formatCombo } from './hooks/useHotkeys'
import { useCommands } from './hooks/useCommands'
import { ViewMenu, LanguageMenu } from './components/ShellControls'
import { CommandPalette } from './components/CommandPalette'
import { ShortcutsDialog } from './components/ShortcutsDialog'
import { NotificationProvider } from './context/NotificationContext'
import { NotificationBell, ToastStack } from './components/Notifications'
import { useRunNotifications } from './hooks/useRunNotifications'
import { Logo, LogoMark } from './components/Logo'
import ExperimentsPage    from './pages/ExperimentsPage'
import FrontierPage       from './pages/FrontierPage'
import ProductionPage     from './pages/ProductionPage'
import RunPage            from './pages/RunPage'
import ComparisonPage     from './pages/ComparisonPage'
import NewExperimentPage  from './pages/NewExperimentPage'
import ChatPage           from './pages/ChatPage'
import PromptsPage        from './pages/PromptsPage'
import JudgmentsPage      from './pages/JudgmentsPage'
import CorpusPage         from './pages/CorpusPage'
import DatasetsPage       from './pages/DatasetsPage'
import GuidePage          from './pages/GuidePage'
import DomainPacksPage    from './pages/DomainPacksPage'
import RealmsPage         from './pages/RealmsPage'
import AtlasPage from './pages/AtlasPage'
import RealmResourcesPage from './pages/RealmResourcesPage'
import OverviewPage        from './pages/OverviewPage'
import WelcomePage         from './pages/WelcomePage'

function RealmSwitcher() {
  const { realms, activeRealm, activeRealmId, setActiveRealmId, loaded } = useRealm()
  const { t } = useTranslation()
  const toRealm = useRealmPath()
  const [open, setOpen] = useState(false)

  // No "realm-less" state — every Realm carries its own runs/resources/
  // settings, so there is nothing meaningful to test without one. Zero
  // Realms means the switcher just points at the creation page instead of
  // opening a dropdown (see the design notes).
  if (loaded && realms.length === 0) {
    return (
      <div className="realm-switcher">
        <NavLink to="/realms" className="realm-switcher-trigger realm-switcher-empty">
          <span className="realm-switcher-dot" />
          <span className="realm-switcher-label">{t('realm.createFirst')}</span>
        </NavLink>
      </div>
    )
  }

  return (
    <div className="realm-switcher">
      <button className="realm-switcher-trigger" onClick={() => setOpen(o => !o)}>
        <span className={`realm-switcher-dot ${activeRealmId ? 'active' : ''}`} />
        {/* The name, and under it the id with a resource count. A realm has
            two names: the one people call it by and the one it is written as
            in URLs and exports. The first alone does not tell you whether
            this is the realm you meant. */}
        <span className="realm-switcher-label">
          <span className="realm-switcher-name">{activeRealm?.name ?? '…'}</span>
          {activeRealm && (
            <span className="realm-switcher-id">
              {activeRealm.id} · {t('realm.resourceCount', { count: activeRealm.resources?.length ?? 0 })}
            </span>
          )}
        </span>
        <ChevronDown size={13} className={`realm-switcher-chevron ${open ? 'open' : ''}`} />
      </button>
      {open && (
        <>
          <div className="realm-switcher-backdrop" onClick={() => setOpen(false)} />
          <div className="realm-switcher-menu">
            {realms.map(r => (
              <div
                key={r.id}
                className={`realm-switcher-item ${r.id === activeRealmId ? 'active' : ''}`}
                onClick={() => { setActiveRealmId(r.id); setOpen(false) }}
              >
                <span className="realm-switcher-dot active" />
                {r.name}
              </div>
            ))}
            <div className="realm-switcher-footer">
              <NavLink to={toRealm('/realms')} onClick={() => setOpen(false)} className="realm-switcher-manage">
                <Settings2 size={12} />{t('realm.manage')}
              </NavLink>
            </div>
          </div>
        </>
      )}
    </div>
  )
}

function NavSection({ label }: { label: string }) {
  const { collapsed } = useShell()
  // Collapsed, the separator stays: the groups still exist, only their names
  // no longer fit. Dropping it too would merge fifteen icons into one
  // structureless column.
  if (collapsed) return <div className="nav-section-rule" />
  return (
    <div style={{ fontSize: '10px', fontWeight: 700, letterSpacing: '0.08em',
      color: 'var(--text-muted)', textTransform: 'uppercase', padding: '8px 10px 4px', marginTop: '4px' }}>
      {label}
    </div>
  )
}

// `matchPaths` exists for an entry that stands for several routes. The corpus
// entry is one item covering four tabs (Upload/Content/Health/Graph),
// and without it the entry would go dark the moment a reader switched tab —
// telling them they had left the section they are plainly still in.
function NavItem({ to, label, icon: Icon, disabled, matchPaths, draft }: {
  to: string; label: string; icon: React.ElementType
  disabled?: boolean; matchPaths?: string[]
  /** The section works but has not been checked all the way through. The mark
   *  appears both in the menu and in the section's own header: it has to be
   *  seen by somebody choosing where to go and by somebody who arrived
   *  through a link. */
  draft?: boolean
}) {
  const { t } = useTranslation()
  const toRealm = useRealmPath()
  const location = useLocation()
  const { collapsed } = useShell()
  // Collapsed, the label moves into `title` rather than hiding behind
  // `display: none`: hidden text stays in the accessibility tree and gets read
  // aloud twice.
  const caption = collapsed ? null : label
  const draftMark = draft && !collapsed
    ? <span className="nav-draft" title={t('draft.tooltip')}>{t('draft.short')}</span>
    : null
  if (disabled) {
    return (
      <span className="nav-link nav-link-disabled" title={t('realm.needRealm')} aria-label={label}>
        <Icon size={15} className="nav-icon" />
        {caption}
        {draftMark}
      </span>
    )
  }
  const alsoActive = (matchPaths ?? []).some(p => location.pathname === p)
  return (
    <NavLink
      to={toRealm(to)}
      title={collapsed ? label : undefined}
      aria-label={collapsed ? label : undefined}
      className={({ isActive }) => 'nav-link' + (isActive || alsoActive ? ' active' : '')}
    >
      <Icon size={15} className="nav-icon" />
      {caption}
      {draftMark}
    </NavLink>
  )
}

// The handle that drags the menu's right edge.
//
// Pointer events rather than mouse events: a pen and a trackpad send the same
// thing, and `setPointerCapture` carries the gesture through even when the
// cursor leaves the window. Without it, releasing the button outside left the
// menu stuck to the cursor.
function SidebarResizer() {
  const { width, setWidth, collapsed } = useShell()
  const { t } = useTranslation()
  const dragging = useRef(false)

  const onPointerDown = useCallback((e: React.PointerEvent<HTMLDivElement>) => {
    if (collapsed) return
    dragging.current = true
    e.currentTarget.setPointerCapture(e.pointerId)
    document.body.style.cursor = 'col-resize'
    document.body.style.userSelect = 'none'
  }, [collapsed])

  const onPointerMove = useCallback((e: React.PointerEvent) => {
    if (dragging.current) setWidth(e.clientX)
  }, [setWidth])

  const stop = useCallback(() => {
    dragging.current = false
    document.body.style.cursor = ''
    document.body.style.userSelect = ''
  }, [])

  if (collapsed) return null
  return (
    <div
      className="sidebar-resizer" role="separator" aria-orientation="vertical"
      aria-label={t('shell.resize')} aria-valuenow={width}
      aria-valuemin={SIDEBAR_MIN} aria-valuemax={SIDEBAR_MAX} tabIndex={0}
      onPointerDown={onPointerDown} onPointerMove={onPointerMove}
      onPointerUp={stop} onPointerCancel={stop}
      onKeyDown={e => {
        // From the keyboard too: a width that can only be dragged with a
        // mouse is a setting unavailable to anyone who does not use one.
        if (e.key === 'ArrowLeft') { e.preventDefault(); setWidth(width - 16) }
        if (e.key === 'ArrowRight') { e.preventDefault(); setWidth(width + 16) }
      }}
    />
  )
}

function Sidebar({ onOpenCommands, onOpenShortcuts }: {
  onOpenCommands: () => void; onOpenShortcuts: () => void
}) {
  const { realms, loaded } = useRealm()
  const { t } = useTranslation()
  const toRealm = useRealmPath()
  const { collapsed } = useShell()
  const noRealms = loaded && realms.length === 0

  return (
    <nav
      className={`sidebar${collapsed ? ' collapsed' : ''}`}
      aria-label={t('nav.sectionChat')}
    >
      {/* `app-drag` marks the window-drag region of the desktop build. One
          class now, so the header need not be relaid out later; in a browser
          it does nothing. */}
      <div className="sidebar-header app-drag">
        {collapsed ? <LogoMark size={20} /> : <Logo />}
        {!collapsed && (
          <div className="sidebar-header-actions">
            <NotificationBell />
            <ViewMenu />
            <LanguageMenu />
          </div>
        )}
      </div>

      <RealmSwitcher />

      {/* Only the item list scrolls. The footer below stays put: it is the
          one thing that tells a reader the shortcuts exist. */}
      <div className="sidebar-scroll">
        {/* Overview comes before every section and carries no heading: it is
            an entry point rather than a section, and a heading over a single
            item is not a heading. */}
        <NavItem to="/overview" label={t('nav.overview')} icon={LayoutDashboard} disabled={noRealms} />

        {/* Chat sits apart from runs. Both ask the system a question, but
            chat answers one immediately while a run measures over a hundred
            and forty. The work behind them differs, and a one-item section
            costs less here than an item in the wrong place. */}
        <NavSection label={t('nav.sectionChat')} />
        <NavItem to="/chat"        label={t('nav.chat')}         icon={MessageSquare} disabled={noRealms} />

        <NavSection label={t('nav.sectionRuns')} />
        <NavItem to="/new"         label={t('nav.newRun')}       icon={Plus} disabled={noRealms} />
        <NavItem to="/experiments" label={t('nav.experiments')}  icon={FlaskConical} disabled={noRealms} />
        <NavItem to="/compare"     label={t('nav.comparison')}   icon={GitCompare} disabled={noRealms} />
        <NavItem to="/frontier"    label={t('nav.frontier')}     icon={ChartScatter} disabled={noRealms} draft />

        {/* Upload, content, health and graph are four views of one corpus
            rather than four things, so they are one item with tabs inside it.
            They used to be four entries in the list, three of which named
            different categories of thing: one action and two objects. */}
        <NavSection label={t('nav.sectionData')} />
        <NavItem
          to="/data/upload" label={t('nav.dataCorpus')} icon={Database} disabled={noRealms}
          matchPaths={['/data/content', '/data/health', '/data/graph']}
        />
        <NavItem to="/data/qa"        label={t('nav.dataQa')}        icon={HelpCircle} disabled={noRealms} />
        <NavItem to="/data/judgments" label={t('nav.dataJudgments')} icon={Scale} disabled={noRealms} draft />
        <NavItem to="/production"     label={t('nav.production')}    icon={SatelliteDish} disabled={noRealms} draft />

        {/* Prompts used to sit with no section heading straight after the
            data items, and read as a continuation of them. An answer template
            configures how the system replies, which makes it a setting. */}
        <NavSection label={t('nav.sectionSettings')} />
        <NavItem
          to="/prompts" label={t('nav.prompts')} icon={FileEdit} disabled={noRealms}
          matchPaths={['/data/presets']}
        />
        <NavItem to="/settings/resources" label={t('nav.resources')} icon={Cpu} disabled={noRealms} />
        <NavItem to="/atlas"          label={t('nav.atlas')}         icon={BookOpen} />
        <NavItem to="/domain-packs"   label={t('nav.domainPacks')}   icon={Puzzle} />
        <NavItem to="/realms"         label={t('nav.realms')}        icon={Settings} />

      </div>

      {/* The footer. "Keyboard shortcuts ?" stays here permanently, and it is
          the only way to find that dialog without already knowing it exists:
          somebody who has never pressed "?" would learn of the key from
          nowhere else. */}
      <div className="sidebar-foot">
        <button type="button" className="sidebar-foot-btn" onClick={onOpenCommands}>
          <SearchIcon size={13} />
          {!collapsed && <><span>{t('commands.title')}</span><kbd className="kbd">{formatCombo('mod+k')}</kbd></>}
        </button>
        <button type="button" className="sidebar-foot-btn" onClick={onOpenShortcuts}>
          <Keyboard size={13} />
          {!collapsed && <><span>{t('shortcuts.title')}</span><kbd className="kbd">?</kbd></>}
        </button>
        <NavLink to={toRealm('/guide')} className="sidebar-foot-btn" title={t('nav.guide')}>
          <BookOpen size={13} />
          {!collapsed && <span>{t('nav.guide')}</span>}
        </NavLink>
      </div>

      <SidebarResizer />
    </nav>
  )
}

// `<Navigate to="/x" replace />` drops the current query string by default —
// same `?realm=` loss as plain Link/NavLink (see useRealmPath), but for
// route-level redirects (root `/`, `/data`, and the legacy path aliases
// below), so a bookmarked/typed URL like `/corpus?realm=demo` doesn't
// bounce the user to whatever Realm sorts first once it lands on `/data/upload`.
function PreserveSearchRedirect({ to }: { to: string }) {
  const location = useLocation()
  return <Navigate to={{ pathname: to, search: location.search }} replace />
}

function AppShell() {
  const { realms, loaded } = useRealm()
  const noRealms = loaded && realms.length === 0
  const [cmdOpen, setCmdOpen] = useState(false)
  const [shortcutsOpen, setShortcutsOpen] = useState(false)

  const openShortcuts = useCallback(() => { setCmdOpen(false); setShortcutsOpen(true) }, [])
  const commands = useCommands(openShortcuts)

  // Runs take minutes. The completion notice comes from a state transition
  // rather than from a poll response (see useRunNotifications).
  useRunNotifications()

  // Two bindings rather than one. `⌘K` is always live: it both opens the
  // palette and closes it, and disabling it along with the rest would mean the
  // palette could not be closed by the shortcut that opened it. Found live.
  useHotkeys([{ combo: 'mod+k', run: () => { setShortcutsOpen(false); setCmdOpen(o => !o) } }])

  // The rest come from the command registry, the single source of bindings: a
  // shortcut written both here and in the palette diverges on the second edit.
  // They stay disabled while any dialog is open, because a page sits beneath
  // it and `n` must not start a run while somebody is reading the shortcuts.
  useHotkeys(
    commands.filter(c => c.combo).map(c => ({ combo: c.combo!, run: c.run })),
    !cmdOpen && !shortcutsOpen,
  )

  return (
    <div className="layout">
      <Sidebar onOpenCommands={() => setCmdOpen(true)} onOpenShortcuts={openShortcuts} />
      <main className="main" role="main">
        {noRealms ? (
          <Routes>
            <Route path="/guide" element={<GuidePage />} />
            <Route path="/realms" element={<RealmsPage />} />
            {/* Every path used to render the realm management page. That page
                answers "how many are there", and somebody opening the platform
                for the first time is asking "where do I start". */}
            <Route path="*" element={<WelcomePage />} />
          </Routes>
        ) : (
          <Routes>
            {/* The entry point is the realm's overview rather than the run
                list: a history of what has been done answers neither "does
                everything work" nor "what next". */}
            <Route path="/"                   element={<PreserveSearchRedirect to="/overview" />} />
            <Route path="/overview"           element={<OverviewPage />} />
            <Route path="/experiments"        element={<ExperimentsPage />} />
            <Route path="/experiments/:runId" element={<RunPage />} />
            <Route path="/compare"            element={<ComparisonPage />} />
            <Route path="/frontier"           element={<FrontierPage />} />
            <Route path="/production"         element={<ProductionPage />} />
            <Route path="/new"                element={<NewExperimentPage />} />
            <Route path="/data/qa"            element={<DatasetsPage />} />
            <Route path="/data/presets"       element={<PromptsPage />} />
            <Route path="/data/judgments"     element={<JudgmentsPage />} />
            <Route path="/data/:tab"          element={<CorpusPage />} />
            <Route path="/data"               element={<PreserveSearchRedirect to="/data/upload" />} />
            <Route path="/prompts"            element={<PromptsPage />} />
            <Route path="/chat"               element={<ChatPage />} />
            <Route path="/guide"              element={<GuidePage />} />
            <Route path="/atlas"              element={<AtlasPage />} />
            <Route path="/domain-packs"       element={<DomainPacksPage />} />
            <Route path="/realms"             element={<RealmsPage />} />
            <Route path="/settings/resources" element={<RealmResourcesPage />} />
            {/* legacy redirects */}
            <Route path="/external-rags"      element={<PreserveSearchRedirect to="/settings/resources" />} />
            <Route path="/corpus"             element={<PreserveSearchRedirect to="/data/upload" />} />
            <Route path="/datasets"           element={<PreserveSearchRedirect to="/data/qa" />} />
            {/* `/panels` is no longer its own page: the tool catalogue merged
                into the resources page, because "does everything work" is one
                question rather than two. */}
            <Route path="/panels"             element={<PreserveSearchRedirect to="/settings/resources" />} />
            {/* `/dashboard` had a route and no menu item, so it was reachable
                only by typing the URL. Its page was a weaker copy of the run
                list and read every realm's runs at once. */}
            <Route path="/dashboard"          element={<PreserveSearchRedirect to="/overview" />} />
          </Routes>
        )}
      </main>

      <ToastStack />
      <CommandPalette commands={commands} open={cmdOpen} onClose={() => setCmdOpen(false)} />
      <ShortcutsDialog commands={commands} open={shortcutsOpen} onClose={() => setShortcutsOpen(false)} />
    </div>
  )
}

export default function App() {
  return (
    <PaletteProvider>
      <ShellProvider>
        <NotificationProvider>
          <RealmProvider>
            <AppShell />
          </RealmProvider>
        </NotificationProvider>
      </ShellProvider>
    </PaletteProvider>
  )
}
