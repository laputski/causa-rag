import { createContext, useContext, useState, useEffect, ReactNode } from 'react'
import { useSearchParams } from 'react-router-dom'

interface Realm {
  id: string
  name: string
  description?: string
  resources: unknown[]
  /** The realm's key metrics: one list governing both the run table's columns
   *  and the number band on the run screen. */
  key_metrics?: string[]
  created_at: string
}

interface RealmContextValue {
  realms: Realm[]
  activeRealm: Realm | null
  activeRealmId: string | null
  setActiveRealmId: (id: string | null) => void
  reload: () => Promise<void>
  // True once the first GET /realms has returned — lets callers distinguish
  // "still loading" from "confirmed zero Realms exist" (see App.tsx, which
  // gates the whole nav on this rather than flashing an empty state).
  loaded: boolean
}

const RealmContext = createContext<RealmContextValue>({
  realms: [],
  activeRealm: null,
  activeRealmId: null,
  setActiveRealmId: () => {},
  reload: async () => {},
  loaded: false,
})

export function RealmProvider({ children }: { children: ReactNode }) {
  const [realms, setRealms] = useState<Realm[]>([])
  const [loaded, setLoaded] = useState(false)
  const [searchParams, setSearchParams] = useSearchParams()
  const realmParam = searchParams.get('realm')
  const [activeRealmId, setActiveRealmIdState] = useState<string | null>(realmParam)

  const reload = async () => {
    try {
      const res = await fetch('/api/realms')
      if (res.ok) setRealms(await res.json())
    } catch {
      // non-fatal
    } finally {
      setLoaded(true)
    }
  }

  useEffect(() => { reload() }, [])

  // sync URL → state on mount
  useEffect(() => {
    if (realmParam !== activeRealmId) setActiveRealmIdState(realmParam)
  }, [realmParam])  // eslint-disable-line

  const setActiveRealmId = (id: string | null) => {
    setActiveRealmIdState(id)
    setSearchParams(prev => {
      const next = new URLSearchParams(prev)
      if (id) next.set('realm', id)
      else next.delete('realm')
      return next
    }, { replace: true })
  }

  // There is no meaningful "no Realm" working state — experiments and
  // settings need a Realm to be scoped to (see the design notes "Experiment
  // and settings scoping"). Once Realms are loaded, auto-pick one if the
  // current activeRealmId is unset or points at a Realm that no longer
  // exists (e.g. it was deleted, or the URL had a stale ?realm= param).
  useEffect(() => {
    if (!loaded || realms.length === 0) return
    if (activeRealmId && realms.some(r => r.id === activeRealmId)) return
    setActiveRealmId(realms[0].id)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [loaded, realms, activeRealmId])

  const activeRealm = realms.find(r => r.id === activeRealmId) ?? null

  return (
    <RealmContext.Provider value={{ realms, activeRealm, activeRealmId, setActiveRealmId, reload, loaded }}>
      {children}
    </RealmContext.Provider>
  )
}

export function useRealm() {
  return useContext(RealmContext)
}

// Plain <Link to="/new">/<NavLink> targets don't carry the current location's
// query string, so every internal navigation used to drop `?realm=` — which
// then made RealmProvider's "unset ⇒ auto-pick realms[0]" fallback fire on
// every click, silently bouncing the user back to whichever Realm sorts
// first (see the design notes "Realm selection & UI shell"). This hook
// re-attaches the active Realm to a path so navigation actually preserves
// the selected Realm, merging with any query string the path already has
// (e.g. `/compare?a=X&b=Y`).
export function useRealmPath() {
  const { activeRealmId } = useRealm()
  return (path: string) => {
    if (!activeRealmId) return path
    const [base, query] = path.split('?')
    const params = new URLSearchParams(query)
    params.set('realm', activeRealmId)
    return `${base}?${params.toString()}`
  }
}
