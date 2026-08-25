import { useQueries, useQuery } from '@tanstack/react-query'
import { api } from '../api/client'
import { useRealm } from '../context/RealmContext'

// The state of the installation: the realm's infrastructure, its RAG endpoints
// and its panels.
//
// Fanned out from the browser rather than through one server route. All three
// checks already exist and are already used by the resources page; a fourth,
// summarising route would have to be kept consistent with those three, and such
// a pair always diverges toward the summary, because a summary lies more
// quietly.
//
// One react-query cache for all of them: the sidebar summary, the overview band
// and the status page read the same requests, so three places cannot show three
// different numbers.

export type HealthState = 'ok' | 'warn' | 'down' | 'unknown'

export interface HealthItem {
  id: string
  label: string
  kind: 'resource' | 'rag' | 'panel'
  state: HealthState
  detail?: string
}

export interface RealmHealth {
  items: HealthItem[]
  /** The worst state among those checked; the summary takes its colour. */
  worst: HealthState
  ok: number
  total: number
  loading: boolean
  refetch: () => void
}

interface RealmResource { type: string }

export function useRealmHealth(): RealmHealth {
  const { activeRealm, activeRealmId } = useRealm()
  const resources = ((activeRealm?.resources ?? []) as RealmResource[]).map(r => r.type)

  const resourceQueries = useQueries({
    queries: resources.map(type => ({
      queryKey: ['resource-test', activeRealmId, type],
      queryFn: () => api.realms.testResource(activeRealmId!, type),
      enabled: Boolean(activeRealmId),
      // A resource is a separate process an operator starts and stops. One
      // minute: more often is pointless, less often shows a half-hour-old
      // state.
      staleTime: 60_000,
      retry: false,
    })),
  })

  const ragsQuery = useQuery({
    queryKey: ['external-rags', activeRealmId],
    queryFn: () => api.externalRags.list(activeRealmId),
    enabled: Boolean(activeRealmId),
    staleTime: 60_000,
  })

  const panelsQuery = useQuery({
    queryKey: ['panels-status'],
    queryFn: api.panelsStatus,
    staleTime: 15_000,
  })

  const items: HealthItem[] = []

  resources.forEach((type, i) => {
    const q = resourceQueries[i]
    const data = q.data as { status?: string; detail?: string } | undefined
    items.push({
      id: `resource:${type}`, label: type, kind: 'resource',
      state: q.isLoading ? 'unknown' : q.isError || data?.status !== 'ok' ? 'down' : 'ok',
      detail: data?.detail,
    })
  })

  for (const rag of ragsQuery.data ?? []) {
    // An endpoint's capabilities are known only after a check, and checking
    // them in this fan-out is not allowed: that is a request to somebody else's
    // system rather than to our own. Hence "unknown" rather than "working".
    items.push({
      id: `rag:${rag.id}`, label: rag.name, kind: 'rag',
      state: rag.capabilities ? 'ok' : 'unknown',
    })
  }

  for (const panel of panelsQuery.data ?? []) {
    items.push({
      id: `panel:${panel.id}`, label: panel.id, kind: 'panel',
      // Refusing to be embedded is the tool's own decision rather than a
      // fault. Its own state, because amber and red call for different things
      // here, and amber calls for nothing.
      state: !panel.reachable ? 'down' : panel.embeddable ? 'ok' : 'warn',
      detail: panel.blocked_by ?? undefined,
    })
  }

  // "Unchecked" is excluded from the denominator: a ratio whose numerator and
  // denominator count different things reads as a score, and this is not one.
  const counted = items.filter(i => i.state !== 'unknown')
  const worst: HealthState =
    counted.some(i => i.state === 'down') ? 'down'
    : counted.some(i => i.state === 'warn') ? 'warn'
    : 'ok'
  return {
    items,
    // Refusing to be embedded is the tool's decision rather than a fault, so it
    // counts in the numerator. The band's colour still follows the worst state:
    // "11 of 11" in green above two amber rows would be two different pictures
    // on one screen.
    worst,
    ok: counted.filter(i => i.state === 'ok' || i.state === 'warn').length,
    total: counted.length,
    loading: resourceQueries.some(q => q.isLoading) || panelsQuery.isLoading,
    refetch: () => {
      resourceQueries.forEach(q => q.refetch())
      panelsQuery.refetch()
      ragsQuery.refetch()
    },
  }
}
