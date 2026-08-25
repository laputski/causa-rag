import { useEffect, useRef } from 'react'
import { useQuery } from '@tanstack/react-query'
import { useTranslation } from 'react-i18next'
import { api, type ExperimentItem } from '../api/client'
import { useRealm, useRealmPath } from '../context/RealmContext'
import { useNotifications } from '../context/NotificationContext'
import { classifyDelta } from '../lib/metricMeta'

// Where run notifications come from.
//
// The run list is already polled every three seconds while any run is going
// (`ExperimentsPage`). A notification is born from a state **transition**,
// `running` → anything else, rather than from a response. The response arrives
// dozens of times and the transition once, and that is exactly the difference
// easily lost: subscribing to the response gives one notification per poll.
//
// The previous state is kept by `run_id` in a ref rather than in component
// state: changing that map must not trigger a re-render.

export function useRunNotifications() {
  const { activeRealmId } = useRealm()
  const toRealm = useRealmPath()
  const { t } = useTranslation()
  const { notify } = useNotifications()
  const seen = useRef<Map<string, string>>(new Map())
  /** The first response only fills the map. Otherwise opening the platform
   *  would rain notifications about every run that finished last week. */
  const primed = useRef(false)

  const { data } = useQuery({
    queryKey: ['experiments', activeRealmId],
    queryFn: () => api.experiments.list({ realmId: activeRealmId }),
    enabled: Boolean(activeRealmId),
    refetchInterval: q => (q.state.data?.some(e => e.status === 'running') ? 3000 : 30_000),
  })

  useEffect(() => {
    if (!data) return
    const baseline = data.find(e => e.is_baseline)

    if (!primed.current) {
      for (const run of data) seen.current.set(run.run_id, run.status ?? 'unknown')
      primed.current = true
      return
    }

    for (const run of data) {
      const was = seen.current.get(run.run_id)
      const now = run.status ?? 'unknown'
      seen.current.set(run.run_id, now)
      if (was !== 'running' || now === 'running') continue

      const href = toRealm(`/experiments/${run.run_id}`)
      // The list knows only `running` and `done`: a failed run arrives as
      // `done` with empty metrics, because there is nothing to save. The
      // difference is read from that rather than from a status that does not
      // exist.
      const failed = now === 'done' && Object.keys(run.aggregate_metrics ?? {}).length === 0 && !run.stopped
      notify({
        id: `run:${run.run_id}:done`,
        kind: failed ? 'error' : run.stopped ? 'warning' : 'success',
        title: failed
          ? t('notifications.runFailed', { name: run.name })
          : run.stopped
            ? t('notifications.runStopped', { name: run.name })
            : t('notifications.runDone', { name: run.name }),
        body: t('notifications.runBody', { id: run.run_id.slice(0, 8), n: run.n_questions ?? 0 }),
        href, hrefLabel: t('notifications.open'), toast: true,
      })

      // A regression is its own event rather than a postscript to the first: it
      // answers a different question, whether things got worse, and somebody who
      // has seen "run finished" should not have to read on to learn about a
      // drop.
      if (!failed && baseline && baseline.run_id !== run.run_id) {
        const regressed = Object.entries(run.aggregate_metrics ?? {}).filter(([key, value]) => {
          const before = baseline.aggregate_metrics?.[key]
          return typeof before === 'number' && typeof value === 'number'
            && classifyDelta(before, value, key) === 'regressed'
        })
        if (regressed.length > 0) {
          notify({
            id: `run:${run.run_id}:regression`,
            kind: 'warning',
            title: t('notifications.regression', { name: run.name }),
            body: t('notifications.regressionBody', {
              count: regressed.length,
              metrics: regressed.slice(0, 3).map(([k]) => k).join(', '),
            }),
            href: toRealm(`/compare?a=${baseline.run_id}&b=${run.run_id}`),
            hrefLabel: t('notifications.compare'), toast: true,
          })
        }
      }
    }
  }, [data, notify, t, toRealm])
}

export type { ExperimentItem }
