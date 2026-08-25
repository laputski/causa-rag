import { useEffect, useState } from 'react'

export interface ProgressEvent {
  type: string
  stage?: string
  processed?: number
  total?: number
  metrics?: Record<string, number>
  run_id?: string
}

export function useProgress(runId: string | null) {
  const [events, setEvents] = useState<ProgressEvent[]>([])
  const [done, setDone] = useState(false)

  useEffect(() => {
    if (!runId) return
    setEvents([])
    setDone(false)

    const ws = new WebSocket(`ws://${location.host}/api/experiments/${runId}/progress`)
    ws.onmessage = (e) => {
      const ev: ProgressEvent = JSON.parse(e.data)
      setEvents(prev => [...prev, ev])
      if (ev.type === 'done') { setDone(true); ws.close() }
    }
    ws.onerror = () => setDone(true)
    return () => ws.close()
  }, [runId])

  const latest = events[events.length - 1]
  const progress = latest?.total && latest?.processed != null
    ? latest.processed / latest.total
    : null

  return { events, done, progress, latest }
}
