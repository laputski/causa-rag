import { useState } from 'react'
import { api } from '../api/client'

// Found live: a run picking a slow model (or hitting a stuck external RAG)
// had no way to be interrupted short of waiting out every remaining
// question. Cooperative, not instant — ExperimentRunner only checks between
// questions, so a question already in flight still finishes; already-
// answered questions are kept (see api.experiments.stop). Shared by
// NewExperimentPage's ProgressWidget (WS-driven `done`) and RunPage (polling-
// driven `status`) — those two differ in how they detect "still running",
// but the stop request itself doesn't depend on that, so it lives here once.
export function useStopExperiment(runId: string | null | undefined) {
  const [stopping, setStopping] = useState(false)
  const [stopError, setStopError] = useState<string | null>(null)

  async function stop() {
    if (!runId) return
    setStopping(true)
    setStopError(null)
    try {
      await api.experiments.stop(runId)
    } catch (err) {
      setStopError(String(err))
      setStopping(false)
    }
  }

  return { stopping, stopError, stop }
}
