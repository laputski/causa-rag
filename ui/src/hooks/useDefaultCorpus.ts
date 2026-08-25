import { useEffect } from 'react'
import { useQuery } from '@tanstack/react-query'
import { api } from '../api/client'

/**
 * Chooses which corpus a screen opens on when nobody has said.
 *
 * Screens used to open on `default`, an id always present in the list and only
 * sometimes holding anything, so the first thing a reader saw was "no data"
 * with a populated corpus on the next line.
 *
 * The rule: the URL wins, then the corpus of the last successful ingest that
 * actually indexed something, and only then the first live entry in the
 * registry. The registry stores no size, so "the largest" cannot be chosen,
 * whereas the ingest log knows where documents were really put.
 *
 * The hook never overrides a choice already made: it fires only while the value
 * is empty. `default` no longer counts as empty — where a corpus by that name
 * exists it is an ordinary registry corpus, and choosing it deserves
 * respect.
 */
export function useDefaultCorpus(
  realmId: string | null,
  current: string,
  onPick: (corpusId: string) => void,
  { skip = false }: { skip?: boolean } = {},
) {
  const history = useQuery({
    queryKey: ['corpus', realmId],
    queryFn: () => api.corpus.list(realmId),
    enabled: Boolean(realmId) && !skip,
  })
  const registry = useQuery({
    queryKey: ['corpus-collections', realmId],
    queryFn: () => api.corpus.collections(realmId),
    enabled: Boolean(realmId) && !skip,
  })

  useEffect(() => {
    if (skip || current) return
    const filled = (history.data ?? []).find(
      h => h.status === 'done' && (h.n_chunks ?? 0) > 0 && h.corpus_id,
    )
    const pick = filled?.corpus_id ?? (registry.data ?? []).find(c => !c.deleted_at)?.corpus_id
    if (pick && pick !== current) onPick(pick)
    // `onPick` changes every render for callers passing an inline arrow, so it
    // is deliberately kept out of the dependency list.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [history.data, registry.data, current, skip])
}
