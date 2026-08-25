import { useEffect, useMemo, useState } from 'react'
import { X } from 'lucide-react'
import { SigmaContainer, useLoadGraph, useSigma } from '@react-sigma/core'
import '@react-sigma/core/lib/react-sigma.min.css'
import Graph from 'graphology'
import { useQuery } from '@tanstack/react-query'
import { useTranslation } from 'react-i18next'
import { token } from '../lib/themeTokens'
import SelectBox from './SelectBox'
import { api, type GraphCommunitiesResult, type GraphCommunity } from '../api/client'

// One color per dominant source_code bucket would need an unbounded
// palette (this corpus alone has ~27 codes) — instead we color by HOW
// MIXED a community is, which is the actual diagnostic question this view
// exists to answer (see adapters/neo4j_graph.py module docstring): does a
// community line up with one source, or is it shared-keyword noise?
//
// The steps come from the diagram palette. The four literals that stood here
// outlived the design they were chosen for and followed neither the theme nor
// the palette: on the light theme the green drowned in white. Sigma draws in
// WebGL and accepts a string alone, so the token is read out, not referenced.
function colorForCommunity(c: GraphCommunity): string {
  if (c.distinct_source_count <= 1) return token('--diag-green')    // one source: clean
  if (c.dominant_source_share >= 0.6) return token('--diag-cyan')   // one clearly dominates
  if (c.dominant_source_share >= 0.3) return token('--diag-yellow') // mixed, with one leading
  return token('--diag-orange')                                     // heavily mixed: lexical noise
}

function GraphLoader({ result, onNodeClick }: {
  result: GraphCommunitiesResult
  onNodeClick: (communityId: number) => void
}) {
  const { t } = useTranslation()
  const loadGraph = useLoadGraph()
  const sigma = useSigma()

  useEffect(() => {
    const graph = new Graph()
    // Singleton communities (no RELATED edges at all) are the overwhelming
    // majority on this corpus (~96%, see adapters/neo4j_graph.py module
    // comment) and add nothing but visual noise — they're not "structure",
    // they're absence of it. Render only communities with >1 node.
    const nonTrivial = result.communities.filter(c => c.size > 1)
    const maxSize = Math.max(1, ...nonTrivial.map(c => c.size))
    // Even spacing for an unknown N needs an irrational angular step — a
    // fixed 2π/N step (the previous approach) put points at nearly the
    // same angle as radius grows, collapsing the whole set into a spiral
    // line instead of filling the plane. The golden angle is the standard
    // fix (phyllotaxis / "sunflower" placement).
    const goldenAngle = Math.PI * (3 - Math.sqrt(5))

    nonTrivial.forEach((c, i) => {
      const radius = 4 * Math.sqrt(i + 1)
      graph.addNode(String(c.community_id), {
        x: radius * Math.cos(i * goldenAngle),
        y: radius * Math.sin(i * goldenAngle),
        size: 2 + 18 * Math.sqrt(c.size / maxSize),
        color: colorForCommunity(c),
        label: t('graphCommunityView.nodeLabel', { id: c.community_id, size: c.size }),
      })
    })

    const maxWeight = Math.max(1, ...result.inter_community_edges.map(e => e.weight))
    result.inter_community_edges.forEach(e => {
      const a = String(e.community_a)
      const b = String(e.community_b)
      if (graph.hasNode(a) && graph.hasNode(b) && !graph.hasEdge(a, b)) {
        graph.addEdge(a, b, {
          size: 0.3 + 2.5 * (e.weight / maxWeight),
          color: token('--diag-line'),
        })
      }
    })

    loadGraph(graph)
  }, [result, loadGraph, t])

  useEffect(() => {
    const handler = (e: { node: string }) => onNodeClick(Number(e.node))
    sigma.on('clickNode', handler)
    return () => {
      sigma.removeListener('clickNode', handler)
    }
  }, [sigma, onNodeClick])

  return null
}

function CommunityDetailPanel({ corpusId, communityId, algorithm, edgeType, realmId, onClose }: {
  corpusId: string
  communityId: number
  algorithm: 'leiden' | 'louvain'
  edgeType: 'lexical' | 'semantic'
  realmId?: string | null
  onClose: () => void
}) {
  const { t } = useTranslation()
  const { data, isLoading } = useQuery({
    queryKey: ['graph-community-detail', corpusId, communityId, algorithm, edgeType, realmId],
    queryFn: () => api.corpus.graphCommunityDetail(corpusId, communityId, algorithm, edgeType, realmId),
  })

  return (
    /* A community's detail is a section under the graph with a ruled heading,
       rather than a card on top of a card. The fragment table keeps each
       structural path on one truncated line: a path can run past a hundred
       characters, and wrapping stretched every row to three, which made any
       pattern across thirty fragments invisible — and the pattern is exactly
       what the column is read for. */
    <section className="graph-detail">
      <div className="section-rule">
        <h3 className="section-title">
          {t('graphCommunityView.communityTitle', { id: communityId })}
        </h3>
        <span className="section-meta">
          {data && t('graphCommunityView.subgraphSummary', { nodes: data.nodes.length, edges: data.edges.length })}
        </span>
        <button type="button" className="icon-btn" onClick={onClose} aria-label={t('graphCommunityView.close')}>
          <X size={14} />
        </button>
      </div>

      {isLoading && <p className="loading">{t('graphCommunityView.loadingSubgraph')}</p>}
      {data && (
        <>
          <table>
            <thead>
              <tr>
                <th className="col-130">chunk_id</th>
                <th>{t('graphCommunityView.path')}</th>
              </tr>
            </thead>
            <tbody>
              {data.nodes.slice(0, 30).map(n => (
                <tr key={n.chunk_id}>
                  <td><code className="mono-sm">{n.chunk_id.slice(0, 12)}…</code></td>
                  <td className="path-cell" title={n.path || undefined}>{n.path || '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {data.nodes.length > 30 && (
            <p className="hint-line">{t('graphCommunityView.subgraphTruncated')}</p>
          )}
        </>
      )}
    </section>
  )
}

export default function GraphCommunityView({ corpusId, realmId }: { corpusId: string; realmId?: string | null }) {
  const { t } = useTranslation()
  const [algorithm, setAlgorithm] = useState<'leiden' | 'louvain'>('leiden')
  // Two independent signals: "lexical" = RELATED (shared-keyword),
  // "semantic" = RELATED_SEMANTIC (embedding cosine). Kept as a separate
  // choice, not auto-merged, so the user can directly compare whether
  // community structure differs between the two.
  const [edgeType, setEdgeType] = useState<'lexical' | 'semantic'>('lexical')
  const [selected, setSelected] = useState<number | null>(null)

  const { data, isLoading, error, refetch, isFetching } = useQuery({
    queryKey: ['graph-communities', corpusId, algorithm, edgeType, realmId],
    queryFn: () => api.corpus.graphCommunities(corpusId, algorithm, edgeType, realmId),
    staleTime: Infinity, // GDS run is not cheap to repeat on every render — explicit refetch only
  })

  const sortedCommunities = useMemo(
    () => data ? [...data.communities].sort((a, b) => b.size - a.size) : [],
    [data],
  )

  return (
    <div className="section">
      <div className="section-rule flush">
        <h2 className="section-title">{t('graphCommunityView.heading')}</h2>
        <span className="section-meta">{t('graphCommunityView.scopeMeta')}</span>
        <button className="btn btn-sm push" onClick={() => refetch()} disabled={isFetching}>
          {isFetching ? t('graphCommunityView.computing') : t('graphCommunityView.recompute')}
        </button>
      </div>
      <div className="form-grid">
        <div className="form-group">
          <label>{t('graphCommunityView.algorithm')}</label>
          <SelectBox value={algorithm} onChange={e => setAlgorithm(e.target.value as 'leiden' | 'louvain')}
                     aria-label={t('graphCommunityView.algorithm')}>
            <option value="leiden">Leiden</option>
            <option value="louvain">Louvain</option>
          </SelectBox>
        </div>
        <div className="form-group">
          <label>{t('graphCommunityView.edgeType')}</label>
          <SelectBox value={edgeType} onChange={e => setEdgeType(e.target.value as 'lexical' | 'semantic')}
                     aria-label={t('graphCommunityView.edgeType')}>
            <option value="lexical">{t('graphCommunityView.edgeTypeLexical')}</option>
            <option value="semantic">{t('graphCommunityView.edgeTypeSemantic')}</option>
          </SelectBox>
        </div>
      </div>

      {/* Choosing a corpus on this tab does not narrow the graph: Neo4j
          nodes carry no corpus_id property, so clustering always runs over
          the realm's whole graph. The corpus decides only where the source
          labels come from. Staying quiet about that would pass the whole
          realm's numbers off as the selected corpus's. */}
      {data?.realm_scoped !== false && (
        <p className="hint-line mt-4">{t('graphCommunityView.scopeNote')}</p>
      )}

      {isLoading && <p className="text-muted">{t('graphCommunityView.computingDetection')}</p>}
      {error && <p className="badge badge-danger">{t('graphCommunityView.unavailable', { error: String(error) })}</p>}

      {/* A realm with no Neo4j of its own has no graph of its own: the one
          named by the environment is shared, and its communities belong to
          another realm. Showing them under this realm's name would be untrue,
          so the screen declines instead of adding a caveat. */}
      {data && data.realm_scoped === false && (
        <p className="conn-status conn-status-warn">{t('graphCommunityView.notScoped')}</p>
      )}
      {data && data.realm_scoped !== false && data.community_count === 0 && (
        <p className="empty">{t('graphCommunityView.noEdges')}</p>
      )}
      {data && data.realm_scoped !== false && data.community_count > 0 && (
        <>
          <div className="stat-band">
            <div className="stat-cell">
              <div className="eyebrow">{t('graphCommunityView.communities')}</div>
              <div className="metric-val">{data.community_count}</div>
            </div>
            <div className="stat-cell">
              <div className="eyebrow">{t('graphCommunityView.modularity')}</div>
              <div className={`metric-val tone-${(data.modularity ?? 0) >= 0.4 ? 'ok' : 'warn'}`}>
                {data.modularity?.toFixed(3) ?? '—'}
              </div>
              <div className="stat-sub">{t('graphCommunityView.modularityHint')}</div>
            </div>
            <div className="stat-cell">
              <div className="eyebrow">{t('graphCommunityView.nonTrivial')}</div>
              <div className="metric-val">{sortedCommunities.filter(c => c.size > 1).length}</div>
              <div className="stat-sub">{t('graphCommunityView.nonTrivialHint')}</div>
            </div>
          </div>

          {/* The legend runs on tokens. The literals that used to sit here,
              #1d9e75/#ba7517/#d85a30, changed with neither palette nor theme,
              while the graph's own nodes take the same values from code: the
              two would have drifted apart silently. */}
          <div className="graph-legend">
            {([['singleSource', 'ok'], ['dominant', 'ok2'], ['mixed', 'warn'], ['heavilyMixed', 'bad']] as const).map(([key, tone]) => (
              <span key={key}>
                <i className={`legend-dot legend-${tone}`} aria-hidden="true" />
                {t(`graphCommunityView.legend.${key}`)}
              </span>
            ))}
          </div>

          <div className="graph-box">
            <SigmaContainer className="graph-sigma"
              settings={{ renderLabels: false, allowInvalidContainer: true }}>
              <GraphLoader result={data} onNodeClick={setSelected} />
            </SigmaContainer>
          </div>
          <p className="text-muted sm mt-4">
            {t('graphCommunityView.graphHint')}
            {' '}{t('graphCommunityView.singletonHint', { count: data.communities.filter(c => c.size === 1).length, total: data.community_count })}
          </p>

          {selected !== null && (
            <CommunityDetailPanel corpusId={corpusId} communityId={selected} algorithm={algorithm} edgeType={edgeType} realmId={realmId} onClose={() => setSelected(null)} />
          )}
        </>
      )}
    </div>
  )
}
