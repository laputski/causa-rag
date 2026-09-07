import { useState } from 'react'
import { Trans, useTranslation } from 'react-i18next'
import { DraftNote } from '../components/DraftNotice'
import { Link, useSearchParams } from 'react-router-dom'
import {
  Layers, Zap, Database, Search, Cpu, Bot, BarChart2,
  AlertTriangle, HelpCircle, Scissors, GitMerge, Clock,
  CheckCircle, XCircle, Info, ChevronRight, Target, UserCog, ShieldAlert,
  SlidersHorizontal, Share2, ArrowUpDown, ShieldCheck, Boxes, Plug, GitCompare, Scale, Puzzle,
  Command,
  Wrench, ClipboardList, ChartScatter, SatelliteDish, Package, Sparkles,
} from 'lucide-react'

/* ── Platform service topology — high-level view of all services & their roles ── */
function PlatformServiceDiagram() {
  const { t } = useTranslation()
  const R = (x: number, y: number, w: number, h: number, color: string, dashed = false) => (
    <rect key={`r${x}-${y}`} x={x} y={y} width={w} height={h} rx="6"
      fill={color} fillOpacity="var(--diag-fill-opacity)" stroke={color} strokeWidth="1.4"
      strokeDasharray={dashed ? '5,3' : undefined} />
  )
  const T = (x: number, y: number, t: string, color: string, sz = 9.5) => (
    <text key={`t${x}-${y}`} x={x} y={y} textAnchor="middle" fontSize={sz}
      fill={color} fontWeight="700" fontFamily="system-ui,sans-serif">{t}</text>
  )
  const N = (x: number, y: number, t: string) => (
    <text key={`n${t}`} x={x} y={y} textAnchor="middle" fontSize="8"
      fill="var(--diag-muted)" fontFamily="system-ui,sans-serif">{t}</text>
  )
  const CL = (x: number, t: string) => (
    <text key={`cl${t}`} x={x} y={13} textAnchor="middle" fontSize="7.5" fill="var(--diag-line)"
      fontFamily="system-ui,sans-serif" letterSpacing="0.5">{t}</text>
  )
  // straight arrow
  const A = (x1: number, y1: number, x2: number, y2: number, dashed = false) => (
    <line key={`a${x1}${y1}${x2}${y2}`} x1={x1} y1={y1} x2={x2} y2={y2}
      stroke="var(--diag-line)" strokeWidth="1.3" strokeDasharray={dashed ? '3,2' : undefined}
      markerEnd="url(#svc-arr)" />
  )
  // elbow arrow: goes horizontal to xMid, then vertical to y2, then horizontal to x2
  const E = (x1: number, y1: number, xMid: number, y2: number, x2: number) => (
    <polyline key={`e${x1}${y1}${x2}${y2}`}
      points={`${x1},${y1} ${xMid},${y1} ${xMid},${y2} ${x2},${y2}`}
      fill="none" stroke="var(--diag-line)" strokeWidth="1.3" markerEnd="url(#svc-arr)" />
  )

  // layout constants
  const GX = 185, GW = 150, GCX = GX + GW / 2  // gateway: x=185..335, centre=260
  const AIX = 375, AIW = 148, AIC = AIX + AIW / 2  // AI col: x=375..523, centre=449
  const OX  = 550, OW  = 145, OC  = OX + OW / 2   // ops col: x=550..695, centre=622
  const ELBOW = OX - 15  // x=535, right of AI col (523), left of ops col (550)

  return (
    <DiagFrame
      id="svc" viewBox="0 0 755 305"
      caption={<>
        {t('guidePage.serviceDiagram.captionPart1')} <em>{t('guidePage.serviceDiagram.captionEm')}</em>{' '}
        {t('guidePage.serviceDiagram.captionPart2')} <strong>{t('guidePage.serviceDiagram.captionStrong')}</strong>{' '}
        {t('guidePage.serviceDiagram.captionPart3')}
      </>}
    >
        {CL(72,  t('guidePage.serviceDiagram.client'))}
        {CL(GCX, 'API GATEWAY')}
        {CL(AIC, t('guidePage.serviceDiagram.aiSearch'))}
        {CL(OC,  t('guidePage.serviceDiagram.dataOps'))}
        {/* subtitle under AI col — clarify this is in-process only */}
        <text x={AIC} y={21} textAnchor="middle" fontSize="6.5" fill="var(--diag-muted)"
          fontFamily="system-ui,sans-serif">{t('guidePage.serviceDiagram.inProcessOnly')}</text>

        {/* ── client ── */}
        {R(8, 22, 128, 34, 'var(--diag-violet)')}
        {T(72, 36, 'React UI', 'var(--diag-violet)')}
        {N(72, 48, 'Vite · TypeScript · React Query')}

        {R(8, 70, 128, 34, 'var(--diag-blue)')}
        {T(72, 84, 'Browser / Dev', 'var(--diag-blue)')}
        {N(72, 96, 'REST · WebSocket')}

        {/* ── API Gateway (tall) ── */}
        {R(GX, 20, GW, 186, 'var(--diag-cyan)')}
        {T(GCX, 36,  'FastAPI :8081',       'var(--diag-cyan)')}
        {N(GCX, 51,  'Pydantic v2 · asyncio')}
        {N(GCX, 64,  'structlog · httpx')}
        {N(GCX, 80,  'ExperimentRunner')}
        {N(GCX, 93,  'HttpPipeline')}
        {N(GCX, 106, 'CompositeEvaluator')}
        {N(GCX, 119, 'ComponentRegistry')}
        {N(GCX, 135, 'IngestCLI · CorpusRouter')}
        {N(GCX, 148, 'WebSocket /progress')}
        {N(GCX, 161, 'ExternalRagsRouter')}
        {N(GCX, 174, 'DomainPacksRouter')}

        {/* client → gateway: horizontal-first elbow */}
        {A(136, 38, GX, 85)}
        {A(136, 87, GX, 97)}

        {/* ── AI / search ── */}
        {R(AIX, 22, AIW, 30, 'var(--diag-cyan)')}
        {T(AIC, 34,  'BGE-M3',          'var(--diag-cyan)')}
        {N(AIC, 45,  'in-process · 1024-dim L2')}

        {R(AIX, 62, AIW, 30, 'var(--diag-green)')}
        {T(AIC, 74,  'Qdrant :6333',    'var(--diag-green)')}
        {N(AIC, 85,  'dense cosine')}

        {R(AIX, 102, AIW, 30, 'var(--diag-orange)')}
        {T(AIC, 114, 'OpenSearch :9200','var(--diag-orange)')}
        {N(AIC, 125, 'BM25 sparse')}

        {R(AIX, 142, AIW, 30, 'var(--diag-yellow)')}
        {T(AIC, 154, 'Neo4j :7687',     'var(--diag-yellow)')}
        {N(AIC, 165, 'knowledge graph')}

        {R(AIX, 182, AIW, 30, 'var(--diag-yellow)')}
        {T(AIC, 194, 'Ollama :11434',   'var(--diag-yellow)')}
        {N(AIC, 205, 'LLM · qwen3 / gemma')}

        {/* gateway → AI: short horizontal arrows from gateway right edge */}
        {A(GX + GW, 55,  AIX, 37)}
        {A(GX + GW, 78,  AIX, 77)}
        {A(GX + GW, 108, AIX, 117)}
        {A(GX + GW, 138, AIX, 157)}
        {A(GX + GW, 162, AIX, 197)}

        {/* ── data / ops ── */}
        {R(OX, 36, OW, 30, 'var(--diag-pink)')}
        {T(OC, 48,  'MongoDB :27017',  'var(--diag-pink)')}
        {N(OC, 59,  t('guidePage.serviceDiagram.mongoRunsPrompts'))}

        {R(OX, 76, OW, 30, 'var(--diag-blue)')}
        {T(OC, 88,  'Docker Compose',  'var(--diag-blue)')}
        {N(OC, 99,  'Helm · ArgoCD · TF')}

        {/* gateway → ops: elbow connectors that go right past AI column */}
        {E(GX + GW, 90, ELBOW, 51, OX)}
        {E(GX + GW, 120, ELBOW, 91, OX)}

        {/* ── External RAG (dashed) — left box: HTTP contract ── */}
        {R(GX, 232, GW + 65, 54, 'var(--diag-cyan)', true)}
        {T(GX + (GW + 65) / 2, 248, t('guidePage.serviceDiagram.externalRagLabel'), 'var(--diag-cyan)', 9)}
        {N(GX + (GW + 65) / 2, 261, 'POST / → {answer, sources[]}')}
        {N(GX + (GW + 65) / 2, 273, 'POST /retrieve → {sources[]}')}

        {/* ── External RAG own infra — right box (dashed, muted) ── */}
        {R(GX + GW + 80, 232, 130, 54, 'var(--diag-line)', true)}
        {T(GX + GW + 80 + 65, 248, t('guidePage.serviceDiagram.ownInfraLabel'), 'var(--diag-line)', 8)}
        {N(GX + GW + 80 + 65, 261, t('guidePage.serviceDiagram.anyDbVectorStore'))}
        {N(GX + GW + 80 + 65, 273, t('guidePage.serviceDiagram.platformUnaware'))}
        {/* connector between ext-rag and its own infra */}
        <line x1={GX + GW + 65} y1={259} x2={GX + GW + 80} y2={259}
          stroke="var(--diag-surface)" strokeWidth="1" strokeDasharray="3,2" />

        {/* gateway ↔ external RAG */}
        {A(GCX - 10, 206, GCX - 10, 232)}
        {A(GCX + 10, 232, GCX + 10, 206, true)}
        <text x={GCX + 55} y={216} fontSize="7.5" fill="var(--diag-line)" fontFamily="system-ui,sans-serif">corpus_id</text>
        <text x={GCX + 55} y={226} fontSize="7.5" fill="var(--diag-line)" fontFamily="system-ui,sans-serif">pipeline_id</text>
    </DiagFrame>
  )
}

/* ── Code layer diagram — module structure ── */
function CodeLayerDiagram() {
  const { t } = useTranslation()
  const layer = (y: number, h: number, title: string, color: string, items: string[]) => (
    <g key={title}>
      <rect x={10} y={y} width={740} height={h} rx="5"
        fill={color} fillOpacity="var(--diag-fill-opacity)" stroke={color} strokeWidth="1.3" />
      <text x={22} y={y + 16} fontSize="9.5" fill={color} fontWeight="700"
        fontFamily="system-ui,monospace">{title}</text>
      {items.map((item, i) => (
        <text key={i} x={22 + (i % 2) * 370} y={y + 30 + Math.floor(i / 2) * 14}
          fontSize="8" fill="var(--diag-muted)" fontFamily="system-ui,monospace">{item}</text>
      ))}
    </g>
  )
  const connector = (y: number) => (
    <line key={`c${y}`} x1={380} y1={y} x2={380} y2={y + 10}
      stroke="var(--diag-line)" strokeWidth="1.1" markerEnd="url(#layer-arr)" />
  )
  return (
    <DiagFrame id="layer" viewBox="0 0 760 310" className="diag-spaced-top" caption={<><code className="inline-code">core/</code> {t('guidePage.codeLayer.note1')} <code className="inline-code">adapters/</code> {t('guidePage.codeLayer.note2')} <code className="inline-code">domain_packs/</code> {t('guidePage.codeLayer.note3')}</>}>{layer(4,  34, 'ui/',          'var(--diag-violet)', [
          'React + TypeScript + Vite · React Query · Lucide',
          'ExperimentsPage · RunPage · RealmResourcesPage · GuidePage ...',
        ])}
        {connector(38)}
        {layer(48, 40, 'services/',    'var(--diag-blue)', [
          'api_gateway/main.py    FastAPI + routers (experiments, external_rags, corpus ...)',
          'ingestion/cli.py       chunk → embed → Qdrant / OpenSearch / Neo4j',
          t('guidePage.codeLayer.referenceRagServerItem'),
        ])}
        {connector(88)}
        {layer(98, 84, 'core/',        'var(--diag-cyan)', [
          'models.py         Pydantic v2: Answer, SourceRef, StageTrace, ExperimentConfig ...',
          'pipeline.py       NaivePipeline / ConfigurablePipeline',
          'experiment/       ExperimentConfig (config_hash) + ExperimentRunner',
          'eval/             retrieval, semantic, funnel, regression, answerability',
          'retrieval/        HybridRetriever (RRF/weighted) · GraphHybridRetriever',
          'sdk.py            wrap_retriever / wrap_generator (white-box in-process)',
        ])}
        {connector(182)}
        {layer(192, 70, 'adapters/',    'var(--diag-green)', [
          'bge_m3.py            BGE-M3 multilingual embedder (USE_REAL_BGE_M3=true)',
          'qdrant.py            QdrantRetriever — dense cosine, namespace-per-strategy',
          'opensearch.py        OpenSearchRetriever — BM25 sparse, ru/be analyzers',
          'reranker.py          CrossEncoderReranker (optional [reranker] extra)',
          'http_pipeline.py     HttpPipeline — tier-1 native · tier-2 JSONPath',
          'jsonpath_mapping.py  {{query}}/{{corpus_id}} template → request body',
        ])}
        {connector(262)}
        {layer(272, 34, 'domain_packs/',  'var(--diag-orange)', [
          'manuals/    routing policy · question types · masks · refusal · scorer',
          'generic_qa/ demo pack — minimal implementation of all hooks',
        ])}
      </DiagFrame>
  )
}

/* ── Architecture SVG diagram ──
 * pipeline_id picks ONE retrieval backend combo, not all three at once:
 * naive=Qdrant only; hybrid_rrf/hybrid_weighted=Qdrant+OpenSearch; graph=Neo4j+Qdrant
 * (graph never touches OpenSearch — it's its own pipeline, not a hybrid variant). */
function ArchDiagram() {
  const { t } = useTranslation()
  const nodes = [
    { x: 20,  y: 80,  w: 100, label: t('guidePage.archDiagram.corpus'), sub: '.txt / .md',          color: 'var(--diag-blue)' },
    { x: 150, y: 80,  w: 100, label: 'Chunker',     sub: 'fixed / sent / para', color: 'var(--diag-violet)' },
    { x: 280, y: 80,  w: 100, label: 'BGE-M3',      sub: '1024-dim L2',        color: 'var(--diag-cyan)' },
    { x: 410, y: 30,  w: 110, label: 'Qdrant',      sub: 'dense cosine',       color: 'var(--diag-green)' },
    { x: 410, y: 80,  w: 110, label: 'OpenSearch',  sub: 'BM25 sparse',        color: 'var(--diag-orange)' },
    { x: 410, y: 130, w: 110, label: 'Neo4j',       sub: t('guidePage.archDiagram.graphOptional'), color: 'var(--diag-yellow)' },
    { x: 560, y: 80,  w: 110, label: 'Merge',       sub: t('guidePage.archDiagram.dependsOnPipelineId'), color: 'var(--diag-pink)' },
    { x: 700, y: 80,  w: 100, label: 'LLM',         sub: 'Ollama qwen3:8b',    color: 'var(--diag-yellow)' },
    { x: 830, y: 80,  w: 80,  label: t('guidePage.archDiagram.answer'), sub: '',                   color: 'var(--diag-green)' },
  ]
  return (
    <DiagFrame id="arch" viewBox="0 0 930 180" caption={<><code className="inline-code">pipeline_id</code> {t('guidePage.archDiagram.captionIntro')} <code className="inline-code">naive</code> {t('guidePage.archDiagram.captionNaive')}{' '}
        <code className="inline-code">hybrid_rrf</code>/<code className="inline-code">hybrid_weighted</code> {t('guidePage.archDiagram.captionHybrid')}{' '}
        <code className="inline-code">graph</code> {t('guidePage.archDiagram.captionGraph')}</>}>{/* straight arrows 0→1→2 */}
        {[0,1,2].map(i => (
          <line key={`l${i}`} x1={nodes[i].x+nodes[i].w} y1={95} x2={nodes[i+1].x} y2={95}
            stroke="var(--diag-line)" strokeWidth="1.5" markerEnd="url(#arch-arr)" />
        ))}
        {/* 2 → Qdrant / OpenSearch / Neo4j */}
        <line x1={380} y1={88} x2={410} y2={48}  stroke="var(--diag-line)" strokeWidth="1.5" markerEnd="url(#arch-arr)" />
        <line x1={380} y1={92} x2={410} y2={92}  stroke="var(--diag-line)" strokeWidth="1.5" markerEnd="url(#arch-arr)" />
        <line x1={380} y1={98} x2={410} y2={130} stroke="var(--diag-line)" strokeWidth="1.5" markerEnd="url(#arch-arr)" />
        {/* Qdrant / OpenSearch / Neo4j → Merge */}
        <line x1={520} y1={48}  x2={560} y2={72}  stroke="var(--diag-line)" strokeWidth="1.5" markerEnd="url(#arch-arr)" />
        <line x1={520} y1={92}  x2={560} y2={92}  stroke="var(--diag-line)" strokeWidth="1.5" markerEnd="url(#arch-arr)" />
        <line x1={520} y1={130} x2={560} y2={96} stroke="var(--diag-line)" strokeWidth="1.5" markerEnd="url(#arch-arr)" />
        {/* Merge → LLM → Answer */}
        {[6,7].map(i => (
          <line key={`l${i}r`} x1={nodes[i].x+nodes[i].w} y1={95} x2={nodes[i+1].x} y2={95}
            stroke="var(--diag-line)" strokeWidth="1.5" markerEnd="url(#arch-arr)" />
        ))}
        {/* nodes */}
        {nodes.map((n, i) => (
          <g key={i}>
            <rect x={n.x} y={n.y-18} width={n.w} height={36} rx="6"
              fill={n.color} fillOpacity="var(--diag-fill-opacity)" stroke={n.color} strokeWidth="1.5" />
            <text x={n.x+n.w/2} y={n.y-2} textAnchor="middle" fontSize="10.5" fill={n.color}
              fontWeight="700" fontFamily="system-ui,sans-serif">{n.label}</text>
            {n.sub && <text x={n.x+n.w/2} y={n.y+11} textAnchor="middle" fontSize="8.5" fill="var(--diag-line)"
              fontFamily="system-ui,sans-serif">{n.sub}</text>}
          </g>
        ))}
      </DiagFrame>
  )
}

/* ── Metric bar ── */
function MetricRow({ name, formula, good, warn, note }: {
  name: string; formula: string; good: string; warn: string; note: string
}) {
  return (
    <div className="guide-metric-row">
      <div className="guide-metric-name"><code className="inline-code">{name}</code></div>
      <div className="guide-metric-formula"><code className="inline-code">{formula}</code></div>
      <div className="guide-metric-thresholds">
        <span className="guide-threshold ok"><CheckCircle size={11} />{good}</span>
        <span className="guide-threshold warn"><AlertTriangle size={11} />{warn}</span>
      </div>
      <div className="guide-metric-note">{note}</div>
    </div>
  )
}

/* ── Problem card ── */
function ProblemCard({ title, symptom, cause, fix }: {
  title: string; symptom: string; cause: string; fix: string
}) {
  const { t } = useTranslation()
  const [open, setOpen] = useState(false)
  return (
    <div className="guide-problem-card">
      <button className="guide-problem-header" onClick={() => setOpen(o => !o)}>
        <AlertTriangle size={14} className="guide-problem-icon" />
        <span>{title}</span>
        <ChevronRight size={13} className={`guide-chevron ${open ? 'open' : ''}`} />
      </button>
      {open && (
        <div className="guide-problem-body">
          <div className="guide-problem-row"><span className="guide-problem-label">{t('guidePage.problemCard.symptom')}</span><span>{symptom}</span></div>
          <div className="guide-problem-row"><span className="guide-problem-label">{t('guidePage.problemCard.cause')}</span><span>{cause}</span></div>
          <div className="guide-problem-row fix"><span className="guide-problem-label">{t('guidePage.problemCard.fix')}</span><span>{fix}</span></div>
        </div>
      )}
    </div>
  )
}

/* ── Chunking table ── */
function ChunkingTable() {
  const { t } = useTranslation()
  const rows = [
    { id: 'fixed',           when: t('guidePage.chunkingTable.fixed.when'),           pro: t('guidePage.chunkingTable.fixed.pro'),           con: t('guidePage.chunkingTable.fixed.con') },
    { id: 'structure_aware', when: t('guidePage.chunkingTable.structureAware.when'),  pro: t('guidePage.chunkingTable.structureAware.pro'),  con: t('guidePage.chunkingTable.structureAware.con') },
    { id: 'sentence',        when: t('guidePage.chunkingTable.sentence.when'),        pro: t('guidePage.chunkingTable.sentence.pro'),        con: t('guidePage.chunkingTable.sentence.con') },
    { id: 'paragraph',       when: t('guidePage.chunkingTable.paragraph.when'),       pro: t('guidePage.chunkingTable.paragraph.pro'),       con: t('guidePage.chunkingTable.paragraph.con') },
  ]
  return (
    <div className="guide-chunk-table">
      {rows.map(r => (
        <div key={r.id} className="guide-chunk-row">
          <code className="inline-code guide-chunk-id">{r.id}</code>
          <div className="guide-chunk-when"><Info size={10} />{r.when}</div>
          <div className="guide-chunk-pro"><CheckCircle size={10} />{r.pro}</div>
          <div className="guide-chunk-con"><XCircle size={10} />{r.con}</div>
        </div>
      ))}
    </div>
  )
}

/* ── Funnel diagnosis diagram ──
 * Mirrors core/eval/funnel.py:diagnose_question decision order exactly —
 * the combined check (suspected_ungrounded_answer) runs BEFORE isolated
 * per-layer checks, on purpose (see Phase 1 finding in the callout above). */
function FunnelDiagram() {
  const { t } = useTranslation()
  const decision = (y: number, label: string) => (
    <g>
      <rect x={40} y={y} width={280} height={34} rx="6" fill="var(--diag-surface)" stroke="var(--diag-line)" strokeWidth="1.2" strokeDasharray="3,2" />
      <text x={180} y={y + 21} textAnchor="middle" fontSize="10" fill="var(--diag-text)" fontFamily="system-ui,sans-serif">{label}</text>
    </g>
  )
  const outcome = (y: number, label: string, color: string) => (
    <g>
      <rect x={400} y={y} width={260} height={34} rx="6" fill={color} fillOpacity="var(--diag-fill-opacity)" stroke={color} strokeWidth="1.5" />
      <text x={530} y={y + 21} textAnchor="middle" fontSize="10.5" fill={color} fontWeight="700" fontFamily="system-ui,sans-serif">{label}</text>
    </g>
  )
  const vArrow = (x: number, y1: number, y2: number, dashed = false) => (
    <line x1={x} y1={y1} x2={x} y2={y2} stroke="var(--diag-line)" strokeWidth="1.5" markerEnd="url(#funnel-arr)" strokeDasharray={dashed ? '3,2' : undefined} />
  )
  const branchArrow = (y: number, label: string, color: string) => (
    <g>
      <line x1={320} y1={y} x2={400} y2={y} stroke={color} strokeWidth="1.5" markerEnd="url(#funnel-arr)" />
      <text x={360} y={y - 6} textAnchor="middle" fontSize="9" fill={color} fontFamily="system-ui,sans-serif">{label}</text>
    </g>
  )
  // "no" from a decision falls through to an outcome one row below — down,
  // then right, with the label on the vertical leg (kept clear of the
  // decision/outcome box borders on both sides).
  const fallThrough = (yFrom: number, yTo: number, label: string, color: string) => (
    <g>
      <line x1={180} y1={yFrom} x2={180} y2={yTo} stroke={color} strokeWidth="1.5" />
      <line x1={180} y1={yTo} x2={400} y2={yTo} stroke={color} strokeWidth="1.5" markerEnd="url(#funnel-arr)" />
      <text x={196} y={(yFrom + yTo) / 2 + 3} fontSize="9" fill={color} fontFamily="system-ui,sans-serif">{label}</text>
    </g>
  )
  return (
    <DiagFrame id="funnel" viewBox="0 0 700 480" caption={<>{t('guidePage.funnelDiagram.captionPrefix')}<code className="inline-code">good</code>/<code className="inline-code">warn</code>{t('guidePage.funnelDiagram.captionSuffix')}</>}>{decision(10,  'answerability == "answerable" ?')}
        {branchArrow(27, t('guidePage.funnelDiagram.no'), 'var(--diag-muted)')}
        {outcome(10, 'not_applicable', 'var(--diag-muted)')}
        {vArrow(180, 44, 74)}

        {decision(74, t('guidePage.funnelDiagram.decisionUngrounded'))}
        {branchArrow(91, t('guidePage.funnelDiagram.yes'), 'var(--diag-red)')}
        {outcome(74, 'suspected_ungrounded_answer', 'var(--diag-red)')}
        {vArrow(180, 108, 138)}

        {decision(138, 'recall < 0.3 ?')}
        <text x={196} y={194} fontSize="9" fill="var(--diag-orange)" fontFamily="system-ui,sans-serif">{t('guidePage.funnelDiagram.yes')}</text>
        {vArrow(180, 172, 210)}
        {/* "no" (recall ≥ 0.3) bypasses the rerank/retrieval sub-decision
            entirely and skips straight to the grounded/similarity check —
            routed left of the boxes so it never crosses a border. */}
        <path d="M 100 172 L 100 185 L 16 198 L 16 363 L 40 363" fill="none" stroke="var(--diag-muted)" strokeWidth="1.5" strokeDasharray="3,2" markerEnd="url(#funnel-arr)" />
        <text x={20} y={280} fontSize="9" fill="var(--diag-muted)" fontFamily="system-ui,sans-serif" transform="rotate(-90 20 280)">{t('guidePage.funnelDiagram.no')}</text>

        {decision(210, t('guidePage.funnelDiagram.decisionRerank'))}
        {branchArrow(227, t('guidePage.funnelDiagram.yes'), 'var(--diag-yellow)')}
        {outcome(210, t('guidePage.funnelDiagram.outcomeRerankDropped'), 'var(--diag-yellow)')}
        {fallThrough(244, 297, t('guidePage.funnelDiagram.no'), 'var(--diag-orange)')}
        {outcome(280, t('guidePage.funnelDiagram.outcomeRetrievalMissed'), 'var(--diag-orange)')}

        {decision(346, t('guidePage.funnelDiagram.decisionGeneration'))}
        {branchArrow(363, t('guidePage.funnelDiagram.yes'), 'var(--diag-violet)')}
        {outcome(346, 'generation', 'var(--diag-violet)')}
        {fallThrough(380, 433, t('guidePage.funnelDiagram.no'), 'var(--diag-green)')}
        {outcome(416, 'ok', 'var(--diag-green)')}
      </DiagFrame>
  )
}

/* ── Corpus namespace diagram ──
 * Qdrant/OpenSearch are partitioned per corpus_id (separate collections/
 * indices); Neo4j is NOT — one shared graph regardless of corpus_id. */
function CorpusNamespaceDiagram() {
  const { t } = useTranslation()
  const corpusBox = (x: number, name: string, color: string) => (
    <g>
      <rect x={x} y={10} width={200} height="120" rx="8" fill={color} fillOpacity="var(--diag-fill-opacity)" stroke={color} strokeWidth="1.5" strokeDasharray="4,3" />
      <text x={x + 100} y={28} textAnchor="middle" fontSize="11" fill={color} fontWeight="700" fontFamily="system-ui,sans-serif">corpus_id={name}</text>
      <rect x={x + 15} y={40} width={170} height="30" rx="5" fill="var(--diag-green)" fillOpacity="var(--diag-fill-opacity)" stroke="var(--diag-green)" strokeWidth="1.2" />
      <text x={x + 100} y={59} textAnchor="middle" fontSize="9.5" fill="var(--diag-green)" fontFamily="system-ui,sans-serif">Qdrant collection</text>
      <rect x={x + 15} y={78} width={170} height="30" rx="5" fill="var(--diag-orange)" fillOpacity="var(--diag-fill-opacity)" stroke="var(--diag-orange)" strokeWidth="1.2" />
      <text x={x + 100} y={97} textAnchor="middle" fontSize="9.5" fill="var(--diag-orange)" fontFamily="system-ui,sans-serif">OpenSearch index</text>
    </g>
  )
  return (
    <DiagFrame id="corpus" viewBox="0 0 480 230">
        {corpusBox(10,  'default',    'var(--diag-blue)')}
        {corpusBox(270, 'handbook', 'var(--diag-violet)')}

        {/* both -> shared Neo4j */}
        <line x1={110} y1={130} x2={240} y2={185} stroke="var(--diag-yellow)" strokeWidth="1.5" markerEnd="url(#corpus-arr)" />
        <line x1={370} y1={130} x2={240} y2={185} stroke="var(--diag-yellow)" strokeWidth="1.5" markerEnd="url(#corpus-arr)" />
        <rect x={140} y="190" width={200} height="34" rx="6" fill="var(--diag-yellow)" fillOpacity="var(--diag-fill-opacity)" stroke="var(--diag-yellow)" strokeWidth="1.5" />
        <text x={240} y="207" textAnchor="middle" fontSize="10.5" fill="var(--diag-yellow)" fontWeight="700" fontFamily="system-ui,sans-serif">{t('guidePage.corpusNamespaceDiagram.sharedGraph')}</text>
        <text x={240} y="219" textAnchor="middle" fontSize="8" fill="var(--diag-muted)" fontFamily="system-ui,sans-serif">{t('guidePage.corpusNamespaceDiagram.noCorpusIdField')}</text>
      </DiagFrame>
  )
}

/* ── Graph community detection (GDS Leiden/Louvain) ── */
function GraphCommunityDiagram() {
  const { t } = useTranslation()
  const legend = [
    // The same tokens as the legend on the graph page itself. Hex literals
    // changed with neither palette nor theme, so the guide described colours
    // that were not on the screen.
    { color: 'var(--diag-green)', label: t('guidePage.graphCommunityDiagram.legendSingleSource') },
    { color: 'var(--diag-cyan)', label: t('guidePage.graphCommunityDiagram.legendDominant') },
    { color: 'var(--diag-orange)', label: t('guidePage.graphCommunityDiagram.legendMixed') },
    { color: 'var(--diag-red)', label: t('guidePage.graphCommunityDiagram.legendHeavilyMixed') },
  ]
  return (
    <DiagFrame id="graph" viewBox="0 0 480 220"><text x={10} y={18} fontSize="10.5" fill="var(--diag-muted)" fontFamily="system-ui,sans-serif">{t('guidePage.graphCommunityDiagram.navPath')}</text>
        <rect x={10} y={28} width={460} height={130} rx="8" fill="var(--diag-surface)" fillOpacity="var(--diag-fill-opacity)" stroke="var(--diag-line)" strokeWidth="1.2" strokeDasharray="3,2" />
        {/* a handful of community nodes scattered, sized by chunk count, colored by mix */}
        {[
          { cx: 90, cy: 70, r: 22, color: 'var(--diag-red)' },
          { cx: 170, cy: 100, r: 16, color: 'var(--diag-orange)' },
          { cx: 230, cy: 60, r: 10, color: 'var(--diag-cyan)' },
          { cx: 300, cy: 95, r: 26, color: 'var(--diag-red)' },
          { cx: 370, cy: 65, r: 8, color: 'var(--diag-green)' },
          { cx: 410, cy: 110, r: 13, color: 'var(--diag-orange)' },
        ].map((n, i) => <circle key={i} cx={n.cx} cy={n.cy} r={n.r} fill={n.color} opacity={0.85} />)}
        <line x1={112} y1={75} x2={154} y2={95} stroke="var(--diag-muted)" strokeWidth="1" opacity={0.5} />
        <line x1={186} y1={95} x2={218} y2={68} stroke="var(--diag-muted)" strokeWidth="1" opacity={0.5} />
        <line x1={186} y1={102} x2={278} y2={97} stroke="var(--diag-muted)" strokeWidth="1.6" opacity={0.6} />
        <line x1={326} y1={88} x2={362} y2={70} stroke="var(--diag-muted)" strokeWidth="1" opacity={0.5} />
        <line x1={326} y1={98} x2={398} y2={108} stroke="var(--diag-muted)" strokeWidth="1" opacity={0.5} />
        <text x={240} y={150} textAnchor="middle" fontSize="9" fill="var(--diag-muted)" fontFamily="system-ui,sans-serif">{t('guidePage.graphCommunityDiagram.nodeEdgeLegend')}</text>

        <g transform="translate(10, 175)">
          {legend.map((l, i) => (
            <g key={i} transform={`translate(${(i % 2) * 230}, ${Math.floor(i / 2) * 22})`}>
              <circle cx={6} cy={6} r={6} fill={l.color} />
              <text x={18} y={10} fontSize="9.5" fill="var(--diag-text)" fontFamily="system-ui,sans-serif">{l.label}</text>
            </g>
          ))}
        </g>
      </DiagFrame>
  )
}

/* ── pipeline_id decision tree ── */
function PipelineDecisionTree() {
  const { t } = useTranslation()
  const q = (x: number, y: number, w: number, label: string) => (
    <g>
      <rect x={x} y={y} width={w} height={32} rx="6" fill="var(--diag-surface)" stroke="var(--diag-line)" strokeWidth="1.2" strokeDasharray="3,2" />
      <text x={x + w / 2} y={y + 20} textAnchor="middle" fontSize="9.5" fill="var(--diag-text)" fontFamily="system-ui,sans-serif">{label}</text>
    </g>
  )
  const leaf = (x: number, y: number, w: number, label: string, color: string) => (
    <g>
      <rect x={x} y={y} width={w} height={32} rx="6" fill={color} fillOpacity="var(--diag-fill-opacity)" stroke={color} strokeWidth="1.5" />
      <text x={x + w / 2} y={y + 20} textAnchor="middle" fontSize="10" fill={color} fontWeight="700" fontFamily="system-ui,sans-serif">{label}</text>
    </g>
  )
  const edge = (x1: number, y1: number, x2: number, y2: number, label: string, color = 'var(--diag-line)') => (
    <g>
      <line x1={x1} y1={y1} x2={x2} y2={y2} stroke={color} strokeWidth="1.5" markerEnd="url(#tree-arr)" />
      <text x={(x1 + x2) / 2 + (x2 > x1 ? 10 : -10)} y={(y1 + y2) / 2 - 4} textAnchor="middle" fontSize="9" fill={color} fontFamily="system-ui,sans-serif">{label}</text>
    </g>
  )
  return (
    <DiagFrame id="tree" viewBox="0 0 700 260">{q(250, 5, 200, t('guidePage.pipelineDecisionTree.needMultiHop'))}
        {edge(350, 37, 620, 90, t('guidePage.pipelineDecisionTree.yes'), 'var(--diag-yellow)')}
        {leaf(530, 90, 140, 'graph', 'var(--diag-yellow)')}
        {edge(350, 37, 170, 90, t('guidePage.pipelineDecisionTree.no'))}

        {q(10, 90, 320, t('guidePage.pipelineDecisionTree.needKeywordSearch'))}
        {edge(170, 122, 150, 165, t('guidePage.pipelineDecisionTree.yes'))}
        {q(10, 165, 280, t('guidePage.pipelineDecisionTree.rrfOrWeighted'))}
        {edge(150, 197, 55, 220, 'rrf', 'var(--diag-green)')}
        {edge(150, 197, 245, 220, 'weighted', 'var(--diag-green)')}
        {leaf(0, 224, 110, 'hybrid_rrf', 'var(--diag-green)')}
        {leaf(170, 224, 150, 'hybrid_weighted', 'var(--diag-green)')}

        {edge(170, 122, 375, 165, t('guidePage.pipelineDecisionTree.no'))}
        {leaf(330, 165, 110, 'naive', 'var(--diag-blue)')}
      </DiagFrame>
  )
}

/* ── Regression/baseline workflow diagram ── */
function RegressionWorkflowDiagram() {
  const { t } = useTranslation()
  const steps = [
    { label: t('guidePage.regressionWorkflowDiagram.runA'),   sub: 'baseline',                                             color: 'var(--diag-blue)' },
    { label: 'PUT /baseline',    sub: t('guidePage.regressionWorkflowDiagram.pin'),         color: 'var(--diag-violet)' },
    { label: t('guidePage.regressionWorkflowDiagram.runB'),   sub: t('guidePage.regressionWorkflowDiagram.newConfig'),     color: 'var(--diag-cyan)' },
    { label: 'regression.py',    sub: t('guidePage.regressionWorkflowDiagram.diffMetrics'),  color: 'var(--diag-yellow)' },
    { label: t('guidePage.regressionWorkflowDiagram.uiBadge'), sub: 'pass / fail 5%',    color: 'var(--diag-green)' },
  ]
  return (
    <DiagFrame id="regr" viewBox="0 0 930 90" className="diag-spaced-bottom">{steps.map((s, i) => {
          const x = 10 + i * 185
          return (
            <g key={s.label}>
              <rect x={x} y={20} width={160} height="50" rx="7" fill={s.color} fillOpacity="var(--diag-fill-opacity)" stroke={s.color} strokeWidth="1.5" />
              <text x={x + 80} y={42} textAnchor="middle" fontSize="11" fill={s.color} fontWeight="700" fontFamily="system-ui,sans-serif">{s.label}</text>
              <text x={x + 80} y={58} textAnchor="middle" fontSize="9" fill="var(--diag-muted)" fontFamily="system-ui,sans-serif">{s.sub}</text>
              {i < steps.length - 1 && (
                <line x1={x + 160} y1={45} x2={x + 185} y2={45} stroke="var(--diag-line)" strokeWidth="1.5" markerEnd="url(#regr-arr)" />
              )}
            </g>
          )
        })}
      </DiagFrame>
  )
}

/* ── External RAG B/C comparison diagram ── */
function ExternalRagDiagram() {
  const { t } = useTranslation()
  const box = (x: number, y: number, w: number, h: number, label: string, sub: string, color: string, dashed = false) => (
    <g>
      <rect x={x} y={y} width={w} height={h} rx="6" fill={color} fillOpacity="var(--diag-fill-opacity)" stroke={color} strokeWidth="1.4" strokeDasharray={dashed ? '3,2' : undefined} />
      <text x={x + w / 2} y={y + h / 2 - (sub ? 4 : -3)} textAnchor="middle" fontSize="10" fill={color} fontWeight="700" fontFamily="system-ui,sans-serif">{label}</text>
      {sub && <text x={x + w / 2} y={y + h / 2 + 11} textAnchor="middle" fontSize="8" fill="var(--diag-muted)" fontFamily="system-ui,sans-serif">{sub}</text>}
    </g>
  )
  const arrow = (x1: number, y1: number, x2: number, y2: number, color = 'var(--diag-line)') => (
    <line x1={x1} y1={y1} x2={x2} y2={y2} stroke={color} strokeWidth="1.4" markerEnd="url(#ext-arr)" />
  )
  // Orthogonal (elbow) routed arrow — goes right/left then up/down then
  // right/left again, so it never cuts diagonally through an unrelated box
  // sitting between source and target (the straight-diagonal version did).
  const elbow = (points: [number, number][], color: string) => (
    <polyline points={points.map(p => p.join(',')).join(' ')} fill="none" stroke={color} strokeWidth="1.4" markerEnd="url(#ext-arr)" />
  )
  return (
    <DiagFrame id="ext" viewBox="0 0 700 230">{box(10, 15, 150, 36, t('guidePage.externalRagDiagram.modelB'), 'core/sdk.py', 'var(--diag-cyan)')}
        {arrow(160, 33, 230, 33)}
        {box(230, 15, 170, 36, 'wrap_retriever/generator', 'in-process', 'var(--diag-cyan)', true)}
        {arrow(400, 33, 470, 35)}
        {box(470, 10, 160, 50, 'White-box', t('guidePage.externalRagDiagram.alwaysFullStageTrace'), 'var(--diag-green)')}

        {box(10, 95, 150, 36, t('guidePage.externalRagDiagram.modelC'), 'adapters/http_pipeline.py', 'var(--diag-yellow)')}
        {arrow(160, 113, 230, 113)}
        {box(230, 95, 170, 36, 'HTTP POST /query', t('guidePage.externalRagDiagram.externalService'), 'var(--diag-yellow)', true)}
        {arrow(315, 131, 315, 150)}
        {box(230, 150, 170, 40, t('guidePage.externalRagDiagram.hasTraceField'), '', 'var(--diag-muted)', true)}

        {elbow([[400, 160], [440, 160], [440, 35], [470, 35]], 'var(--diag-green)')}
        <text x={447} y={95} fontSize="9" fill="var(--diag-green)" fontFamily="system-ui,sans-serif">{t('guidePage.externalRagDiagram.yes')}</text>
        {arrow(400, 182, 470, 178, 'var(--diag-red)')}
        <text x={420} y={200} fontSize="9" fill="var(--diag-red)" fontFamily="system-ui,sans-serif">{t('guidePage.externalRagDiagram.no')}</text>
        {box(470, 150, 160, 50, 'Black-box', t('guidePage.externalRagDiagram.answerOnlyOutcome'), 'var(--diag-red)')}
      </DiagFrame>
  )
}

/* ── Trace visual ── */
function TraceExplainer() {
  // The tint is named, and not passed as a property string: a class takes
  // the colour, and the stages are drawn from the same palette as every other
  // diagram in the guide.
  const stages = [
    { label: 'Embed',    time: '12ms',  note: 'BGE-M3 · 1024-dim',    hue: 'green' },
    { label: 'Dense',    time: '45ms',  note: 'Qdrant · 10 chunks',   hue: 'blue' },
    { label: 'Sparse',   time: '38ms',  note: 'OpenSearch · 10',      hue: 'orange' },
    { label: 'Merge',    time: '2ms',   note: 'RRF → 7 chunks',       hue: 'pink' },
    { label: 'Generate', time: '2.1s',  note: 'qwen3:8b · 187 tok',   hue: 'yellow' },
  ]
  return (
    <div className="guide-trace-example">
      {stages.map((s, i) => (
        <div key={s.label} className="inline-4">
          <div className={`guide-trace-block hue-${s.hue}`}>
            <div className="guide-trace-label">{s.label}</div>
            <div className="guide-trace-ms">{s.time}</div>
            <div className="guide-trace-detail">{s.note}</div>
          </div>
          {i < stages.length - 1 && <div className="guide-trace-arrow">→</div>}
        </div>
      ))}
    </div>
  )
}

/* ── MongoDB collections ── */
function MongoCollections() {
  const { t } = useTranslation()
  const cols = [
    { name: 'prompts',          ops: 'CRUD + activate',  note: t('guidePage.mongoCollections.prompts') },
    { name: 'experiment_runs',  ops: 'Create + Delete',  note: t('guidePage.mongoCollections.experimentRuns') },
    { name: 'datasets',         ops: 'CRUD',             note: t('guidePage.mongoCollections.datasets') },
    { name: 'corpora',          ops: 'Create + Read',    note: t('guidePage.mongoCollections.corpora') },
    { name: 'deepeval_results', ops: 'Create + Read',    note: t('guidePage.mongoCollections.deepevalResults') },
    { name: 'corpus_ingests',   ops: 'Create + Delete',  note: t('guidePage.mongoCollections.corpusIngests') },
    { name: 'settings',         ops: 'Read + Update',    note: t('guidePage.mongoCollections.settings') },
    { name: 'external_rags',    ops: 'CRUD',             note: t('guidePage.mongoCollections.externalRags') },
  ]
  return (
    <div className="guide-mongo-grid">
      {cols.map(c => (
        <div key={c.name} className="guide-mongo-card">
          <Database size={13} className="guide-mongo-icon" />
          <code className="inline-code xs">{c.name}</code>
          <div className="guide-mongo-ops">{c.ops}</div>
          <div className="guide-mongo-note">{c.note}</div>
        </div>
      ))}
    </div>
  )
}

/* ── The three things one relevance judgment turns into ── */
function JudgmentUsesDiagram() {
  const { t } = useTranslation()
  const box = (x: number, y: number, w: number, h: number, label: string, color: string) => (
    <g key={`ju${x}-${y}`}>
      <rect x={x} y={y} width={w} height={h} rx={6} fill="none" stroke={color} strokeWidth={1.5} />
      <text x={x + w / 2} y={y + h / 2 + 4} textAnchor="middle" fontSize="10" fill="currentColor" fontFamily="system-ui,sans-serif">{label}</text>
    </g>
  )
  const arrow = (x1: number, y1: number, x2: number, y2: number) => (
    // All four coordinates: three arrows leave one point here, so a key
    // without y2 made them collide and React warned that it might drop some.
    <line key={`ja${x1}-${y1}-${x2}-${y2}`} x1={x1} y1={y1} x2={x2} y2={y2} stroke="var(--diag-muted)" strokeWidth={1.2} markerEnd="url(#judg-arr)" />
  )
  return (
    <DiagFrame id="judg" viewBox="0 0 700 220">{box(10, 90, 180, 44, t('guidePage.judgmentsSection.diagramJudgment'), 'var(--diag-blue)')}
        {arrow(190, 112, 250, 40)}
        {arrow(190, 112, 250, 112)}
        {arrow(190, 112, 250, 184)}
        {box(255, 18, 200, 44, t('guidePage.judgmentsSection.diagramTest'), 'var(--diag-green)')}
        {box(255, 90, 200, 44, t('guidePage.judgmentsSection.diagramPair'), 'var(--diag-cyan)')}
        {box(255, 162, 200, 44, t('guidePage.judgmentsSection.diagramCause'), 'var(--diag-yellow)')}
        {arrow(455, 40, 510, 40)}
        {arrow(455, 112, 510, 112)}
        {arrow(455, 184, 510, 184)}
        {box(515, 18, 175, 44, t('guidePage.judgmentsSection.diagramTestEffect'), 'var(--diag-muted)')}
        {box(515, 90, 175, 44, t('guidePage.judgmentsSection.diagramPairEffect'), 'var(--diag-muted)')}
        {box(515, 162, 175, 44, t('guidePage.judgmentsSection.diagramCauseEffect'), 'var(--diag-muted)')}
      </DiagFrame>
  )
}

/* ── Feedback triage → human decision → (optionally) a pin ── */
function TriageWorkflowDiagram() {
  const { t } = useTranslation()
  const steps = [
    { label: t('guidePage.triageWorkflowDiagram.comment'), color: 'var(--diag-blue)' },
    { label: t('guidePage.triageWorkflowDiagram.classify'), color: 'var(--diag-yellow)' },
    { label: t('guidePage.triageWorkflowDiagram.crossCheck'), color: 'var(--diag-cyan)' },
    { label: t('guidePage.triageWorkflowDiagram.lever'), color: 'var(--diag-violet)' },
  ]
  const outcomes = [
    { label: t('guidePage.triageWorkflowDiagram.confirm'), color: 'var(--diag-green)', y: 15 },
    { label: t('guidePage.triageWorkflowDiagram.reject'), color: 'var(--diag-red)', y: 55 },
    { label: t('guidePage.triageWorkflowDiagram.recordJudgment'), color: 'var(--diag-pink)', y: 95 },
  ]
  return (
    <DiagFrame id="triage" viewBox="0 0 950 140">{steps.map((s, i) => {
          const x = 10 + i * 185
          return (
            <g key={s.label}>
              <rect x={x} y={35} width={160} height="45" rx="7" fill={s.color} fillOpacity="var(--diag-fill-opacity)" stroke={s.color} strokeWidth="1.5" />
              <text x={x + 80} y={62} textAnchor="middle" fontSize="10.5" fill={s.color} fontWeight="700" fontFamily="system-ui,sans-serif">{s.label}</text>
              <line x1={x + 160} y1={57} x2={x + 185} y2={57} stroke="var(--diag-line)" strokeWidth="1.5" markerEnd="url(#triage-arr)" />
            </g>
          )
        })}
        {outcomes.map(o => (
          <g key={o.label}>
            <rect x={770} y={o.y} width={170} height="30" rx="6" fill={o.color} fillOpacity="var(--diag-fill-opacity)" stroke={o.color} strokeWidth="1.4" />
            <text x={855} y={o.y + 20} textAnchor="middle" fontSize="10" fill={o.color} fontWeight="700" fontFamily="system-ui,sans-serif">{o.label}</text>
          </g>
        ))}
        <line x1={750} y1={57} x2={765} y2={30} stroke="var(--diag-line)" strokeWidth="1.4" markerEnd="url(#triage-arr)" />
        <line x1={750} y1={57} x2={765} y2={70} stroke="var(--diag-line)" strokeWidth="1.4" markerEnd="url(#triage-arr)" />
        <line x1={750} y1={57} x2={765} y2={110} stroke="var(--diag-line)" strokeWidth="1.4" markerEnd="url(#triage-arr)" />
      </DiagFrame>
  )
}

/* ── Shared diagram primitives ──
 * Every diagram below draws the same three things: a labelled box, an arrow
 * and a caption. They were being redrawn from scratch in each function, which
 * is why nine diagrams had nine slightly different corner radii and nine
 * copies of the same arrow marker. Colours come from the `--diag-*` tokens so
 * a diagram follows the theme; before this they were hex literals chosen
 * against the dark background, and every box in the light theme was dark navy
 * with pale grey text on it. */
function DiagFrame({ id, viewBox, children, caption, className }: {
  id: string; viewBox: string; children: React.ReactNode; caption?: React.ReactNode; className?: string
}) {
  return (
    <div className={`guide-diagram-wrap${className ? ' ' + className : ''}`}>
      <svg viewBox={viewBox} xmlns="http://www.w3.org/2000/svg" className="plot-svg" role="img">
        <defs>
          <marker id={`${id}-arr`} markerWidth="7" markerHeight="7" refX="6" refY="3" orient="auto">
            <path d="M0,0 L6,3 L0,6 Z" fill="var(--diag-line)" />
          </marker>
        </defs>
        {children}
      </svg>
      {caption && <p className="guide-diagram-caption">{caption}</p>}
    </div>
  )
}

// SVG text does not wrap, and nothing warns when it runs past its box — the
// label simply lies across whatever is next to it. Found on the production and
// bundle diagrams, where Russian labels are half again as long as the English
// ones they were laid out for.
//
// Width per character is approximated rather than measured: measuring needs a
// laid-out DOM node, and the accuracy is not worth a layout pass here. 0.54 em
// is close for the sans faces in use and errs toward breaking early.
function wrapLabel(label: string, boxWidth: number, fontSize: number): string[] {
  const maxChars = Math.max(6, Math.floor((boxWidth - 14) / (fontSize * 0.54)))
  if (label.length <= maxChars) return [label]
  const lines: string[] = []
  let line = ''
  for (const word of label.split(' ')) {
    if (!line) line = word
    else if ((line + ' ' + word).length <= maxChars) line += ' ' + word
    else { lines.push(line); line = word }
  }
  if (line) lines.push(line)
  return lines
}

function DBox({ x, y, w, h, label, color, sub, dashed }: {
  x: number; y: number; w: number; h: number; label: string
  color?: string; sub?: string; dashed?: boolean
}) {
  const c = color ?? 'var(--diag-muted)'
  const lines = wrapLabel(label, w, 10.5)
  const subLines = sub ? wrapLabel(sub, w, 9.5) : []
  // The whole text block is centred as one unit, so a two-line label stays in
  // the middle of the box instead of drifting below it.
  const lineHeight = 12
  const subHeight = 11
  const total = lines.length * lineHeight + subLines.length * subHeight
  const top = y + h / 2 - total / 2 + 9
  return (
    <g>
      <rect
        x={x} y={y} width={w} height={h} rx={7}
        fill={c} fillOpacity="var(--diag-fill-opacity)" stroke={c} strokeWidth={1.5}
        strokeDasharray={dashed ? '4,3' : undefined}
      />
      <text
        x={x + w / 2} textAnchor="middle"
        fontSize="10.5" fontWeight="700" fill={c} fontFamily="system-ui,sans-serif"
      >
        {lines.map((line, i) => (
          <tspan key={i} x={x + w / 2} y={top + i * lineHeight}>{line}</tspan>
        ))}
      </text>
      {subLines.length > 0 && (
        <text x={x + w / 2} textAnchor="middle" fontSize="9.5"
              fill="var(--diag-muted)" fontFamily="system-ui,sans-serif">
          {subLines.map((line, i) => (
            <tspan key={i} x={x + w / 2} y={top + lines.length * lineHeight + i * subHeight}>
              {line}
            </tspan>
          ))}
        </text>
      )}
    </g>
  )
}

function DArrow({ id, x1, y1, x2, y2, label, dashed, labelY }: {
  id: string; x1: number; y1: number; x2: number; y2: number; label?: string; dashed?: boolean
  // Explicit vertical placement for a label that will not fit in the gap the
  // arrow spans. Measured, not guessed: a label centred on a horizontal arrow
  // sits at the vertical centre of the boxes it runs between, so anything
  // wider than the gap lies across them.
  labelY?: number
}) {
  return (
    <g>
      <line
        x1={x1} y1={y1} x2={x2} y2={y2} stroke="var(--diag-line)" strokeWidth={1.4}
        markerEnd={`url(#${id}-arr)`} strokeDasharray={dashed ? '4,3' : undefined}
      />
      {label && (() => {
        // Wrapped against the gap the arrow actually spans, with no slack: the
        // first attempt added 26px of it and a label 96px wide stayed on one
        // line inside an 82px gap.
        const lines = wrapLabel(label, Math.abs(x2 - x1), 9)
        const base = labelY != null
          ? labelY - (lines.length - 1) * 10
          : (y1 + y2) / 2 - 6 - (lines.length - 1) * 10
        return (
          <text textAnchor="middle" fontSize="9" fill="var(--diag-muted)"
                fontFamily="system-ui,sans-serif">
            {lines.map((line, i) => (
              <tspan key={i} x={(x1 + x2) / 2} y={base + i * 10}>{line}</tspan>
            ))}
          </text>
        )
      })()}
    </g>
  )
}

/* ── One retrieval failure splits into four causes, each pointing at one lever ── */
function RootCauseFanDiagram() {
  const { t } = useTranslation()
  const rows = [
    { cause: t('guidePage.diagnosisSection.causeDataMissing'), lever: t('guidePage.diagnosisSection.leverIngest'), color: 'var(--diag-red)' },
    { cause: t('guidePage.diagnosisSection.causeChunking'), lever: t('guidePage.diagnosisSection.leverChunking'), color: 'var(--diag-orange)' },
    { cause: t('guidePage.diagnosisSection.causeRanking'), lever: t('guidePage.diagnosisSection.leverRanking'), color: 'var(--diag-yellow)' },
    { cause: t('guidePage.diagnosisSection.causeNotRetrievable'), lever: t('guidePage.diagnosisSection.leverVocabulary'), color: 'var(--diag-violet)' },
  ]
  return (
    <DiagFrame id="rcf" viewBox="0 0 720 250" caption={t('guidePage.diagnosisSection.fanCaption')}>
      <DBox x={8} y={95} w={150} h={54} label={t('guidePage.diagnosisSection.fanVerdict')}
            sub="funnel = retrieval" color="var(--diag-blue)" />
      {rows.map((r, i) => {
        const y = 12 + i * 58
        return (
          <g key={r.cause}>
            <DArrow id="rcf" x1={160} y1={122} x2={218} y2={y + 21} />
            <DBox x={222} y={y} w={230} h={42} label={r.cause} color={r.color} />
            <DArrow id="rcf" x1={454} y1={y + 21} x2={492} y2={y + 21} />
            <DBox x={496} y={y} w={216} h={42} label={r.lever} color="var(--diag-muted)" />
          </g>
        )
      })}
    </DiagFrame>
  )
}

/* ── The whole improvement loop, one lap ── */
function ImprovementLoopDiagram() {
  const { t } = useTranslation()
  const steps = [
    { label: t('guidePage.diagnosisSection.loopRun'), color: 'var(--diag-blue)' },
    { label: t('guidePage.diagnosisSection.loopCause'), color: 'var(--diag-orange)' },
    { label: t('guidePage.diagnosisSection.loopPriority'), color: 'var(--diag-yellow)' },
    { label: t('guidePage.diagnosisSection.loopPayoff'), color: 'var(--diag-cyan)' },
    { label: t('guidePage.diagnosisSection.loopDocument'), color: 'var(--diag-violet)' },
    { label: t('guidePage.diagnosisSection.loopAcceptance'), color: 'var(--diag-green)' },
  ]
  return (
    <DiagFrame id="loop" viewBox="0 0 950 150" caption={t('guidePage.diagnosisSection.loopCaption')}>
      {steps.map((s, i) => {
        const x = 8 + i * 158
        return (
          <g key={s.label}>
            <DBox x={x} y={30} w={140} h={46} label={s.label} color={s.color} />
            {i < steps.length - 1 && <DArrow id="loop" x1={x + 140} y1={53} x2={x + 156} y2={53} />}
          </g>
        )
      })}
      {/* The lap closes: an accepted fix is measured by the next run, not by
          the reviewer's impression of it. */}
      <path d="M 900 78 L 900 118 L 78 118 L 78 80" fill="none" stroke="var(--diag-line)"
            strokeWidth={1.4} strokeDasharray="4,3" markerEnd="url(#loop-arr)" />
      <text x={489} y={132} textAnchor="middle" fontSize="9" fill="var(--diag-muted)"
            fontFamily="system-ui,sans-serif">{t('guidePage.diagnosisSection.loopBack')}</text>
    </DiagFrame>
  )
}

/* ── Quality against latency, with the dominated points greyed out ── */
function FrontierPlotDiagram() {
  const { t } = useTranslation()
  // Illustrative, not measured: the shape is the point, and a real run's
  // numbers would tie the picture to one corpus.
  const onFrontier = [{ x: 90, y: 200 }, { x: 190, y: 130 }, { x: 330, y: 80 }, { x: 520, y: 55 }]
  const dominated = [{ x: 220, y: 205 }, { x: 300, y: 175 }, { x: 430, y: 150 }, { x: 480, y: 195 }]
  return (
    <DiagFrame id="fr" viewBox="0 0 700 296" caption={t('guidePage.frontierSection.plotCaption')}>
      <line x1={60} y1={20} x2={60} y2={230} stroke="var(--diag-line)" strokeWidth={1.2} />
      <line x1={60} y1={230} x2={660} y2={230} stroke="var(--diag-line)" strokeWidth={1.2} />
      <text x={16} y={26} fontSize="10" fill="var(--diag-muted)" fontFamily="system-ui,sans-serif">
        {t('guidePage.frontierSection.axisQuality')}
      </text>
      <text x={600} y={250} fontSize="10" fill="var(--diag-muted)" fontFamily="system-ui,sans-serif">
        {t('guidePage.frontierSection.axisLatency')}
      </text>
      <polyline
        points={onFrontier.map(p => `${p.x},${p.y}`).join(' ')}
        fill="none" stroke="var(--diag-green)" strokeWidth={1.6} strokeDasharray="5,3"
      />
      {dominated.map(p => (
        <circle key={`d${p.x}`} cx={p.x} cy={p.y} r={6} fill="var(--diag-muted)" fillOpacity={0.5} />
      ))}
      {onFrontier.map(p => (
        <circle key={`f${p.x}`} cx={p.x} cy={p.y} r={7} fill="var(--diag-green)"
                fillOpacity="var(--diag-fill-opacity)" stroke="var(--diag-green)" strokeWidth={1.8} />
      ))}
      {/* The legend sits under the axis rather than inside the plot: at the
          top right it covered the very point that carries the highest
          quality, which is the one a reader looks for first. */}
      <DBox x={60} y={252} w={280} h={36} label={t('guidePage.frontierSection.plotFrontier')} color="var(--diag-green)" />
      <DBox x={356} y={252} w={304} h={36} label={t('guidePage.frontierSection.plotDominated')} color="var(--diag-muted)" />
    </DiagFrame>
  )
}

/* ── Pull, never push: production traffic becomes a test ── */
function ProductionLoopDiagram() {
  const { t } = useTranslation()
  return (
    // Geometry chosen around the labels rather than the other way round: the
    // gaps are wide enough for a two-line Russian arrow label, and the return
    // path runs above a band left clear for it.
    <DiagFrame id="prod" viewBox="0 0 960 210" caption={t('guidePage.productionSection.loopCaption')}>
      <DBox x={8} y={74} w={166} h={56} label={t('guidePage.productionSection.boxServed')}
            sub={t('guidePage.productionSection.boxServedSub')} color="var(--diag-cyan)" />
      <DArrow id="prod" x1={176} y1={102} x2={258} y2={102} labelY={62}
              label={t('guidePage.productionSection.arrowPull')} />
      <DBox x={262} y={74} w={166} h={56} label={t('guidePage.productionSection.boxCandidates')}
            sub={t('guidePage.productionSection.boxCandidatesSub')} color="var(--diag-blue)" />
      <DArrow id="prod" x1={430} y1={102} x2={512} y2={102} labelY={62}
              label={t('guidePage.productionSection.arrowReview')} />
      <DBox x={516} y={74} w={166} h={56} label={t('guidePage.productionSection.boxGolden')}
            sub={t('guidePage.productionSection.boxGoldenSub')} color="var(--diag-green)" />
      <DArrow id="prod" x1={684} y1={102} x2={766} y2={102} />
      <DBox x={770} y={74} w={182} h={56} label={t('guidePage.productionSection.boxRun')} color="var(--diag-violet)" />
      {/* The platform never reaches into the served system; it only ever
          reads its log, and only when an operator asks. */}
      <path d="M 861 74 L 861 30 L 91 30 L 91 72" fill="none" stroke="var(--diag-line)"
            strokeWidth={1.4} strokeDasharray="4,3" markerEnd="url(#prod-arr)" />
      <text x={476} y={22} textAnchor="middle" fontSize="9" fill="var(--diag-muted)"
            fontFamily="system-ui,sans-serif">{t('guidePage.productionSection.arrowFix')}</text>
      <DBox x={262} y={152} w={420} h={44} dashed
            label={t('guidePage.productionSection.boxCoverage')} color="var(--diag-orange)" />
      <DArrow id="prod" x1={472} y1={132} x2={472} y2={150} />
    </DiagFrame>
  )
}

/* ── A correction travels as a document, applied once at startup ── */
function BundleDiagram() {
  const { t } = useTranslation()
  return (
    <DiagFrame id="bnd" viewBox="0 0 960 210" caption={t('guidePage.bundleSection.diagramCaption')}>
      <DBox x={8} y={62} w={166} h={56} label={t('guidePage.bundleSection.boxPlatform')} color="var(--diag-blue)" />
      <DArrow id="bnd" x1={176} y1={90} x2={258} y2={90} labelY={48}
              label={t('guidePage.bundleSection.arrowPublish')} />
      <DBox x={262} y={62} w={166} h={56} label={t('guidePage.bundleSection.boxFile')}
            sub={t('guidePage.bundleSection.boxFileSub')} color="var(--diag-violet)" />
      <DArrow id="bnd" x1={430} y1={90} x2={512} y2={90} labelY={48}
              label={t('guidePage.bundleSection.arrowLoad')} />
      <DBox x={516} y={62} w={166} h={56} label={t('guidePage.bundleSection.boxResolve')}
            sub={t('guidePage.bundleSection.boxResolveSub')} color="var(--diag-cyan)" />
      <DArrow id="bnd" x1={684} y1={90} x2={766} y2={90} />
      <DBox x={770} y={62} w={182} h={56} label={t('guidePage.bundleSection.boxServe')} color="var(--diag-green)" />
      {/* The line that is not there: no query-time call back to the platform.
          Drawn crossed out because its absence is the whole design. The label
          sits on its own row under the crossing, which is where it stops
          colliding with the return path. */}
      <path d="M 861 118 L 861 156 L 91 156 L 91 120" fill="none" stroke="var(--diag-red)"
            strokeWidth={1.3} strokeDasharray="4,3" opacity={0.6} />
      <line x1={452} y1={146} x2={484} y2={166} stroke="var(--diag-red)" strokeWidth={1.6} />
      <line x1={484} y1={146} x2={452} y2={166} stroke="var(--diag-red)" strokeWidth={1.6} />
      <text x={476} y={186} textAnchor="middle" fontSize="9.5" fill="var(--diag-red)"
            fontFamily="system-ui,sans-serif">
        {t('guidePage.bundleSection.noCallback')}
      </text>
    </DiagFrame>
  )
}


/* ── Where the platform stands among existing tools ──
 * A compressed form of the market survey. Only the rows that
 * separate the classes are here: a full table of thirty-five functions belongs
 * in the report, and a guide that reprints it teaches nothing extra.
 *
 * The classes carry hints rather than only names, because "Ragas" tells a
 * reader nothing unless they already know it. */
const TOOL_CLASSES: { key: string; tools: string }[] = [
  { key: 'metrics', tools: 'Ragas, DeepEval, TruLens' },
  { key: 'observability', tools: 'LangSmith, Langfuse, Arize Phoenix, Braintrust' },
  { key: 'optimizers', tools: 'AutoRAG, syftr, RAGBuilder' },
  { key: 'relevance', tools: 'Quepid, Rated Ranking Evaluator, OpenSearch SRW' },
]

// "yes" | "partial" | "no" per class, then ours.
const MATRIX: { key: string; marks: string[]; ours: string }[] = [
  { key: 'rowRetrievalMetrics', marks: ['yes', 'partial', 'partial', 'yes'], ours: 'yes' },
  { key: 'rowLayerVerdict',     marks: ['partial', 'no', 'no', 'no'], ours: 'yes' },
  { key: 'rowCause',            marks: ['no', 'no', 'no', 'no'], ours: 'yes' },
  { key: 'rowLever',            marks: ['no', 'no', 'no', 'no'], ours: 'yes' },
  { key: 'rowPriority',         marks: ['no', 'no', 'no', 'no'], ours: 'yes' },
  { key: 'rowPayoff',           marks: ['no', 'no', 'partial', 'no'], ours: 'yes' },
  { key: 'rowFrontier',         marks: ['no', 'no', 'yes', 'partial'], ours: 'yes' },
  { key: 'rowPrescription',     marks: ['no', 'no', 'no', 'no'], ours: 'yes' },
  { key: 'rowAcceptance',       marks: ['no', 'no', 'no', 'no'], ours: 'yes' },
  { key: 'rowJudgmentFile',     marks: ['no', 'partial', 'no', 'yes'], ours: 'yes' },
  { key: 'rowBundle',           marks: ['no', 'no', 'no', 'no'], ours: 'yes' },
  { key: 'rowProduction',       marks: ['no', 'yes', 'no', 'no'], ours: 'yes' },
  { key: 'rowCoverage',         marks: ['no', 'partial', 'no', 'no'], ours: 'yes' },
  { key: 'rowImplicit',         marks: ['no', 'partial', 'no', 'yes'], ours: 'no' },
  { key: 'rowQueue',            marks: ['no', 'yes', 'no', 'partial'], ours: 'no' },
  { key: 'rowCi',               marks: ['yes', 'yes', 'partial', 'partial'], ours: 'no' },
]

/** A services table: name, role, and a third column for where the service
 *  takes part. Two such tables stood side by side, written out separately with
 *  eleven inline styles each, differing only in the third column's header and
 *  in whether it was set in monospace. */
function ServiceTable({ roleHeader, thirdHeader, rows, mono }: {
  roleHeader: string
  thirdHeader: string
  rows: [string, string, string][]
  mono?: boolean
}) {
  const { t } = useTranslation()
  return (
    <table className="guide-table">
      <thead>
        <tr>
          <th>{t('guidePage.archSection.tableHeaderService')}</th>
          <th>{roleHeader}</th>
          <th>{thirdHeader}</th>
        </tr>
      </thead>
      <tbody>
        {rows.map(([svc, role, third]) => (
          <tr key={svc}>
            <td className="gt-name">{svc}</td>
            <td className="gt-role">{role}</td>
            <td className={mono ? 'gt-endpoint' : 'gt-where'}>{third}</td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}

const MARK_GLYPH: Record<string, string> = { yes: '●', partial: '◐', no: '—' }

function CapabilityMatrix() {
  const { t } = useTranslation()
  return (
    <div>
      <div className="chart-scroll">
        <table className="cap-matrix">
          <thead>
            <tr>
              <th className="left">{t('guidePage.marketSection.colFunction')}</th>
              {TOOL_CLASSES.map(c => (
                <th key={c.key} title={c.tools}>
                  {t(`guidePage.marketSection.class_${c.key}`)}
                  <span className="cap-matrix-tools">{c.tools}</span>
                </th>
              ))}
              <th className="cap-matrix-ours">{t('guidePage.marketSection.colOurs')}</th>
            </tr>
          </thead>
          <tbody>
            {MATRIX.map(row => (
              <tr key={row.key}>
                <th scope="row">{t(`guidePage.marketSection.${row.key}`)}</th>
                {row.marks.map((mark, i) => (
                  <td key={i} className={`mark-${mark}`}>{MARK_GLYPH[mark]}</td>
                ))}
                <td className={`cap-matrix-ours mark-${row.ours}`}>
                  {MARK_GLYPH[row.ours]}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="cap-matrix-legend">
        <span className="mark-yes">●</span> {t('guidePage.marketSection.legendYes')}
        {'   '}
        <span className="mark-partial">◐</span> {t('guidePage.marketSection.legendPartial')}
        {'   '}
        <span className="mark-no">—</span> {t('guidePage.marketSection.legendNo')}
      </p>
    </div>
  )
}

/* ── Sections ──
 * Ordered by what a reader needs when, and grouped so twenty-two entries in
 * one flat list stop reading as an inventory. The order was wrong in three
 * places before this: the quick start sat behind the architecture, so the one
 * section that says what to press first was reachable only past the deepest
 * topic; the corpus came after the regression guard, though a corpus is
 * prepared before anything is run; and the whole diagnosis chain sat after
 * operational reference material.
 *
 * `group` is a label, not a level: every section is still one click away, and
 * the groups only tell a reader which part of the work they are in. */
type GuideGroup = 'intro' | 'shell' | 'data' | 'run' | 'diagnose' | 'deliver' | 'reference'

const SECTIONS: {
  id: string; label: string; icon: React.ElementType; group: GuideGroup
  content: () => JSX.Element
}[] = [
  {
    id: 'purpose', label: 'guidePage.nav.purpose', icon: Target, group: 'intro',
    content: () => {
      const { t } = useTranslation()
      const h3 = { fontSize: 'var(--fs-sm)', fontWeight: 700, color: 'var(--color-text)', margin: '24px 0 8px' } as const
      return (
      <div>
        <p className="guide-lead">{t('guidePage.purposeSection.lead')}</p>
        <div className="guide-component-list">
          <div className="guide-component-row">
            <UserCog size={15} className="who-icon blue" />
            <div><strong>{t('guidePage.purposeSection.mainUserTitle')}</strong><span className="guide-component-desc">{t('guidePage.purposeSection.mainUserDesc')}</span></div>
          </div>
          <div className="guide-component-row">
            <ShieldAlert size={15} className="who-icon orange" />
            <div><strong>{t('guidePage.purposeSection.painTitle')}</strong><span className="guide-component-desc">{t('guidePage.purposeSection.painDesc')}</span></div>
          </div>
        </div>
        <div className="guide-callout info mt-16">
          <Info size={14} />
          <div>{t('guidePage.purposeSection.calloutInfoPart1')}<code className="inline-code">docs/DEBUGGING.md</code>{t('guidePage.purposeSection.calloutInfoPart2')}</div>
        </div>
        <div className="guide-callout warn mt-12">
          <AlertTriangle size={14} />
          <div>{t('guidePage.purposeSection.calloutWarnPart1')}<code className="inline-code">ExperimentConfig</code>{t('guidePage.purposeSection.calloutWarnPart2')}<code className="inline-code">config_hash</code>{t('guidePage.purposeSection.calloutWarnPart3')}<strong>{t('guidePage.purposeSection.anySingular')}</strong>{t('guidePage.purposeSection.calloutWarnPart4')}<strong>{t('guidePage.purposeSection.anyPlural')}</strong>{t('guidePage.purposeSection.calloutWarnPart5')}</div>
        </div>

        {/* Where the platform stands among the tools a reader may already use.
            It belongs in the purpose section because "what is this for" is not
            answerable without "and what does it do that the others do not". */}
        <h3 style={h3}>{t('guidePage.marketSection.heading')}</h3>
        <p>{t('guidePage.marketSection.para1')}</p>
        <p>{t('guidePage.marketSection.para2')}</p>
        <CapabilityMatrix />
        <p>{t('guidePage.marketSection.para3')}</p>
        <div className="guide-callout warn mt-12">
          <AlertTriangle size={14} />
          <div>{t('guidePage.marketSection.gapsCallout')}</div>
        </div>
      </div>
      )
    },
  },
  {
    id: 'arch', label: 'guidePage.nav.arch', icon: Layers, group: 'intro',
    content: () => {
      const { t } = useTranslation()
      return (
      <div>
        <p className="guide-lead">{t('guidePage.archSection.lead')}</p>

        <h3 className="guide-h3">{t('guidePage.archSection.servicesTopologyHeading')}</h3>
        <PlatformServiceDiagram />

        <h3 className="guide-h3 wide">{t('guidePage.archSection.codeLayersHeading')}</h3>
        <CodeLayerDiagram />

        <div className="guide-callout info mt-16">
          <Boxes size={14} />
          <div>
            <strong>{t('guidePage.archSection.keyInvariantStrong')}</strong> <code className="inline-code">core/</code>{t('guidePage.archSection.keyInvariantRest')}
          </div>
        </div>

        <h3 className="guide-h3 wide">{t('guidePage.archSection.whatPlatformUsesHeading')}</h3>
        <p className="guide-note">
          {t('guidePage.archSection.whatPlatformUsesParagraph')}
        </p>
        <ServiceTable
          roleHeader={t('guidePage.archSection.tableHeaderRoleForPlatform')}
          thirdHeader="Endpoint"
          mono
          rows={[
            ['BGE-M3', t('guidePage.archSection.table1Row1Role'), 'CompositeEvaluator'],
            ['Qdrant :6333', t('guidePage.archSection.table1Row2Role'), 'GET /corpus/{id}/chunks\nGET /corpus/{id}/health\nPOST /corpus/{id}/diagnostics/*'],
            ['Neo4j :7687', t('guidePage.archSection.table1Row3Role'), 'GET /corpus/{id}/graph/communities\nGET /corpus/{id}/graph/communities/{cid}'],
            ['Ollama :11434', t('guidePage.archSection.table1Row4Role'), 'POST /corpus/{id}/diagnostics/ragas\nPOST /corpus/{id}/diagnostics/trulens\nPOST /corpus/{id}/diagnostics/chunk-coherence'],
          ]}
        />

        <h3 className="guide-h3 wide">{t('guidePage.archSection.whatBundledRagUsesHeading')}</h3>
        <p className="guide-note">
          {t('guidePage.archSection.bundledRagParagraph')}
        </p>
        <ServiceTable
          roleHeader={t('guidePage.archSection.tableHeaderRoleInBundled')}
          thirdHeader="Pipeline"
          rows={[
            ['BGE-M3', t('guidePage.archSection.table2Row1Role'), 'naive, hybrid, graph'],
            ['Qdrant :6333', t('guidePage.archSection.table2Row2Role'), 'naive, hybrid_rrf, hybrid_weighted, graph'],
            ['OpenSearch :9200', t('guidePage.archSection.table2Row3Role'), 'hybrid_rrf, hybrid_weighted'],
            ['Neo4j :7687', t('guidePage.archSection.table2Row4Role'), 'graph'],
            ['Ollama :11434', t('guidePage.archSection.table2Row5Role'), t('guidePage.archSection.table2Row5Used')],
          ]}
        />

        <h3 className="guide-h3 wide">{t('guidePage.archSection.ragPipelineInsideHeading')}</h3>
        <p className="guide-lead sm">
          <code className="inline-code">pipeline_id</code>{t('guidePage.archSection.ragPipelineParagraph')}
        </p>
        <ArchDiagram />

        <h3 className="guide-h3 wide">{t('guidePage.archSection.externalIntegrationHeading')}</h3>
        <p className="guide-note md">
          {t('guidePage.archSection.externalIntegrationIntro')}
        </p>
        <div className="guide-component-list">
          {[
            { icon: Plug, color: 'green', name: 'Tier 1 — native contract', desc: t('guidePage.archSection.tier1Desc') },
            { icon: SlidersHorizontal, color: 'yellow', name: 'Tier 2 — JSONPath mapping', desc: t('guidePage.archSection.tier2DescPart1') + '{{query}}/{{corpus_id}}/{{top_k}}' + t('guidePage.archSection.tier2DescPart2') },
            { icon: Boxes, color: 'orange', name: 'Tier 3 — code adapter (escape hatch)', desc: t('guidePage.archSection.tier3Desc') },
          ].map(({ icon: Icon, color, name, desc }) => (
            <div key={name} className="guide-component-row">
              <Icon size={15} className={`who-icon ${color}`} />
              <div><strong>{name}</strong><span className="guide-component-desc">{desc}</span></div>
            </div>
          ))}
        </div>

        <h3 className="guide-h3 wide">{t('guidePage.archSection.whichPipelineIdHeadingPrefix')}<code className="inline-code">pipeline_id</code>{t('guidePage.archSection.whichPipelineIdHeadingSuffix')}</h3>
        <PipelineDecisionTree />
      </div>
      )
    },
  },
  {
    id: 'quickstart', label: 'guidePage.nav.quickstart', icon: Zap, group: 'intro',
    content: () => {
      const { t } = useTranslation()
      return (
      <div>
        <p className="guide-lead">{t('guidePage.quickstartSection.lead')}</p>
        {/* This section described five manual steps: bring up the containers,
            ingest a corpus through the CLI, start the gateway, start the UI,
            and "migrate data into MongoDB (once)". A new reader never had any
            use for the last one, a one-off transfer out of an era they never
            saw. The first four are what `./install.sh` does, and the README
            opens with it. The guide describes the same path. */}
        {[
          { n: '1', title: t('guidePage.quickstartSection.step1Title'), code: 'git clone https://github.com/laputski/causa-rag && cd causa-rag && ./install.sh' },
          { n: '2', title: t('guidePage.quickstartSection.step2Title'), code: './install.sh --check' },
          { n: '3', title: t('guidePage.quickstartSection.step3Title'), code: 'open http://localhost:5173' },
          { n: '4', title: t('guidePage.quickstartSection.step4Title'), code: 'make doctor' },
        ].map(({ n, title, code }) => (
          <div key={n} className="guide-step">
            <div className="guide-step-num">{n}</div>
            <div className="guide-step-body">
              <div className="guide-step-title">{title}</div>
              <pre className="code-block">{code}</pre>
            </div>
          </div>
        ))}
      </div>
      )
    },
  },
  {
    id: 'chunking', label: 'guidePage.nav.chunking', icon: Scissors, group: 'data',
    content: () => {
      const { t } = useTranslation()
      return (
      <div>
        <p className="guide-lead">{t('guidePage.chunkingSection.lead')}</p>
        <ChunkingTable />
        <div className="guide-callout info mt-16">
          <Info size={14} />
          <div><strong>{t('guidePage.chunkingSection.recommendationStrong')}</strong> <code className="inline-code">structure_aware</code>{t('guidePage.chunkingSection.part1')}<code className="inline-code">--structure npa_article</code>{t('guidePage.chunkingSection.part2')}<code className="inline-code">{t('guidePage.chunkingSection.exampleXpath')}</code>{t('guidePage.chunkingSection.part3')}<code className="inline-code">--structure npa_article</code>{t('guidePage.chunkingSection.part4')}<code className="inline-code">"root"</code>{t('guidePage.chunkingSection.part5')}<code className="inline-code">chunk_size=512</code>{t('guidePage.chunkingSection.part6')}</div>
        </div>
      </div>
      )
    },
  },
  {
    id: 'config', label: 'guidePage.nav.config', icon: SlidersHorizontal, group: 'run',
    content: () => {
      const { t } = useTranslation()
      return (
      <div>
        <p className="guide-lead">{t('guidePage.configSection.lead')}</p>
        <div className="guide-component-list">
          {[
            { icon: ArrowUpDown, color: 'var(--diag-violet)', name: t('guidePage.configSection.rerankerName'), desc: t('guidePage.configSection.rerankerDesc') },
            { icon: ShieldCheck, color: 'var(--diag-green)', name: 'Grounding', desc: t('guidePage.configSection.groundingDesc') },
            { icon: GitMerge,    color: 'var(--diag-cyan)', name: 'Routing',  desc: t('guidePage.configSection.routingDesc') },
            { icon: Share2,      color: 'var(--diag-yellow)', name: 'GraphRAG', desc: t('guidePage.configSection.graphRagDesc') },
          ].map(({ icon: Icon, color, name, desc }) => (
            <div key={name} className="guide-component-row">
              <Icon size={15} className={`who-icon ${color}`} />
              <div><strong>{name}</strong><span className="guide-component-desc">{desc}</span></div>
            </div>
          ))}
        </div>
        <div className="guide-callout info mt-16">
          <Info size={14} />
          <div><strong>{t('guidePage.configSection.reproducibilityStrong')}</strong>{t('guidePage.configSection.reproducibilityRest')}</div>
        </div>
        <div className="guide-callout info mt-12">
          <Info size={14} />
          <div><strong>GraphRAG end-to-end:</strong> <code className="inline-code">docker compose --profile graph up -d neo4j</code>{t('guidePage.configSection.graphE2EPart1')}<code className="inline-code">USE_GRAPH=true</code>{t('guidePage.configSection.graphE2EPart2')}<strong>{t('guidePage.configSection.graphE2EStrong')}</strong>{t('guidePage.configSection.graphE2EPart3')}<code className="inline-code">pip install '.[reranker]' '.[graph]'</code>{t('guidePage.configSection.graphE2EPart4')}</div>
        </div>
      </div>
      )
    },
  },
  {
    id: 'regression', label: 'guidePage.nav.regression', icon: GitCompare, group: 'run',
    content: () => {
      const { t } = useTranslation()
      return (
      <div>
        <p className="guide-lead">{t('guidePage.regressionSection.leadPart1')}<strong>baseline</strong>{t('guidePage.regressionSection.leadPart2')}</p>
        <div className="guide-callout info mb-16">
          <Info size={14} />
          <div>
            <strong>{t('guidePage.regressionSection.pinBaselineStrong')}</strong> <code className="inline-code">PUT /experiments/{'{id}'}/baseline</code>{t('guidePage.regressionSection.part1')}<code className="inline-code">settings.baseline_run_id</code>{t('guidePage.regressionSection.part1b')}
            <br /><strong>{t('guidePage.regressionSection.autoCompareStrong')}</strong> <code className="inline-code">GET /experiments/{'{id}'}</code>{t('guidePage.regressionSection.part2')}<code className="inline-code">regression</code>{t('guidePage.regressionSection.part3')}<code className="inline-code">core/eval/regression.py</code>{t('guidePage.regressionSection.part4')}
          </div>
        </div>
        <RegressionWorkflowDiagram />
        <div className="guide-component-list">
          {[
            { icon: AlertTriangle, color: 'var(--diag-red)', name: t('guidePage.regressionSection.stubEmbedderName'),  desc: t('guidePage.regressionSection.stubEmbedderDesc') },
            { icon: AlertTriangle, color: 'var(--diag-yellow)', name: t('guidePage.regressionSection.topKDupesName'),     desc: t('guidePage.regressionSection.topKDupesDesc') },
            { icon: AlertTriangle, color: 'var(--diag-orange)', name: t('guidePage.regressionSection.headerOnlyName'),    desc: t('guidePage.regressionSection.headerOnlyDesc') },
            { icon: AlertTriangle, color: 'var(--diag-violet)', name: t('guidePage.regressionSection.bm25DominanceName'), desc: t('guidePage.regressionSection.bm25DominanceDesc') },
            { icon: AlertTriangle, color: 'var(--diag-blue)', name: t('guidePage.regressionSection.emptyAnswersName'),  desc: t('guidePage.regressionSection.emptyAnswersDesc') },
          ].map(({ icon: Icon, color, name, desc }) => (
            <div key={name} className="guide-component-row">
              <Icon size={15} className={`who-icon ${color}`} />
              <div><strong>{name}</strong><span className="guide-component-desc">{desc}</span></div>
            </div>
          ))}
        </div>
        <div className="guide-callout info mt-12">
          <Info size={14} />
          <div>{t('guidePage.regressionSection.detectorsPart1')}<code className="inline-code">core/eval/detectors.py</code>{t('guidePage.regressionSection.detectorsPart2')}<code className="inline-code">RunDiagnostics.tsx</code>{t('guidePage.regressionSection.detectorsPart3')}<code className="inline-code">RunCharts.tsx</code>{t('guidePage.regressionSection.detectorsPart4')}</div>
        </div>
      </div>
      )
    },
  },
  {
    id: 'judgments', label: 'guidePage.nav.judgments', icon: Scale, group: 'diagnose',
    content: () => {
      const { t } = useTranslation()
      const h3 = { fontSize: 'var(--fs-sm)', fontWeight: 700, color: 'var(--color-text)', margin: '20px 0 8px' } as const
      return (
      <div>
        <DraftNote />
        <p className="guide-lead">{t('guidePage.judgmentsSection.lead')}</p>
        <p className="guide-lead">{t('guidePage.judgmentsSection.leadPara2')}</p>

        <h3 style={h3}>{t('guidePage.judgmentsSection.whatHeading')}</h3>
        <p>{t('guidePage.judgmentsSection.whatPara1')}</p>
        <p>{t('guidePage.judgmentsSection.whatPara2')}</p>
        <p>{t('guidePage.judgmentsSection.whatPara3')}</p>
        <p>{t('guidePage.judgmentsSection.whatPara4')}</p>

        <h3 style={h3}>{t('guidePage.judgmentsSection.usesHeading')}</h3>
        <p>{t('guidePage.judgmentsSection.usesPara1')}</p>
        <p>{t('guidePage.judgmentsSection.usesPara2')}</p>
        <p>{t('guidePage.judgmentsSection.usesPara3')}</p>

        <JudgmentUsesDiagram />
        <p className="diag-caption">{t('guidePage.judgmentsSection.usesDiagramCaption')}</p>

        <h3 style={h3}>{t('guidePage.judgmentsSection.formHeading')}</h3>
        <p>{t('guidePage.judgmentsSection.formPara1')}</p>
        <p>{t('guidePage.judgmentsSection.formPara2')}</p>
        <p>{t('guidePage.judgmentsSection.formPara3')}</p>
        <p>{t('guidePage.judgmentsSection.formPara4')}</p>
        <p>{t('guidePage.judgmentsSection.formPara5')}</p>

        <h3 style={h3}>{t('guidePage.judgmentsSection.offHeading')}</h3>
        <p>{t('guidePage.judgmentsSection.offPara1')}</p>
        <p>{t('guidePage.judgmentsSection.offPara2')}</p>
        <p>{t('guidePage.judgmentsSection.offPara3')}</p>

        <h3 style={h3}>{t('guidePage.judgmentsSection.confirmDiffHeading')}</h3>
        <div className="guide-callout info mb-8">
          <Info size={14} />
          <div>{t('guidePage.judgmentsSection.confirmDiffPara1')}</div>
        </div>
        <p>{t('guidePage.judgmentsSection.confirmDiffPara2')}</p>
        <p>{t('guidePage.judgmentsSection.confirmDiffPara3')}</p>
        <p>{t('guidePage.judgmentsSection.confirmDiffPara4')}</p>
        <p>{t('guidePage.judgmentsSection.confirmDiffPara5')}</p>
      </div>
      )
    },
  },
  {
    // Split out of the judgments section: two mechanisms with different UI
    // locations were reading as one long topic, and a reader looking for the
    // triage panel had to scroll past everything about judgments to find it.
    id: 'triage', label: 'guidePage.nav.triage', icon: Sparkles, group: 'diagnose',
    content: () => {
      const { t } = useTranslation()
      const h3 = { fontSize: 'var(--fs-sm)', fontWeight: 700, color: 'var(--color-text)', margin: '20px 0 8px' } as const
      return (
      <div>
        <p className="guide-lead">{t('guidePage.judgmentsSection.triageDefinitionPara1')}</p>
        <p>{t('guidePage.judgmentsSection.triageDefinitionPara2')}</p>
        <p>{t('guidePage.judgmentsSection.triageDefinitionPara3')}</p>
        <p>{t('guidePage.judgmentsSection.triageDefinitionPara4')}</p>

        <h3 style={h3}>{t('guidePage.judgmentsSection.triageMechanicsHeading')}</h3>
        <p>{t('guidePage.judgmentsSection.triageMechanicsPara1')}</p>
        <p>{t('guidePage.judgmentsSection.triageMechanicsPara2')}</p>
        <p>{t('guidePage.judgmentsSection.triageMechanicsPara3')}</p>

        <TriageWorkflowDiagram />

        <h3 style={h3}>{t('guidePage.judgmentsSection.triageUiHeading')}</h3>
        <p>{t('guidePage.judgmentsSection.triageUiPara1')}</p>
        <p>{t('guidePage.judgmentsSection.triageUiPara2')}</p>
        <p>{t('guidePage.judgmentsSection.triageUiPara3')}</p>
        <p>{t('guidePage.judgmentsSection.triageUiPara4')}</p>

        <div className="guide-callout warn mt-12">
          <AlertTriangle size={14} />
          <div>{t('guidePage.judgmentsSection.triageNeverAppliesPara1')}</div>
        </div>
      </div>
      )
    },
  },
  {
    id: 'packs', label: 'guidePage.nav.packs', icon: Puzzle, group: 'shell',
    content: () => {
      const { t } = useTranslation()
      return (
        <div>
          <p className="guide-lead">{t('guidePage.packsSection.lead')}</p>
          <p>{t('guidePage.packsSection.whatIs')}</p>

          <h3 className="guide-h3">{t('guidePage.packsSection.whereHeading')}</h3>
          <p>{t('guidePage.packsSection.where1')}</p>
          <pre className="code-block">{`CAUSA_DOMAIN_PACKS=/opt/acme/packs uvicorn services.api_gateway.main:app`}</pre>
          <p>{t('guidePage.packsSection.where2')}</p>

          <h3 className="guide-h3">{t('guidePage.packsSection.addHeading')}</h3>
          <p>{t('guidePage.packsSection.add1')}</p>
          <pre className="code-block">{`acme_support/
  pack.yaml          # id, version, display_name, exported_kinds
  __init__.py        # def register(registry, settings)
  structure_parser.py`}</pre>
          <p>{t('guidePage.packsSection.add2')}</p>

          <div className="guide-callout warn">
            <AlertTriangle size={14} />
            <div>{t('guidePage.packsSection.calloutExternal')}</div>
          </div>

          <div className="guide-callout info">
            <Info size={14} />
            <div>{t('guidePage.packsSection.calloutRestart')}</div>
          </div>
        </div>
      )
    },
  },
  {
    id: 'realm', label: 'guidePage.nav.realm', icon: Boxes, group: 'shell',
    content: () => {
      const { t } = useTranslation()
      return (
        <div>
          <p className="guide-lead">{t('guidePage.realmSection.lead')}</p>
          <p>{t('guidePage.realmSection.whatIs')}</p>
          <p>{t('guidePage.realmSection.switching')}</p>

          <div className="guide-callout warn">
            <AlertTriangle size={14} />
            <div>{t('guidePage.realmSection.calloutScope')}</div>
          </div>

          <h3 className="guide-h3">{t('guidePage.realmSection.transferHeading')}</h3>
          <p>{t('guidePage.realmSection.transfer1')}</p>
          <p>{t('guidePage.realmSection.transfer2')}</p>
          <div className="guide-callout info">
            <Info size={14} />
            <div>{t('guidePage.realmSection.calloutCorpus')}</div>
          </div>

          <h3 className="guide-h3">{t('guidePage.realmSection.metricsHeading')}</h3>
          <p>{t('guidePage.realmSection.metrics1')}</p>
          <p>{t('guidePage.realmSection.metrics2')}</p>
        </div>
      )
    },
  },
  {
    id: 'shell', label: 'guidePage.nav.shell', icon: Command, group: 'shell',
    content: () => {
      const { t } = useTranslation()
      return (
        <div>
          <p className="guide-lead">{t('guidePage.shellSection.lead')}</p>

          <h3 className="guide-h3">{t('guidePage.shellSection.paletteHeading')}</h3>
          <p>{t('guidePage.shellSection.palette1')}</p>
          <p>{t('guidePage.shellSection.palette2')}</p>

          <h3 className="guide-h3">{t('guidePage.shellSection.notifyHeading')}</h3>
          <p>{t('guidePage.shellSection.notify1')}</p>
          <div className="guide-callout info">
            <Info size={14} />
            <div>{t('guidePage.shellSection.calloutNotify')}</div>
          </div>

          <h3 className="guide-h3">{t('guidePage.shellSection.viewHeading')}</h3>
          <p>{t('guidePage.shellSection.view1')}</p>
        </div>
      )
    },
  },
  {
    id: 'corpus', label: 'guidePage.nav.corpus', icon: Boxes, group: 'data',
    content: () => {
      const { t } = useTranslation()
      return (
      <div>
        <p className="guide-lead">{t('guidePage.corpusSection.leadPart1')}<code className="inline-code">(realm_id, corpus_id)</code>{t('guidePage.corpusSection.leadPart2')}</p>
        <div className="guide-callout info mb-16">
          <Info size={14} />
          <div>
            <strong>{t('guidePage.corpusSection.dimensionsStrong')}</strong>{t('guidePage.corpusSection.dimensionsPart1')}<em>{t('guidePage.corpusSection.storageEm')}</em>{t('guidePage.corpusSection.dimensionsPart2')}<em>{t('guidePage.corpusSection.ownershipEm')}</em>{t('guidePage.corpusSection.dimensionsPart3')}
          </div>
        </div>
        <div className="guide-callout info mb-16">
          <Info size={14} />
          <div>{t('guidePage.corpusSection.namingPart1')}<code className="inline-code">corpus_id="default"</code>{t('guidePage.corpusSection.namingPart2')}<code className="inline-code">{'{realm_id}__'}</code>{t('guidePage.corpusSection.namingPart3')}<code className="inline-code">corpus_id</code>{t('guidePage.corpusSection.namingPart4')}<code className="inline-code">corpora</code>{t('guidePage.corpusSection.namingPart5')}<code className="inline-code">GET /corpus/collections?realm_id=</code>{t('guidePage.corpusSection.namingPart6')}</div>
        </div>
        <CorpusNamespaceDiagram />
        <div className="guide-callout warn my-16">
          <AlertTriangle size={14} />
          <div><strong>{t('guidePage.corpusSection.neo4jNotPartitionedStrong')}</strong>{t('guidePage.corpusSection.neo4jPart1')}<code className="inline-code">link_related</code>{t('guidePage.corpusSection.neo4jPart2')}<code className="inline-code">realm_id</code>{t('guidePage.corpusSection.neo4jPart3')}<code className="inline-code">Neo4jGraphRetriever.clear()</code>{t('guidePage.corpusSection.neo4jPart4')}</div>
        </div>

        <p className="guide-lead sm">{t('guidePage.corpusSection.graphTabIntro')}</p>
        <GraphCommunityDiagram />
        <div className="guide-callout warn my-16">
          <AlertTriangle size={14} />
          <div><strong>{t('guidePage.corpusSection.realResultStrong')}</strong>{t('guidePage.corpusSection.realResultPart1')}<code className="inline-code">RELATED</code>{t('guidePage.corpusSection.realResultPart2')}</div>
        </div>
        <p className="guide-lead sm">{t('guidePage.corpusSection.sidebarItemsPart1')}<code className="inline-code">uses_realm_resources</code>{t('guidePage.corpusSection.sidebarItemsPart2')}</p>
        <div className="guide-component-list">
          <div className="guide-component-row">
            <Search size={15} className="who-icon blue" />
            <div><strong>{t('guidePage.corpusSection.contentTabStrong')}</strong><span className="guide-component-desc">{t('guidePage.corpusSection.contentTabDescPart1')}<code className="inline-code">GET /corpus/{'{id}'}/chunks</code>{t('guidePage.corpusSection.contentTabDescPart2')}</span></div>
          </div>
          <div className="guide-component-row">
            <ShieldCheck size={15} className="who-icon green" />
            <div><strong>{t('guidePage.corpusSection.healthTabStrong')}</strong><span className="guide-component-desc">{t('guidePage.corpusSection.healthTabDescPart1')}<code className="inline-code">core/eval/corpus_health.py</code>{t('guidePage.corpusSection.healthTabDescPart2')}</span></div>
          </div>
        </div>
        <div className="guide-component-list mt-12">
          {[
            { id: 'duplicates',                   desc: t('guidePage.corpusSection.detectorDuplicates') },
            { id: 'header_only',                  desc: t('guidePage.corpusSection.detectorHeaderOnly') },
            { id: 'too_short',                     desc: t('guidePage.corpusSection.detectorTooShort') },
            { id: 'duplicate_structural_numbers',  desc: t('guidePage.corpusSection.detectorDuplicateStructuralNumbers') },
            { id: 'missing_structural_numbers',    desc: t('guidePage.corpusSection.detectorMissingStructuralNumbers') },
            { id: 'near_duplicates',               desc: t('guidePage.corpusSection.detectorNearDuplicates') },
            { id: 'mixed_language',                desc: t('guidePage.corpusSection.detectorMixedLanguage') },
          ].map(r => (
            <div key={r.id} className="guide-component-row">
              <Info size={15} className="who-icon blue" />
              <div><code className="inline-code">{r.id}</code><span className="guide-component-desc">{r.desc}</span></div>
            </div>
          ))}
        </div>
      </div>
      )
    },
  },
  {
    id: 'deep-diagnostics', label: 'guidePage.nav.deepDiagnostics', icon: ShieldAlert, group: 'data',
    content: () => {
      const { t } = useTranslation()
      return (
      <div>
        <p className="guide-lead">{t('guidePage.deepDiagnosticsSection.lead')}</p>
        <div className="guide-callout warn mb-16">
          <AlertTriangle size={14} />
          <div><strong>{t('guidePage.deepDiagnosticsSection.whyFourStrong')}</strong>{t('guidePage.deepDiagnosticsSection.whyFourRest')}</div>
        </div>
        <div className="guide-component-list">
          <div className="guide-component-row">
            <Bot size={15} className="who-icon blue" />
            <div><strong>DeepEval</strong> (<code className="inline-code">eval/deepeval_runner.py</code>)<span className="guide-component-desc">{t('guidePage.deepDiagnosticsSection.deepEvalDescPart1')}<code className="inline-code">run_id</code>{t('guidePage.deepDiagnosticsSection.deepEvalDescPart2')}</span></div>
          </div>
          <div className="guide-component-row">
            <BarChart2 size={15} className="who-icon green" />
            <div><strong>Ragas</strong> (<code className="inline-code">eval/ragas_runner.py</code>)<span className="guide-component-desc">{t('guidePage.deepDiagnosticsSection.ragasDesc')}</span></div>
          </div>
          <div className="guide-component-row">
            <Search size={15} className="who-icon yellow" />
            <div><strong>TruLens</strong> (<code className="inline-code">eval/trulens_runner.py</code>)<span className="guide-component-desc">{t('guidePage.deepDiagnosticsSection.truLensDesc')}</span></div>
          </div>
          <div className="guide-component-row">
            <Scissors size={15} className="who-icon pink" />
            <div><strong>{t('guidePage.deepDiagnosticsSection.chunkCoherenceStrong')}</strong> (<code className="inline-code">eval/chunk_coherence_judge.py</code>)<span className="guide-component-desc">{t('guidePage.deepDiagnosticsSection.chunkCoherenceDesc')}</span></div>
          </div>
        </div>
        <div className="guide-callout info mt-12">
          <Info size={14} />
          <div><strong>{t('guidePage.deepDiagnosticsSection.razdelStrong')}</strong>{t('guidePage.deepDiagnosticsSection.razdelPart1')}<code className="inline-code">core/chunking/{'{structure_aware,sentence}'}.py</code>{t('guidePage.deepDiagnosticsSection.razdelPart2')}<code className="inline-code">razdel</code>{t('guidePage.deepDiagnosticsSection.razdelPart3')}</div>
        </div>
      </div>
      )
    },
  },
  {
    id: 'external', label: 'guidePage.nav.external', icon: Plug, group: 'run',
    content: () => {
      const { t } = useTranslation()
      return (
      <div>
        <p className="guide-lead">{t('guidePage.externalSection.lead')}</p>
        <div className="guide-component-list">
          <div className="guide-component-row">
            <Cpu size={15} className="who-icon cyan" />
            <div><strong>{t('guidePage.externalSection.modelBStrong')}</strong><span className="guide-component-desc">{t('guidePage.externalSection.modelBDesc')}</span></div>
          </div>
          <div className="guide-component-row">
            <Plug size={15} className="who-icon yellow" />
            <div><strong>{t('guidePage.externalSection.modelCStrong')}</strong><span className="guide-component-desc">{t('guidePage.externalSection.modelCDesc')}</span></div>
          </div>
        </div>
        <ExternalRagDiagram />
        <pre className="code-block mt-12">{`from core.sdk import wrap_retriever, wrap_generator, build_pipeline

retriever = wrap_retriever(lambda query, k, filters=None: my_rag.search(query, k))
generator = wrap_generator(my_rag.answer)
pipeline = build_pipeline(retriever, generator)`}</pre>
        <div className="guide-callout info mt-12">
          <Info size={14} />
          <div>
            <strong>{t('guidePage.externalSection.contractStrong')}</strong>{t('guidePage.externalSection.contractPart1')}<code className="inline-code">answer</code>{t('guidePage.externalSection.contractPart2')}<code className="inline-code">trace</code>{t('guidePage.externalSection.contractPart3')}<code className="inline-code">stage_trace</code>/<code className="inline-code">sources</code>/<code className="inline-code">rendered_prompt</code>{t('guidePage.externalSection.contractPart4')}<strong>{t('guidePage.externalSection.blackBoxStrong')}</strong>{t('guidePage.externalSection.contractPart5')}
          </div>
        </div>
        <div className="guide-callout warn mt-12">
          <AlertTriangle size={14} />
          <div>{t('guidePage.externalSection.egressPart1')}<code className="inline-code">RAG_HTTP_ALLOWLIST</code>{t('guidePage.externalSection.egressPart2')}<code className="inline-code">config_hash</code>{t('guidePage.externalSection.egressPart3')}<code className="inline-code">pipeline_source</code>/<code className="inline-code">http_endpoint</code>{t('guidePage.externalSection.egressPart4')}<code className="inline-code">corpus_id</code>{t('guidePage.externalSection.egressPart5')}</div>
        </div>
        <div className="guide-callout info mt-12">
          <Plug size={14} />
          <div>
            <strong>{t('guidePage.externalSection.notInRepoStrong')}</strong>{t('guidePage.externalSection.clientPart1')}<code className="inline-code">rag-platform-client</code>{t('guidePage.externalSection.clientPart2')}<code className="inline-code">clients/python/</code>{t('guidePage.externalSection.clientPart3')}<code className="inline-code">register_rag</code> / <code className="inline-code">upload_dataset</code> / <code className="inline-code">run_experiment</code> / <code className="inline-code">wait_for_completion</code>{t('guidePage.externalSection.clientPart4')}<code className="inline-code">serve(retrieve_fn, generate_fn)</code>{t('guidePage.externalSection.clientPart5')}<code className="inline-code">/external-rag-spec</code>{t('guidePage.externalSection.clientPart6')}<code className="inline-code">docs/connector-guide.md</code>{t('guidePage.externalSection.clientPart7')}<code className="inline-code">examples/connector_quickstart/</code>{t('guidePage.externalSection.clientPart8')}
          </div>
        </div>
      </div>
      )
    },
  },
  {
    id: 'metrics', label: 'guidePage.nav.metrics', icon: BarChart2, group: 'run',
    content: () => {
      const { t } = useTranslation()
      return (
      <div>
        <p className="guide-lead">{t('guidePage.metricsSection.leadPart1')}<code className="inline-code">_CompositeEvaluator</code>{t('guidePage.metricsSection.leadPart2')}<code className="inline-code">article_refs</code>{t('guidePage.metricsSection.leadPart3')}<code className="inline-code">faithfulness</code>/<code className="inline-code">answer_relevancy</code>/<code className="inline-code">reference_overlap</code>{t('guidePage.metricsSection.leadPart4')}</p>
        <div className="guide-callout warn mb-16">
          <AlertTriangle size={14} />
          <div><strong>{t('guidePage.metricsSection.gatingStrong')}</strong> <code className="inline-code">retrieval_recall_at_k</code>/<code className="inline-code">precision</code>/<code className="inline-code">answer_similarity</code>/<code className="inline-code">context_support</code>{t('guidePage.metricsSection.gatingPart1')}<code className="inline-code">answerable</code>{t('guidePage.metricsSection.gatingPart2')}<code className="inline-code">correct_refusal</code>{t('guidePage.metricsSection.gatingPart3')}<code className="inline-code">uncovered</code>/<code className="inline-code">out_of_scope</code>{t('guidePage.metricsSection.gatingPart4')}</div>
        </div>
        <MetricRow name="retrieval_recall_at_k"      formula={t('guidePage.metricsSection.formulaRecall')}        good="> 0.6" warn="< 0.3" note={t('guidePage.metricsSection.noteRecall')} />
        <MetricRow name="retrieval_precision_at_k"   formula={t('guidePage.metricsSection.formulaPrecision')}      good="> 0.3" warn="< 0.1" note={t('guidePage.metricsSection.notePrecision')} />
        <MetricRow name="answer_similarity"          formula={t('guidePage.metricsSection.formulaAnswerSimilarity')}     good="> 0.7" warn="< 0.4" note={t('guidePage.metricsSection.noteAnswerSimilarity')} />
        <MetricRow name="context_support"            formula={t('guidePage.metricsSection.formulaContextSupport')}   good="> 0.6" warn="< 0.3" note={t('guidePage.metricsSection.noteContextSupport')} />
        <MetricRow name="grounded_in_correct_source"  formula={t('guidePage.metricsSection.formulaGrounded')} good="> 0.6" warn="< 0.4" note={t('guidePage.metricsSection.noteGrounded')} />
        <MetricRow name="correct_refusal"             formula={t('guidePage.metricsSection.formulaCorrectRefusal')}   good="> 0.8" warn="< 0.6" note={t('guidePage.metricsSection.noteCorrectRefusal')} />
        <div className="guide-callout warn mt-12">
          <AlertTriangle size={14} />
          <div><strong>{t('guidePage.metricsSection.blindSpotStrong')}</strong>{t('guidePage.metricsSection.blindSpotPart1')}<code className="inline-code">retrieval_recall_at_k</code>{t('guidePage.metricsSection.blindSpotPart2')}<code className="inline-code">answer_similarity</code>{t('guidePage.metricsSection.blindSpotPart3')}</div>
        </div>
        <div className="guide-callout info mt-12">
          <Info size={14} />
          <div>{t('guidePage.metricsSection.deepEvalPart1')}<code className="inline-code">make test-eval</code>{t('guidePage.metricsSection.deepEvalPart2')}</div>
        </div>
      </div>
      )
    },
  },
  {
    id: 'funnel', label: 'guidePage.nav.funnel', icon: Layers, group: 'diagnose',
    content: () => {
      const { t } = useTranslation()
      return (
      <div>
        <p className="guide-lead">{t('guidePage.funnelSection.leadPart1')}<code className="inline-code">core/eval/funnel.py:diagnose_question</code>{t('guidePage.funnelSection.leadPart2')}</p>
        <FunnelDiagram />
        <div className="guide-callout warn mt-16">
          <AlertTriangle size={14} />
          <div><strong>{t('guidePage.funnelSection.findingStrong')}</strong>{t('guidePage.funnelSection.findingPart1')}<code className="inline-code">retrieval_recall_at_k=0</code>{t('guidePage.funnelSection.findingPart2')}<code className="inline-code">answer_similarity</code>{t('guidePage.funnelSection.findingPart3')}</div>
        </div>
        <div className="guide-component-list mt-12">
          <div className="guide-component-row">
            <Share2 size={15} className="who-icon yellow" />
            <div><strong>retrieval vs rerank</strong><span className="guide-component-desc">{t('guidePage.funnelSection.retrievalVsRerankPart1')}<code className="inline-code">Answer.pre_rerank_source_refs</code>{t('guidePage.funnelSection.retrievalVsRerankPart2')}</span></div>
          </div>
          <div className="guide-component-row">
            <XCircle size={15} className="who-icon red" />
            <div><strong>not_applicable</strong><span className="guide-component-desc"><code className="inline-code">uncovered</code>/<code className="inline-code">out_of_scope</code>{t('guidePage.funnelSection.notApplicablePart1')}<code className="inline-code">correct_refusal</code>{t('guidePage.funnelSection.notApplicablePart2')}</span></div>
          </div>
        </div>
      </div>
      )
    },
  },
  {
    // Phases 1.2 to 3 — what the platform does after the funnel names a layer.
    // The funnel section above says *where* a question failed; nothing said
    // *why*, *what to fix first*, or *what the fix is worth* until this.
    id: 'diagnosis', label: 'guidePage.nav.diagnosis', icon: Wrench, group: 'diagnose',
    content: () => {
      const { t } = useTranslation()
      const h3 = { fontSize: 'var(--fs-sm)', fontWeight: 700, color: 'var(--color-text)', margin: '20px 0 8px' } as const
      return (
      <div>
        <p className="guide-lead">{t('guidePage.diagnosisSection.lead')}</p>

        <h3 style={h3}>{t('guidePage.diagnosisSection.causeHeading')}</h3>
        <p>{t('guidePage.diagnosisSection.causePara1')}</p>
        <p>{t('guidePage.diagnosisSection.causePara2')}</p>
        <RootCauseFanDiagram />
        <p>{t('guidePage.diagnosisSection.causePara3')}</p>

        <h3 style={h3}>{t('guidePage.diagnosisSection.priorityHeading')}</h3>
        <p>{t('guidePage.diagnosisSection.priorityPara1')}</p>
        <p>{t('guidePage.diagnosisSection.priorityPara2')}</p>
        <div className="guide-callout info mt-12">
          <Info size={14} />
          <div>{t('guidePage.diagnosisSection.priorityCallout')}</div>
        </div>

        <h3 style={h3}>{t('guidePage.diagnosisSection.payoffHeading')}</h3>
        <p>{t('guidePage.diagnosisSection.payoffPara1')}</p>
        <p>{t('guidePage.diagnosisSection.payoffPara2')}</p>
        <p>{t('guidePage.diagnosisSection.payoffPara3')}</p>

        <h3 style={h3}>{t('guidePage.diagnosisSection.loopHeading')}</h3>
        <p>{t('guidePage.diagnosisSection.loopPara1')}</p>
        <ImprovementLoopDiagram />
        <p>{t('guidePage.diagnosisSection.loopPara2')}</p>
      </div>
      )
    },
  },
  {
    // Phase 4 — the document, and the verdict on whether it was carried out.
    id: 'prescription', label: 'guidePage.nav.prescription', icon: ClipboardList, group: 'deliver',
    content: () => {
      const { t } = useTranslation()
      const h3 = { fontSize: 'var(--fs-sm)', fontWeight: 700, color: 'var(--color-text)', margin: '20px 0 8px' } as const
      return (
      <div>
        <p className="guide-lead">{t('guidePage.prescriptionSection.lead')}</p>

        <h3 style={h3}>{t('guidePage.prescriptionSection.whyHeading')}</h3>
        <p>{t('guidePage.prescriptionSection.whyPara1')}</p>
        <p>{t('guidePage.prescriptionSection.whyPara2')}</p>

        <h3 style={h3}>{t('guidePage.prescriptionSection.contentHeading')}</h3>
        <p>{t('guidePage.prescriptionSection.contentPara1')}</p>
        <p>{t('guidePage.prescriptionSection.contentPara2')}</p>
        <p>{t('guidePage.prescriptionSection.contentPara3')}</p>

        <h3 style={h3}>{t('guidePage.prescriptionSection.acceptanceHeading')}</h3>
        <p>{t('guidePage.prescriptionSection.acceptancePara1')}</p>
        <p>{t('guidePage.prescriptionSection.acceptancePara2')}</p>
        <div className="guide-callout warn mt-12">
          <AlertTriangle size={14} />
          <div>{t('guidePage.prescriptionSection.acceptanceCallout')}</div>
        </div>

        <h3 style={h3}>{t('guidePage.prescriptionSection.whereHeading')}</h3>
        <p>{t('guidePage.prescriptionSection.wherePara1')}</p>
      </div>
      )
    },
  },
  {
    // Phase 5 — choosing between configurations, and checking the metric that
    // does the choosing.
    id: 'frontier', label: 'guidePage.nav.frontier', icon: ChartScatter, group: 'diagnose',
    content: () => {
      const { t } = useTranslation()
      const h3 = { fontSize: 'var(--fs-sm)', fontWeight: 700, color: 'var(--color-text)', margin: '20px 0 8px' } as const
      return (
      <div>
        <DraftNote />
        <p className="guide-lead">{t('guidePage.frontierSection.lead')}</p>
        <FrontierPlotDiagram />

        <h3 style={h3}>{t('guidePage.frontierSection.readHeading')}</h3>
        <p>{t('guidePage.frontierSection.readPara1')}</p>
        <p>{t('guidePage.frontierSection.readPara2')}</p>

        <h3 style={h3}>{t('guidePage.frontierSection.latencyHeading')}</h3>
        <div className="guide-callout warn mb-8">
          <AlertTriangle size={14} />
          <div>{t('guidePage.frontierSection.latencyCallout')}</div>
        </div>
        <p>{t('guidePage.frontierSection.latencyPara1')}</p>

        <h3 style={h3}>{t('guidePage.frontierSection.trustHeading')}</h3>
        <p>{t('guidePage.frontierSection.trustPara1')}</p>
        <p>{t('guidePage.frontierSection.trustPara2')}</p>
        <p>{t('guidePage.frontierSection.trustPara3')}</p>
      </div>
      )
    },
  },
  {
    // Phase 7 — the questions nobody thought of in advance.
    id: 'production', label: 'guidePage.nav.production', icon: SatelliteDish, group: 'deliver',
    content: () => {
      const { t } = useTranslation()
      const h3 = { fontSize: 'var(--fs-sm)', fontWeight: 700, color: 'var(--color-text)', margin: '20px 0 8px' } as const
      return (
      <div>
        <DraftNote />
        <p className="guide-lead">{t('guidePage.productionSection.lead')}</p>
        <ProductionLoopDiagram />

        <h3 style={h3}>{t('guidePage.productionSection.pullHeading')}</h3>
        <p>{t('guidePage.productionSection.pullPara1')}</p>
        <p>{t('guidePage.productionSection.pullPara2')}</p>

        <h3 style={h3}>{t('guidePage.productionSection.candidateHeading')}</h3>
        <p>{t('guidePage.productionSection.candidatePara1')}</p>
        <div className="guide-callout info mt-12">
          <Info size={14} />
          <div>{t('guidePage.productionSection.candidateCallout')}</div>
        </div>

        <h3 style={h3}>{t('guidePage.productionSection.promoteHeading')}</h3>
        <p>{t('guidePage.productionSection.promotePara1')}</p>
        <p>{t('guidePage.productionSection.promotePara2')}</p>

        <h3 style={h3}>{t('guidePage.productionSection.coverageHeading')}</h3>
        <p>{t('guidePage.productionSection.coveragePara1')}</p>
        <p>{t('guidePage.productionSection.coveragePara2')}</p>
      </div>
      )
    },
  },
  {
    // Phase 6 — the artefact, reachable from the judgments page's toolbar.
    id: 'bundle', label: 'guidePage.nav.bundle', icon: Package, group: 'deliver',
    content: () => {
      const { t } = useTranslation()
      const h3 = { fontSize: 'var(--fs-sm)', fontWeight: 700, color: 'var(--color-text)', margin: '20px 0 8px' } as const
      return (
      <div>
        <p className="guide-lead">{t('guidePage.bundleSection.lead')}</p>
        <BundleDiagram />

        <h3 style={h3}>{t('guidePage.bundleSection.anchorHeading')}</h3>
        <p>{t('guidePage.bundleSection.anchorPara1')}</p>
        <p>{t('guidePage.bundleSection.anchorPara2')}</p>

        <h3 style={h3}>{t('guidePage.bundleSection.thresholdHeading')}</h3>
        <p>{t('guidePage.bundleSection.thresholdPara1')}</p>
        <p>{t('guidePage.bundleSection.thresholdPara2')}</p>
        <div className="guide-callout warn mt-12">
          <AlertTriangle size={14} />
          <div>{t('guidePage.bundleSection.thresholdCallout')}</div>
        </div>

        <h3 style={h3}>{t('guidePage.bundleSection.whereHeading')}</h3>
        <p>{t('guidePage.bundleSection.wherePara1')}</p>
      </div>
      )
    },
  },
  {
    id: 'trace', label: 'guidePage.nav.trace', icon: Clock, group: 'reference',
    content: () => {
      const { t } = useTranslation()
      return (
      <div>
        <p className="guide-lead">{t('guidePage.traceSection.leadPart1')}<code className="inline-code">stage_trace</code>{t('guidePage.traceSection.leadPart2')}</p>
        <TraceExplainer />
        <div className="guide-trace-legend">
          {[
            { tone: 'ok', label: t('guidePage.traceSection.legendBelowThreshold') },
            { tone: 'warn', label: t('guidePage.traceSection.legendModeratelySlow') },
            { tone: 'bad', label: t('guidePage.traceSection.legendBottleneck') },
          ].map(({ tone, label }) => (
            <span key={label} className="guide-trace-legend-item">
              <span className={`guide-trace-legend-dot tone-${tone}`} />{label}
            </span>
          ))}
        </div>
        <div className="guide-callout info mt-12">
          <Info size={14} />
          <div>
            <strong>Generate &gt; 80%</strong>{t('guidePage.traceSection.calloutPart1')}<br />
            <strong>embed_ms &gt; 500ms</strong>{t('guidePage.traceSection.calloutPart2')}<br />
            {t('guidePage.traceSection.calloutPart3')}<br />
            {t('guidePage.traceSection.calloutPart4')}<code className="inline-code">GET /trace/latest</code>{t('guidePage.traceSection.calloutPart5')}
          </div>
        </div>
      </div>
      )
    },
  },
  {
    id: 'mongodb', label: 'guidePage.nav.mongodb', icon: Database, group: 'reference',
    content: () => {
      const { t } = useTranslation()
      return (
      <div>
        <p className="guide-lead">{t('guidePage.mongodbSection.lead')}</p>
        <div className="guide-callout info mb-16">
          <Info size={14} />
          <div>
            <strong>MongoDB Compass:</strong> <code className="inline-code">mongodb://localhost:27017</code> → database: <code className="inline-code">ragplatform</code>
          </div>
        </div>
        <MongoCollections />
        <div className="guide-callout warn mt-16">
          <AlertTriangle size={14} />
          <div>{t('guidePage.mongodbSection.notInMongoPart1')}<strong>{t('guidePage.mongodbSection.notInMongoStrong')}</strong>{t('guidePage.mongodbSection.notInMongoPart2')}<code className="inline-code">corpus/demo_handbook/</code>{t('guidePage.mongodbSection.notInMongoPart3')}</div>
        </div>
      </div>
      )
    },
  },
  {
    id: 'problems', label: 'guidePage.nav.problems', icon: AlertTriangle, group: 'reference',
    content: () => {
      const { t } = useTranslation()
      return (
      <div>
        <p className="guide-lead">{t('guidePage.problemsSection.lead')}</p>
        {/* Two catalogues, and they are not the same catalogue. What a served
            system can suffer belongs to the failure atlas, where an entry
            claims a signal catches it and the build refuses that claim without
            a bait. What is left here is this platform's own history: defects
            of the software you are reading and of running it. Fourteen cards
            stood here that belonged in the atlas, and a reader had no way to
            tell which kind a card was. */}
        <p className="guide-lead">
          <Trans i18nKey="guidePage.problemsSection.elsewhere">
            <Link to="/atlas" />
          </Trans>
        </p>
        <div className="stack-8">
          <ProblemCard title={t('guidePage.problemsSection.problem5Title')} symptom={t('guidePage.problemsSection.problem5Symptom')} cause={t('guidePage.problemsSection.problem5Cause')} fix={t('guidePage.problemsSection.problem5Fix')} />
          <ProblemCard title={t('guidePage.problemsSection.problem14Title')} symptom={t('guidePage.problemsSection.problem14Symptom')} cause={t('guidePage.problemsSection.problem14Cause')} fix={t('guidePage.problemsSection.problem14Fix')} />
        </div>
      </div>
      )
    },
  },
  {
    id: 'faq', label: 'guidePage.nav.faq', icon: HelpCircle, group: 'reference',
    content: () => {
      const { t } = useTranslation()
      return (
      <div>
        <p className="guide-lead">{t('guidePage.faqSection.lead')}</p>
        <div className="guide-faq-list">
          {[
            { q: t('guidePage.faqSection.q1'), a: t('guidePage.faqSection.a1') },
            { q: t('guidePage.faqSection.q2'), a: t('guidePage.faqSection.a2') },
            { q: t('guidePage.faqSection.q3'), a: t('guidePage.faqSection.a3') },
            { q: t('guidePage.faqSection.q4'), a: t('guidePage.faqSection.a4') },
            { q: t('guidePage.faqSection.q5'), a: t('guidePage.faqSection.a5') },
            { q: t('guidePage.faqSection.q6'), a: t('guidePage.faqSection.a6') },
            { q: t('guidePage.faqSection.q7'), a: t('guidePage.faqSection.a7') },
            { q: t('guidePage.faqSection.q8'), a: t('guidePage.faqSection.a8') },
          ].map(({ q, a }) => (
            <div key={q} className="guide-faq-item">
              <div className="guide-faq-q"><HelpCircle size={12} />{q}</div>
              <div className="guide-faq-a">{a}</div>
            </div>
          ))}
        </div>
      </div>
      )
    },
  },
]

// Reading order, stated once and used by both the navigation and the map at
// the top of the first section. An id missing from this list still renders, at
// the end of its group — a section must never disappear from the navigation
// because somebody forgot to add it here.
const READING_ORDER: string[] = [
  'purpose', 'quickstart', 'arch',
  'realm', 'shell',
  'corpus', 'chunking', 'deep-diagnostics',
  'config', 'external', 'metrics', 'regression',
  'funnel', 'diagnosis', 'judgments', 'triage', 'frontier',
  'prescription', 'production', 'bundle',
  'trace', 'mongodb', 'problems', 'faq',
]

const GROUP_ORDER: GuideGroup[] = ['intro', 'shell', 'data', 'run', 'diagnose', 'deliver', 'reference']

const GROUP_LABEL: Record<GuideGroup, string> = {
  intro: 'guidePage.groups.intro',
  shell: 'guidePage.groups.shell',
  data: 'guidePage.groups.data',
  run: 'guidePage.groups.run',
  diagnose: 'guidePage.groups.diagnose',
  deliver: 'guidePage.groups.deliver',
  reference: 'guidePage.groups.reference',
}

function orderedSections() {
  const rank = (id: string) => {
    const i = READING_ORDER.indexOf(id)
    return i === -1 ? READING_ORDER.length : i
  }
  return GROUP_ORDER.map(group => ({
    group,
    items: SECTIONS.filter(s => s.group === group).sort((a, b) => rank(a.id) - rank(b.id)),
  })).filter(g => g.items.length > 0)
}

/* ── Where to start, and what follows ──
 * Five groups as five stops, each linking into its own first section. It earns
 * its place at the top of the guide because twenty-two sections give a reader
 * no way to tell which four they actually need first. */
function ReadingMapDiagram({ onGo }: { onGo: (id: string) => void }) {
  const { t } = useTranslation()
  const stops: { group: GuideGroup; first: string; color: string }[] = [
    { group: 'intro', first: 'purpose', color: 'var(--diag-blue)' },
    { group: 'shell', first: 'realm', color: 'var(--diag-cyan)' },
    { group: 'data', first: 'corpus', color: 'var(--diag-cyan)' },
    { group: 'run', first: 'config', color: 'var(--diag-violet)' },
    { group: 'diagnose', first: 'funnel', color: 'var(--diag-orange)' },
    { group: 'deliver', first: 'prescription', color: 'var(--diag-orange)' },
    { group: 'reference', first: 'problems', color: 'var(--diag-green)' },
  ]
  return (
    <div className="reading-map">
      {stops.map((stop, i) => (
        <button
          key={stop.group} type="button" className="reading-map-stop"
          // The guide's one inline style, and it is not about a size: a stop
          // on the map passes its own tint as a property that the stylesheet
          // reads in four places. As a class over seven groups this would be
          // seven rules instead of one.
          style={{ ['--stop-color' as string]: stop.color }}
          onClick={() => onGo(stop.first)}
        >
          <span className="reading-map-num">{i + 1}</span>
          <span className="reading-map-title">{t(GROUP_LABEL[stop.group])}</span>
          <span className="reading-map-desc">{t(`guidePage.groups.${stop.group}Desc`)}</span>
          <span className="reading-map-go">{t('guidePage.groups.start')} →</span>
        </button>
      ))}
    </div>
  )
}

export default function GuidePage() {
  const { t } = useTranslation()
  const [searchParams] = useSearchParams()
  // Deep-link support (?section=judgments) — e.g. JudgmentsPage.tsx links
  // here so a reviewer lands directly on the relevant section instead of
  // "Overview". Read once at mount, same "don't clobber a manual nav click
  // afterward" convention JudgmentsPage.tsx's own URL-prefill already uses.
  // The parameter is a query, not a hash: a link written as /guide#judgments
  // silently lands on the overview, which is how this was found.
  const [active, setActive] = useState(() => searchParams.get('section') || 'purpose')
  const current = SECTIONS.find(s => s.id === active) ?? SECTIONS[0]
  const Content = current.content

  return (
    <div className="page page-wide">
      <h1 className="page-title mb-16">{t('guidePage.title')}</h1>
      {/* Three full-width columns rather than two under a 1200px cap. The
          prose inside keeps its measure (`--measure`) while diagrams, tables
          and code samples take whatever is left: before this the funnel
          diagram squeezed to half size to fit the width of a paragraph. */}
      <div className="guide-grid">
        <nav className="guide-nav" aria-label={t('guidePage.title')}>
          {orderedSections().map(({ group, items }) => (
            <div key={group}>
              <div className="guide-nav-group">{t(GROUP_LABEL[group])}</div>
              {items.map(s => {
                const Icon = s.icon
                return (
                  <button key={s.id} onClick={() => setActive(s.id)}
                    className={`guide-nav-item ${active === s.id ? 'active' : ''}`}>
                    <Icon size={13} />
                    {t(s.label)}
                  </button>
                )
              })}
            </div>
          ))}
        </nav>
        <div className="guide-content">
          <h2 className="guide-section-title">
            <current.icon size={18} className="dim-icon" />
            {t(current.label)}
          </h2>
          {/* The map opens the guide rather than sitting inside the purpose
              section's own text: it answers "where do I start", which is a
              question about the guide and not about the platform. */}
          {active === 'purpose' && <ReadingMapDiagram onGo={setActive} />}
          <Content />
        </div>
        {/* "On this page". It disappears below 1100px: a column you have to
            scroll to see what it indexes is not an index. */}
        <aside className="guide-aside" aria-label={t('guidePage.onThisPage')}>
          <div className="eyebrow toc-eyebrow">{t('guidePage.onThisPage')}</div>
          <div className="guide-aside-meta">
            {t('guidePage.sectionOf', {
              n: SECTIONS.findIndex(x => x.id === active) + 1, total: SECTIONS.length,
              group: t(GROUP_LABEL[current.group]),
            })}
          </div>
        </aside>
      </div>
    </div>
  )
}
