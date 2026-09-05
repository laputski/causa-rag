import { useState, useRef, useEffect } from 'react'
import { Link, useParams, useSearchParams, useNavigate } from 'react-router-dom'
import { useQuery, useMutation, useQueryClient, keepPreviousData } from '@tanstack/react-query'
import { useTranslation } from 'react-i18next'
import GuideLink from '../components/GuideLink'
import { Upload, Database, Heart, Network } from 'lucide-react'
import SelectBox from '../components/SelectBox'
import { useDefaultCorpus } from '../hooks/useDefaultCorpus'
import { api, type CorpusIngest, type CorpusHealthItem, type CorpusRegistryEntry, type DiagnosticsMetricsResult } from '../api/client'
import { useConfirm } from '../components/ConfirmDialog'
import GraphCommunityView from '../components/GraphCommunityView'
import { useRealm, useRealmPath } from '../context/RealmContext'

// Their own keys rather than the sidebar's: these are tab labels now, and the
// sidebar has one entry above them. Reusing `nav.dataUpload` would have named
// the tab "Corpus" inside a page already called "Corpus".
const TAB_META: Record<CorpusTab, { icon: React.ElementType; key: string }> = {
  upload:  { icon: Upload,   key: 'corpusPage.tabs.upload' },
  content: { icon: Database, key: 'corpusPage.tabs.content' },
  health:  { icon: Heart,    key: 'corpusPage.tabs.health' },
  graph:   { icon: Network,  key: 'corpusPage.tabs.graph' },
}

const VALID_TABS = ['upload', 'content', 'health', 'graph'] as const
type CorpusTab = typeof VALID_TABS[number]

const API_BASE = import.meta.env.VITE_API_BASE ?? 'http://localhost:8081'
const WS_BASE = API_BASE.replace(/^http/, 'ws')

interface ProgressEvent {
  type: 'start' | 'progress' | 'done' | 'error'
  message?: string
  n_chunks?: number
  hit_ratio?: number
}

// The upload form's original mode (chunk raw .txt/.md through the
// pipeline) is one of four now; the other three load a dump already
// prepared in one backend's own shape directly (see corpus.py's "Alternate
// dump ingestion" endpoints) — each has genuinely different mechanics and
// file format, not just a different endpoint URL.
const DUMP_TYPES = ['text', 'qdrant-snapshot', 'neo4j-cypher', 'neo4j-chunks', 'opensearch-dump'] as const
type SortKey = 'document' | 'shortest' | 'longest'

type DumpType = typeof DUMP_TYPES[number]

const DUMP_ENDPOINTS: Record<Exclude<DumpType, 'text'>, string> = {
  'qdrant-snapshot': '/corpus/ingest/qdrant-snapshot',
  'neo4j-cypher': '/corpus/ingest/neo4j-cypher',
  'neo4j-chunks': '/corpus/ingest/neo4j-chunks',
  'opensearch-dump': '/corpus/ingest/opensearch-dump',
}

const DUMP_FILE_ACCEPT: Record<DumpType, string> = {
  text: '.txt,.md',
  'qdrant-snapshot': '.snapshot',
  'neo4j-cypher': '.cypher,.cql,.txt',
  // Found live: raw Cypher-script uploads inline free-text as literal
  // string values, which breaks on embedded quotes a real-world exporter
  // didn't escape (see neo4j-chunks' hint text below) — this mode instead
  // sends the same :Chunk shape through Neo4j driver parameters, same
  // NDJSON/JSON/gzip/zip acceptance as the OpenSearch dump mode.
  'neo4j-chunks': '.ndjson,.jsonl,.json,.gz,.zip',
  'opensearch-dump': '.ndjson,.jsonl,.json,.gz,.zip',
}

function StatusBadge({ status }: { status: CorpusIngest['status'] }) {
  const map = { running: ['🔄', 'badge-warn'], done: ['✅', 'badge-success'], error: ['❌', 'badge-danger'] } as const
  const [icon, cls] = map[status] ?? ['?', '']
  return <span className={`badge ${cls}`}>{icon} {status}</span>
}

export default function CorpusPage() {
  const { confirm, dialog } = useConfirm()
  const { t } = useTranslation()
  const qc = useQueryClient()
  const { tab: tabParam } = useParams<{ tab?: string }>()
  const navigate = useNavigate()
  const toRealm = useRealmPath()
  // The four views came back into this page as tabs. As four separate sidebar
  // items they claimed to be four subjects while being four views of one
  // corpus, and they were named in two different categories at once: one
  // action (Upload) beside three things (Content, Health, Graph).
  // The routes are unchanged, so every existing link and bookmark still lands
  // on the same view.
  const tab: CorpusTab = (VALID_TABS as readonly string[]).includes(tabParam ?? '') ? (tabParam as CorpusTab) : 'upload'
  const { activeRealmId } = useRealm()
  const [searchParams] = useSearchParams()
  const { data: history = [], refetch } = useQuery({
    queryKey: ['corpus', activeRealmId],
    queryFn: () => api.corpus.list(activeRealmId),
    staleTime: 5000,
  })
  // The `corpora` registry, not ingest-job history above: this is
  // what actually knows about corpora ingested outside the upload form (CLI,
  // the migration script) — history alone left this dropdown empty for
  // every real corpus that predates or bypasses /corpus/ingest.
  const { data: collections = [] } = useQuery({
    queryKey: ['corpus-collections', activeRealmId],
    queryFn: () => api.corpus.collections(activeRealmId),
    staleTime: 5000,
  })
  const { data: registry } = useQuery({ queryKey: ['registry', activeRealmId], queryFn: () => api.registry(activeRealmId) })

  const [files, setFiles] = useState<File[]>([])
  const [strategy, setStrategy] = useState('structure_aware')
  const [chunkSize, setChunkSize] = useState(512)
  const [overlap, setOverlap] = useState(64)
  const [exclude, setExclude] = useState('full.txt')
  const [corpusId, setCorpusId] = useState('')
  const [dragging, setDragging] = useState(false)
  const [progress, setProgress] = useState<ProgressEvent[]>([])
  const [_jobId, setJobId] = useState<string | null>(null)
  const [uploading, setUploading] = useState(false)
  const [dumpType, setDumpType] = useState<DumpType>('text')
  const [confirmCypher, setConfirmCypher] = useState(false)
  const [replaceIndex, setReplaceIndex] = useState(false)
  // Neo4j has no corpus_id partitioning (see the Cypher endpoint's own
  // docstring) — unlike replaceIndex above, this wipes the WHOLE graph,
  // not just this corpus's slice.
  const [replaceGraph, setReplaceGraph] = useState(false)
  const [dumpResult, setDumpResult] = useState<{ ok: boolean; message: string } | null>(null)
  // A shortcut link (e.g. from an ExternalRag card on "Resources"
  // with uses_realm_resources) can pre-select which corpus_id to open here.
  const [browseCorpusId, setBrowseCorpusId] = useState(searchParams.get('corpus_id') || '')
  const fileRef = useRef<HTMLInputElement>(null)
  const logRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    logRef.current?.scrollTo(0, logRef.current.scrollHeight)
  }, [progress])

  const deleteMut = useMutation({
    mutationFn: api.corpus.delete,
    onSuccess: () => qc.invalidateQueries({ queryKey: ['corpus'] }),
  })

  const chunkers: string[] = registry?.chunker ?? ['fixed', 'structure_aware', 'sentence', 'paragraph']

  useDefaultCorpus(activeRealmId, browseCorpusId, setBrowseCorpusId, {
    skip: Boolean(searchParams.get('corpus_id')),
  })
  // Only corpora that actually exist, in the registry or in the ingest log.
  // `default` used to be added unconditionally, so on a realm without such a
  // corpus the list offered an id leading to an empty collection, with nothing
  // to explain what it meant. Where a corpus by that name does exist, it
  // arrives from the registry with a description of its own.
  const knownCorpusIds = Array.from(new Set([
    ...collections.map(c => c.corpus_id),
    ...history.map(h => h.corpus_id).filter((id): id is string => !!id),
  ]))

  function onDrop(e: React.DragEvent) {
    e.preventDefault(); setDragging(false)
    const dropped = Array.from(e.dataTransfer.files)
    setFiles(dumpType === 'text' ? dropped.filter(f => f.name.endsWith('.txt') || f.name.endsWith('.md')) : dropped.slice(0, 1))
  }

  async function startIngest() {
    if (!files.length || uploading) return
    setUploading(true)
    setProgress([])
    setJobId(null)

    const form = new FormData()
    files.forEach(f => form.append('files', f))
    form.append('strategy', strategy)
    form.append('chunk_size', String(chunkSize))
    form.append('overlap', String(overlap))
    form.append('exclude', exclude)
    form.append('corpus_id', corpusId)
    if (activeRealmId) form.append('realm_id', activeRealmId)

    try {
      const res = await fetch(`${API_BASE}/corpus/ingest`, { method: 'POST', body: form })
      const data = await res.json()
      const jid: string = data.job_id
      setJobId(jid)

      const ws = new WebSocket(`${WS_BASE}/corpus/progress/${jid}`)
      ws.onmessage = (e) => {
        const ev: ProgressEvent = JSON.parse(e.data)
        setProgress(prev => [...prev, ev])
        if (ev.type === 'done' || ev.type === 'error') {
          setUploading(false)
          refetch()
          ws.close()
        }
      }
      ws.onerror = () => { setUploading(false); ws.close() }
    } catch (err) {
      setProgress(prev => [...prev, { type: 'error', message: String(err) }])
      setUploading(false)
    }
  }

  // The three alternate dump kinds (qdrant-snapshot/neo4j-cypher/opensearch-
  // dump) return one JSON response directly — no background job/WebSocket,
  // unlike text-chunking above, which can run minutes over many files.
  async function startAltIngest() {
    if (dumpType === 'text' || !files.length || uploading) return
    setUploading(true)
    setDumpResult(null)

    // strategy/embedder are naming components for the target collection/
    // index (corpus.py's _collection_name/_index_name), not chunking
    // settings — nothing is chunked on these paths. Left unset here, the
    // backend's own defaults (structure_aware/bge_m3, the same defaults the
    // query side assumes elsewhere) apply, rather than exposing a picker the
    // user has no principled way to choose from for a pre-made dump.
    const form = new FormData()
    form.append('file', files[0])
    form.append('corpus_id', corpusId)
    if (activeRealmId) form.append('realm_id', activeRealmId)
    if (dumpType === 'neo4j-cypher') {
      form.append('confirm_arbitrary_cypher', String(confirmCypher))
      form.append('replace', String(replaceGraph))
    } else if (dumpType === 'neo4j-chunks') {
      form.append('replace', String(replaceGraph))
    } else if (dumpType === 'opensearch-dump') {
      form.append('replace', String(replaceIndex))
    }

    try {
      const res = await fetch(`${API_BASE}${DUMP_ENDPOINTS[dumpType]}`, { method: 'POST', body: form })
      const data = await res.json()
      if (!res.ok) {
        // `detail` is usually a plain string (the backend's own, untranslated
        // system message — nothing to localize about e.g. a raw Qdrant/Neo4j
        // error). For failures with an actionable next step, the backend
        // instead sends {message, hint} — message stays as-is, but `hint` is
        // an i18n key under dumpErrorHints, looked up here so the
        // explanation follows the UI's selected language rather than the
        // backend hardcoding one language into the response.
        const detail = data.detail
        if (detail && typeof detail === 'object' && detail.hint) {
          throw new Error(`${detail.message}\n\n${t(`corpusPage.upload.dumpErrorHints.${detail.hint}`)}`)
        }
        throw new Error(detail || res.statusText)
      }

      const message =
        dumpType === 'qdrant-snapshot' ? t('corpusPage.upload.dumpSuccessQdrant', { collection: data.collection }) :
        dumpType === 'neo4j-cypher' ? t('corpusPage.upload.dumpSuccessNeo4j', { count: data.statements_executed }) +
          (data.replaced ? ` ${t('corpusPage.upload.dumpSuccessNeo4jReplaced')}` : '') :
        dumpType === 'neo4j-chunks' ? t('corpusPage.upload.dumpSuccessNeo4jChunks', { written: data.n_written, skipped: data.n_skipped }) +
          (data.replaced ? ` ${t('corpusPage.upload.dumpSuccessNeo4jReplaced')}` : '') :
        t('corpusPage.upload.dumpSuccessOpensearch', { indexed: data.n_indexed, skipped: data.n_skipped, errors: data.n_errors }) +
          (data.replaced ? ` ${t('corpusPage.upload.dumpSuccessOpensearchReplaced')}` : '')
      setDumpResult({ ok: true, message })
      qc.invalidateQueries({ queryKey: ['corpus-collections'] })
      qc.invalidateQueries({ queryKey: ['corpus-collections-managed'] })
    } catch (err) {
      setDumpResult({ ok: false, message: String(err) })
    } finally {
      setUploading(false)
    }
  }

  return (
    <div className="page">
      {dialog}
      <h1 className="page-title">{t('corpusPage.title')}</h1>

      {/* The tab keeps its own address (`/data/<tab>`) rather than living in
          component state, so a view stays linkable and the browser's back
          button still moves between views. */}
      <div className="data-tab-bar" role="tablist">
        {VALID_TABS.map(name => {
          const { icon: Icon, key } = TAB_META[name]
          return (
            <button
              key={name} type="button" role="tab" aria-selected={tab === name}
              className={`data-tab${tab === name ? ' active' : ''}`}
              onClick={() => navigate(toRealm(`/data/${name}`))}
            >
              <Icon size={14} aria-hidden="true" /> {t(key)}
            </button>
          )
        })}
      </div>

      {tab === 'content' && (
        <CorpusContentTab corpusId={browseCorpusId} setCorpusId={setBrowseCorpusId} knownIds={knownCorpusIds} realmId={activeRealmId} />
      )}
      {tab === 'health' && (
        <CorpusHealthTab corpusId={browseCorpusId} setCorpusId={setBrowseCorpusId} knownIds={knownCorpusIds} realmId={activeRealmId} />
      )}
      {tab === 'graph' && (
        <>
          <div className="table-toolbar">
            <CorpusIdSelector corpusId={browseCorpusId} setCorpusId={setBrowseCorpusId} knownIds={knownCorpusIds} />
          </div>
          <GraphCommunityView corpusId={browseCorpusId} realmId={activeRealmId} />
        </>
      )}

      {tab === 'upload' && <>
        <div className="page-head">
          <h1 className="page-title">{t('corpusPage.upload.pageTitle')}</h1>
          <p className="page-sub">{browseCorpusId}</p>
          <span className="page-act">
            <GuideLink section="corpus" />
            <CorpusIdSelector corpusId={browseCorpusId} setCorpusId={setBrowseCorpusId} knownIds={knownCorpusIds} bare />
          </span>
        </div>

        <div className="section">
          <div className="section-rule flush">
            <h2 className="section-title">{t('corpusPage.upload.heading')}</h2>
          </div>
          <div className="form-grid upload-grid">

          <div className="form-group">
            <label htmlFor="dump-type">{t('corpusPage.upload.dumpTypeLabel')}</label>
            <select
              id="dump-type" value={dumpType} className="form-select"
              onChange={e => { setDumpType(e.target.value as DumpType); setFiles([]); setDumpResult(null) }}
            >
              {DUMP_TYPES.map(dt => <option key={dt} value={dt}>{t(`corpusPage.upload.dumpType.${dt}`)}</option>)}
            </select>
            <p className="upload-note">
              {t(`corpusPage.upload.dumpTypeHint.${dumpType}`)}
            </p>
          </div>

          <div
            className={`drop-zone ${dragging ? 'dragging' : ''}`}
            onDragOver={e => { e.preventDefault(); setDragging(true) }}
            onDragLeave={() => setDragging(false)}
            onDrop={onDrop}
            onClick={() => fileRef.current?.click()}
          >
            {files.length > 0
              ? <><strong>{files.length}</strong> {t('corpusPage.upload.filesSelected')}<br /><span className="text-muted">{files.map(f => f.name).join(', ').slice(0, 80)}</span></>
              : <>{t('corpusPage.upload.dropHere', { ext: DUMP_FILE_ACCEPT[dumpType] })}<br /><span className="text-muted">{t('corpusPage.upload.orClickToSelect')}</span></>
            }
            <input
              ref={fileRef} type="file" accept={DUMP_FILE_ACCEPT[dumpType]} multiple={dumpType === 'text'} hidden
              onChange={e => setFiles(Array.from(e.target.files ?? []).slice(0, dumpType === 'text' ? undefined : 1))}
            />
          </div>

          {dumpType === 'text' && (
            <div className="form-group gen-block">
              <label>{t('corpusPage.upload.chunkingStrategy')}</label>
              <select value={strategy} onChange={e => setStrategy(e.target.value)} className="form-select">
                {chunkers.map(c => <option key={c} value={c}>{c}</option>)}
              </select>
            </div>
          )}

          {dumpType === 'text' && <div className="form-pair">
            <div className="form-group">
              <label>{t('corpusPage.upload.chunkSize')}</label>
              <input type="number" value={chunkSize} onChange={e => setChunkSize(Number(e.target.value))}
                className="form-input" min={128} max={2048} step={64} />
            </div>
            <div className="form-group">
              <label>{t('corpusPage.upload.overlap')}</label>
              <input type="number" value={overlap} onChange={e => setOverlap(Number(e.target.value))}
                className="form-input" min={0} max={256} step={16} />
            </div>
          </div>}

          {dumpType === 'text' && (
            <div className="form-group">
              <label>{t('corpusPage.upload.excludeFiles')}</label>
              <input type="text" value={exclude} onChange={e => setExclude(e.target.value)}
                className="form-input" placeholder="full.txt" />
            </div>
          )}

          {dumpType === 'neo4j-cypher' && (
            <>
              <label className="form-group check-row">
                <input type="checkbox" checked={confirmCypher} onChange={e => setConfirmCypher(e.target.checked)} />
                <span>{t('corpusPage.upload.confirmCypher')}</span>
              </label>
              <label className="form-group check-row">
                <input type="checkbox" checked={replaceGraph} onChange={e => setReplaceGraph(e.target.checked)} />
                <span>{t('corpusPage.upload.replaceNeo4jGraph')}</span>
              </label>
            </>
          )}

          {dumpType === 'neo4j-chunks' && (
            <label className="form-group check-row">
              <input type="checkbox" checked={replaceGraph} onChange={e => setReplaceGraph(e.target.checked)} />
              <span>{t('corpusPage.upload.replaceNeo4jChunks')}</span>
            </label>
          )}

          {dumpType === 'opensearch-dump' && (
            <label className="form-group check-row">
              <input type="checkbox" checked={replaceIndex} onChange={e => setReplaceIndex(e.target.checked)} />
              <span>{t('corpusPage.upload.replaceOpensearchIndex')}</span>
            </label>
          )}

          <div className="form-group">
            <label>{t('corpusPage.upload.corpusIdLabel')}</label>
            <UploadCorpusIdField value={corpusId} onChange={setCorpusId} knownIds={knownCorpusIds} />
          </div>
          </div>

          <button
            className="btn-primary full-w"
            onClick={dumpType === 'text' ? startIngest : startAltIngest}
            disabled={!files.length || uploading || (dumpType === 'neo4j-cypher' && !confirmCypher)}
          >
            {uploading ? t('corpusPage.upload.indexing') : t('corpusPage.upload.uploadAndIndex')}
          </button>

          {dumpType !== 'text' && dumpResult && (
            <p className={`dump-result${dumpResult.ok ? ' ok' : ' bad'}`}>
              {dumpResult.message}
            </p>
          )}

          {progress.length > 0 && (
            <div ref={logRef} className="ingest-log">
              {progress.map((ev, i) => (
                <div key={i} className={`log-line log-${ev.type}`}>
                  {ev.type === 'done'
                    ? `✅ ${t('corpusPage.upload.logDone', { chunks: ev.n_chunks, hitRatio: ((ev.hit_ratio ?? 0) * 100).toFixed(0) })}`
                    : ev.type === 'error' ? `❌ ${ev.message}`
                    : ev.message?.includes('ingest.file_done') ? `· ${ev.message.split('file=')[1]?.split(' ')[0] ?? ''} (${ev.n_chunks} chunks total)`
                    : ev.message}
                </div>
              ))}
            </div>
          )}
        </div>

        {/* The ingest history runs full width: it has five columns, and in
            half the width they do not fit. */}
        <div className="section">
          <div className="section-rule">
            <h2 className="section-title">{t('corpusPage.history.heading')}</h2>
          </div>
          {history.length === 0
            ? <p className="text-muted">{t('corpusPage.history.empty')}</p>
            : (
              <table className="table table-sm">
                <thead>
                  <tr>
                    <th>{t('corpusPage.history.date')}</th><th>{t('corpusPage.history.strategy')}</th><th>{t('corpusPage.history.files')}</th><th>{t('corpusPage.history.chunks')}</th><th>{t('corpusPage.history.status')}</th><th></th>
                  </tr>
                </thead>
                <tbody>
                  {history.map((h: CorpusIngest) => (
                    <tr key={h.job_id}>
                      <td>{h.started_at ? new Date(h.started_at).toLocaleString('ru') : '—'}</td>
                      <td><code>{h.strategy}</code></td>
                      <td>{h.n_files}</td>
                      <td>{h.n_chunks}</td>
                      <td><StatusBadge status={h.status} /></td>
                      <td>
                        <button className="btn-sm btn-danger"
                          onClick={async () => { if (await confirm({ title: t('corpusPage.history.confirmDelete', { jobId: h.job_id }), danger: true })) deleteMut.mutate(h.job_id) }}>
                          ✕
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          <div className="corpus-foot">
            {t('corpusPage.history.reindexHint')}
            <br />{t('corpusPage.history.upsertHint')}
          </div>
        </div>
        <CorpusManagementSection realmId={activeRealmId} />
      </>}
    </div>
  )
}

// ── Corpus registry management (CRUD, was create+read only) ─────────

function CorpusManagementSection({ realmId }: { realmId: string | null }) {
  const { confirm, dialog } = useConfirm()
  const { t } = useTranslation()
  const qc = useQueryClient()
  const [showDeleted, setShowDeleted] = useState(false)
  const { data: corpora = [] } = useQuery({
    queryKey: ['corpus-collections-managed', realmId, showDeleted],
    queryFn: () => api.corpus.collections(realmId, showDeleted),
  })
  const [editingId, setEditingId] = useState<string | null>(null)
  const [editDescription, setEditDescription] = useState('')

  const invalidate = () => {
    qc.invalidateQueries({ queryKey: ['corpus-collections-managed'] })
    qc.invalidateQueries({ queryKey: ['corpus-collections'] })
  }
  const updateMut = useMutation({
    mutationFn: (vars: { id: string; description: string }) => api.corpus.updateCollection(vars.id, { description: vars.description }),
    onSuccess: () => { invalidate(); setEditingId(null) },
  })
  const deleteMut = useMutation({
    mutationFn: (id: string) => api.corpus.deleteCollection(id),
    onSuccess: invalidate,
  })
  const restoreMut = useMutation({
    mutationFn: (id: string) => api.corpus.restoreCollection(id),
    onSuccess: invalidate,
  })

  return (
    <div className="section">
      {dialog}
      <div className="section-rule flush">
        <h2 className="section-title">{t('corpusPage.management.heading')}</h2>
        <label className="rule-toggle">
          <input type="checkbox" checked={showDeleted} onChange={e => setShowDeleted(e.target.checked)} />
          {t('corpusPage.management.showDeleted')}
        </label>
      </div>

      {corpora.length === 0 ? (
        <p className="text-muted">{t('corpusPage.management.empty')}</p>
      ) : (
        <table className="table table-sm">
          <thead>
            <tr>
              <th>Corpus ID</th>
              <th>{t('corpusPage.management.storageType')}</th>
              <th>{t('corpusPage.management.backends')}</th>
              <th>{t('corpusPage.management.description')}</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {corpora.map((c: CorpusRegistryEntry) => (
              <tr key={c.id} style={c.deleted_at ? { opacity: 0.5 } : undefined}>
                <td><code>{c.corpus_id}</code></td>
                <td>{c.storage_type}</td>
                <td>{Object.keys(c.backends || {}).join(', ') || '—'}</td>
                <td>
                  {editingId === c.id ? (
                    <input
                      className="form-input" value={editDescription}
                      onChange={e => setEditDescription(e.target.value)}
                    />
                  ) : (c.description || <span className="text-muted">—</span>)}
                </td>
                <td className="nowrap">
                  {c.deleted_at ? (
                    <button className="btn-sm" onClick={() => restoreMut.mutate(c.id)} disabled={restoreMut.isPending}>
                      {t('corpusPage.management.restore')}
                    </button>
                  ) : editingId === c.id ? (
                    <>
                      <button
                        className="btn-sm btn-primary" disabled={updateMut.isPending}
                        onClick={() => updateMut.mutate({ id: c.id, description: editDescription })}
                      >
                        {t('corpusPage.management.save')}
                      </button>{' '}
                      <button className="btn-sm" onClick={() => setEditingId(null)}>{t('corpusPage.management.cancel')}</button>
                    </>
                  ) : (
                    <>
                      <button className="btn-sm" onClick={() => { setEditingId(c.id); setEditDescription(c.description || '') }}>
                        {t('corpusPage.management.edit')}
                      </button>{' '}
                      <button
                        className="btn-sm btn-danger"
                        onClick={async () => { if (await confirm({ title: t('corpusPage.management.confirmDelete', { corpusId: c.corpus_id }), danger: true })) deleteMut.mutate(c.id) }}
                      >
                        {t('corpusPage.management.delete')}
                      </button>
                    </>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  )
}

const CUSTOM_CORPUS_ID = '__custom_corpus_id__'

/** Real `<select>` of known corpus_ids + a "custom id" escape hatch for a
 * brand-new one — replaces a plain text `<input list="...">` (`<datalist>`)
 * that read as an empty text box pre-filled with "default", not an
 * intentional dropdown (most browsers only show datalist suggestions once
 * you start typing) — found live, same discoverability bug
 * DatasetsPage.tsx#CorpusSelect already fixed for the generation panel,
 * missed here on the upload form itself. */
function UploadCorpusIdField({ value, onChange, knownIds }: {
  value: string
  onChange: (v: string) => void
  knownIds: string[]
}) {
  const { t } = useTranslation()
  const isKnown = knownIds.includes(value)
  const [customMode, setCustomMode] = useState(value !== '' && !isKnown)

  if (customMode) {
    return (
      <div className="field-with-btn">
        <input
          type="text" className="form-input grow" value={value} autoFocus
          onChange={e => onChange(e.target.value)}
          placeholder="my-corpus-id"
        />
        <button
          type="button" className="btn-sm"
          onClick={() => { setCustomMode(false); onChange(knownIds[0] ?? '') }}
        >
          {t('corpusPage.upload.fromList')}
        </button>
      </div>
    )
  }

  return (
    <select
      id="upload-corpus-id" className="form-select" value={isKnown ? value : ''}
      onChange={e => {
        if (e.target.value === CUSTOM_CORPUS_ID) { setCustomMode(true); onChange('') }
        else onChange(e.target.value)
      }}
    >
      {knownIds.map(id => <option key={id} value={id}>{id}</option>)}
      <option value={CUSTOM_CORPUS_ID}>{t('corpusPage.upload.customCorpusOption')}</option>
    </select>
  )
}

// ── Corpus content browser ───────────────────────────────

function CorpusIdSelector({ corpusId, setCorpusId, knownIds, bare }: {
  corpusId: string; setCorpusId: (id: string) => void; knownIds: string[]; bare?: boolean
}) {
  return (
    <div className="form-group corpus-pick">
      {/* A label only where the picker sits among other fields. In the page
          header the corpus id is already named on the line below the title,
          and a second label above it spends a line for nothing. */}
      {!bare && <label>Corpus ID</label>}
      <SelectBox value={corpusId} onChange={e => setCorpusId(e.target.value)} aria-label="Corpus ID">
        {knownIds.map(id => <option key={id} value={id}>{id}</option>)}
      </SelectBox>
    </div>
  )
}

function CorpusContentTab({ corpusId, setCorpusId, knownIds, realmId }: {
  corpusId: string; setCorpusId: (id: string) => void; knownIds: string[]; realmId?: string | null
}) {
  const { t } = useTranslation()
  const [q, setQ] = useState('')
  const [flag, setFlag] = useState<'all' | 'short' | 'duplicate' | 'nopath'>('all')
  const [sort, setSort] = useState<SortKey>('document')
  const [expanded, setExpanded] = useState<string | null>(null)
  const { data, isLoading, error } = useQuery({
    queryKey: ['corpus-chunks', corpusId, q, realmId],
    queryFn: () => api.corpus.chunks(corpusId, { limit: 50, q, realmId }),
    // The previous rows stay on screen while new ones load. Without this,
    // switching corpus or typing in the search empties the table for a moment,
    // and instead of waiting the reader sees "no chunks": an answer that was
    // never given.
    placeholderData: keepPreviousData,
  })

  // A fragment under two hundred characters is usually a heading separated
  // from its text: search finds it, and there is nothing in it to answer from.
  const items = data?.items ?? []
  const isShort = (c: { length: number }) => c.length < 200
  // `root` is what the chunker marks a document with when it found no tree:
  // nothing parses structure out of plain text until a domain parser is
  // enabled. Counting it as a path means reporting zero unstructured fragments
  // on a corpus that is entirely unstructured.
  const noPath = (c: { structural_path?: string }) => !c.structural_path || c.structural_path === 'root'
  const flagCounts = {
    all: items.length,
    short: items.filter(isShort).length,
    duplicate: items.filter(c => c.duplicate).length,
    nopath: items.filter(noPath).length,
  }
  const filtered = items.filter(c => {
    if (flag === 'short') return isShort(c)
    if (flag === 'duplicate') return !!c.duplicate
    if (flag === 'nopath') return noPath(c)
    return true
  })
  // Document order is whatever order the index returned; the other two answer
  // "where is the damage": shortest first, or longest first.
  const visible = sort === 'document'
    ? filtered
    : [...filtered].sort((a, b) => (sort === 'shortest' ? a.length - b.length : b.length - a.length))

  return (
    <div>
      <div className="page-head">
        <h1 className="page-title">{t('corpusPage.content.heading')}</h1>
        <p className="page-sub">
          {corpusId}
          {data && ` · ${t('corpusPage.content.shownOf', { shown: items.length })}`}
        </p>
        <span className="page-act">
          <CorpusIdSelector corpusId={corpusId} setCorpusId={setCorpusId} knownIds={knownIds} bare />
        </span>
      </div>

      <div className="table-toolbar">
        <input
          className="toolbar-search" type="search" value={q} onChange={e => setQ(e.target.value)}
          placeholder={t('corpusPage.content.searchPlaceholder')}
          aria-label={t('corpusPage.content.searchLabel')}
        />
        {/* Three kinds of damage, each with its own count: how many fragments
            fall under the threshold, how many are duplicates, how many have no
            parsed structure. Without these, damage is only found by paging
            through all fifty rows. */}
        <div className="chips">
          {(['all', 'short', 'duplicate', 'nopath'] as const).map(f => (
            <button
              key={f} type="button"
              className={`chip${flag === f ? ' active' : ''}`}
              onClick={() => setFlag(f)}
            >
              {t(`corpusPage.content.flag.${f}`)}
              <span className="chip-count">{flagCounts[f]}</span>
            </button>
          ))}
        </div>
        <SelectBox
          value={sort} onChange={e => setSort(e.target.value as SortKey)}
          aria-label={t('corpusPage.content.sortLabel')}
        >
          {(['document', 'shortest', 'longest'] as const).map(k => (
            <option key={k} value={k}>{t(`corpusPage.content.sort.${k}`)}</option>
          ))}
        </SelectBox>
      </div>

      {isLoading && <p className="text-muted">{t('corpusPage.loading')}</p>}
      {error && <p className="conn-status conn-status-err">{t('corpusPage.indexUnavailable', { error: String(error) })}</p>}
      {data && visible.length === 0 && <p className="empty">{t('corpusPage.content.empty')}</p>}
      {data && visible.length > 0 && (
        <table className="chunk-table">
          <thead>
            <tr>
              <th className="c-id">{t('corpusPage.content.chunk')}</th>
              <th className="c-path">{t('corpusPage.content.path')}</th>
              <th>{t('corpusPage.content.text')}</th>
              <th className="c-len num">{t('corpusPage.content.length')}</th>
            </tr>
          </thead>
          <tbody>
            {visible.map(c => {
              const open = expanded === c.chunk_id
              return (
                <tr
                  key={c.chunk_id}
                  className={open ? 'open' : undefined}
                  onClick={() => setExpanded(open ? null : c.chunk_id)}
                >
                  <td className="c-id"><code>{c.chunk_id.slice(0, 8)}</code></td>
                  <td className="c-path" title={c.structural_path || undefined}>
                    {noPath(c)
                      ? <span className="q-noref">{t('corpusPage.content.noPath')}</span>
                      : <code>{c.structural_path}</code>}
                    {c.header_only && <span className="badge badge-warn">header-only</span>}
                    {c.duplicate && <span className="badge badge-danger">{t('corpusPage.content.duplicate')}</span>}
                  </td>
                  {/* A fragment's text can run to a thousand characters. On one
                      line it gives a table you can scan; in full it gives a
                      table you have to page through. A row expands on click,
                      and exactly one is ever open. */}
                  <td className={`c-text${open ? ' open' : ''}`}>{c.text}</td>
                  <td className={`c-len num${isShort(c) ? ' len-short' : ''}`}>{c.length}</td>
                </tr>
              )
            })}
          </tbody>
        </table>
      )}
    </div>
  )
}

// ── Corpus health ────────────────────────────────────────

function CorpusHealthTab({ corpusId, setCorpusId, knownIds, realmId }: {
  corpusId: string; setCorpusId: (id: string) => void; knownIds: string[]; realmId?: string | null
}) {
  const { t } = useTranslation()
  const toRealm = useRealmPath()
  const { data, isLoading, error } = useQuery({
    queryKey: ['corpus-health', corpusId, realmId],
    queryFn: () => api.corpus.health(corpusId, realmId),
  })

  const SEVERITY_ORDER: Record<CorpusHealthItem['severity'], number> = { error: 0, warn: 1, info: 2, ok: 3 }
  const sortedItems = [...(data?.items ?? [])].sort((a, b) => SEVERITY_ORDER[a.severity] - SEVERITY_ORDER[b.severity])

  return (
    <div>
      <div className="page-head">
        <h1 className="page-title">{t('corpusPage.health.heading')}</h1>
        <p className="page-sub">{corpusId}</p>
        <span className="page-act">
          <CorpusIdSelector corpusId={corpusId} setCorpusId={setCorpusId} knownIds={knownIds} bare />
        </span>
      </div>

      {isLoading && <p className="text-muted">{t('corpusPage.loading')}</p>}
      {error && <p className="conn-status conn-status-err">{t('corpusPage.indexUnavailable', { error: String(error) })}</p>}
      {data && (
        <>
          <div className="stat-band">
            <div className="stat-cell">
              <div className="eyebrow">{t('corpusPage.health.totalChunks')}</div>
              <div className="metric-val">{data.n_chunks}</div>
            </div>
            <div className="stat-cell">
              <div className="eyebrow">{t('corpusPage.health.avgLength')}</div>
              <div className="metric-val">{data.avg_length.toFixed(0)}</div>
              <div className="stat-sub">
                {t('corpusPage.health.rangeSub', { min: data.min_length, max: data.max_length })}
              </div>
            </div>
            <div className="stat-cell">
              <div className="eyebrow">{t('corpusPage.health.duplicates')}</div>
              <div className="metric-val" style={data.n_duplicates > 0 ? { color: 'var(--color-warning)' } : undefined}>
                {data.n_duplicates}
              </div>
            </div>
            <div className="stat-cell">
              <div className="eyebrow">{t('corpusPage.health.noStructure')}</div>
              <div className="metric-val" style={(data.n_missing_path ?? 0) > 0 ? { color: 'var(--color-warning)' } : undefined}>
                {data.n_missing_path ?? 0}
              </div>
              <div className="stat-sub">{t('corpusPage.health.noStructureSub')}</div>
            </div>
            <div className="stat-cell">
              <div className="eyebrow">{t('corpusPage.health.headerOnly')}</div>
              <div className="metric-val">{data.n_header_only}</div>
              <div className="stat-sub">{t('corpusPage.health.headerOnlySub')}</div>
            </div>
          </div>

          <div className="ov-split">
            <section>
              <div className="section-rule flush">
                <h2 className="section-title">{t('corpusPage.health.distHeading')}</h2>
                <span className="section-meta">{t('corpusPage.health.distMeta')}</span>
              </div>
              {/* The shape of the distribution rather than its mean: a corpus
                  of headings and walls of text has the same mean as an even
                  one, and only the shape tells them apart. */}
              {(data.length_deciles?.length ?? 0) > 0 ? (
                <>
                  <div className="decile-bars">
                    {data.length_deciles!.map((n, i) => {
                      const peak = Math.max(...data.length_deciles!)
                      const lo = data.min_length + ((data.max_length - data.min_length) * i) / 10
                      return (
                        <div
                          key={i}
                          className={`decile-bar${lo < 200 ? ' below' : ''}`}
                          style={{ height: `${Math.max(2, (n / peak) * 100)}%` }}
                          title={t('corpusPage.health.bucketTitle', { n, from: Math.round(lo) })}
                        />
                      )
                    })}
                  </div>
                  <div className="decile-axis">
                    <span>{data.min_length}</span>
                    <span>{data.avg_length.toFixed(0)}</span>
                    <span>{data.max_length}</span>
                  </div>
                  <p className="hint-line">{t('corpusPage.health.distHint')}</p>
                </>
              ) : (
                <p className="text-muted">{t('corpusPage.health.noDist')}</p>
              )}
            </section>

            <section>
              <div className="section-rule flush">
                <h2 className="section-title">{t('corpusPage.health.findingsHeading')}</h2>
                <span className="section-meta">{t('corpusPage.health.bySeverity')}</span>
              </div>
              {/* A palette dot rather than an emoji: an emoji is drawn by the
                  system font and follows neither theme nor palette. */}
              {sortedItems.map((item, i) => (
                <div key={i} className="find-row">
                  <span className={`find-dot find-${item.severity}`} aria-hidden="true" />
                  <div>
                    {/* Findings arrive from the server as English strings,
                        composed by `core/eval/corpus_health.py`. That shows on
                        a localised screen, so the title is looked up by the
                        finding's id and the server string stays as the
                        fallback for ids not listed here. The detail and the
                        action carry numbers and are left as they came. */}
                    <div className="find-title">
                      {t(`corpusPage.health.finding.${item.id}`, { defaultValue: item.title })}
                      {/* Which catalogue entries this finding is evidence for.
                          A reader could see a finding and had no way to tell
                          whether it named a failure the platform knows or was
                          a sentence about this corpus alone. */}
                      {(item.failure_ids ?? []).map(id => (
                        <Link key={id} className="link-btn mono-sm ml-8" to={toRealm(`/atlas?entry=${id}`)}>
                          {id}
                        </Link>
                      ))}
                    </div>
                    <p className="find-detail">{item.detail}</p>
                    {item.action && <p className="find-detail find-action">{item.action}</p>}
                  </div>
                </div>
              ))}
            </section>
          </div>

          <DeepDiagnosticsSection corpusId={corpusId} realmId={realmId} />
        </>
      )}
    </div>
  )
}

// ── Deep diagnostics — LLM-judge sample reports, on demand only ──────────────
//
// Unlike the fast detectors above (run automatically on tab open), these
// each make many LLM calls (one per question/chunk) — slow, so the user
// triggers them explicitly rather than the page loading them eagerly. The
// three runners are independent cross-checks, not one consolidated score
// (see eval/ragas_runner.py module docstring) — shown side by side so
// agreement/disagreement between judges is visible.

type DiagKind = 'ragas' | 'trulens' | 'chunk-coherence'

// Values are i18n keys (resolved with t() at the render site), not display
// text, except 'ragas'/'trulens' which are proper-noun tool names.
const diagLabels: Record<DiagKind, string> = {
  ragas: 'Ragas',
  trulens: 'TruLens',
  'chunk-coherence': 'corpusPage.deepDiagnostics.chunkCoherenceLabel',
}

// How to read each metric — shown inline next to the number so the user
// doesn't have to cross-reference the guide while looking at a result.
// Numbers alone ("0.62") are not self-explanatory for a judge-scored metric.
// Values are i18n keys, resolved with t() at the render site.
const metricHints: Record<string, string> = {
  context_precision: 'corpusPage.deepDiagnostics.hints.contextPrecision',
  context_recall: 'corpusPage.deepDiagnostics.hints.contextRecall',
  answer_relevancy: 'corpusPage.deepDiagnostics.hints.answerRelevancy',
  context_relevance: 'corpusPage.deepDiagnostics.hints.contextRelevance',
  groundedness: 'corpusPage.deepDiagnostics.hints.groundedness',
  answer_relevance: 'corpusPage.deepDiagnostics.hints.answerRelevance',
  avg_coherence: 'corpusPage.deepDiagnostics.hints.avgCoherence',
  n_incoherent: 'corpusPage.deepDiagnostics.hints.nIncoherent',
}

// Default values mirror the backend's own defaults (services/api_gateway/
// routers/corpus.py: _DEFAULT_DIAGNOSTICS_DATASET/_MAX_QUESTIONS/pipeline_id
// "naive"/top_k=5/_DEFAULT_CHUNK_COHERENCE_SAMPLE_SIZE) — changing a control
// below opts into something other than what ran before this was configurable.
const _DEFAULT_MAX_QUESTIONS = 10
const _DEFAULT_PIPELINE_ID = 'naive'
const _DEFAULT_TOP_K = 5
const _DEFAULT_SAMPLE_SIZE = 150

function DeepDiagnosticsSection({ corpusId, realmId }: { corpusId: string; realmId?: string | null }) {
  const { t } = useTranslation()
  const diagLabel = (kind: DiagKind) => kind === 'chunk-coherence' ? t(diagLabels[kind]) : diagLabels[kind]
  const [results, setResults] = useState<Partial<Record<DiagKind, DiagnosticsMetricsResult>>>({})
  const [errors, setErrors] = useState<Partial<Record<DiagKind, string>>>({})
  const [running, setRunning] = useState<DiagKind | null>(null)
  // Native `title` hover tooltips turned out not to be discoverable in
  // practice (reported live) — click-to-toggle is explicit and works the
  // same on touch/embedded-webview setups where hover doesn't fire.
  const [openHint, setOpenHint] = useState<string | null>(null)

  // Empty, and not `handbook.v0.fast.jsonl`. That one belongs to the demo
  // Realm, so on any other Realm it was absent from the list beside the field
  // yet still travelled in the request: a corpus was diagnosed with questions
  // about a handbook it does not contain, honestly returning context_recall = 0.
  // The dataset is picked from the ones the Realm actually has, the same rule
  // the default corpus follows.
  const [dataset, setDataset] = useState('')
  const [maxQuestions, setMaxQuestions] = useState(_DEFAULT_MAX_QUESTIONS)
  const [pipelineId, setPipelineId] = useState(_DEFAULT_PIPELINE_ID)
  const [topK, setTopK] = useState(_DEFAULT_TOP_K)
  const [sampleSize, setSampleSize] = useState(_DEFAULT_SAMPLE_SIZE)

  // Reuse the same data sources the "New run" form already pulls from
  // — no separate preset list needed for a 3-knob diagnostics form (unlike
  // NewExperimentPage's full 8+ field config, where named presets earn
  // their keep).
  const { data: datasets = [] } = useQuery({
    queryKey: ['datasets', realmId],
    queryFn: () => api.datasets.list(realmId),
  })
  const { data: registry } = useQuery({ queryKey: ['registry', realmId], queryFn: () => api.registry(realmId) })
  useEffect(() => {
    if (dataset || datasets.length === 0) return
    setDataset(datasets[0].filename)
  }, [datasets, dataset])
  const pipelineIds = registry?.pipeline ?? ['naive', 'hybrid_rrf', 'hybrid_weighted', 'graph']

  async function runOne(kind: DiagKind) {
    setRunning(kind)
    setErrors(prev => ({ ...prev, [kind]: undefined }))
    try {
      const result = kind === 'ragas'
        ? await api.corpus.diagnosticsRagas(corpusId, { pipeline_id: pipelineId, top_k: topK, dataset, max_questions: maxQuestions, realm_id: realmId })
        : kind === 'trulens'
        ? await api.corpus.diagnosticsTrulens(corpusId, { pipeline_id: pipelineId, top_k: topK, dataset, max_questions: maxQuestions, realm_id: realmId })
        : await api.corpus.diagnosticsChunkCoherence(corpusId, { sample_size: sampleSize, realm_id: realmId })
      setResults(prev => ({ ...prev, [kind]: result }))
    } catch (err) {
      setErrors(prev => ({ ...prev, [kind]: String(err) }))
    } finally {
      setRunning(null)
    }
  }

  async function runAll() {
    for (const kind of ['ragas', 'trulens', 'chunk-coherence'] as DiagKind[]) {
      await runOne(kind)
    }
  }

  return (
    <div className="deep-diag">
      <div className="flex-row deep-diag-head">
        <h3>{t('corpusPage.deepDiagnostics.heading')}</h3>
        <button className="btn" onClick={runAll} disabled={running !== null || !dataset}>
          {running ? t('corpusPage.deepDiagnostics.running', { label: diagLabel(running) }) : t('corpusPage.deepDiagnostics.runAll')}
        </button>
      </div>
      <p className="deep-diag-lead">
        {t('corpusPage.deepDiagnostics.intro')}
      </p>

      <div className="diag-params">
        <div className="form-group">
          <label>{t('corpusPage.deepDiagnostics.datasetLabel')}</label>
          <select value={dataset} onChange={e => setDataset(e.target.value)} className="form-select">
            {datasets.length > 0
              ? datasets.map(d => <option key={d.filename} value={d.filename}>{d.filename}</option>)
              : <option value="">{t('corpusPage.deepDiagnostics.noDatasets')}</option>}
          </select>
        </div>
        <div className="form-group">
          <label>{t('corpusPage.deepDiagnostics.questionsLabel')}</label>
          <input type="number" min={1} max={200} value={maxQuestions}
            onChange={e => setMaxQuestions(Number(e.target.value))} className="form-input" />
        </div>
        <div className="form-group">
          <label>{t('corpusPage.deepDiagnostics.pipelineLabel')}</label>
          <select value={pipelineId} onChange={e => setPipelineId(e.target.value)} className="form-select">
            {pipelineIds.map(p => <option key={p} value={p}>{p}</option>)}
          </select>
        </div>
        <div className="form-group">
          <label>top_k</label>
          <input type="number" min={1} max={50} value={topK}
            onChange={e => setTopK(Number(e.target.value))} className="form-input" />
        </div>
        <div className="form-group">
          <label>{t('corpusPage.deepDiagnostics.coherenceSampleLabel')}</label>
          <input type="number" min={10} max={500} value={sampleSize}
            onChange={e => setSampleSize(Number(e.target.value))} className="form-input" />
        </div>
      </div>

      <div className="grid-3 diag-tiles">
        {(['ragas', 'trulens', 'chunk-coherence'] as DiagKind[]).map(kind => (
          <div key={kind} className="card diag-tile">
            <div className="flex-row diag-tile-head">
              <strong className="diag-tile-title">{diagLabel(kind)}</strong>
              <button className="btn-sm" onClick={() => runOne(kind)} disabled={running !== null || !dataset}>
                {running === kind ? '…' : '▶'}
              </button>
            </div>
            {errors[kind] && <p className="badge badge-danger">{errors[kind]}</p>}
            {results[kind] && (
              <>
                {Object.entries(results[kind]!.metrics).map(([k, v]) => {
                  const hintKey = `${kind}:${k}`
                  return (
                    <div key={k}>
                      <div className="diag-metric">
                        <span className="text-muted inline-4">
                          {k}
                          {metricHints[k] && (
                            <button
                              type="button"
                              onClick={() => setOpenHint(prev => prev === hintKey ? null : hintKey)}
                              className="what-btn"
                              aria-label={t('corpusPage.deepDiagnostics.whatDoesThisMean', { metric: k })}
                            >
                              ?
                            </button>
                          )}
                        </span>
                        <strong className="no-shrink">{typeof v === 'number' ? v.toFixed(2) : String(v)}</strong>
                      </div>
                      {openHint === hintKey && (
                        <p className="diag-hint">
                          {t(metricHints[k])}
                        </p>
                      )}
                    </div>
                  )
                })}
                {results[kind]!.sample_size !== undefined && (
                  <p className="res-note">
                    {t('corpusPage.deepDiagnostics.bySample', { size: results[kind]!.sample_size })}
                  </p>
                )}
              </>
            )}
            {!results[kind] && !errors[kind] && running !== kind && (
              <p className="text-muted field-note">{t('corpusPage.deepDiagnostics.notRun')}</p>
            )}
          </div>
        ))}
      </div>

      {results['chunk-coherence']?.incoherent_chunks && results['chunk-coherence']!.incoherent_chunks!.length > 0 && (
        <div className="card diag-sample">
          <strong className="diag-tile-title">
            {t('corpusPage.deepDiagnostics.incoherentChunks', { count: results['chunk-coherence']!.incoherent_chunks!.length })}
          </strong>
          <p className="diag-sample-note">
            {t('corpusPage.deepDiagnostics.incoherentChunksHint')}
          </p>
          <table className="table chunk-table-fixed">
            <colgroup>
              <col className="col-8" />
              <col className="col-18" />
              <col className="col-74" />
            </colgroup>
            <thead><tr><th>{t('corpusPage.deepDiagnostics.score')}</th><th>chunk_id</th><th>{t('corpusPage.deepDiagnostics.textStart')}</th></tr></thead>
            <tbody>
              {results['chunk-coherence']!.incoherent_chunks!.map(c => (
                <tr key={c.chunk_id}>
                  <td><span className={`badge ${c.score === 0 ? 'badge-danger' : 'badge-warn'}`}>{c.score}/5</span></td>
                  <td><code>{c.chunk_id.slice(0, 12)}…</code></td>
                  <td className="wrap-cell">{c.text_preview}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
