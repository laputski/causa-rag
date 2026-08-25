import { Fragment, useEffect, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import { useTranslation, Trans } from 'react-i18next'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Trash2, Plus, Zap, Database, Waypoints, Search, Cpu, Server, CircleCheck, CircleX, Rocket, ExternalLink, Frame, Ban } from 'lucide-react'
import { useRealm, useRealmPath } from '../context/RealmContext'
import { api, type Dataset, type ExternalRag, type ExternalRagTestResult, type PanelStatus } from '../api/client'

// Icon per connector type — falls back to a generic server icon for any
// future connector_type the backend seeds that isn't listed here yet, so a
// new connector never renders as a blank/broken icon.
const RESOURCE_ICONS: Record<string, React.ElementType> = {
  qdrant: Database,
  neo4j: Waypoints,
  opensearch: Search,
  ollama: Cpu,
}

// Resource params (host/port/user/...) and connector-test responses can both
// carry credentials (Neo4j password, API keys) — masked everywhere a raw
// value would otherwise render in plain text.
const SENSITIVE_KEY_RE = /password|secret|token|api[_-]?key/i

function displayValue(key: string, value: unknown): string {
  return SENSITIVE_KEY_RE.test(key) ? '••••••••' : String(value)
}

// Renders a connector/RAG connection-test result as a proper panel instead
// of stuffing raw JSON into a pill-shaped `.badge` — that class's
// border-radius: 99px only looks right for a short one-line label, and
// visibly broke (a stretched, oddly-rounded box) once results started
// including array fields (Qdrant collections, Ollama models).
function ConnStatus({ ok, detail, fields, children }: {
  ok: boolean
  detail?: string
  fields?: [string, unknown][]
  children?: React.ReactNode
}) {
  const { t } = useTranslation()
  return (
    <div className={`conn-status ${ok ? 'conn-status-ok' : 'conn-status-err'}`}>
      <div className="conn-status-head">
        {ok ? <CircleCheck size={14} /> : <CircleX size={14} />}
        {ok ? t('realmResourcesPage.connStatus.ok') : t('realmResourcesPage.connStatus.error')}
      </div>
      {detail && <p className="conn-status-plain">{detail}</p>}
      {fields && fields.length > 0 && (
        <div className="conn-status-detail">
          {fields.map(([k, v]) => (
            <div key={k} className="conn-status-row">
              <span className="conn-status-key">{k}</span>
              {Array.isArray(v) ? (
                v.length === 0 ? (
                  <span>{t('realmResourcesPage.connStatus.empty')}</span>
                ) : (
                  // A Qdrant probe returns every collection at once, and on a
                  // working installation there are a dozen with fifty-character names.
                  // The tile grew ten times taller than its neighbours and the names
                  // ran past its edge. Six of them, plus a count of the rest.
                  <span className="conn-chip-list">
                    {v.slice(0, 6).map((item, i) => (
                      <code key={i} className="conn-chip" title={String(displayValue(k, item))}>
                        {displayValue(k, item)}
                      </code>
                    ))}
                    {v.length > 6 && (
                      <span className="conn-chip-more" title={v.map(x => displayValue(k, x)).join('\n')}>
                        {t('realmResourcesPage.connStatus.more', { count: v.length - 6 })}
                      </span>
                    )}
                  </span>
                )
              ) : (
                <code className="conn-status-value">{displayValue(k, v)}</code>
              )}
            </div>
          ))}
        </div>
      )}
      {children}
    </div>
  )
}

interface ConnectorType {
  type: string
  label: string
  description: string
  params_schema: Record<string, { type: string; default?: unknown }>
}

interface Resource {
  type: string
  [key: string]: unknown
}

interface TestResult {
  status: 'ok' | 'error'
  detail?: string
  [key: string]: unknown
}

interface Neo4jProvisionStatus {
  status: 'running' | 'done' | 'error' | 'not_needed'
  service_name?: string
  uri?: string
  detail?: string
}

// ── RAG endpoints (formerly the standalone "External RAG implementations" page) ──
// Merged here because registering a RAG endpoint's URL and configuring the
// Realm's shared backend resources are the same workflow in practice: both
// describe what this Realm's RAG implementation(s) talk to.

function CapabilitiesBadge({ caps }: { caps: ExternalRag['capabilities'] }) {
  const { t } = useTranslation()
  if (!caps) {
    return <span className="badge">{t('realmResourcesPage.capabilities.notTested')}</span>
  }
  return (
    <span className="caps-stack">
      <span className="caps-row">
        <span className="badge">{caps.supports_trace ? t('realmResourcesPage.capabilities.whiteBox') : t('realmResourcesPage.capabilities.blackBox')}</span>
        {caps.source_ref_granularity && (
          <span className="badge">{t('realmResourcesPage.capabilities.granularity', { value: caps.source_ref_granularity === 'chunk' ? t('realmResourcesPage.capabilities.granularityChunk') : t('realmResourcesPage.capabilities.granularityDocument') })}</span>
        )}
        <span className="badge">{caps.supports_retrieval_only ? t('realmResourcesPage.capabilities.hasRetrievalOnly') : t('realmResourcesPage.capabilities.noRetrievalOnly')}</span>
        {caps.supported_params && caps.supported_params.length > 0 && (
          <span className="badge" title={t('realmResourcesPage.capabilities.paramsHint')}>
            params: {caps.supported_params.join(', ')}
          </span>
        )}
      </span>
      {/* Reflects the LAST test() probe, visible without a new
          click (see the design notes) — the platform never guesses at
          an embedder, this is purely a stored comparison result. */}
      {caps.embedder_mismatch_warning && (
        <span className="badge badge-warn" title={caps.embedder_mismatch_warning}>
          ⚠ {t('realmResourcesPage.capabilities.embedderMismatch')}
        </span>
      )}
      {!caps.embedder_mismatch_warning && caps.embedder_hint && (
        <span className="badge dim" title={caps.embedder_hint}>
          ⓘ {t('realmResourcesPage.capabilities.embedderUnspecified')}
        </span>
      )}
    </span>
  )
}

function TestResultBadge({ result }: { result: ExternalRagTestResult }) {
  const { t } = useTranslation()
  if (!result.ok) {
    return <ConnStatus ok={false} detail={result.error} />
  }
  const detail = t('realmResourcesPage.testResult.answer', { preview: result.answer_preview })
    + ' ' + (result.has_trace
      ? t('realmResourcesPage.testResult.traceSources', { count: result.n_sources })
      : t('realmResourcesPage.testResult.noTrace'))
    + (result.connection_tier === 'mapped' ? ` ${t('realmResourcesPage.testResult.tierMapped')}` : '')
  return (
    <ConnStatus
      ok
      detail={detail}
    >
      {result.connection_tier === 'mapped' && result.parsed_sources_preview && result.parsed_sources_preview.length > 0 && (
        <pre className="code-block">
          {JSON.stringify(result.parsed_sources_preview, null, 2)}
        </pre>
      )}
      {result.capabilities?.embedder_mismatch_warning && (
        <p className="res-note warn">
          ⚠ {result.capabilities.embedder_mismatch_warning}
        </p>
      )}
      {!result.capabilities?.embedder_mismatch_warning && result.capabilities?.embedder_hint && (
        <p className="res-note">
          ⓘ {result.capabilities.embedder_hint}
        </p>
      )}
    </ConnStatus>
  )
}

// reads/writes the same unified `datasets` collection the
// platform's own golden sets live in (via source_rag_id, not a dedicated
// per-RAG collection) — see api/client.ts's `datasets` methods and
// services/api_gateway/routers/datasets.py's module docstring.
function DatasetsSection({ ragId, realmId }: { ragId: string; realmId: string }) {
  const { t } = useTranslation()
  const qc = useQueryClient()
  const [jsonl, setJsonl] = useState('')
  const [filename, setFilename] = useState('')
  const [error, setError] = useState<string | null>(null)

  const queryKey = ['datasets', realmId, ragId]
  const { data: datasets = [] } = useQuery({
    queryKey,
    queryFn: () => api.datasets.list(realmId, ragId),
  })

  const uploadMut = useMutation({
    mutationFn: () => {
      const questions = jsonl
        .split('\n')
        .map(l => l.trim())
        .filter(Boolean)
        .map(l => JSON.parse(l))
      return api.datasets.create({ filename, realm_id: realmId, questions, source_rag_id: ragId })
    },
    onSuccess: () => {
      qc.invalidateQueries({ queryKey })
      setJsonl('')
      setFilename('')
      setError(null)
    },
    onError: () => setError(t('realmResourcesPage.datasets.parseError')),
  })

  const deleteMut = useMutation({
    mutationFn: (datasetId: string) => api.datasets.delete(datasetId),
    onSuccess: () => qc.invalidateQueries({ queryKey }),
  })

  return (
    <div className="res-sub">
      <p className="res-sub-title">{t('realmResourcesPage.datasets.heading', { count: datasets.length })}</p>
      {datasets.map((d: Dataset) => (
        <div key={d.id} className="res-list-row">
          <span>{d.filename}{d.count != null ? ` — ${t('realmResourcesPage.datasets.questionsCount', { count: d.count })}` : ''}</span>
          <button type="button" className="btn-sm"
            onClick={() => d.id && deleteMut.mutate(d.id)} disabled={deleteMut.isPending}>
            {t('realmResourcesPage.delete')}
          </button>
        </div>
      ))}
      <form className="res-sub-form" onSubmit={e => { e.preventDefault(); uploadMut.mutate() }}>
        <input type="text" value={filename} onChange={e => setFilename(e.target.value)}
          placeholder={t('realmResourcesPage.datasets.filenamePlaceholder')} required
          />
        <textarea value={jsonl} onChange={e => setJsonl(e.target.value)}
          placeholder={t('realmResourcesPage.datasets.jsonlPlaceholder')}
          rows={3} className="mono-area" required />
        <button type="submit" className="btn-sm res-sub-btn" disabled={uploadMut.isPending}>
          {uploadMut.isPending ? t('realmResourcesPage.datasets.uploading') : t('realmResourcesPage.datasets.upload')}
        </button>
        {error && <p className="form-error">{error}</p>}
      </form>
    </div>
  )
}

// default_corpus_id/pipeline_id/reranker_id/params this RAG
// should be pre-filled with on "New run" (see ExternalRag.default_corpus_id
// in client.ts for why: the form used to hardcode 'handbook' regardless of
// which RAG was picked). uses_realm_resources travels here too — same
// "advisory, set after registration" pattern.
function DefaultConfigSection({ rag, realmId }: { rag: ExternalRag; realmId: string }) {
  const { t } = useTranslation()
  const qc = useQueryClient()
  const [editing, setEditing] = useState(false)
  const [corpusId, setCorpusId] = useState(rag.default_corpus_id ?? '')
  const [pipelineId, setPipelineId] = useState(rag.default_pipeline_id ?? '')
  const [rerankerId, setRerankerId] = useState(rag.default_reranker_id ?? '')
  const [usesRealmResources, setUsesRealmResources] = useState(rag.uses_realm_resources ?? false)
  const [paramsText, setParamsText] = useState(
    rag.default_params && Object.keys(rag.default_params).length > 0 ? JSON.stringify(rag.default_params) : '',
  )
  const [error, setError] = useState<string | null>(null)

  const { data: collections = [] } = useQuery({
    queryKey: ['corpus-collections', realmId],
    queryFn: () => api.corpus.collections(realmId),
    enabled: editing,
  })

  const updateMut = useMutation({
    mutationFn: () => {
      const default_params = paramsText.trim() ? JSON.parse(paramsText) : {}
      return api.externalRags.update(rag.id, {
        default_corpus_id: corpusId || undefined,
        default_pipeline_id: pipelineId || undefined,
        default_reranker_id: rerankerId || undefined,
        default_params,
        uses_realm_resources: usesRealmResources,
      })
    },
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['external-rags'] })
      setEditing(false)
      setError(null)
    },
    onError: () => setError(t('realmResourcesPage.defaultConfig.parseError')),
  })

  const has = rag.default_corpus_id || rag.default_pipeline_id || rag.default_reranker_id
    || rag.uses_realm_resources
    || (rag.default_params && Object.keys(rag.default_params).length > 0)

  if (!editing) {
    return (
      <div className="res-kv">
        {has ? (
          <span className="text-muted">
            {t('realmResourcesPage.defaultConfig.summary', { value: [
              rag.default_corpus_id && `corpus_id=${rag.default_corpus_id}`,
              rag.default_pipeline_id && `pipeline_id=${rag.default_pipeline_id}`,
              rag.default_reranker_id && `reranker_id=${rag.default_reranker_id}`,
              rag.uses_realm_resources && t('realmResourcesPage.defaultConfig.usesRealmResourcesShort'),
            ].filter(Boolean).join(', ') })}
          </span>
        ) : (
          <span className="text-muted">{t('realmResourcesPage.defaultConfig.none')}</span>
        )}
        {' '}
        <button type="button" className="btn-sm" onClick={() => setEditing(true)}>{t('realmResourcesPage.edit')}</button>
      </div>
    )
  }

  return (
    <div className="res-sub">
      <p className="res-sub-title">
        {t('realmResourcesPage.defaultConfig.heading')}
      </p>
      <div className="grid-2">
        <div className="form-group">
          <label>corpus_id {t('realmResourcesPage.optional')}</label>
          <input type="text" list="rag-known-corpus-ids" value={corpusId} onChange={e => setCorpusId(e.target.value)}
            placeholder={t('realmResourcesPage.defaultConfig.corpusIdPlaceholder')} />
          <datalist id="rag-known-corpus-ids">
            {collections.map(c => <option key={c.corpus_id} value={c.corpus_id} />)}
          </datalist>
        </div>
        <div className="form-group">
          <label>pipeline_id {t('realmResourcesPage.optional')}</label>
          <input type="text" value={pipelineId} onChange={e => setPipelineId(e.target.value)}
            placeholder="naive / hybrid_rrf / hybrid_weighted" />
        </div>
      </div>
      <div className="grid-2">
        <div className="form-group">
          <label>reranker_id {t('realmResourcesPage.optional')}</label>
          <input type="text" value={rerankerId} onChange={e => setRerankerId(e.target.value)}
            placeholder={t('realmResourcesPage.defaultConfig.rerankerIdPlaceholder')} />
        </div>
        <div className="form-group">
          <label>params (JSON, {t('realmResourcesPage.optionalShort')})</label>
          <input type="text" value={paramsText} onChange={e => setParamsText(e.target.value)}
            placeholder='{"alpha": 0.7}' className="mono-field" />
        </div>
      </div>
      <div className="form-group">
        <label className="check-label">
          <input type="checkbox" checked={usesRealmResources} onChange={e => setUsesRealmResources(e.target.checked)} />
          {t('realmResourcesPage.defaultConfig.usesRealmResourcesLabel')}
        </label>
        <p className="field-note">
          {t('realmResourcesPage.defaultConfig.usesRealmResourcesHint')}
        </p>
      </div>
      {error && <p className="form-error">{error}</p>}
      <div className="flex-row res-actions">
        <button type="button" className="btn-sm btn-primary" onClick={() => updateMut.mutate()} disabled={updateMut.isPending}>
          {updateMut.isPending ? t('realmResourcesPage.saving') : t('realmResourcesPage.save')}
        </button>
        <button type="button" className="btn-sm" onClick={() => setEditing(false)}>{t('realmResourcesPage.cancel')}</button>
      </div>
    </div>
  )
}

function RagCard({ rag, realmId }: { rag: ExternalRag; realmId: string }) {
  const { t } = useTranslation()
  const qc = useQueryClient()
  const toRealm = useRealmPath()
  const [testResult, setTestResult] = useState<ExternalRagTestResult | null>(null)

  const testMut = useMutation({
    mutationFn: () => api.externalRags.test(rag.id),
    onSuccess: result => {
      setTestResult(result)
      qc.invalidateQueries({ queryKey: ['external-rags'] })  // invalidate all realm variants
    },
  })
  const deleteMut = useMutation({
    mutationFn: () => api.externalRags.delete(rag.id),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['external-rags'] }),  // invalidate all realm variants
  })

  return (
    <div className="card rag-card">
      <div className="flex-between">
        <div>
          <strong>{rag.name}</strong>
          <p className="rag-url">{rag.url}</p>
          {rag.description && <p className="rag-desc">{rag.description}</p>}
          {rag.response_mapping && (
            <p className="field-note">
              {t('realmResourcesPage.ragCard.mappedTier')}
            </p>
          )}
          <div className="rag-caps"><CapabilitiesBadge caps={rag.capabilities ?? null} /></div>
        </div>
        <div className="rag-buttons">
          {rag.uses_realm_resources && rag.default_corpus_id && (
            <Link
              className="btn"
              to={toRealm(`/data/content?corpus_id=${encodeURIComponent(rag.default_corpus_id)}`)}
              title={t('realmResourcesPage.ragCard.inspectCorpusTitle')}
            >
              {t('realmResourcesPage.ragCard.inspectCorpus')}
            </Link>
          )}
          <button type="button" className="btn" onClick={() => testMut.mutate()} disabled={testMut.isPending}>
            {testMut.isPending ? t('realmResourcesPage.ragCard.testing') : t('realmResourcesPage.ragCard.test')}
          </button>
          <button type="button" className="btn" onClick={() => deleteMut.mutate()} disabled={deleteMut.isPending}>
            {t('realmResourcesPage.delete')}
          </button>
        </div>
      </div>
      {testResult && <div className="rag-test-result"><TestResultBadge result={testResult} /></div>}
      <DefaultConfigSection rag={rag} realmId={realmId} />
      <DatasetsSection ragId={rag.id} realmId={realmId} />
    </div>
  )
}

function RagEndpointsSection({ activeRealmId }: { activeRealmId: string }) {
  const { t } = useTranslation()
  const qc = useQueryClient()
  const { data: rags = [], isLoading } = useQuery({
    queryKey: ['external-rags', activeRealmId],
    queryFn: () => api.externalRags.list(activeRealmId),
  })
  const [form, setForm] = useState({ name: '', url: '', description: '', retrieve_endpoint: '' })
  // Only ever one endpoint is open: three expanded cards under the table are
  // the same list a second time.
  const [openRag, setOpenRag] = useState<string | null>(null)
  const [adding, setAdding] = useState(false)
  const [mappingError, setMappingError] = useState<string | null>(null)
  const [requestTemplateText, setRequestTemplateText] = useState('')
  const [responseMappingText, setResponseMappingText] = useState('')
  const [supportedParamsText, setSupportedParamsText] = useState('')

  const createMut = useMutation({
    mutationFn: () => {
      const request_template = requestTemplateText.trim() ? JSON.parse(requestTemplateText) : undefined
      const response_mapping = responseMappingText.trim() ? JSON.parse(responseMappingText) : undefined
      const supported_params = supportedParamsText.trim()
        ? supportedParamsText.split(',').map(s => s.trim()).filter(Boolean)
        : undefined
      return api.externalRags.create({
        ...form, retrieve_endpoint: form.retrieve_endpoint || undefined, request_template, response_mapping,
        supported_params,
        realm_id: activeRealmId,
      })
    },
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['external-rags', activeRealmId] })
      setForm({ name: '', url: '', description: '', retrieve_endpoint: '' })
      setRequestTemplateText('')
      setResponseMappingText('')
      setSupportedParamsText('')
      setMappingError(null)
    },
    onError: () => setMappingError(t('realmResourcesPage.ragEndpoints.mappingParseError')),
  })

  return (
    <div className="section">
      <div className="section-rule">
        <h2 className="section-title">{t('realmResourcesPage.ragEndpoints.heading')}</h2>
        <span className="section-meta">{t('realmResourcesPage.ragEndpoints.meta')}</span>
        {/* A button, and not a disclosure: few people add an endpoint and
            they do it once, while a "▸ Add RAG endpoint" row sat under the
            table for everybody, always, collecting two more disclosures
            beneath it. */}
        <button type="button" className="btn btn-sm btn-primary push"
                onClick={() => setAdding(v => !v)}>
          <Plus size={13} aria-hidden="true" />
          {adding ? t('realmResourcesPage.cancel') : t('realmResourcesPage.ragEndpoints.addHeading')}
        </button>
      </div>

      {!isLoading && rags.length === 0 && <p className="empty">{t('realmResourcesPage.ragEndpoints.noneSaved')}</p>}
      {rags.length > 0 && (
        <table className="cmp-table">
          <thead>
            <tr>
              <th>{t('realmResourcesPage.ragEndpoints.colName')}</th>
              <th>{t('realmResourcesPage.ragEndpoints.colUrl')}</th>
              <th>{t('realmResourcesPage.ragEndpoints.colCaps')}</th>
              <th>{t('realmResourcesPage.ragEndpoints.colState')}</th>
            </tr>
          </thead>
          <tbody>
            {rags.map(rag => (
              <Fragment key={rag.id}>
              <tr
                className={`row-click${openRag === rag.id ? ' open' : ''}`}
                onClick={() => setOpenRag(openRag === rag.id ? null : rag.id)}
              >
                <td><strong>{rag.name}</strong></td>
                <td className="cmp-val"><code>{rag.url}</code></td>
                <td className="cmp-means">
                  {rag.capabilities
                    ? [rag.capabilities.supports_trace ? t('realmResourcesPage.capabilities.whiteBox') : t('realmResourcesPage.capabilities.blackBox'),
                       rag.capabilities.source_ref_granularity].filter(Boolean).join(' · ')
                    : '—'}
                  {/* An embedder mismatch shows on the row itself and not in the
                      expanded card alone: a warning you have to click to find
                      is a warning nobody sees. Under a mismatch the retrieval
                      metrics are comparing different things. */}
                  {rag.capabilities?.embedder_mismatch_warning && (
                    <span className="badge badge-warn ml-6" title={rag.capabilities.embedder_mismatch_warning}>
                      ⚠ {t('realmResourcesPage.capabilities.embedderMismatch')}
                    </span>
                  )}
                  {!rag.capabilities?.embedder_mismatch_warning && rag.capabilities?.embedder_hint && (
                    <span className="badge dim ml-6" title={rag.capabilities.embedder_hint}>
                      ⓘ {t('realmResourcesPage.capabilities.embedderUnspecified')}
                    </span>
                  )}
                </td>
                <td>
                  <span className={`flag ${rag.capabilities ? 'flag-ok' : ''}`}>
                    <i className="flag-dot" aria-hidden="true" />
                    {rag.capabilities ? t('overview.state.ok') : t('overview.state.unknown')}
                  </span>
                </td>
              </tr>
              {/* The detail opens as a row, and not as three cards under the
                  table: the table has already listed them all, and repeating
                  that list as expanded blocks shows one thing twice. */}
              {openRag === rag.id && (
                <tr>
                  <td colSpan={4} className="rag-detail-cell">
                    <RagCard rag={rag} realmId={activeRealmId} />
                  </td>
                </tr>
              )}
              </Fragment>
            ))}
          </tbody>
        </table>
      )}

      {adding && (
      <div className="rag-add">
        <p className="field-note rag-add-intro">{t('realmResourcesPage.ragEndpoints.introShort')}</p>
        <details className="rag-ref">
          <summary>{t('realmResourcesPage.ragEndpoints.apiRequirements')}</summary>
          <div className="contract-body">
            <p><strong>{t('realmResourcesPage.ragEndpoints.requestHeading')}</strong> {t('realmResourcesPage.ragEndpoints.requestSub')}</p>
            <pre className="code-block">
              {t('realmResourcesPage.ragEndpoints.requestExample')}
            </pre>
            <p>{t('realmResourcesPage.ragEndpoints.traceIdHeader')} <code>X-Trace-Id</code> {t('realmResourcesPage.ragEndpoints.traceIdHeaderSuffix')}</p>
            <p className="contract-p"><strong>{t('realmResourcesPage.ragEndpoints.responseHeading')}</strong> {t('realmResourcesPage.ragEndpoints.responseSub')}</p>
            <pre className="code-block">
              {t('realmResourcesPage.ragEndpoints.responseExample')}
            </pre>
            <ul className="contract-list">
              <li><Trans i18nKey="realmResourcesPage.ragEndpoints.rule1" t={t}>The only required response field is <code>answer</code>.</Trans></li>
              <li><Trans i18nKey="realmResourcesPage.ragEndpoints.rule2" t={t}><code>trace</code> is entirely optional; so is each of the three fields inside it.</Trans></li>
              <li><Trans i18nKey="realmResourcesPage.ragEndpoints.rule3" t={t}>If you provide <code>sources</code>, each item only requires <code>doc_id</code>; <code>chunk_id</code> is optional (include it if your granularity is chunk-level). For retrieval metrics, add <code>source_code</code>/<code>article_no</code> (or their equivalents) — the platform checks these against the ground truth.</Trans></li>
              <li>{t('realmResourcesPage.ragEndpoints.rule4')}</li>
              <li>{t('realmResourcesPage.ragEndpoints.rule5')}</li>
              <li><Trans i18nKey="realmResourcesPage.ragEndpoints.rule6" t={t}>The URL must be in the allowlist (<code>RAG_HTTP_ALLOWLIST</code> on the platform server), otherwise the request is blocked before it reaches the network.</Trans></li>
            </ul>
          </div>
        </details>

        {/* The form was a flat list of seven consecutive fields: the
            required, the optional and the format mapping all in one heap,
            and it did not read what had to be filled in before "Save" would
            work. Three sections with rules, as on the new-run form: where to
            send the request, what may optionally be refined, and what to do
            when the response format does not match the contract. */}
        <form className="rag-form" onSubmit={e => { e.preventDefault(); createMut.mutate() }}>
          <section className="section">
            <div className="section-rule flush">
              <h4 className="section-title">{t('realmResourcesPage.ragEndpoints.formWhereHeading')}</h4>
              <span className="section-meta">{t('realmResourcesPage.ragEndpoints.formRequired')}</span>
            </div>
            <div className="form-grid">
              <div className="form-group">
                <label htmlFor="rag-name">{t('realmResourcesPage.ragEndpoints.nameLabel')}</label>
                <input id="rag-name" type="text" value={form.name}
                  onChange={e => setForm(f => ({ ...f, name: e.target.value }))}
                  placeholder="my-team-rag" required />
              </div>
              <div className="form-group">
                <label htmlFor="rag-url">URL</label>
                <input id="rag-url" type="text" value={form.url}
                  onChange={e => setForm(f => ({ ...f, url: e.target.value }))}
                  placeholder="https://external.example/query" required />
              </div>
            </div>
          </section>

          <section className="section">
            <div className="section-rule">
              <h4 className="section-title">{t('realmResourcesPage.ragEndpoints.formOptionalHeading')}</h4>
              <span className="section-meta">{t('realmResourcesPage.ragEndpoints.formOptionalMeta')}</span>
            </div>
            <div className="form-grid">
              <div className="form-group form-span">
                <label htmlFor="rag-desc">{t('realmResourcesPage.ragEndpoints.descriptionLabel')}</label>
                <input id="rag-desc" type="text" value={form.description}
                  onChange={e => setForm(f => ({ ...f, description: e.target.value }))} />
              </div>
              <div className="form-group">
                <label htmlFor="rag-retrieve">{t('realmResourcesPage.ragEndpoints.retrieveOnlyLabel')}</label>
                <input id="rag-retrieve" type="text" value={form.retrieve_endpoint}
                  onChange={e => setForm(f => ({ ...f, retrieve_endpoint: e.target.value }))}
                  placeholder="https://external.example/retrieve" />
                <p className="field-note">{t('realmResourcesPage.ragEndpoints.retrieveOnlyHint')}</p>
              </div>
              <div className="form-group">
                <label htmlFor="rag-supported-params">{t('realmResourcesPage.ragEndpoints.supportedParamsLabel')}</label>
                <input id="rag-supported-params" type="text" value={supportedParamsText}
                  onChange={e => setSupportedParamsText(e.target.value)}
                  placeholder="fetch_k, temperature, prompt_version" />
                <p className="field-note">{t('realmResourcesPage.ragEndpoints.supportedParamsHint')}</p>
              </div>
            </div>
          </section>

          {/* The format mapping is needed by a minority, those whose response
              does not match the contract, and open it took a third of the
              form. */}
          <details className="rag-mapping rag-ref">
            <summary>{t('realmResourcesPage.ragEndpoints.mappingSummary')}</summary>
            <div className="contract-body">
              <p>{t('realmResourcesPage.ragEndpoints.mappingIntro')}</p>
              <div className="form-group">
                <label htmlFor="rag-req-template">{t('realmResourcesPage.ragEndpoints.requestTemplateLabel')}</label>
                <textarea id="rag-req-template" value={requestTemplateText}
                  onChange={e => setRequestTemplateText(e.target.value)}
                  placeholder={'{"q": "{{query}}", "k": "{{top_k}}"}'}
                  rows={3} className="mono-area" />
                <p className="field-note">
                  <Trans i18nKey="realmResourcesPage.ragEndpoints.requestTemplateHint" t={t}>
                    Placeholders: <code>{'{{query}}'}</code>, <code>{'{{top_k}}'}</code>,{' '}
                    <code>{'{{filters}}'}</code>, <code>{'{{trace_id}}'}</code> — in your RAG's own request format.
                  </Trans>
                </p>
              </div>
              <div className="form-group">
                <label htmlFor="rag-resp-mapping">{t('realmResourcesPage.ragEndpoints.responseMappingLabel')}</label>
                <textarea id="rag-resp-mapping" value={responseMappingText}
                  onChange={e => setResponseMappingText(e.target.value)}
                  placeholder={'{"answer": "$.result.text", "sources": "$.result.docs", "source_doc_id": "$.id", "source_text": "$.content"}'}
                  rows={3} className="mono-area" />
                <p className="field-note">
                  <Trans i18nKey="realmResourcesPage.ragEndpoints.responseMappingHint" t={t}>
                    Keys: <code>answer</code>, <code>sources</code> (the path to the list), and for each item in it —{' '}
                    <code>source_doc_id</code>/<code>source_chunk_id</code>/<code>source_score</code>/<code>source_text</code>/
                    <code>source_structural_path</code>/<code>source_code</code>/<code>source_article_no</code>.
                  </Trans>
                </p>
              </div>
              {mappingError && <p className="form-error">{mappingError}</p>}
            </div>
          </details>

          <div className="rag-form-foot">
            <button type="submit" className="btn btn-primary" disabled={createMut.isPending || !form.name || !form.url}>
              {createMut.isPending ? t('realmResourcesPage.saving') : t('realmResourcesPage.save')}
            </button>
            <span className="field-note">{t('realmResourcesPage.ragEndpoints.formFoot')}</span>
          </div>
        </form>
      </div>
      )}

      {isLoading && <p className="text-muted">{t('realmResourcesPage.loading')}</p>}
    </div>
  )
}

// ── Realm resources (Qdrant/Neo4j/OpenSearch/Ollama) ────────────────────────


// ── Tool panels ─────────────────────────────────────────────────────────────
// There used to be a separate `/panels` page. It merged in here because "does
// everything work" is one question rather than two: the realm's infrastructure,
// its RAG endpoints and the third-party panels answer it together, and going to
// two places for the answer means comparing two screens by hand.
//
// The reason the catalogue does not embed everything is unchanged: Langfuse
// serves `frame-ancestors 'none'` and Qdrant serves `X-Frame-Options: DENY`.
// That is their own decision rather than a fault in the platform, and a reader
// looking at an empty rectangle reads it as a broken platform.

const PANEL_META: Record<string, { label: string; purposeKey: string }> = {
  qdrant:     { label: 'Qdrant',                purposeKey: 'panelsPage.purposeQdrant' },
  opensearch: { label: 'OpenSearch Dashboards', purposeKey: 'panelsPage.purposeOpensearch' },
  langfuse:   { label: 'Langfuse',              purposeKey: 'panelsPage.purposeLangfuse' },
  deepeval:   { label: 'DeepEval',              purposeKey: 'panelsPage.purposeDeepeval' },
  neo4j:      { label: 'Neo4j Browser',         purposeKey: 'panelsPage.purposeNeo4j' },
}

function PanelStateBadge({ panel }: { panel: PanelStatus }) {
  const { t } = useTranslation()
  if (!panel.reachable) {
    return (
      <span className="badge badge-warn" title={t('panelsPage.stateDownHint')}>
        <CircleX size={11} aria-hidden="true" /> {t('panelsPage.stateDown')}
      </span>
    )
  }
  if (!panel.embeddable) {
    return (
      <span className="badge badge-info" title={panel.blocked_by ?? ''}>
        <Ban size={11} aria-hidden="true" /> {t('panelsPage.stateNoFrame')}
      </span>
    )
  }
  return (
    <span className="badge badge-success">
      <CircleCheck size={11} aria-hidden="true" /> {t('panelsPage.stateUp')}
    </span>
  )
}

function ToolsSection() {
  const { t } = useTranslation()
  const [framed, setFramed] = useState<string | null>(null)
  const { data: panels = [], isLoading } = useQuery({
    queryKey: ['panels-status'],
    queryFn: api.panelsStatus,
    // The tools are separate processes an operator starts and stops, so a page
    // left open would otherwise be showing an hour-old state.
    staleTime: 15_000,
  })

  const shown = panels.find(p => p.id === framed) ?? null

  return (
    <section>
      <div className="section-rule">
        <h2 className="section-title">{t('panelsPage.title')}</h2>
        <span className="section-meta">{t('realmResourcesPage.panelsNote')}</span>
      </div>
      {isLoading && <div className="loading">{t('panelsPage.loadingPanels')}</div>}
      {!isLoading && panels.length === 0 && <p className="empty">{t('panelsPage.noPanelsBody')}</p>}

      <div className="tool-grid">
        {panels.map(panel => {
          const meta = PANEL_META[panel.id]
          return (
            <div key={panel.id} className="card tool-card">
              <div className="tool-card-head">
                <span className="tool-card-name">{meta?.label ?? panel.id}</span>
                <PanelStateBadge panel={panel} />
              </div>
              <p className="tool-card-purpose">{meta ? t(meta.purposeKey) : t('panelsPage.purposeUnknown')}</p>
              <code className="tool-card-url">{panel.url}</code>
              <div className="tool-card-actions">
                <a className="btn-sm" href={panel.url} target="_blank" rel="noopener noreferrer">
                  <ExternalLink size={12} aria-hidden="true" /> {t('panelsPage.openInNewTab')}
                </a>
                {panel.reachable && panel.embeddable && (
                  <button type="button" className="btn-sm"
                          onClick={() => setFramed(framed === panel.id ? null : panel.id)}>
                    <Frame size={12} aria-hidden="true" />
                    {framed === panel.id ? t('panelsPage.hideHere') : t('panelsPage.showHere')}
                  </button>
                )}
              </div>
              {panel.reachable && !panel.embeddable && (
                <div className="tool-card-note">{t('panelsPage.noFrameNote')}</div>
              )}
            </div>
          )
        })}
      </div>

      {shown && (
        <div className="card frame-card">
          <div className="jd-toolbar frame-bar">
            <span className="jd-toolbar-label">{PANEL_META[shown.id]?.label ?? shown.id}</span>
            <div className="jd-toolbar-spacer" />
            <button type="button" className="btn-sm" onClick={() => setFramed(null)}>
              {t('panelsPage.hideHere')}
            </button>
          </div>
          <iframe
            key={shown.id} src={shown.url}
            title={PANEL_META[shown.id]?.label ?? shown.id}
            className="panel-iframe"
            sandbox="allow-scripts allow-same-origin allow-forms"
          />
        </div>
      )}
    </section>
  )
}

export default function RealmResourcesPage() {
  const { t } = useTranslation()
  const { activeRealm, activeRealmId, reload: reloadRealms } = useRealm()
  const [connectorTypes, setConnectorTypes] = useState<ConnectorType[]>([])
  const [resources, setResources] = useState<Resource[]>([])
  const [addingType, setAddingType] = useState<string | null>(null)
  const [params, setParams] = useState<Record<string, string>>({})
  const [testResults, setTestResults] = useState<Record<string, TestResult>>({})
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState('')
  const [neo4jProvision, setNeo4jProvision] = useState<Neo4jProvisionStatus | null>(null)
  const [provisioning, setProvisioning] = useState(false)
  const provisionPollRef = useRef<ReturnType<typeof setInterval> | null>(null)

  useEffect(() => {
    fetch('/api/connector-types').then(r => r.json()).then(setConnectorTypes).catch(() => {})
  }, [])

  useEffect(() => {
    if (activeRealm) setResources(activeRealm.resources as Resource[])
  }, [activeRealm])

  useEffect(() => () => {
    if (provisionPollRef.current) clearInterval(provisionPollRef.current)
  }, [])

  // Self-service front-end for the manual `tools/generate_realm_neo4j_compose.py`
  // + `docker compose up -d` two-step — POST kicks off a background job on the
  // gateway (image pull +
  // healthcheck can exceed a request timeout), this polls its status.
  const provisionNeo4j = async () => {
    if (!activeRealmId) return
    setProvisioning(true)
    try {
      const res = await fetch(`/api/realms/${activeRealmId}/neo4j/provision`, { method: 'POST' })
      const data: Neo4jProvisionStatus = await res.json()
      setNeo4jProvision(data)
      if (data.status !== 'running') {
        setProvisioning(false)
        return
      }
      provisionPollRef.current = setInterval(async () => {
        const r = await fetch(`/api/realms/${activeRealmId}/neo4j/provision`)
        if (!r.ok) return
        const d: Neo4jProvisionStatus = await r.json()
        setNeo4jProvision(d)
        if (d.status !== 'running') {
          if (provisionPollRef.current) clearInterval(provisionPollRef.current)
          setProvisioning(false)
          if (d.status === 'done') await reloadRealms()
        }
      }, 2000)
    } catch {
      setProvisioning(false)
    }
  }

  if (!activeRealmId) {
    return (
      <div className="page">
        <h1 className="page-title">{t('realmResourcesPage.title')}</h1>
        <div className="card">
          <p className="text-muted">{t('realmResourcesPage.selectRealmFirst')}</p>
        </div>
      </div>
    )
  }

  const save = async (updated: Resource[]) => {
    if (!activeRealmId) return
    setSaving(true)
    setError('')
    try {
      const res = await fetch(`/api/realms/${activeRealmId}/resources`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(updated),
      })
      if (!res.ok) {
        const d = await res.json()
        setError(d.detail ?? t('realmResourcesPage.saveError'))
      } else {
        setResources(updated)
        await reloadRealms()
      }
    } finally {
      setSaving(false)
    }
  }

  const removeResource = (idx: number) => {
    const updated = resources.filter((_, i) => i !== idx)
    save(updated)
  }

  const addResource = async () => {
    if (!addingType) return
    const ct = connectorTypes.find(c => c.type === addingType)
    if (!ct) return
    const newResource: Resource = { type: addingType }
    for (const [k, v] of Object.entries(ct.params_schema)) {
      newResource[k] = params[k] ?? (v.default !== undefined ? String(v.default) : '')
    }
    const updated = [...resources.filter(r => r.type !== addingType), newResource]
    await save(updated)
    setAddingType(null)
    setParams({})
  }

  const testResource = async (type: string) => {
    if (!activeRealmId) return
    const res = await fetch(`/api/realms/${activeRealmId}/resources/test`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ type }),
    })
    const data = await res.json()
    setTestResults(prev => ({ ...prev, [type]: data }))
  }

  // "Check everything" fans out over the same three checks as the individual
  // buttons. Sequentially rather than at once: four simultaneous requests to
  // one local machine compete with each other, and one of them times out
  // falsely. A screen that manufactures its own failure is useless.
  const queryClient = useQueryClient()
  const [testingAll, setTestingAll] = useState(false)
  const testAll = async () => {
    setTestingAll(true)
    try {
      for (const r of resources) await testResource(r.type)
      await queryClient.invalidateQueries({ queryKey: ['panels-status'] })
    } finally {
      setTestingAll(false)
    }
  }

  const selectedCt = connectorTypes.find(c => c.type === addingType)

  // A resource counts as checked once an answer has arrived: "unchecked" and
  // "not responding" are different things, and adding them into one denominator
  // passes ignorance off as a measurement.
  const checkStats = resources.reduce(
    (acc, r) => {
      const tr = testResults[r.type]
      if (!tr) acc.unknown += 1
      else if (tr.status === 'ok') acc.ok += 1
      else acc.down += 1
      return acc
    },
    { total: resources.length, ok: 0, down: 0, unknown: 0 },
  )

  return (
    <div className="page page-wide">
      <div className="page-head">
        <h1 className="page-title">{t('realmResourcesPage.title')}</h1>
        <p className="page-sub">{activeRealm?.name ?? activeRealmId}</p>
        <span className="page-act">
          <button type="button" className="btn btn-primary" onClick={testAll} disabled={testingAll}>
            <Zap size={13} />
            {testingAll ? t('realmResourcesPage.testingAll') : t('realmResourcesPage.testAll')}
          </button>
        </span>
      </div>

      {/* How many checks passed, above the list rather than as marks scattered
          across the tiles: "two not responding" reads at a glance, whereas
          twelve tiles have to be walked. */}
      <div className="stat-band">
        <div className="stat-cell">
          <div className="eyebrow">{t('realmResourcesPage.band.total')}</div>
          <div className="metric-val">{checkStats.total}</div>
        </div>
        <div className="stat-cell">
          <div className="eyebrow">{t('realmResourcesPage.band.ok')}</div>
          <div className="metric-val ok">{checkStats.ok}</div>
        </div>
        <div className="stat-cell">
          <div className="eyebrow">{t('realmResourcesPage.band.down')}</div>
          <div className="metric-val" style={checkStats.down > 0 ? { color: 'var(--color-danger)' } : undefined}>
            {checkStats.down}
          </div>
        </div>
        <div className="stat-cell">
          <div className="eyebrow">{t('realmResourcesPage.band.unknown')}</div>
          <div className="metric-val">{checkStats.unknown}</div>
        </div>
      </div>

      <div className="section-rule flush">
        <h2 className="section-title">{t('realmResourcesPage.resourcesHeading')}</h2>
        <span className="section-meta">{t('realmResourcesPage.resourcesNote')}</span>
      </div>
      <div className="res-grid">
      {resources.length === 0 && (
        <div className="card resource-empty">
          <Server size={22} className="dim-icon" />
          <p className="text-muted flush">{t('realmResourcesPage.noResources')}</p>
        </div>
      )}

      {resources.map((r, idx) => {
        const tr = testResults[r.type]
        const Icon = RESOURCE_ICONS[r.type] ?? Server
        const entries = Object.entries(r).filter(([k]) => k !== 'type')
        return (
          <div key={r.type} className="res-tile">
            <div className="resource-card-head">
              <div className="resource-card-title">
                <span className="resource-icon"><Icon size={16} /></span>
                <strong className="resource-card-name">
                  {connectorTypes.find(c => c.type === r.type)?.label ?? r.type}
                </strong>
                {/* The state sits beside the name rather than only in the
                    check result below: a tile is read for "is it alive", and
                    that should not require reading to the end. */}
                <span className={`flag ${tr ? (tr.status === 'ok' ? 'flag-ok' : 'flag-bad') : ''}`}>
                  <i className="flag-dot" aria-hidden="true" />
                  {tr
                    ? t(`overview.state.${tr.status === 'ok' ? 'ok' : 'down'}`)
                    : t('overview.state.unknown')}
                </span>
              </div>
              <div className="flex-row res-tile-actions">
                <button className="btn-sm" onClick={() => testResource(r.type)}>
                  <Zap size={12} className="btn-icon" />{t('realmResourcesPage.testShort')}
                </button>
                <button className="btn-sm btn-danger" onClick={() => removeResource(idx)} title={t('realmResourcesPage.deleteResource')}>
                  <Trash2 size={13} />
                </button>
              </div>
            </div>
            {entries.length > 0 && (
              <div className="resource-params">
                {entries.map(([k, v]) => (
                  <span key={k}><span className="k">{k}</span><span className="v">{displayValue(k, v)}</span></span>
                ))}
              </div>
            )}
            {r.type === 'neo4j' && (
              <>
                <p className="res-note">
                  {t('realmResourcesPage.neo4j.isolationNote')}
                </p>
                <button
                  className="btn-sm res-sub-btn"
                  onClick={provisionNeo4j} disabled={provisioning}
                >
                  <Rocket size={12} className="btn-icon" />
                  {provisioning ? t('realmResourcesPage.neo4j.provisioning') : t('realmResourcesPage.neo4j.provisionButton')}
                </button>
                {neo4jProvision && neo4jProvision.status === 'not_needed' && (
                  <p className="res-note">{neo4jProvision.detail}</p>
                )}
                {neo4jProvision && neo4jProvision.status === 'running' && (
                  <p className="res-note">
                    <Trans i18nKey="realmResourcesPage.neo4j.launching" t={t} values={{ serviceName: neo4jProvision.service_name }}>
                      {'Starting container '}<code className="inline-code">{'{{serviceName}}'}</code>{' — this can take a minute while the image is pulled for the first time…'}
                    </Trans>
                  </p>
                )}
                {neo4jProvision && neo4jProvision.status === 'done' && (
                  <ConnStatus ok detail={t('realmResourcesPage.neo4j.healthyDetail', { uri: neo4jProvision.uri ? ` — bolt: ${neo4jProvision.uri}` : '' })} />
                )}
                {neo4jProvision && neo4jProvision.status === 'error' && (
                  <ConnStatus ok={false} detail={neo4jProvision.detail} />
                )}
              </>
            )}
            {tr && (
              tr.status === 'ok'
                ? <ConnStatus ok fields={Object.entries(tr).filter(([k]) => !['status', 'type'].includes(k))} />
                : <ConnStatus ok={false} detail={tr.detail} />
            )}
          </div>
        )
      })}
      </div>

      {error && <p className="conn-status conn-status-err page-error">{error}</p>}

      {addingType ? (
        <div className="section">
          <div className="section-rule flush">
            <h2 className="section-title">
            {t('realmResourcesPage.addResourceHeading', { label: connectorTypes.find(c => c.type === addingType)?.label })}
          </h2>
          </div>
          {selectedCt && Object.entries(selectedCt.params_schema).map(([k, schema]) => (
            <div key={k} className="form-group">
              <label>{k}</label>
              <input
                className="form-input"
                value={params[k] ?? String(schema.default ?? '')}
                onChange={e => setParams(p => ({ ...p, [k]: e.target.value }))}
                type={schema.type === 'integer' ? 'number' : k === 'password' ? 'password' : 'text'}
              />
            </div>
          ))}
          <div className="flex-row res-actions">
            <button className="btn-primary" onClick={addResource} disabled={saving}>{t('realmResourcesPage.save')}</button>
            <button className="btn" onClick={() => { setAddingType(null); setParams({}) }}>
              {t('realmResourcesPage.cancel')}
            </button>
          </div>
        </div>
      ) : (
        <div className="resource-add-row">
          {connectorTypes
            .filter(ct => !resources.some(r => r.type === ct.type))
            .map(ct => {
              const AddIcon = RESOURCE_ICONS[ct.type] ?? Server
              return (
                <button key={ct.type} className="btn" onClick={() => setAddingType(ct.type)}>
                  <Plus size={13} className="btn-icon" />
                  <AddIcon size={14} className="btn-icon bare" />
                  {ct.label}
                </button>
              )
            })}
        </div>
      )}

      <RagEndpointsSection activeRealmId={activeRealmId} />
      <ToolsSection />
    </div>
  )
}
