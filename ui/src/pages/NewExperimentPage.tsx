import { useEffect, useRef, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import type { PipelineDescription } from '../api/client'
import { useNavigate, useSearchParams, Link } from 'react-router-dom'
import { useTranslation, Trans } from 'react-i18next'
import GuideLink from '../components/GuideLink'
import { Play } from 'lucide-react'
import SelectBox from '../components/SelectBox'
import { api } from '../api/client'
import { useProgress } from '../hooks/useProgress'
import { useStopExperiment } from '../hooks/useStopExperiment'
import { useRealm, useRealmPath } from '../context/RealmContext'
import i18n from '../i18n'

// ── Presets ───────────────────────────────────────────────────────────────────

interface Preset {
  label: string
  descriptionKey: string
  form: Partial<Record<string, string>>
}

// Presets vary only fields that actually change pipeline behavior
// (retrievalType/top_k/reranker/grounding) — chunker/embedder/generator/seed
// used to appear here too, but ExperimentRunner._build_pipeline never reads
// them for an in_process run (see the design notes), so a preset
// like the old "Structure-Aware" was a lie: it looked like it changed
// chunking, but every preset silently ran identical retrieval. Dropped
// rather than kept as a no-op choice.
const PRESETS: Preset[] = [
  {
    label: 'Baseline Dense',
    descriptionKey: 'newExperimentPage.presets.baselineDense.description',
    form: { name: 'baseline-dense', retrievalType: 'naive', top_k: '5' },
  },
  {
    label: 'Hybrid RRF',
    descriptionKey: 'newExperimentPage.presets.hybridRrf.description',
    form: { name: 'hybrid-rrf', retrievalType: 'hybrid_rrf', top_k: '10' },
  },
  {
    label: 'Hybrid Weighted',
    descriptionKey: 'newExperimentPage.presets.hybridWeighted.description',
    form: { name: 'hybrid-weighted', retrievalType: 'hybrid_weighted', top_k: '10' },
  },
  {
    label: 'High Recall (top-k=15)',
    descriptionKey: 'newExperimentPage.presets.highRecall.description',
    form: { name: 'high-recall-k15', retrievalType: 'hybrid_rrf', top_k: '15' },
  },
  {
    label: 'GraphRAG (multi-hop)',
    descriptionKey: 'newExperimentPage.presets.graphragMultihop.description',
    form: { name: 'graphrag-multihop', retrievalType: 'graph', top_k: '10' },
  },
  {
    label: 'GraphRAG + Rerank',
    descriptionKey: 'newExperimentPage.presets.graphragRerank.description',
    form: { name: 'graphrag-rerank', retrievalType: 'graph', reranker: 'cross_encoder_local', top_k: '15' },
  },
  {
    label: 'GraphRAG + Grounding',
    descriptionKey: 'newExperimentPage.presets.graphragGrounding.description',
    form: { name: 'graphrag-grounded', retrievalType: 'graph', grounding: 'token_overlap', top_k: '10' },
  },
]

// Registry pipeline_id -> human label. Falls back to the raw id for any
// future pipeline_id the registry offers that isn't listed here yet — a new
// archetype registered on the backend shows up immediately, just unlabeled
// (matches the page's own "the form is built from the registry" principle).
const PIPELINE_LABELS: Record<string, string> = {
  // A live getter (not a plain string) so it re-resolves through i18next on
  // every access instead of freezing whatever language was active at
  // module-load time — same reasoning as lib/metricMeta.ts's METRIC_META.
  get naive() { return i18n.t('newExperimentPage.pipelineLabels.naive') },
  hybrid_rrf: 'Hybrid — RRF',
  hybrid_weighted: 'Hybrid — Weighted',
  graph: 'GraphRAG (Neo4j)',
}

// ── Config builder ────────────────────────────────────────────────────────────

// The contract's own extensible-params object, plus the Model select
// (see DEFAULT_FORM.model) merged in under the same "model" key an external
// RAG already reads there — one shared shape for both
// pipeline_source values instead of a second field just for in_process.
// Invalid hand-typed JSON in the params textarea silently becomes "ignore
// the textarea, keep the Model select's own value" rather than dropping
// the model choice too.
function _buildParams(form: Record<string, string>): Record<string, unknown> | undefined {
  let params: Record<string, unknown> = {}
  if (form.params?.trim()) {
    try {
      params = JSON.parse(form.params)
    } catch {
      params = {}
    }
  }
  if (form.model) params.model = form.model
  return Object.keys(params).length > 0 ? params : undefined
}

function buildConfig(form: Record<string, string>, pipelines: Record<string, PipelineDescription> = {}) {
  // 'Retrieval type' IS pipeline_id directly now — no more deriving it from two
  // separate selects (retriever + merge). That old scheme could never
  // produce 'naive' (the plain select always had a truthy merge value), so
  // dense-only retrieval was unreachable from this form even though the
  // registry has always had it — see the design notes
  const pipeline_id = form.retrievalType || 'naive'

  const isHttp = form.pipeline_source === 'http' && !!form.external_rag_id

  const cfg: Record<string, unknown> = {
    name: form.name || 'experiment',
    // Never varied from this form, and the reason has narrowed. The build
    // reads none of chunking_strategy/generator/seed for an in_process run,
    // so those stay fixed literals that keep the config valid without
    // pretending to be choices. `embedder` is no longer one of them: the
    // build resolves it now and a run naming another model queries with it.
    // Offering it here is a screen that has to be designed before it is
    // written, so this sends the one the gateway registers and says why,
    // instead of claiming nothing reads it.
    seed: 42,
    chunking_strategy: { kind: 'chunker', component_id: 'fixed' },
    embedder: { kind: 'embedder', component_id: 'bge_m3' },
    // Read from what the server holds under this pipeline id, never derived
    // from the id itself: the old expression named a dense retriever for
    // every pipeline but the graph one, so a hybrid run was recorded as
    // dense and any architecture registered later would be too.
    retrievers: [{ kind: 'retriever', component_id: pipelines[pipeline_id]?.retriever || 'qdrant_dense' }],
    generator: { kind: 'generator', component_id: 'ollama' },
    top_k: parseInt(form.top_k || '5', 10),
    pipeline_id,
    // Inert: services/api_gateway/routers/experiments.py's create_experiment
    // always overwrites this with the request's own top-level dataset_name
    // (`cfg_data["dataset_name"] = body.dataset_name`) before hashing/running
    // — kept in sync with `form.dataset` anyway so nothing misleading sits
    // in the object that gets serialized into the POST body.
    dataset_name: form.dataset,
    dataset_version: '',
  }

  if (isHttp) {
    // Resolve url/headers/retrieve_endpoint/mapping from
    // the saved ExternalRag record (not a raw URL field — that also enables
    // tier-2 declarative mapping and per-rag datasets, see _load_dataset).
    cfg.pipeline_source = 'http'
    cfg.external_rag_id = form.external_rag_id
    // corpus_id AND reranker travel through the
    // native HTTP contract now (adapters/http_pipeline.py,
    // services/reference_rag_server) — omitting corpus_id was a real bug
    // (near-zero recall on a dataset built for a different corpus, see
    // the design notes). `pipeline_id` (now the single 'Retrieval type'
    // choice) was always in `cfg` already — this just makes it actually
    // reach the external RAG too, so presets like "GraphRAG (multi-hop)"/
    // "GraphRAG + Rerank" now drive reference_rag_server's pipeline_id/
    // reranker_id, not just top_k. grounding/route_policy/scorer/
    // mask_engine/refusal_policy remain omitted — reference_rag_server has
    // no equivalent for those, and a third-party external RAG has no
    // declared way to accept them either.
    cfg.corpus_id = form.corpus_id || 'default'
    if (form.reranker) cfg.reranker = { kind: 'reranker', component_id: form.reranker }
    // retrieval-only: scores retrieval without paying
    // for generation on every measurement. No-op for in_process (there is
    // no separate retrieval-only path there), so only sent for http.
    if (form.retrievalOnly === 'true') cfg.retrieval_only = true
    const httpParams = _buildParams(form)
    if (httpParams) cfg.params = httpParams
    return cfg
  }

  // merge_strategy is not bookkeeping any more, and the comment that said so
  // outlived the change: `_rebind_merge` applies it, rebuilding the hybrid
  // wrapper around the same two retrievers. So the old expression, which
  // wrote 'rrf' for every pipeline but hybrid_weighted, applied rank fusion
  // to anything new that arrived. Taken from what the pipeline was actually
  // built with, and omitted when the retriever merges nothing: a dense
  // pipeline fuses no sources, and 'rrf' would record a fusion it never did.
  const merge = pipelines[pipeline_id]?.merge_strategy
  if (merge) cfg.merge_strategy = merge
  cfg.corpus_id = form.corpus_id || 'default'
  // Optional, config-driven steps. Omitted when unset so config_hash
  // stays backward compatible with historical runs.
  if (form.reranker) cfg.reranker = { kind: 'reranker', component_id: form.reranker }
  if (form.grounding) cfg.grounding = { kind: 'grounder', component_id: form.grounding }
  if (form.routing) cfg.route_policy = { kind: 'route_policy', component_id: form.routing }
  // domain-pack-provided steps. Omitted when unset so config_hash
  // stays backward compatible with runs made before domain packs existed.
  if (form.scorer) cfg.scorer = { kind: 'scorer', component_id: form.scorer }
  if (form.maskEngine) cfg.mask_engine = { kind: 'mask_engine', component_id: form.maskEngine }
  if (form.refusalPolicy) cfg.refusal_policy = { kind: 'refusal', component_id: form.refusalPolicy }
  const params = _buildParams(form)
  if (params) cfg.params = params
  return cfg
}

const CUSTOM_CORPUS_ID = '__custom_corpus_id__'

/** Real `<SelectBox>` of the active Realm's known corpus_ids + a "+ custom id..."
 * escape hatch, mirroring DatasetsPage.tsx#CorpusSelect and
 * CorpusPage.tsx#UploadCorpusIdField — replaces a plain `<input list="...">`
 * + `<datalist>` that read as an empty text box, not a dropdown, and (found
 * live) kept showing a hardcoded, no-longer-valid corpus_id ('handbook')
 * after switching to a Realm that never had it. `value=''` renders as a
 * disabled placeholder (no auto-picked corpus_id — see canSubmit above,
 * there's no single safe universal default across Realms). The escape
 * hatch stays needed for http pipelines, where corpus_id is an opaque
 * string for the external RAG, not one of this Realm's own registered
 * corpora. */
function CorpusIdField({ id, value, onChange, knownIds }: {
  id: string
  value: string
  onChange: (v: string) => void
  knownIds: string[]
}) {
  const { t } = useTranslation()
  const isKnown = knownIds.includes(value)
  const [customMode, setCustomMode] = useState(value !== '' && !isKnown)
  // Unlike CorpusSelect/UploadCorpusIdField, `value` here can change from
  // OUTSIDE this field's own onChange — picking a saved external RAG
  // auto-fills corpus_id from its default_corpus_id (see externalRags
  // select below), which is virtually never one of this Realm's own
  // registered corpus_ids. Without this effect, that auto-fill would
  // silently render as a blank select (isKnown false) instead of showing
  // the RAG's actual corpus_id in the custom-input escape hatch.
  useEffect(() => {
    if (value !== '' && !isKnown) setCustomMode(true)
  }, [value, isKnown])

  if (customMode) {
    return (
      <div className="field-with-btn">
        <input
          id={id} type="text" className="grow" value={value} autoFocus
          onChange={e => onChange(e.target.value)}
        />
        <button
          type="button" className="btn-sm"
          onClick={() => { setCustomMode(false); onChange('') }}
        >
          {t('newExperimentPage.form.corpusFromList')}
        </button>
      </div>
    )
  }

  return (
    <SelectBox
      id={id} value={isKnown ? value : ''}
      onChange={e => {
        if (e.target.value === CUSTOM_CORPUS_ID) { setCustomMode(true); onChange('') }
        else onChange(e.target.value)
      }}
    >
      <option value="" disabled>{t('newExperimentPage.form.corpusSelectPrompt')}</option>
      {knownIds.map(cid => <option key={cid} value={cid}>{cid}</option>)}
      <option value={CUSTOM_CORPUS_ID}>{t('newExperimentPage.form.corpusCustomOption')}</option>
    </SelectBox>
  )
}

function ProgressWidget({ runId, onDone }: { runId: string; onDone: () => void }) {
  const { t } = useTranslation()
  // create_experiment now returns before the run
  // finishes (job model), so navigation must wait for the WS "done" event
  // instead of a fixed setTimeout guessing how long the run will take.
  const { done, progress, latest } = useProgress(runId)
  const { stopping, stopError, stop } = useStopExperiment(runId)
  useEffect(() => {
    if (done) onDone()
  }, [done, onDone])

  return (
    <div className="card mt-16">
      <h2>{t('newExperimentPage.progress.title')}</h2>
      <div className="progress-bar">
        <div className="progress-fill" style={{ width: `${(progress ?? 0) * 100}%` }} />
      </div>
      <div className="flex-between align-start">
        <p className="text-muted sm">
          {done ? t('newExperimentPage.progress.done')
            : stopping ? t('newExperimentPage.progress.stopping')
            : latest?.stage ? t('newExperimentPage.progress.stage', { stage: latest.stage })
            : t('newExperimentPage.progress.starting')}
          {latest?.processed != null && latest?.total != null && (
            <> — {t('newExperimentPage.progress.questionsCount', { processed: latest.processed, total: latest.total })}</>
          )}
        </p>
        {!done && (
          <button type="button" className="btn-sm btn-danger" onClick={stop} disabled={stopping}>
            {stopping ? t('newExperimentPage.progress.stopping') : t('newExperimentPage.progress.stopButton')}
          </button>
        )}
      </div>
      {stopError && <div className="badge badge-danger mt-8">{stopError}</div>}
    </div>
  )
}

// ── Page ──────────────────────────────────────────────────────────────────────

const DEFAULT_FORM: Record<string, string> = {
  name: '', top_k: '5',
  // The one real structural lever for in_process runs — see buildConfig().
  // Defaults to the platform's historical default shape (dense+sparse RRF),
  // not 'naive', so existing muscle memory / comparisons against older runs
  // don't shift just because 'naive' became reachable.
  retrievalType: 'hybrid_rrf',
  // These remain genuinely optional — empty here means the step is omitted
  // from the config entirely (buildConfig's `if (form.x) cfg.x = ...`), not
  // "use some hidden default", so "— none —" stays accurate.
  reranker: '', grounding: '', routing: '',
  scorer: '', maskEngine: '', refusalPolicy: '',
  dataset: 'handbook.v2.full.jsonl',
  // Found live: this used to be hardcoded to 'handbook', a demo-only
  // corpus, so switching to any other Realm still showed/
  // submitted a corpus_id that Realm never had. A universal fallback like
  // 'default' isn't safe either — for demo itself, 'default' is an
  // older/smaller demo corpus, not what the also-defaulted handbook.v2 dataset
  // (143 questions) was actually built against; silently picking either
  // hardcoded value risks the same "retrieval searches the wrong index"
  // failure corpusHintBuiltin below already warns about. Left empty on
  // purpose — the submit button stays disabled (see canSubmit) until the
  // user makes an explicit choice for in_process runs.
  corpus_id: '',
  pipeline_source: 'in_process', external_rag_id: '',
  // retrieval-only path, http only (see buildConfig()).
  retrievalOnly: '',
  // Extensible, capabilities-gated knobs (JSON object).
  // Empty here means "omitted from config entirely", same convention as
  // the optional component selectors above.
  params: '',
  // Empty = no override, falls back to whatever the target already uses
  // (the Realm/global active model for in_process, the RAG's own default
  // for http) — same "additive only" convention as the optional selectors
  // above, not a forced choice.
  model: '',
}

export default function NewExperimentPage() {
  const { t } = useTranslation()
  const navigate = useNavigate()
  const { activeRealmId } = useRealm()
  const toRealm = useRealmPath()
  const [form, setForm] = useState<Record<string, string>>(DEFAULT_FORM)
  const [activePreset, setActivePreset] = useState<string | null>(null)
  const [runId, setRunId] = useState<string | null>(null)
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // "New run based on this" (RunDiagnostics.tsx / RunPage.tsx toolbar)
  // navigates here with ?from=<run_id> — found live: this page never read
  // that param at all, so the button silently opened a blank form. Fetches
  // the source run's own config once and prefills the form from it; only
  // enabled when `from` is present so pages/tests that open this route
  // without the param never call api.experiments.get at all.
  const [searchParams] = useSearchParams()
  const fromRunId = searchParams.get('from')
  const didPrefillRef = useRef(false)
  const { data: sourceRun } = useQuery({
    queryKey: ['experiment', fromRunId],
    queryFn: () => api.experiments.get(fromRunId!),
    enabled: !!fromRunId,
  })

  const { data: registry, isLoading: regLoading } = useQuery({
    queryKey: ['registry', activeRealmId],
    queryFn: () => api.registry(activeRealmId),
  })
  // What each pipeline is made of. One per process and not per realm, since
  // the registry is one per process.
  const { data: pipelines } = useQuery({
    queryKey: ['pipelines'],
    queryFn: () => api.pipelines(),
  })
  const { data: datasets } = useQuery({
    queryKey: ['datasets', activeRealmId],
    queryFn: () => api.datasets.list(activeRealmId),
  })
  const { data: externalRags = [] } = useQuery<import('../api/client').ExternalRag[]>({
    queryKey: ['external-rags', activeRealmId],
    queryFn: () => api.externalRags.list(activeRealmId),
  })
  const { data: corpusHistory = [] } = useQuery({
    queryKey: ['corpus', activeRealmId],
    queryFn: () => api.corpus.list(activeRealmId),
  })
  // The `corpora` registry, not just ingest-job history: knows
  // about corpora ingested outside the upload form (CLI, the migration
  // script) — this is what used to leave 'handbook' hardcoded below as
  // the only non-history fallback.
  const { data: corpusCollections = [] } = useQuery({
    queryKey: ['corpus-collections', activeRealmId],
    queryFn: () => api.corpus.collections(activeRealmId),
  })
  // Which model generates this run's answers — used to be completely
  // decorative for in_process runs (ExperimentConfig.generator was never
  // read by _build_pipeline, see the design notes "Decorative")
  // and only reachable for http runs by hand-typing {"model": "..."} into
  // the raw params JSON below. Both paths now read this from the SAME
  // params.model key (buildConfig merges it in) — an external RAG already
  // read params["model"] itself, this just gives in_process
  // runs the identical, working lever instead of a second field shape.
  const { data: models = [] } = useQuery({ queryKey: ['models', 'completion'], queryFn: () => api.models(true) })
  const isHttp = form.pipeline_source === 'http'
  // Only to warn that a chosen external RAG declared no retrieve_endpoint,
  // in which case the platform quietly runs the full generation path.
  const retrievalOnlyRag = externalRags.find(r => r.id === form.external_rag_id)
  // in_process runs need an explicit corpus_id — no realm-agnostic default
  // is safe to guess (see DEFAULT_FORM.corpus_id above). http pipelines are
  // exempt: corpus_id there is an opaque, optional hint for the external
  // RAG, not a required platform corpus.
  const canSubmit = isHttp || !!form.corpus_id
  // Datasets are Realm-scoped (see services/api_gateway/routers/
  // datasets.py) as a single unified list regardless of
  // pipeline_source — platform- and RAG-authored datasets are the same
  // collection now, not two separate optgroups. Switching the active Realm
  // can make the form's current `dataset` filename disappear from the list
  // entirely. Clear it rather than silently submitting a run against a
  // dataset the dropdown no longer shows (it would still resolve
  // server-side, since GET-by-filename isn't realm-checked, but that's not
  // what picking "no visible dataset" should mean from this form).
  useEffect(() => {
    if (!datasets) return
    if (form.dataset && !datasets.some(d => d.filename === form.dataset)) {
      setForm(f => ({ ...f, dataset: datasets[0]?.filename ?? '' }))
    }
  }, [datasets])
  // No synthetic 'default': the name was always in the list while the content
  // behind it was not, so picking it meant running against nothing. A corpus by
  // that name, if it exists, arrives from the registry like any other.
  const knownCorpusIds = Array.from(new Set([
    ...corpusCollections.map(c => c.corpus_id),
    ...corpusHistory.flatMap(h => h.corpus_id ? [h.corpus_id] : []),
  ]))
  // Found live: switching the active Realm left `form.corpus_id` on
  // whatever it was before (another Realm's corpus id), even though that
  // corpus doesn't exist in the newly-selected Realm at all — this field
  // had no equivalent of the `dataset` reset effect above. Clears back to
  // '' (forcing an explicit re-choice, see canSubmit) rather than guessing
  // a new default — there's no single corpus_id that's a safe universal
  // fallback across Realms (see DEFAULT_FORM.corpus_id above). Skipped for
  // http pipelines: there, corpus_id is an opaque string meaningful only
  // to the external RAG (see corpusHintHttp below), not one of this
  // Realm's own registered corpora, so it must NOT be silently reset here.
  useEffect(() => {
    if (isHttp) return
    if (form.corpus_id && !knownCorpusIds.includes(form.corpus_id)) {
      setForm(f => ({ ...f, corpus_id: '' }))
    }
  }, [corpusCollections, corpusHistory, isHttp])

  // Applies once per page load (didPrefillRef), not on every refetch of the
  // source run — otherwise a field the user already edited by hand would
  // get silently clobbered back to the source run's value. Every field
  // buildConfig() itself reads is covered, so re-submitting this form
  // reproduces the source run's config exactly unless the user changes
  // something. Fields absent on the source run (optional steps) are set to
  // '' explicitly rather than left alone — "this run didn't use a
  // reranker" is real information, not "no opinion".
  useEffect(() => {
    if (!sourceRun || didPrefillRef.current) return
    didPrefillRef.current = true
    const cfg = sourceRun.config ?? {}
    const { model: _sourceModelParam, ...restParams } = (cfg.params ?? {}) as Record<string, unknown>
    setForm(f => ({
      ...f,
      name: sourceRun.config_name ? `${sourceRun.config_name}-2` : f.name,
      pipeline_source: cfg.pipeline_source || f.pipeline_source,
      external_rag_id: cfg.external_rag_id || '',
      // The real model that answered this run, not the fixed 'ollama'
      // adapter literal — see the design notes#Live levers.
      model: sourceRun.generator_model || (cfg.params?.model as string | undefined) || '',
      retrievalType: cfg.pipeline_id || f.retrievalType,
      reranker: cfg.reranker?.component_id || '',
      top_k: cfg.top_k != null ? String(cfg.top_k) : f.top_k,
      corpus_id: cfg.corpus_id || f.corpus_id,
      // Found live: `sourceRun.dataset_name` is often extension-stripped
      // ("handbook.v2.full") while `cfg.dataset_name` carries the real
      // filename ("handbook.v2.full.jsonl") that matches this <SelectBox>'s own
      // option values (Dataset.filename) — preferring the display-only
      // field left this select silently unselected. Same root cause fixed
      // in RunPage.tsx's runDatasetName.
      dataset: cfg.dataset_name || sourceRun.dataset_name || f.dataset,
      grounding: cfg.grounding?.component_id || '',
      routing: cfg.route_policy?.component_id || '',
      scorer: cfg.scorer?.component_id || '',
      maskEngine: cfg.mask_engine?.component_id || '',
      refusalPolicy: cfg.refusal_policy?.component_id || '',
      retrievalOnly: cfg.retrieval_only ? 'true' : '',
      params: Object.keys(restParams).length > 0 ? JSON.stringify(restParams) : '',
    }))
  }, [sourceRun])

  const set = (key: string) => (e: React.ChangeEvent<HTMLSelectElement | HTMLInputElement>) =>
    setForm(prev => ({ ...prev, [key]: e.target.value }))

  // Fields a preset fully determines — reset to baseline before applying so
  // a value left over from a PREVIOUS preset (e.g. retrievalType: 'graph'
  // from "GraphRAG (multi-hop)") doesn't silently survive into a preset that
  // doesn't mention that field at all (e.g. "Hybrid RRF" never sets
  // `retrievalType`). Without this reset, switching graph → non-graph preset
  // looked like the "Retrieval type" selector had stopped responding — it was
  // the merge-onto-prev pattern keeping the stale value, not a broken select.
  const PRESET_RESET: Record<string, string> = {
    retrievalType: 'hybrid_rrf', reranker: '', grounding: '', top_k: '5',
  }

  const applyPreset = (preset: Preset) => {
    setActivePreset(preset.label)
    setForm(prev => ({ ...prev, ...PRESET_RESET, ...(preset.form as Record<string, string>) }))
  }

  const submit = async (e: React.FormEvent) => {
    e.preventDefault()
    setSubmitting(true)
    setError(null)
    try {
      // No hardcoded dataset fallback here anymore — found live: picking an
      // external RAG with no datasets of its own (dropdown said so) still
      // silently ran the platform's own handbook.v2.full.jsonl (143 real
      // questions) against it, because this literal overrode whatever the
      // dropdown actually showed. An empty dataset_name makes
      // _load_dataset() degrade to a 5-question stub dataset instead —
      // honest (small, obviously synthetic) rather than silently wrong.
      const result = await api.experiments.create(buildConfig(form, pipelines ?? {}), form.dataset, activeRealmId)
      setRunId(result.run_id)
      // Navigation now happens from ProgressWidget's onDone once the WS
      // reports the backgrounded run actually finished —
      // result.run_id is no longer the final payload, just a job handle.
    } catch (err) {
      setError(String(err))
    } finally {
      setSubmitting(false)
    }
  }

  const sel = (key: string, kind: string, label: string, dim = false, optional = false) => (
    <div className="form-group" style={dim ? { opacity: 0.4, pointerEvents: 'none' } : undefined}>
      <label htmlFor={key}>{label}</label>
      <SelectBox id={key} value={form[key]} onChange={set(key)} disabled={regLoading}>
        {/* Optional steps: empty really means "step omitted from config" —
            "— none —" is accurate. Required steps: pre-selected in
            DEFAULT_FORM with the same literal buildConfig() falls back to,
            so this empty option only shows if the user clears the field by
            hand — kept as an explicit "choose one" prompt, not a hidden default. */}
        <option value="">{optional ? t('newExperimentPage.form.none') : t('newExperimentPage.form.selectPrompt')}</option>
        {(registry?.[kind] ?? []).map(id => (
          <option key={id} value={id}>{id}</option>
        ))}
      </SelectBox>
    </div>
  )

  return (
    <div className="page">
      {/* The start button sits in the header rather than below the form.
          The form is long, so at the bottom the button falls off the screen:
          starting a run after editing one field meant scrolling all the way
          down. In the header it stays visible, and the `form` attribute ties it
          to the form. */}
      <div className="page-head">
        <h1 className="page-title">{t('newExperimentPage.title')}</h1>
        <p className="page-sub">{t('newExperimentPage.subtitle')}</p>
        <span className="page-act">
          <GuideLink section="config" />
          <button type="submit" form="new-run-form" className="btn btn-primary" disabled={submitting || !canSubmit}>
            <Play size={13} />
            {submitting ? t('newExperimentPage.submitting') : t('newExperimentPage.submit')}
          </button>
        </span>
      </div>

      <div className="section">
        <div className="section-rule">
          <h2 className="section-title">{t('newExperimentPage.presets.heading')}</h2>
          <span className="section-meta">{t('newExperimentPage.presets.count', { n: PRESETS.length })}</span>
        </div>
        <div className="chips">
          {PRESETS.map(p => (
            <button
              key={p.label} type="button" title={t(p.descriptionKey)}
              className={`chip${activePreset === p.label ? ' active' : ''}`}
              onClick={() => applyPreset(p)}
            >
              {p.label}
            </button>
          ))}
        </div>
        {activePreset && (
          <p className="hint-line bare pad-t4">
            {(() => {
              const key = PRESETS.find(p => p.label === activePreset)?.descriptionKey
              return key ? t(key) : null
            })()}
          </p>
        )}
      </div>

      <form id="new-run-form" onSubmit={submit}>
        <div className="section">
          <div className="section-rule">
            <h2 className="section-title">{t('newExperimentPage.whatRuns')}</h2>
            <span className="section-meta">{t('newExperimentPage.whatRunsHint')}</span>
          </div>
          <div className="form-grid">
          <div className="form-group">
            <label htmlFor="name">{t('newExperimentPage.form.nameLabel')}</label>
            <input id="name" type="text" value={form.name} onChange={set('name')} placeholder="my-experiment" />
          </div>

          {/* Implementation: built-in pipeline vs external RAG by URL. */}
          <div className="form-pair">
            <div className="form-group">
              <label htmlFor="dataset">{t('newExperimentPage.form.datasetLabel')}</label>
              {/* One flat, Realm-scoped list regardless of
                  pipeline_source: platform- and RAG-authored datasets are
                  the same `datasets` collection now (services/api_gateway/
                  routers/datasets.py), not two separate optgroups. */}
              <SelectBox id="dataset" value={form.dataset} onChange={set('dataset')}>
                <option value="">{t('newExperimentPage.form.datasetNone')}</option>
                {(datasets ?? []).map(d => (
                  <option key={d.filename} value={d.filename}>
                    {d.name} {d.version} {d.count != null ? t('newExperimentPage.form.datasetQuestionsCount', { count: d.count }) : `(${d.speed})`}
                  </option>
                ))}
              </SelectBox>
              {(datasets ?? []).length === 0 && (
                <p className="field-hint">
                  {t('newExperimentPage.form.datasetEmptyRealm')}
                  {isHttp && ` ${t('newExperimentPage.form.datasetEmptyRealmHttpHint')}`}
                </p>
              )}
            </div>
            <div className="form-group">
              <label htmlFor="corpus_id">{t('newExperimentPage.form.corpusLabel')}</label>
              <CorpusIdField
                id="corpus_id" value={form.corpus_id} onChange={v => setForm(f => ({ ...f, corpus_id: v }))}
                knownIds={knownCorpusIds}
              />
              {isHttp ? (
                <p className="field-hint">
                  <Trans i18nKey="newExperimentPage.form.corpusHintHttp" t={t}>
                    An opaque string for this RAG; the platform does not validate it. Matching it to the
                    dataset, meaning which sources it expects, is entirely the RAG's responsibility, and a
                    mismatch can look either like zero recall or like an error on every question. Both have
                    happened. The reference point is the <code className="inline-code">default_corpus_id</code>
                    set when this RAG was registered on the Resources page, rather than the list of platform
                    corpora below.
                  </Trans>
                </p>
              ) : (
                <p className="field-hint">
                  {t('newExperimentPage.form.corpusHintBuiltin')}
                </p>
              )}
            </div>
          </div>
          <div className="form-group">
            <label htmlFor="pipeline_source">{t('newExperimentPage.form.implementationLabel')}</label>
            <SelectBox id="pipeline_source" value={form.pipeline_source}
              onChange={e => setForm(f => ({
                // Dataset no longer reset here: the dropdown is
                // one Realm-scoped list regardless of pipeline_source now
                // (see datasets query above), so switching implementation
                // doesn't change which datasets are available.
                ...f, pipeline_source: e.target.value, external_rag_id: '',
              }))}>
              <option value="in_process">{t('newExperimentPage.form.implementationBuiltin')}</option>
              <option value="http">{t('newExperimentPage.form.implementationExternal')}</option>
            </SelectBox>
          </div>
          {/* Same params.model key an external RAG already reads itself
              — see _buildParams above. Empty = no override,
              falls back to whatever the target already uses. */}
          {isHttp && (
            <div className="form-group">
              <label htmlFor="external_rag_id">{t('newExperimentPage.form.externalRagLabel')}</label>
              <SelectBox id="external_rag_id" value={form.external_rag_id}
                onChange={e => {
                  const id = e.target.value
                  // auto-fill this RAG's own default_corpus_id/
                  // pipeline_id/reranker_id/params instead of leaving
                  // whatever the form happened to have before (previously
                  // corpus_id stayed hardcoded to 'handbook' regardless of
                  // which RAG was picked — see ExternalRag.default_corpus_id
                  // in client.ts for the real run this bit).
                  const picked = externalRags.find(r => r.id === id)
                  setForm(f => ({
                    // Dataset no longer reset here: it's the
                    // same Realm-scoped list for every RAG now, not a
                    // per-RAG optgroup that would disappear on RAG change.
                    ...f, external_rag_id: id,
                    corpus_id: picked?.default_corpus_id || f.corpus_id,
                    retrievalType: picked?.default_pipeline_id || f.retrievalType,
                    reranker: picked?.default_reranker_id || f.reranker,
                    params: picked?.default_params && Object.keys(picked.default_params).length > 0
                      ? JSON.stringify(picked.default_params) : f.params,
                  }))
                }}>
                <option value="">{t('newExperimentPage.form.externalRagSelectSaved')}</option>
                {externalRags.map(r => (
                  <option key={r.id} value={r.id}>{r.name} ({r.url})</option>
                ))}
              </SelectBox>
              <p className="field-hint">
                <Trans i18nKey="newExperimentPage.form.externalRagHint" t={t}>
                  The list is managed on the{' '}
                  <Link to={toRealm('/settings/resources')}>Resources and RAG endpoints</Link> page, along with
                  the description of the approach, the test probe, and this RAG's own control questions.
                </Trans>
                {externalRags.length === 0 && ` ${t('newExperimentPage.form.externalRagEmpty')}`}
              </p>
            </div>
          )}

          {isHttp && (() => {
            // The selected RAG's own declaration of
            // which `params` keys it reads (see ExternalRagCapabilities in
            // api/client.ts) — shown so a user knows what's actually worth
            // typing into the field below, instead of guessing.
            const selectedRag = externalRags.find(r => r.id === form.external_rag_id)
            const supportedParams = selectedRag?.capabilities?.supported_params ?? []
            return (
              <div className="form-group">
                <div className="form-group">
                  <label htmlFor="params">{t('newExperimentPage.form.paramsLabel')}</label>
                  <textarea id="params" value={form.params}
                    onChange={e => setForm(f => ({ ...f, params: e.target.value }))}
                    placeholder='{"temperature": 0.2}'
                    rows={2} className="mono-field" />
                  <p className="field-hint">
                    {supportedParams.length > 0
                      ? t('newExperimentPage.form.paramsSupported', { params: supportedParams.join(', ') })
                      : t('newExperimentPage.form.paramsNotDeclared')}
                    {' '}{t('newExperimentPage.form.paramsInvalidJson')}
                  </p>
                </div>
              </div>
            )
          })()}

          </div>
        </div>

        <div className="section">
          <div className="section-rule">
            <h2 className="section-title">{t('newExperimentPage.pipeline')}</h2>
            <span className="section-meta">{t('newExperimentPage.pipelineHint')}</span>
          </div>
          <div className="form-grid">

          {/* 'Retrieval type' now equals pipeline_id directly (see buildConfig),
              the one real structural lever of an in-process run. The options are
              built from the registry (registry.pipeline), so a newly registered
              archetype appears here on its own, even without a human-readable
              label ready for it. */}
          <div className="form-group">
            <label htmlFor="retrievalType">{t('newExperimentPage.form.retrievalTypeLabel')}</label>
            <SelectBox id="retrievalType" value={form.retrievalType} onChange={set('retrievalType')} disabled={regLoading}>
              {(registry?.pipeline ?? []).map(id => (
                <option key={id} value={id}>{PIPELINE_LABELS[id] ?? id}</option>
              ))}
            </SelectBox>
          </div>

          {/* Configurable pipeline steps, built from the registry.
              The reranker is NOT dimmed for an external RAG: buildConfig() now sends
              it (cfg.reranker) and services/reference_rag_server reads
              reranker_id — same wiring as corpus_id/pipeline_id. Grounding/
              Routing stay dimmed: reference_rag_server has no equivalent for
              them, and _build_pipeline's http branch never reads them. */}
          <div className="form-group">
            <label htmlFor="model">{t('newExperimentPage.form.modelLabel')}</label>
            <SelectBox id="model" value={form.model} onChange={set('model')}>
              <option value="">{t('newExperimentPage.form.modelDefaultOption')}</option>
              {models.map(m => <option key={m.name} value={m.name}>{m.name}</option>)}
            </SelectBox>
            {isHttp && (
              <p className="field-hint">
                {t('newExperimentPage.form.modelHttpHint')}
              </p>
            )}
          </div>
          {sel('reranker',  'reranker', t('newExperimentPage.form.rerankerLabel'), false, true)}
          {sel('grounding', 'grounder', t('newExperimentPage.form.groundingLabel'), isHttp, true)}
          {sel('routing', 'route_policy', t('newExperimentPage.form.routingLabel'), isHttp, true)}

          <div className="form-group">
            <label htmlFor="top_k">top-k</label>
            <input id="top_k" type="number" value={form.top_k} onChange={set('top_k')} min="1" max="50" />
          </div>

          {/* domain-pack-provided steps, built from the registry.
              Populated once a domain pack is activated on the "Domain packs" page
              and the gateway is restarted (e.g. manual_grounding/manuals/manuals).
              Also ignored by the runner for an external RAG — dimmed for the
              same reason as the pipeline steps above. */}
          {sel('scorer',     'scorer',      t('newExperimentPage.form.scorerLabel'), isHttp, true)}
          {sel('maskEngine', 'mask_engine', t('newExperimentPage.form.maskEngineLabel'), isHttp, true)}
          {sel('refusalPolicy', 'refusal', t('newExperimentPage.form.refusalPolicyLabel'), isHttp, true)}

          {/* Offered for both pipeline sources. It used to sit inside the
              external-RAG block because an in_process run ignored it
              outright; NaivePipeline.retrieve() now stops before the
              generator, so hiding it here would hide a working control. */}
          <div className="form-group">
            <label className="check-label">
              <input type="checkbox" checked={form.retrievalOnly === 'true'}
                onChange={e => setForm(f => ({ ...f, retrievalOnly: e.target.checked ? 'true' : '' }))} />
              {t('newExperimentPage.form.retrievalOnlyLabel')}
            </label>
            <p className="field-hint">
              {t('newExperimentPage.form.retrievalOnlyHint')}
              {isHttp && retrievalOnlyRag && !retrievalOnlyRag.capabilities?.supports_retrieval_only
                ? ` ${t('newExperimentPage.form.retrievalOnlyNoEndpoint')}`
                : ''}
            </p>
          </div>

          {/* Where the three lists above come from, said once here rather than
              in a parenthesis on every label. */}
          <p className="hint-line form-span-hint">{t('newExperimentPage.packHint')}</p>



          </div>

          <p className="hint-line">{t('newExperimentPage.hashHint')}</p>
          {error && <p className="conn-status conn-status-err mt-12">{error}</p>}
        </div>
      </form>

      {runId && <ProgressWidget runId={runId} onDone={() => navigate(toRealm(`/experiments/${runId}`))} />}
    </div>
  )
}
