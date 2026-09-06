const BASE = '/api'

async function req<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(BASE + path, {
    headers: { 'Content-Type': 'application/json', ...options?.headers },
    ...options,
  })
  if (!res.ok) {
    // The body is where the reason lives. FastAPI puts it in `detail`, and
    // every endpoint in this platform writes something actionable there:
    // "ragas not installed", "no error taxonomy is active for this Realm",
    // "control dataset not found". Throwing the status line alone turned all
    // of them into "503 Service Unavailable: /some/path", which tells the
    // reader that something is wrong and nothing about what to do, on a
    // platform whose subject is diagnosing exactly that.
    const detail = await res
      .json()
      .then(body => {
        const d = (body as { detail?: unknown }).detail
        if (typeof d === 'string') return d
        // A 422 from request validation carries a list of field errors.
        if (Array.isArray(d)) return d.map(e => e?.msg ?? JSON.stringify(e)).join('; ')
        return d ? JSON.stringify(d) : ''
      })
      .catch(() => '')
    throw new Error(detail || `${res.status} ${res.statusText}: ${path}`)
  }
  if (res.status === 204 || res.headers.get('content-length') === '0') return undefined as T
  return res.json() as Promise<T>
}

// ── Types ─────────────────────────────────────────────────────────────────────

export interface PromptTemplate {
  id: string
  name: string
  version: number
  description: string
  is_active: boolean
  created_at: string
  template: string
  realm_id?: string | null
}

export interface GenerationPreset {
  id: string
  name: string
  description: string
  template: string
  realm_id: string | null
  created_at?: string
}

export interface ExperimentItem {
  run_id: string
  name: string
  config_hash: string
  aggregate_metrics: Record<string, number>
  started_at: string
  finished_at: string
  n_questions: number
  dataset_name: string
  prompt_id?: string
  prompt_version?: number
  // the async job-model — present while run is in flight
  status?: 'running' | 'done'
  progress_processed?: number
  progress_total?: number
  realm_id?: string
  // True when a user stopped this run early — n_questions is already the
  // actual answered count in that case (see services/api_gateway/routers/
  // experiments.py#list_experiments), not the dataset's planned total.
  stopped?: boolean
  /** Which run is the baseline. Only `GET /experiments/{id}` used to answer
   *  this, so the list could neither mark the baseline nor compare against
   *  it. */
  is_baseline?: boolean
}

/** The response from both sides of the baseline mark: `null` means there is no
 *  baseline. */
export interface BaselineResult {
  baseline_run_id: string | null
  status: string
}

export interface MissDiagnosis {
  run_id: string
  question_id: string
  found: boolean
  rank: number | null
  score: number | null
  structural_path: string | null
  widened_k: number
}

export interface ExperimentDetail {
  // the async job-model: while the run is in flight the gateway returns
  // status="running" with progress = the last progress event (or null).
  status?: 'running' | 'done'
  progress?: { type: string; processed?: number; total?: number } | null
  config_hash: string
  config_name: string
  run_id: string
  aggregate_metrics: Record<string, number>
  question_results: QuestionResult[]
  // Count of question_results with `error` set — a run can now finish
  // "done" with some questions failed (e.g. an external RAG timing out on
  // a handful of questions) rather than the whole run aborting.
  n_errors?: number
  config?: {
    pipeline_id?: string
    chunking_strategy?: { component_id?: string }
    embedder?: { component_id?: string }
    generator?: { component_id?: string }
    reranker?: { component_id?: string } | null
    corpus_id?: string
    top_k?: number
    merge_strategy?: string
    dataset_name?: string
    seed?: number
    // external/injectable RAG
    pipeline_source?: 'in_process' | 'http'
    http_endpoint?: string | null
    external_rag_id?: string | null
    // Backfilled server-side from the resolved ExternalRag record — see
    // core/experiment/config.py#ExperimentConfig.external_rag_name.
    external_rag_name?: string | null
    // Extensible params (model override, temperature, etc).
    params?: Record<string, unknown> | null
    // Optional pipeline steps, omitted when unset.
    grounding?: { component_id?: string } | null
    route_policy?: { component_id?: string } | null
    scorer?: { component_id?: string } | null
    mask_engine?: { component_id?: string } | null
    refusal_policy?: { component_id?: string } | null
    // http-only, no-op for in_process.
    retrieval_only?: boolean
    // Candidate window, separate from the context size above.
    fetch_k?: number | null
  }
  dataset_name?: string
  n_questions?: number
  started_at?: string
  finished_at?: string
  // True when a user stopped this run early via api.experiments.stop —
  // question_results.length below n_questions is then expected, not a bug.
  stopped?: boolean
  prompt_id?: string
  prompt_version?: number
  // The actual model that generated answers for this run — captured from
  // Answer.metadata (in_process: core/pipeline.py, http: the external RAG's
  // response body). config.generator?.component_id is a fixed adapter-kind literal
  // ("ollama") and never reflects the real model name.
  generator_model?: string
  // backend-provided diagnostics & regression guard
  diagnostics?: DetectorItem[]
  // What this run made impossible to check. Beside the findings and not on
  // another tab: a check that could not run and a check that ran and found
  // nothing are the same absence on a screen unless one of them is named.
  trace_gaps?: { field: string; unavailable: string; remedy: string }[]
  diagnosis_depth?: string
  // How many questions each root cause accounts for, derived
  // server-side from the stored per-question verdicts. This is the number
  // that turns a list of failures into a comparison between kinds of work:
  // one verdict says what went wrong once, the counts say which fix pays
  // off most. Empty for a run with no retrieval failures.
  root_cause_counts?: Record<string, number>
  // Phase 2 — the ordered list of work, derived server-side from the same
  // per-question verdicts the counts come from. Each task is one
  // (cause, document) pair and carries how many questions it would close,
  // which is what makes tasks comparable at all — a per-question verdict
  // never can be. Ordered most-reaching first.
  fix_tasks?: {
    cause: string
    lever: string
    entity: string
    questions: number
    question_ids: string[]
  }[]
  // The share of failed questions that share a cause with at
  // least one other. Whether generalisation is worth starting hangs on this
  // number: near zero means generalisation has nothing to work with.
  clusterable_share?: number
  // Phase 3 — the smallest context size that would capture the whole
  // available payoff, computed from where expected sources actually sat in
  // the run's own ranked lists. Absent when nothing would be gained, so the
  // interface shows advice only where there is advice to give.
  context_size_advice?: {
    current_k: number
    recommended_k: number
    questions_gained: number
    // Questions no cut-off would have closed, because their expected source
    // never appeared at all. Stated so a reader does not take the rest for
    // a tuning problem.
    unreachable: number
  } | null
  is_baseline?: boolean
  regression?: RegressionResult
  // Optional deepeval LLM-judge report for this same run_id (eval/deepeval_runner.py).
  deepeval_report?: { metrics: Record<string, number>; timestamp: number } | null
  // Dataset-averaged per-stage latency (core/experiment/runner.py:avg_stage_trace),
  // for the run-level pipeline diagram. null when no question produced a trace.
  avg_stage_trace?: StageTrace | null
}

// One third-party tool, as GET /panels/status reports it.
export interface PanelStatus {
  id: string
  url: string
  reachable: boolean
  // False when the tool forbids being framed. That is the tool's own choice
  // (Langfuse sends `frame-ancestors 'none'`, Qdrant sends
  // `X-Frame-Options: DENY`) and not something the platform can work around.
  embeddable: boolean
  status: number | null
  blocked_by: string | null
}

export interface DetectorItem {
  // Which catalogue entries this finding is evidence for. Empty when no entry
  // names it, which means the platform is saying something the catalogue has no
  // place for.
  failure_ids?: string[]
  id: string
  severity: 'ok' | 'info' | 'warn' | 'error'
  title: string
  detail: string
  action?: string
}

// Phase 4 — the document an external system's owner works from
// (core/eval/prescription.py). Every field is derived from the run's own
// stored verdicts, so the document and the diagnostics tab cannot disagree
// about a single number.
export interface PrescribedFix {
  cause: string
  lever: string
  entity: string
  questions: number
  examples: string[]
  // Fixed when the prescription is written, which is what makes acceptance
  // judgeable at all: a criterion the recipient can still widen afterwards
  // is not a criterion.
  verification_question_ids: string[]
}

export interface Prescription {
  run_id: string
  dataset_name: string
  n_questions: number
  diagnosis_depth: string
  fixes: PrescribedFix[]
  // What the run could not establish, and what the owner would change to
  // make it establishable (core/eval/trace_completeness.py).
  gaps: { field: string; unavailable: string; remedy: string }[]
  context_size_advice: Record<string, number> | null
  metrics: Record<string, number>
  markdown: string
}

// One finished run reduced to the two numbers a choice is made on.
export interface ConfigPoint {
  run_id: string
  pipeline_source: string
  label: string
  quality: number
  latency_ms: number
  // The third axis, in tokens per question. null means the run
  // carried no counts at all; 0 means it carried them and they were zero,
  // which is a measurement and not a gap.
  tokens: number | null
  config: Record<string, unknown>
}

export interface FrontierResult {
  quality_metric: string
  considered: number
  // Always true, and named rather than implied: the platform's stage trace
  // measures the platform's own work, so for an external call it records the
  // handover and not the remote system's cost.
  latency_comparable_within_source_only: boolean
  // Per group, because the token axis takes part only where every run of that
  // group carries counts. A reader who believes three axes were compared when
  // two were would draw a conclusion the computation never supported.
  tokens_included: Record<string, boolean>
  frontier_by_source: Record<string, ConfigPoint[]>
  dominated: ConfigPoint[]
}

// Works 5.6 and 5.7 — whether the metric that picks a winner can be trusted.
export interface CalibrationResult {
  calibration: {
    metric: string
    threshold: number
    compared: number
    agree: number
    agreement: number
    false_good: string[]
    false_bad: string[]
  }
  position_bias: {
    by_position: Record<string, number>
    counts: Record<string, number>
    spread: number
  }
}

// Phase 7 — one query a real user asked a served system, pulled from its own
// trace endpoint. A candidate, never feedback: whether it was wrong is a
// human judgement nobody has made yet.
export interface ProductionTrace {
  trace_id: string
  realm_id: string
  corpus_id: string
  query: string
  sources: { chunk_id?: string; structural_path?: string }[]
  answer_preview: string
  created_at: string
  collected_at: string
  promoted_question_id: string | null
  // Whether the question this trace became still exists. The mark above
  // records history; this says whether that history still points at
  // anything, so a deleted question does not leave the trace permanently
  // unpromotable.
  promotion_live?: boolean
}

export interface CoverageResult {
  threshold: number
  n_production: number
  n_golden: number
  // Counts every uncovered question, while `uncovered` below is capped —
  // the two are deliberately not the same number.
  uncovered_share: number
  uncovered_total?: number
  uncovered: string[]
  deciles: number[]
}

export interface RegressionResult {
  passed: boolean
  violations: string[]
  baseline_run_id: string
  deltas: { metric: string; baseline: number; current: number; delta: number; regressed: boolean }[]
}

export interface SourceRefView {
  chunk_id: string
  structural_path: string
  score: number
  chunk_text: string
  // Set when this chunk was injected by a matching
  // retrieval_pins entry (core/pins/overlay.py) rather than found by
  // ordinary retrieval. Absent/false for every ordinarily-retrieved chunk.
  pinned?: boolean
}

export interface QuestionResult {
  question_id: string
  question: string
  reference_answer: string
  generated_answer: string
  // Whether this answer refuses, decided by the server so that one rule lives
  // in one place. Absent on a run read before the field existed; a count over
  // it then reports what it can see instead of falling back to a weaker rule
  // of its own, which is what the two sides used to disagree about.
  is_refusal?: boolean
  metrics: Record<string, number>
  source_refs?: SourceRefView[]
  // Ranked list BEFORE the reranker cut it (core/pipeline.py) — empty when
  // no reranker ran. Lets you see what reranking actually changed.
  pre_rerank_source_refs?: SourceRefView[]
  // core/citation.py:compute_citation_labels() — citation derived from
  // retrieved chunk metadata, not from whatever the LLM wrote in the
  // answer text (the LLM's own in-text citation can be wrong even when
  // retrieval found the right chunk — see GuidePage's common-problems section).
  computed_citations?: string[]
  // Eval Measurement Trustworthiness, Phase 1 — per-question layer
  // attribution (core/eval/funnel.py), computed server-side.
  funnel?: { layer: string; detail: string }
  // From core/eval/root_cause.py: why the `retrieval` layer
  // failed, not just that it did, plus the lever that cause points at.
  // Present only for a question whose funnel verdict was `retrieval`, and
  // absent on runs stored before the analysis existed. `evidence` carries
  // what the verdict was derived from (an absent ref, a rank against the
  // run's own top_k), so a reader can disagree without re-running anything.
  root_cause?: {
    cause: 'data_missing' | 'ranking' | 'chunking' | 'not_retrievable' | 'unknown'
    detail: string
    lever: 'ingest' | 'ranking' | 'chunking' | 'vocabulary' | 'chunking_or_vocabulary' | 'verify_index'
    evidence: Record<string, unknown>
  } | null
  // Per-stage latency/counters for this question's pipeline run
  // (core/models.py:StageTrace). null when the pipeline produced no trace
  // (e.g. an external RAG that returns no ExternalTrace.stage_trace).
  stage_trace?: StageTrace | null
  // Set when this question's pipeline call raised (e.g. an external RAG
  // timing out) — the run continues past it instead of aborting entirely
  // (core/experiment/runner.py). generated_answer/metrics stay empty for
  // this row when set.
  error?: string | null
}

// per-question before/after diff (core/eval/regression.py
// :paired_diff), alongside the aggregate-only metric_deltas above: an
// aggregate delta can stay within threshold while some questions improved
// and others regressed, which is exactly what this catches by name.
export interface PairedDiffResult {
  fixed: string[]
  flips: string[]
  unchanged: string[]
  metric_deltas: Record<string, Record<string, number>>
  // Assembled server-side only for ids in fixed/flips (unchanged can be
  // dozens of questions and only needs its count) — a bare question_id
  // like "19349990" says nothing on its own; this is what lets the UI show
  // the actual question text and which funnel layer it moved between.
  questions: Record<string, { question: string; funnel_before: string; funnel_after: string }>
  // Resample-based flip confirmation. A flip
  // `confirm_flips` reclassifies as noise is already moved out of `flips`
  // into `unchanged` server-side; these two fields are transparency only,
  // so the UI can tell "no flips" apart from "resampling quietly absorbed
  // some" instead of both rendering identically.
  resample_attempted?: boolean
  noise_filtered?: string[]
  // The questions that exist on one side only. Pairing skips them, which is
  // right, and reporting nothing about them was not: with zero overlap every
  // group came back empty and read as "the change moved no question".
  // Optional so a stored response from before this field still types.
  only_in_before?: string[]
  only_in_after?: string[]
}

/** One reason the pair may not be measuring the same thing. Same shape as
 *  DetectorItem plus `params`: the UI renders its own sentence from `id` and
 *  `params`, and falls back to the server's English `title` for an id it does
 *  not know yet. */
export interface CompatWarning {
  id: string
  severity: 'ok' | 'info' | 'warn' | 'error'
  title: string
  detail: string
  params?: Record<string, unknown>
  action?: string
}

export interface RunFacts {
  run_id: string
  dataset_name: string
  /** How many questions the run answered, which a stopped run leaves below
   *  the dataset's planned total. */
  n_questions: number
  realm_id: string
  corpus_id: string
  stopped: boolean
}

export interface Comparability {
  comparable: boolean
  matched: number
  only_in_before: number
  only_in_after: number
  before: RunFacts
  after: RunFacts
  warnings: CompatWarning[]
}

export interface CompareResult {
  config_diff: Record<string, { before: unknown; after: unknown }>
  // null where the metric was never measured on that side, which is a
  // different statement from measuring zero.
  metric_deltas: { metric: string; before: number | null; after: number | null; delta: number | null; delta_pct: number | null }[]
  summary: string
  paired_diff: PairedDiffResult
  compatibility?: Comparability
}

export interface RegistryEntry {
  [kind: string]: string[]
}

export interface PipelineDescription {
  retriever: string
  /** Empty for a retriever that merges nothing, which is not the same as
   *  fusing by rank and must not be recorded as it. */
  merge_strategy: string
}

export interface Dataset {
  name: string
  version: string
  speed: string
  filename: string
  // Present once a dataset was registered via POST /datasets
  // (platform- or RAG-authored, same collection now); absent for datasets
  // that still only exist as eval/golden/*.jsonl files on disk.
  id?: string
  source_rag_id?: string | null
  count?: number | null
}

export interface QuestionProvenance {
  origin: 'manual' | 'generated' | 'reviewer_feedback'
  created_at?: string
  updated_at?: string
  edited_manually?: boolean
  // generated-only
  model?: string
  corpus_id?: string
  chunk_id?: string | string[]
  preset_id?: string
  generated_at?: string
  // reviewer_feedback-only — backlinks to the run/
  // question/feedback this golden question was promoted from, so
  // DatasetsPage can link back instead of collapsing it into "manual".
  source_run_id?: string
  source_question_id?: string
  source_feedback_id?: string | null
  // Set when promoting overwrote an existing question (matched by exact
  // text) that already had some other origin — preserves that origin
  // instead of silently discarding it, so "this was auto-generated, then
  // confirmed by a reviewer" stays visible.
  previous_origin?: string
}

// Domain-specific extra fields (e.g. a future non-legal corpus's own
// convention) are allowed server-side (extra="allow", see
// datasets.py#QuestionWriteRequest) but deliberately untyped here — the UI
// only ever reads/writes the fields below, so an index signature isn't
// worth the type-widening friction it causes on Omit/intersection types.
export interface Question {
  id: string
  question: string
  reference_answer: string
  // Datasets authored before this write path existed (e.g. handbook.v0.fast)
  // use this older field name instead — read-only tolerance, mirroring
  // core/experiment/runner.py's own `reference_answer or ground_truth`
  // fallback. Never written by this client.
  ground_truth?: string
  question_type?: string
  article_refs?: string[]
  provenance?: QuestionProvenance
}

// One question's answer text, tolerant of the field-name split above.
export function referenceAnswerOf(q: Partial<Question> | undefined): string {
  return q?.reference_answer ?? q?.ground_truth ?? ''
}

export interface DatasetDetail extends Dataset {
  realm_id?: string | null
  questions: Question[]
}

// id is server-generated — never sent by the client (see datasets.py#QuestionWriteRequest).
export interface QuestionWrite {
  question: string
  reference_answer: string
  question_type?: string
  article_refs?: string[]
  provenance?: QuestionProvenance
}

// Human feedback on one answer — stored separately from the run itself (see
// feedback.py's module docstring). `scores` is an open dict (not fixed
// fields) so a new rating dimension can be added without a schema change;
// the UI renders accuracy/completeness/relevance by default.
export interface Feedback {
  id: string
  run_id: string
  question_id: string
  realm_id?: string | null
  rating?: 'good' | 'bad' | null
  scores: Record<string, number>
  comment?: string | null
  reviewer?: string | null
  created_at: string
  updated_at: string
  // Set once /feedback/triage has run at least once
  // for this question; absent on every question that hasn't been triaged.
  triage_result?: TriageResult
  triage_status?: 'proposed' | 'confirmed' | 'edited' | 'rejected'
  triage_created_at?: string
  triage_decided_at?: string
}

// The structured outcome of classifying one
// feedback comment (core/eval/triage.py#TriageResult.to_dict()).
export interface TriageResult {
  error_classes: string[]
  target_spans: string[]
  expected_refs: string[]
  severity: 'low' | 'medium' | 'high' | 'unknown'
  confidence: number
  verified_refs: Record<string, boolean>
  funnel_layer: string | null
  diagnosis_confirmed: boolean
  needs_manual_review: boolean
  parse_error: string | null
  proposal: {
    primary_lever: string | null
    alternative_levers: string[]
    justification: string
  }
}

// One relevance
// judgment, as returned by services/api_gateway/routers/judgments.py.
//
// A judgment records an observation ("for this question, this chunk is
// relevant, that one is not"), never an instruction. There is deliberately
// no threshold and no similarity vector here: deciding which *other*
// questions a statement should also apply to was the part of the retired
// retrieval-pin mechanism the objective review rejected.
export interface JudgedChunk {
  chunk_id: string
  // Names the source unit rather than a position in the index, so it
  // survives re-indexing. Empty when the index could not resolve the chunk,
  // which is itself worth showing: without it the judgment cannot become a
  // golden question.
  ref_id: string
  doc_id: string
  structural_path: string
  text: string
}

export interface Judgment {
  id: string
  realm_id: string
  corpus_id: string
  question: string
  relevant: JudgedChunk[]
  irrelevant: JudgedChunk[]
  note: string
  author: string
  created_at: string
  status: 'active' | 'retired'
  source_run_id: string
  source_feedback_id: string
  source_question_id: string
  // Server-computed, so the page can show what this judgment is worth
  // without re-deriving the rules: how many training pairs it yields, and
  // whether it can become a test at all.
  preference_pairs: number
  can_become_test: boolean
}

export interface JudgmentWrite {
  realm_id: string
  corpus_id: string
  question: string
  relevant?: string[]
  irrelevant?: string[]
  note?: string
  author?: string
  source_run_id?: string
  source_feedback_id?: string
  source_question_id?: string
}

// A PUT only ever carries the field(s) actually being changed — the
// backend merges into any existing doc, never replaces it wholesale.
export interface FeedbackWrite {
  rating?: 'good' | 'bad'
  scores?: Record<string, number>
  comment?: string
  reviewer?: string
}

export interface RealmRecord {
  id: string
  name: string
  description?: string
  // `proving_ground` marks a realm whose data carries defects on purpose. The
  // interface says so where the realm is named, because red diagnostics there
  // are the expected outcome and read as a broken installation otherwise.
  purpose?: string
  resources: { type: string; [k: string]: unknown }[]
  key_metrics?: string[]
  created_at: string
  deleted_at?: string | null
}

export interface ResourceTestResult {
  status: 'ok' | 'error'
  type: string
  detail?: string
  [k: string]: unknown
}

/** A realm export. The corpus travels as references only: its contents live in
 *  Qdrant and OpenSearch and do not fit in a file, so on the receiving side
 *  this is a list of what has to be ingested again. */
export interface RealmBundle {
  format: string
  exported_at?: string
  realm: RealmRecord
  external_rags: Record<string, unknown>[]
  prompts: Record<string, unknown>[]
  generation_presets: Record<string, unknown>[]
  datasets: Record<string, unknown>[]
  settings: Record<string, unknown>
  corpora: { corpus_id: string; description?: string }[]
  /** Resource fields whose values are masked and have to be entered again. */
  masked_fields: string[]
}

export interface RealmImportReport {
  dry_run: boolean
  realm_id: string
  /** One entry per kind of thing: how many will appear, how many matched. */
  entries: { kind: string; created: number; conflicted: number; skipped: number }[]
  masked_fields: string[]
  warnings: string[]
}

// ── API calls ─────────────────────────────────────────────────────────────────

// ── The failure catalogue ─────────────────────────────────────────────────────
//
// Read-only by design. An entry claims a failure is detected, and the build
// refuses such a claim without a bait; let this side write entries and that
// refusal stops holding.

export interface AtlasSignalRef {
  id: string
  kind: string
  name: string
  side: 'core' | 'ui'
}

export interface AtlasScope {
  code: string
  values: string[]
  dimension: string | null
  url: string
}

export interface AtlasEntry {
  id: string
  title: string
  title_key: string
  atlas_rows: number[]
  stage_origin: string
  stage_visible: string
  severity: { quiet: number; cost: number; prevalence: number; total: number }
  origin: string
  detection: 'detector' | 'visible' | 'none'
  // Decided by the catalogue, never re-derived here: the rule had two copies
  // in two languages, and nothing held them to one answer.
  state: 'caught' | 'visible' | 'unproven' | 'none'
  instrument: string
  signals: AtlasSignalRef[]
  applies_when: AtlasScope[]
  // Three answers, never two. `null` says this architecture records nothing
  // about a coordinate the entry depends on, so the failure is neither possible
  // nor ruled out, and showing it as excluded would state something nobody
  // established.
  applies_here: boolean | null
  // Every architecture this entry can occur in, so a link to it can be
  // honoured instead of guessed at.
  applies_to_points: string[]
  bait: string
  bait_level: 'unit' | 'proving_ground' | null
  not_detected_reason: string
  scope_caveat: string
  shares_signals_with: string[]
  superseded_by: string[]
}

export interface AtlasPoint {
  coordinates: { code: string; value: string; url: string }[]
  applicable: number
}

export interface Atlas {
  entries: AtlasEntry[]
  schema: { source: string; release: string; dimensions: number }
  points: Record<string, AtlasPoint>
  point?: string
  uncovered_coordinates?: { code: string; value: string; dimension: string | null; url: string }[]
}

export interface AtlasSignal {
  id: string
  side: 'core' | 'ui'
  kind: string
  failures: string[]
  singles_out: boolean
  bait_level: string[]
}

export const api = {
  atlas: {
    read: (point?: string) =>
      req<Atlas>(`/atlas${point ? `?point=${encodeURIComponent(point)}` : ''}`),
    signals: () => req<{ signals: AtlasSignal[] }>('/atlas/signals'),
  },
  experiments: {
    list: (params?: { sort_by?: string; dataset?: string; realmId?: string | null }) => {
      const qs = new URLSearchParams()
      if (params?.dataset) qs.set('dataset', params.dataset)
      if (params?.realmId) qs.set('realm_id', params.realmId)
      const q = qs.toString()
      return req<ExperimentItem[]>('/experiments' + (q ? `?${q}` : ''))
    },
    get: (id: string) => req<ExperimentDetail>(`/experiments/${id}`),
    // `progressId` is the client's own, and it has to exist before the request
    // is sent, because the socket it names is what reports the comparison
    // while the request is still open (see ComparisonPage).
    // Whether a pair can be compared at all, asked before comparing. Same
    // rules as the report itself, so the picker and the report cannot
    // disagree about one pair.
    comparePreflight: (a: string, b: string) =>
      req<Comparability>(
        `/experiments/compare/preflight?a=${encodeURIComponent(a)}&b=${encodeURIComponent(b)}`,
      ),
    compare: (ids: string[], progressId?: string) =>
      req<CompareResult>('/experiments/compare', {
        method: 'POST', body: JSON.stringify({ ids, progress_id: progressId }),
      }),
    delete: (id: string) => req<void>(`/experiments/${id}`, { method: 'DELETE' }),
    unsetBaseline: () => req<BaselineResult>(`/experiments/baseline`, { method: 'DELETE' }),
    setBaseline: (id: string) =>
      req<BaselineResult>(`/experiments/${id}/baseline`, { method: 'PUT' }),
    // Returns immediately with a job handle; the run
    // itself happens in the background, so no `metrics` here anymore. Poll
    // `get(run_id)` (status: "running" | "done") or watch the WS progress
    // stream for completion.
    create: (config: Record<string, unknown>, dataset_name: string, realm_id?: string | null) =>
      req<{ run_id: string; config_hash: string; status: string }>(
        '/experiments',
        { method: 'POST', body: JSON.stringify({ config, dataset_name, realm_id: realm_id ?? '' }) },
      ),
    diagnoseMiss: (runId: string, questionId: string, widenedK = 50) =>
      req<MissDiagnosis>(
        `/experiments/${runId}/questions/${questionId}/diagnose-miss?widened_k=${widenedK}`,
        { method: 'POST' },
      ),
    // Found live: a run picking a slow model (or hitting a stuck external
    // RAG) had no way to be interrupted short of waiting out every
    // remaining question. Cooperative — already-in-flight question still
    // finishes, already-answered ones are kept (see ExperimentResult.stopped).
    stop: (runId: string) =>
      req<{ run_id: string; status: string }>(`/experiments/${runId}/stop`, { method: 'POST' }),
    // Phase 4 — the document the owner of the system under test works from.
    // Read-only and derived: asking for it changes nothing about the run.
    prescription: (runId: string) => req<Prescription>(`/experiments/${runId}/prescription`),
    // Whether a prescription was actually carried out, judged over
    // the question ids the prescription itself named.
    acceptance: (body: { before_run_id: string; after_run_id: string; question_ids: string[] }) =>
      req<{
        accepted: boolean
        fixed: string[]
        still_failing: string[]
        regressed: string[]
        unchecked: string[]
        summary: string
      }>('/experiments/acceptance', { method: 'POST', body: JSON.stringify(body) }),
    // Built over runs that already happened, so this starts
    // nothing and costs one read.
    frontier: (realmId: string, qualityMetric?: string) =>
      req<FrontierResult>(
        `/experiments/frontier/${encodeURIComponent(realmId)}${qualityMetric ? `?quality_metric=${encodeURIComponent(qualityMetric)}` : ''}`,
      ),
    calibration: (runId: string, metric?: string, threshold?: number) => {
      const qs = new URLSearchParams()
      if (metric) qs.set('metric', metric)
      if (threshold != null) qs.set('threshold', String(threshold))
      const q = qs.toString()
      return req<CalibrationResult>(`/experiments/${runId}/calibration${q ? `?${q}` : ''}`)
    },
  },
  // Phase 7 — production traffic. Pull, never push: the platform reaches out
  // to a served system's own trace endpoint, so a system that has never heard
  // of the platform is unaffected by its existence.
  production: {
    collect: (body: { realm_id: string; corpus_id: string; url: string; limit?: number }) =>
      req<{ collected: number; stored: number; recording_enabled: boolean }>(
        '/production/collect', { method: 'POST', body: JSON.stringify(body) },
      ),
    list: (realmId: string, corpusId: string, limit?: number) =>
      req<ProductionTrace[]>(
        `/production?realm_id=${encodeURIComponent(realmId)}&corpus_id=${encodeURIComponent(corpusId)}${limit ? `&limit=${limit}` : ''}`,
      ),
    // The question text is the user's own wording, unedited; the reference
    // answer is required and never taken from the trace, because what the
    // system answered is precisely what is being disputed.
    promote: (body: {
      trace_id: string; dataset_id: string; reference_answer: string
      realm_id: string; article_refs?: string[]
    }) =>
      req<{ question_id: string; dataset: string; count: number }>(
        '/production/promote', { method: 'POST', body: JSON.stringify(body) },
      ),
    coverage: (realmId: string, corpusId: string, datasetName: string, threshold?: number) =>
      req<CoverageResult>(
        `/production/coverage?realm_id=${encodeURIComponent(realmId)}&corpus_id=${encodeURIComponent(corpusId)}`
        + `&dataset_name=${encodeURIComponent(datasetName)}${threshold != null ? `&threshold=${threshold}` : ''}`,
      ),
  },
  // One registry per process while pack activation is per realm: without a
  // realm_id the form offers another realm's packs' internals.
  registry: (realmId?: string | null) =>
    req<RegistryEntry>(`/registry${realmId ? `?realm_id=${encodeURIComponent(realmId)}` : ''}`),
  // What each pipeline is made of, read off the objects the server holds.
  // The new-run form used to derive this from a list of three known names.
  pipelines: () => req<Record<string, PipelineDescription>>('/pipelines'),
  datasets: {
    list: (realmId?: string | null, sourceRagId?: string | null) => {
      const qs = new URLSearchParams()
      if (realmId) qs.set('realm_id', realmId)
      if (sourceRagId) qs.set('source_rag_id', sourceRagId)
      const q = qs.toString()
      return req<Dataset[]>(`/datasets${q ? `?${q}` : ''}`)
    },
    get: (filename: string) => req<DatasetDetail>(`/datasets/${filename}`),
    // The first live write path this collection has ever had;
    // usable identically by the platform's own upload form or an external
    // RAG's own ingest pipeline (source_rag_id is provenance only).
    create: (body: { filename: string; realm_id: string; questions: Record<string, unknown>[]; source_rag_id?: string }) =>
      req<Dataset>('/datasets', { method: 'POST', body: JSON.stringify(body) }),
    delete: (datasetId: string, realmId?: string | null) =>
      req<void>(`/datasets/by-id/${datasetId}${realmId ? `?realm_id=${encodeURIComponent(realmId)}` : ''}`, { method: 'DELETE' }),
    // Per-question CRUD — id is always server-generated (see QuestionWrite).
    addQuestion: (datasetId: string, question: QuestionWrite, realmId?: string | null) =>
      req<DatasetDetail>(
        `/datasets/${datasetId}/questions${realmId ? `?realm_id=${encodeURIComponent(realmId)}` : ''}`,
        { method: 'POST', body: JSON.stringify(question) },
      ),
    addQuestionsBatch: (datasetId: string, questions: QuestionWrite[], realmId?: string | null) =>
      req<DatasetDetail>(
        `/datasets/${datasetId}/questions/batch${realmId ? `?realm_id=${encodeURIComponent(realmId)}` : ''}`,
        { method: 'POST', body: JSON.stringify(questions) },
      ),
    updateQuestion: (datasetId: string, questionId: string, question: QuestionWrite, realmId?: string | null) =>
      req<DatasetDetail>(
        `/datasets/${datasetId}/questions/${questionId}${realmId ? `?realm_id=${encodeURIComponent(realmId)}` : ''}`,
        { method: 'PUT', body: JSON.stringify(question) },
      ),
    deleteQuestion: (datasetId: string, questionId: string, realmId?: string | null) =>
      req<DatasetDetail>(
        `/datasets/${datasetId}/questions/${questionId}${realmId ? `?realm_id=${encodeURIComponent(realmId)}` : ''}`,
        { method: 'DELETE' },
      ),
  },
  feedback: {
    // All feedback for one run, keyed by question_id — one call for the
    // whole page (RunPage) instead of one per question row.
    get: (runId: string, realmId?: string | null) =>
      req<Record<string, Feedback>>(
        `/experiments/${runId}/feedback${realmId ? `?realm_id=${encodeURIComponent(realmId)}` : ''}`,
      ),
    upsert: (runId: string, questionId: string, body: FeedbackWrite, realmId?: string | null) =>
      req<Feedback>(
        `/experiments/${runId}/questions/${questionId}/feedback${realmId ? `?realm_id=${encodeURIComponent(realmId)}` : ''}`,
        { method: 'PUT', body: JSON.stringify(body) },
      ),
    delete: (runId: string, questionId: string, realmId?: string | null) =>
      req<void>(
        `/experiments/${runId}/questions/${questionId}/feedback${realmId ? `?realm_id=${encodeURIComponent(realmId)}` : ''}`,
        { method: 'DELETE' },
      ),
    // Turn a reviewed question into a golden-dataset
    // entry (provenance.origin="reviewer_feedback"). article_refs/
    // reference_answer, when omitted, are copied server-side from the
    // run's own source dataset/answer unchanged (no LLM regeneration here).
    // `created: false` in the response means an existing question with the
    // same text was updated in place instead of duplicated.
    promote: (
      runId: string, questionId: string,
      body: { target_dataset_id: string; article_refs?: string[]; reference_answer?: string },
      realmId?: string | null,
    ) =>
      req<DatasetDetail & { created: boolean }>(
        `/experiments/${runId}/questions/${questionId}/feedback/promote${realmId ? `?realm_id=${encodeURIComponent(realmId)}` : ''}`,
        { method: 'POST', body: JSON.stringify(body) },
      ),
    // Classifies the question's CURRENT stored
    // comment via the Realm's own local generator; read-only recommendation,
    // never creates a pin or changes the run itself (see triageDecide below).
    triage: (runId: string, questionId: string, realmId?: string | null) =>
      req<Feedback>(
        `/experiments/${runId}/questions/${questionId}/feedback/triage${realmId ? `?realm_id=${encodeURIComponent(realmId)}` : ''}`,
        { method: 'POST' },
      ),
    triageDecide: (
      runId: string, questionId: string,
      body: { action: 'confirm' | 'reject' | 'edit'; edited_result?: Partial<TriageResult> },
      realmId?: string | null,
    ) =>
      req<Feedback>(
        `/experiments/${runId}/questions/${questionId}/feedback/triage/decide${realmId ? `?realm_id=${encodeURIComponent(realmId)}` : ''}`,
        { method: 'POST', body: JSON.stringify(body) },
      ),
    // Cross-run analysis query — for developers/agents looking for patterns
    // across answers, not used by RunPage itself.
    list: (params?: { realmId?: string | null; runId?: string; rating?: 'good' | 'bad'; limit?: number }) => {
      const qs = new URLSearchParams()
      if (params?.realmId) qs.set('realm_id', params.realmId)
      if (params?.runId) qs.set('run_id', params.runId)
      if (params?.rating) qs.set('rating', params.rating)
      if (params?.limit) qs.set('limit', String(params.limit))
      const q = qs.toString()
      return req<(Feedback & { question: string | null; generated_answer: string | null; run_config_name: string | null })[]>(
        `/feedback${q ? `?${q}` : ''}`,
      )
    },
  },
  // Relevance judgments. Stored as a versioned file per Realm
  // and corpus, not in the platform's database, so realm_id/corpus_id are
  // required on every call rather than optional filters: there is no
  // meaningful unscoped listing to ask for.
  judgments: {
    list: (realmId: string, corpusId: string) =>
      req<Judgment[]>(`/judgments?realm_id=${encodeURIComponent(realmId)}&corpus_id=${encodeURIComponent(corpusId)}`),
    get: (id: string, realmId: string, corpusId: string) =>
      req<Judgment>(`/judgments/${id}?realm_id=${encodeURIComponent(realmId)}&corpus_id=${encodeURIComponent(corpusId)}`),
    create: (body: JudgmentWrite) =>
      req<Judgment>('/judgments', { method: 'POST', body: JSON.stringify(body) }),
    update: (id: string, realmId: string, corpusId: string, body: { status?: string; note?: string }) =>
      req<Judgment>(
        `/judgments/${id}?realm_id=${encodeURIComponent(realmId)}&corpus_id=${encodeURIComponent(corpusId)}`,
        { method: 'PATCH', body: JSON.stringify(body) },
      ),
    delete: (id: string, realmId: string, corpusId: string) =>
      req<void>(
        `/judgments/${id}?realm_id=${encodeURIComponent(realmId)}&corpus_id=${encodeURIComponent(corpusId)}`,
        { method: 'DELETE' },
      ),
    // Phase 6 — the versioned artefact a served system loads at startup.
    // Published rather than applied: the platform never reaches into a
    // served system, so a correction travels as a document.
    bundle: (realmId: string, corpusId: string) =>
      req<{
        format_version: number
        realm_id: string
        corpus_id: string
        created_at: string
        n_entries: number
        calibration: { must_match: string[]; must_not_match_pairs: { a: string; b: string }[]; procedure: string }
        entries: unknown[]
      }>(`/judgments/bundle?realm_id=${encodeURIComponent(realmId)}&corpus_id=${encodeURIComponent(corpusId)}`),
    // The first and most valuable use of a judgment: it becomes a permanent
    // test, so losing the chunk again is caught by the next run rather than
    // by the next complaint.
    toGoldenQuestion: (
      id: string, realmId: string, corpusId: string,
      body: { dataset_id: string; reference_answer: string },
    ) => req<unknown>(
      `/judgments/${id}/golden-question?realm_id=${encodeURIComponent(realmId)}&corpus_id=${encodeURIComponent(corpusId)}`,
      { method: 'POST', body: JSON.stringify(body) },
    ),
  },
  prompts: {
    list: (realmId?: string | null) => req<PromptTemplate[]>(`/prompts${realmId ? `?realm_id=${encodeURIComponent(realmId)}` : ''}`),
    get: (id: string) => req<PromptTemplate>(`/prompts/${id}`),
    create: (body: { name: string; description: string; template: string; realm_id?: string | null }) =>
      req<PromptTemplate>('/prompts', { method: 'POST', body: JSON.stringify(body) }),
    activate: (id: string) => req<PromptTemplate>(`/prompts/${id}/activate`, { method: 'PUT' }),
    delete: (id: string) => req<void>(`/prompts/${id}`, { method: 'DELETE' }),
    // AI-drafted prompt (pre-fills the new-prompt form) — samples real
    // corpus chunks and asks the chosen model, never saves anything itself.
    generate: (body: { realm_id: string; corpus_id: string; model: string }) =>
      req<{ name: string; description: string; template: string }>(
        '/prompts/generate', { method: 'POST', body: JSON.stringify(body) },
      ),
  },
  panels: () => req<Record<string, string>>('/panels'),
  // Reachability and framing permission, answered by the server because
  // neither can be read from the browser: a cross-origin response is opaque
  // to JavaScript, headers included.
  panelsStatus: () => req<PanelStatus[]>('/panels/status'),

  // Realms. The realm and resource pages used to reach these with a bare
  // `fetch`, bypassing the client, which left them without shared error
  // handling and without one place showing which routes exist at all. New calls
  // (resource check, export, import) are declared here; the older ones stay
  // where they are until their own rework.
  realms: {
    list: (includeDeleted?: boolean) =>
      req<RealmRecord[]>(`/realms${includeDeleted ? '?include_deleted=true' : ''}`),
    testResource: (realmId: string, type: string) =>
      req<ResourceTestResult>(`/realms/${realmId}/resources/test`, {
        method: 'POST', body: JSON.stringify({ type }),
      }),
    exportRealm: (realmId: string, includeSecrets = false) =>
      req<RealmBundle>(`/realms/${realmId}/export${includeSecrets ? '?include_secrets=true' : ''}`),
    importRealm: (bundle: RealmBundle, opts: { onConflict?: 'fail' | 'rename'; dryRun?: boolean } = {}) => {
      const qs = new URLSearchParams()
      if (opts.onConflict) qs.set('on_conflict', opts.onConflict)
      if (opts.dryRun) qs.set('dry_run', 'true')
      const suffix = qs.toString() ? `?${qs}` : ''
      return req<RealmImportReport>(`/realms/import${suffix}`, {
        method: 'POST', body: JSON.stringify(bundle),
      })
    },
  },
  health: () => req<{ status: string; components: RegistryEntry }>('/health'),
  models: (completionOnly?: boolean) =>
    req<{ name: string; size_gb: number; modified_at: string; capabilities: string[] }[]>(
      `/models${completionOnly ? '?completion_only=true' : ''}`,
    ),
  generationPresets: {
    list: (realmId?: string | null) =>
      req<GenerationPreset[]>(
        `/generation-presets${realmId ? `?realm_id=${encodeURIComponent(realmId)}` : ''}`,
      ),
    create: (body: { name: string; description?: string; template: string; realm_id?: string | null }) =>
      req<GenerationPreset>(
        '/generation-presets', { method: 'POST', body: JSON.stringify(body) },
      ),
    update: (presetId: string, body: { name: string; description?: string; template: string }, realmId?: string | null) =>
      req<GenerationPreset>(
        `/generation-presets/${presetId}${realmId ? `?realm_id=${encodeURIComponent(realmId)}` : ''}`,
        { method: 'PUT', body: JSON.stringify(body) },
      ),
    delete: (presetId: string, realmId?: string | null) =>
      req<void>(`/generation-presets/${presetId}${realmId ? `?realm_id=${encodeURIComponent(realmId)}` : ''}`, { method: 'DELETE' }),
  },
  generate: {
    // Nothing is persisted by this call — drafts are reviewed/edited in the
    // same UI as manually-added questions and only saved via
    // datasets.addQuestion/addQuestionsBatch (see generation.py's module
    // docstring for why).
    //
    // A batch is one blocking LLM call per sampled chunk group and can run
    // minutes, so this only starts the job and returns a job_id — the
    // frontend follows up on the paired WebSocket (see wsBase() below +
    // generation.py#generate_questions_progress) for live progress and the
    // final drafts/failed payload, instead of blocking on one long request.
    //
    // preset_id is optional — omitted or undefined falls back to a
    // built-in, domain-neutral template server-side (generation.py's
    // _BUILTIN_TEMPLATES), so a Realm with no presets set up yet can still
    // generate.
    questions: (body: {
      realm_id: string; corpus_id: string; model: string; preset_id?: string
      type_counts: { question_type: string; n_questions: number }[]
    }) =>
      req<{ job_id: string; status: string; n_groups: number }>(
        '/generate/questions', { method: 'POST', body: JSON.stringify(body) },
      ),
  },
  settings: {
    get: (realmId?: string | null) =>
      req<{ active_model: string; embedder_mode: string; retriever_mode: string; active_packs: string[] }>(
        `/settings${realmId ? `?realm_id=${encodeURIComponent(realmId)}` : ''}`,
      ),
    setModel: (model: string, realmId?: string | null) =>
      req<{ active_model: string }>(
        `/settings/model${realmId ? `?realm_id=${encodeURIComponent(realmId)}` : ''}`,
        { method: 'PUT', body: JSON.stringify({ model }) },
      ),
  },
  corpus: {
    list: (realmId?: string | null) => req<CorpusIngest[]>(`/corpus${realmId ? `?realm_id=${encodeURIComponent(realmId)}` : ''}`),
    // The `corpora` registry (services/api_gateway/routers/
    // corpus.py#_list_corpora), NOT ingest-job history above: this is what
    // actually knows about corpora ingested outside the upload form (CLI,
    // the migration script) and is what a corpus_id picker should read from.
    collections: (realmId?: string | null, includeDeleted?: boolean) => {
      const qs = new URLSearchParams()
      if (realmId) qs.set('realm_id', realmId)
      if (includeDeleted) qs.set('include_deleted', 'true')
      const suffix = qs.toString()
      return req<CorpusRegistryEntry[]>(`/corpus/collections${suffix ? `?${suffix}` : ''}`)
    },
    // Corpus registry CRUD (was create+read only) — mirrors realms.py's
    // soft-delete-by-default pattern (corpus.py#delete_collection).
    updateCollection: (corpusRegistryId: string, body: { description: string }) =>
      req<CorpusRegistryEntry>(`/corpus/collections/${corpusRegistryId}`, { method: 'PUT', body: JSON.stringify(body) }),
    deleteCollection: (corpusRegistryId: string, hard?: boolean) =>
      req<void>(`/corpus/collections/${corpusRegistryId}${hard ? '?hard=true' : ''}`, { method: 'DELETE' }),
    restoreCollection: (corpusRegistryId: string) =>
      req<CorpusRegistryEntry>(`/corpus/collections/${corpusRegistryId}/restore`, { method: 'POST' }),
    delete: (job_id: string) => req<void>(`/corpus/${job_id}`, { method: 'DELETE' }),
    chunks: (corpusId: string, params: { offset?: string; limit?: number; q?: string; realmId?: string | null } = {}) => {
      const qs = new URLSearchParams()
      if (params.offset) qs.set('offset', params.offset)
      if (params.limit) qs.set('limit', String(params.limit))
      if (params.q) qs.set('q', params.q)
      if (params.realmId) qs.set('realm_id', params.realmId)
      return req<CorpusChunksResult>(`/corpus/${corpusId}/chunks?${qs.toString()}`)
    },
    health: (corpusId: string, realmId?: string | null) =>
      req<CorpusHealthResult>(`/corpus/${corpusId}/health${realmId ? `?realm_id=${encodeURIComponent(realmId)}` : ''}`),
    graphCommunities: (corpusId: string, algorithm: 'leiden' | 'louvain' = 'leiden', edgeType: 'lexical' | 'semantic' = 'lexical', realmId?: string | null) =>
      req<GraphCommunitiesResult>(`/corpus/${corpusId}/graph/communities?algorithm=${algorithm}&edge_type=${edgeType}${realmId ? `&realm_id=${encodeURIComponent(realmId)}` : ''}`),
    graphCommunityDetail: (corpusId: string, communityId: number, algorithm: 'leiden' | 'louvain' = 'leiden', edgeType: 'lexical' | 'semantic' = 'lexical', realmId?: string | null) =>
      req<GraphCommunityDetail>(`/corpus/${corpusId}/graph/communities/${communityId}?algorithm=${algorithm}&edge_type=${edgeType}${realmId ? `&realm_id=${encodeURIComponent(realmId)}` : ''}`),
    diagnosticsRagas: (corpusId: string, params: DiagnosticsRunParams = {}) =>
      req<DiagnosticsMetricsResult>(`/corpus/${corpusId}/diagnostics/ragas?${diagnosticsQuery(params)}`, { method: 'POST' }),
    diagnosticsTrulens: (corpusId: string, params: DiagnosticsRunParams = {}) =>
      req<DiagnosticsMetricsResult>(`/corpus/${corpusId}/diagnostics/trulens?${diagnosticsQuery(params)}`, { method: 'POST' }),
    diagnosticsChunkCoherence: (corpusId: string, params: { sample_size?: number; realm_id?: string | null } = {}) => {
      const qs = new URLSearchParams()
      if (params.sample_size) qs.set('sample_size', String(params.sample_size))
      if (params.realm_id) qs.set('realm_id', params.realm_id)
      return req<DiagnosticsMetricsResult>(`/corpus/${corpusId}/diagnostics/chunk-coherence?${qs.toString()}`, { method: 'POST' })
    },
  },
  trace: {
    latest: () => req<PipelineTrace>('/trace/latest'),
    get: (trace_id: string) => req<PipelineTrace>(`/trace/${trace_id}`),
  },
  externalRags: {
    list: (realm_id?: string | null) => req<ExternalRag[]>(`/external-rags${realm_id ? `?realm_id=${encodeURIComponent(realm_id)}` : ''}`),
    create: (body: { name: string; url: string; description?: string; headers?: Record<string, string>; retrieve_endpoint?: string; request_template?: Record<string, unknown>; response_mapping?: Record<string, string>; supported_params?: string[]; realm_id?: string; default_corpus_id?: string; default_pipeline_id?: string; default_reranker_id?: string; default_params?: Record<string, unknown>; uses_realm_resources?: boolean }) =>
      req<ExternalRag>('/external-rags', { method: 'POST', body: JSON.stringify(body) }),
    update: (id: string, body: Partial<{ name: string; url: string; description: string; headers: Record<string, string>; retrieve_endpoint: string; supported_params: string[]; default_corpus_id: string; default_pipeline_id: string; default_reranker_id: string; default_params: Record<string, unknown>; uses_realm_resources: boolean }>) =>
      req<ExternalRag>(`/external-rags/${id}`, { method: 'PATCH', body: JSON.stringify(body) }),
    delete: (id: string) => req<void>(`/external-rags/${id}`, { method: 'DELETE' }),
    test: (id: string) => req<ExternalRagTestResult>(`/external-rags/${id}/test`, { method: 'POST' }),
  },
  domainPacks: {
    list: (realmId?: string | null) =>
      req<DomainPack[]>(`/domain-packs${realmId ? `?realm_id=${encodeURIComponent(realmId)}` : ''}`),
    setActive: (active_packs: string[], realmId?: string | null) =>
      req<{ active_packs: string[]; status: string }>(
        `/domain-packs${realmId ? `?realm_id=${encodeURIComponent(realmId)}` : ''}`,
        { method: 'PUT', body: JSON.stringify({ active_packs }) },
      ),
  },
}

// Saved external RAG endpoints (model C), so a URL is configured
// once and picked from a list rather than retyped on every new run.
export interface ExternalRagCapabilities {
  supports_trace: boolean
  supports_retrieval_only: boolean
  retrieve_endpoint: string | null
  max_top_k: number | null
  source_ref_granularity: 'chunk' | 'document' | null
  // Declared at registration (not probed; see
  // ExternalRagCreateRequest.supported_params on the backend for why),
  // lets NewExperimentPage show/gate which `params` knobs this RAG
  // actually reads instead of letting a run send one that's silently
  // ignored.
  supported_params?: string[]
  // Derived from the same test() probe, not self-declared.
  // Compared against the corpus's own registered embedder(s)
  // (services/api_gateway/routers/corpus.py's `corpora` registry) so a
  // dimension/embedding-space mismatch surfaces here instead of only as a
  // hard vector-dimension error once a real experiment run reaches it.
  reported_embedders?: string[] | null
  embedder_mismatch_warning?: string | null
  embedder_hint?: string | null
}

export interface ExternalRag {
  id: string
  name: string
  url: string
  description: string
  headers: Record<string, string>
  retrieve_endpoint?: string | null
  // Tier 2 (declarative mapping, config-only). Both
  // null/undefined ⇒ tier 1 (native contract).
  request_template?: Record<string, unknown> | null
  response_mapping?: Record<string, string> | null
  capabilities?: ExternalRagCapabilities | null
  // per-(Realm, ExternalRag) defaults NewExperimentPage auto-fills
  // on RAG selection (see ExternalRagCreateRequest.default_corpus_id on the
  // backend for why this exists — a form that hardcoded corpus_id regardless
  // of which RAG was picked silently ran the wrong corpus against it).
  default_corpus_id?: string | null
  default_pipeline_id?: string | null
  default_reranker_id?: string | null
  default_params?: Record<string, unknown>
  // Was always returned by the backend, just never declared here.
  realm_id?: string | null
  // "Case A" inspection flag, see ExternalRagCreateRequest.
  // uses_realm_resources docstring on the backend.
  uses_realm_resources?: boolean
  created_at: string
}

export interface ExternalRagTestResult {
  ok: boolean
  answer_preview?: string
  has_trace?: boolean
  n_sources?: number
  capabilities?: ExternalRagCapabilities
  error?: string
  connection_tier?: 'native' | 'mapped'
  parsed_sources_preview?: Record<string, unknown>[]
}


// Domain pack plugins, discovered from domain_packs/* manifests.
export interface DomainPack {
  id: string
  version: string
  display_name: string
  description: string
  exported_kinds: string[]
  active: boolean
}

export interface CorpusRegistryEntry {
  id: string
  realm_id: string | null
  corpus_id: string
  storage_type: string
  backends: Record<string, Record<string, unknown>>
  owner: string
  description: string
  created_at?: string
  updated_at?: string
  deleted_at?: string | null
}

export interface CorpusIngest {
  job_id: string
  started_at: string
  finished_at: string | null
  strategy: string
  chunk_size: number
  overlap: number
  files: string[]
  n_files: number
  n_chunks: number
  hit_ratio: number
  status: 'running' | 'done' | 'error'
  error: string | null
  corpus_id?: string
}

// Corpus observability
export interface CorpusChunkItem {
  chunk_id: string
  doc_id: string
  structural_path: string
  text: string
  length: number
  header_only: boolean
  duplicate: boolean
}

export interface CorpusChunksResult {
  corpus_id: string
  items: CorpusChunkItem[]
  next_offset: string | null
}

export interface CorpusHealthItem {
  // See DetectorItem: the catalogue entries this finding is evidence for.
  failure_ids?: string[]
  id: string
  severity: 'ok' | 'info' | 'warn' | 'error'
  title: string
  detail: string
  action?: string
}

export interface CorpusHealthResult {
  /** Ten equal-width buckets across [min_length, max_length]. The shape matters
   *  more than the mean: a corpus of headings and walls of text has the same
   *  mean as an even one. */
  length_deciles?: number[]
  /** Fragments with no parsed structural path. `root` counts as absent: it is
   *  the chunker's mark for a document with no tree. */
  n_missing_path?: number
  corpus_id: string
  n_chunks: number
  n_duplicates: number
  n_header_only: number
  avg_length: number
  min_length: number
  max_length: number
  n_duplicate_numbers?: number
  n_missing_numbers?: number
  language_distribution?: Record<string, number>
  language_sample_size?: number
  items: CorpusHealthItem[]
}

// "Deep diagnostics": on-demand LLM-judge sample reports (Ragas/
// TruLens/chunk-coherence), each independent of the others (see
// eval/ragas_runner.py module docstring for why all are kept separate
// rather than merged into one score).
export interface DiagnosticsMetricsResult {
  corpus_id: string
  metrics: Record<string, number>
  per_question?: { id: string; question: string; actual_output: string }[]
  sample_size?: number
  incoherent_chunks?: { chunk_id: string; score: number; text_preview: string }[]
}

export interface DiagnosticsRunParams {
  pipeline_id?: string
  top_k?: number
  dataset?: string
  max_questions?: number
  realm_id?: string | null
}

function diagnosticsQuery(params: DiagnosticsRunParams): string {
  const qs = new URLSearchParams()
  if (params.pipeline_id) qs.set('pipeline_id', params.pipeline_id)
  if (params.top_k) qs.set('top_k', String(params.top_k))
  if (params.dataset) qs.set('dataset', params.dataset)
  if (params.max_questions) qs.set('max_questions', String(params.max_questions))
  if (params.realm_id) qs.set('realm_id', params.realm_id)
  return qs.toString()
}

// Community detection over the Neo4j graph (GDS Leiden/Louvain), labeled
// with each community's source_code mix — see services/api_gateway/
// routers/corpus.py:_label_communities for what dominant_source_share and
// source_code_distribution mean and why they matter (diagnosing whether
// communities follow corpus/source boundaries or shared-keyword noise).
export interface SourceCodeCount {
  source_code: string
  count: number
}

export interface GraphCommunity {
  community_id: number
  size: number
  source_code_distribution: SourceCodeCount[]
  dominant_source_share: number
  distinct_source_count: number
}

export interface InterCommunityEdge {
  community_a: number
  community_b: number
  weight: number
}

export interface GraphCommunitiesResult {
  /** false when a realm has no neo4j resource of its own: the graph named by
   *  the environment is shared, and what it holds belongs to another realm. */
  realm_scoped?: boolean
  corpus_id: string
  algorithm: string
  edge_type: 'lexical' | 'semantic'
  community_count: number
  modularity: number | null
  communities: GraphCommunity[]
  inter_community_edges: InterCommunityEdge[]
}

export interface GraphCommunityNode {
  chunk_id: string
  doc_id: string
  path: string
}

export interface GraphCommunityDetail {
  corpus_id: string
  community_id: number
  nodes: GraphCommunityNode[]
  edges: { source: string; target: string }[]
}

export interface StageTrace {
  embed_ms: number
  dense_retrieve_ms: number
  sparse_retrieve_ms: number
  merge_ms: number
  generate_ms: number
  total_ms: number
  n_dense: number
  n_sparse: number
  n_merged: number
  n_deduped: number
  input_tokens: number
  output_tokens: number
  context_chars: number
  // the configurable-pipeline stages (0 when the step is disabled).
  rerank_ms?: number
  grounding_ms?: number
  graph_ms?: number
  n_reranked?: number
  n_unsupported?: number
  n_graph?: number
}

export interface PipelineTrace {
  trace_id: string
  query: string
  stage_trace: StageTrace | null
  source_refs: {
    doc_id: string; chunk_id: string; structural_path: string
    score: number; chunk_text: string
    dense_score: number; sparse_score: number; rrf_rank: number
  }[]
  rendered_prompt_preview: string
  answer_preview: string
}
