import { useState, useRef, useEffect } from 'react'
import { useQuery, useMutation } from '@tanstack/react-query'
import { useTranslation } from 'react-i18next'
import { useDefaultCorpus } from '../hooks/useDefaultCorpus'
import SelectBox from '../components/SelectBox'
import { Pin } from 'lucide-react'
import { api, type StageTrace } from '../api/client'
import PipelineTrace from '../components/PipelineTrace'
import { useRealm } from '../context/RealmContext'

interface SourceRef {
  doc_id: string
  chunk_id: string
  structural_path: string
  score: number
  chunk_text?: string
  dense_score?: number
  sparse_score?: number
  rrf_rank?: number
  // Set when this chunk was injected by a matching
  // retrieval_pins entry (core/pins/overlay.py), not found by ordinary
  // retrieval.
  pinned?: boolean
}

interface Message {
  id: string
  role: 'user' | 'assistant'
  text: string
  source_refs?: SourceRef[]
  computed_citations?: string[]
  trace_id?: string
  stage_trace?: StageTrace | null
  rendered_prompt_preview?: string
  loading?: boolean
}

const API_BASE = import.meta.env.VITE_API_BASE ?? 'http://localhost:8081'

const CUSTOM_CORPUS_ID = '__custom_corpus_id__'

/** Real `<select>` of known corpus_ids + a "+ custom id..." escape hatch —
 * replaces a plain text `<input list="...">` (`<datalist>`) that read as an
 * empty text box pre-filled with "default", not an intentional dropdown
 * (found live: reported as "corpus can't be selected" even though typing
 * into it did work — most browsers only surface datalist suggestions once
 * you start typing, with no visual affordance that it's a selector at all,
 * unlike the native `<select>`s right next to it). Same fix as
 * CorpusPage.tsx#UploadCorpusIdField/NewExperimentPage.tsx#CorpusIdField. */
function ChatCorpusIdField({ value, onChange, knownIds }: {
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
          type="text" className="chat-control-input w-90" value={value} autoFocus
          onChange={e => onChange(e.target.value)}
        />
        <button
          type="button" className="btn-sm"
          onClick={() => { setCustomMode(false); onChange(knownIds[0] ?? '') }}
        >
          {t('chatPage.corpusFromList')}
        </button>
      </div>
    )
  }

  return (
    <SelectBox
      value={isKnown ? value : ''} aria-label="corpus_id"
      onChange={e => {
        if (e.target.value === CUSTOM_CORPUS_ID) { setCustomMode(true); onChange('') }
        else onChange(e.target.value)
      }}
    >
      {knownIds.map(id => <option key={id} value={id}>{id}</option>)}
      <option value={CUSTOM_CORPUS_ID}>{t('chatPage.corpusCustomOption')}</option>
    </SelectBox>
  )
}

/** A chat-header chip: label and value on one line, outlined together.
 *  A labelled field takes twice the space and pushes the row off the edge. */
function ChatChip({ label, title, dim, children }: {
  label: string; title?: string; dim?: boolean; children: React.ReactNode
}) {
  // `label`, and not `span`: the chip is its field's label, and a `label`
  // wrapper binds the two without a `for` attribute. Otherwise the field stays
  // nameless both to a screen reader and to a search by label.
  return (
    <label className={`chat-chip${dim ? ' dim' : ''}`} title={title}>
      <span className="chat-chip-label">{label}</span>
      {children}
    </label>
  )
}

export default function ChatPage() {
  const { t } = useTranslation()
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [topK, setTopK] = useState(5)
  const [sending, setSending] = useState(false)
  const bottomRef = useRef<HTMLDivElement>(null)
  const [selectedRagId, setSelectedRagId] = useState('')
  // In-process fallback used to always query whatever corpus_id the gateway
  // happened to boot against ("default"), identically for every Realm — no
  // selector existed at all. Same fix as NewExperimentPage's corpus_id
  // field: explicit, not implicit (see the design notes "Chat routing via Realm").
  // An empty string, and not 'default': what the chat works against is
  // decided by useDefaultCorpus, the last ingest that actually wrote chunks.
  const [corpusId, setCorpusId] = useState('')
  // Chat was permanently dense-only (NaivePipeline, no hybrid
  // merge, no graph) with no way to pick anything else, unlike a real
  // experiment run's retrieval-type selector. Same ids as
  // core/experiment/config.py#ExperimentConfig.pipeline_id.
  const [pipelineId, setPipelineId] = useState('naive')

  const { data: models = [] } = useQuery({ queryKey: ['models'], queryFn: () => api.models(), staleTime: 30000 })
  const { activeRealmId } = useRealm()

  const { data: settings } = useQuery({
    queryKey: ['settings', activeRealmId],
    queryFn: () => api.settings.get(activeRealmId),
    staleTime: 10000,
  })
  const setModelMut = useMutation({
    mutationFn: (model: string) => api.settings.setModel(model, activeRealmId),
    onSuccess: () => { /* settings will refetch */ },
  })
  // Which RAG implementation answers in chat. Empty string =
  // platform's own built-in pipeline (in-process; now reads this Realm's
  // registered resources when present — see the design notes "Chat routing
  // via Realm").
  const { data: externalRags = [] } = useQuery({
    queryKey: ['external-rags', activeRealmId],
    queryFn: () => api.externalRags.list(activeRealmId),
  })
  const { data: corpusHistory = [] } = useQuery({
    queryKey: ['corpus', activeRealmId],
    queryFn: () => api.corpus.list(activeRealmId),
  })
  // The `corpora` registry, not just ingest-job history: knows
  // about corpora ingested outside the upload form (CLI, migration script).
  const { data: corpusCollections = [] } = useQuery({
    queryKey: ['corpus-collections', activeRealmId],
    queryFn: () => api.corpus.collections(activeRealmId),
  })
  // 'default' is not substituted here: the name was always in the list while
  // the content was not, so picking it meant querying nothing. If a corpus by
  // that name really exists, it arrives from the registry like any other.
  const knownCorpusIds = Array.from(new Set([
    ...corpusCollections.map(c => c.corpus_id),
    ...corpusHistory.flatMap(h => h.corpus_id ? [h.corpus_id] : []),
  ]))
  useDefaultCorpus(activeRealmId, corpusId, setCorpusId)
  const { data: registry } = useQuery({ queryKey: ['registry', activeRealmId], queryFn: () => api.registry(activeRealmId) })
  const pipelineIds = registry?.pipeline ?? ['naive', 'hybrid_rrf', 'hybrid_weighted', 'graph']

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  // Reset the pick when the Realm changes (a rag_id from another Realm is
  // meaningless here); auto-pick if there's exactly one, matching the
  // backend's own auto-resolve so the UI shows what would happen
  // even before the user touches the selector.
  useEffect(() => {
    setSelectedRagId(externalRags.length === 1 ? externalRags[0].id : '')
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeRealmId, externalRags.length])

  async function send() {
    const text = input.trim()
    if (!text || sending) return

    const userMsg: Message = { id: crypto.randomUUID(), role: 'user', text }
    const placeholder: Message = { id: crypto.randomUUID(), role: 'assistant', text: '', loading: true }
    setMessages(prev => [...prev, userMsg, placeholder])
    setInput('')
    setSending(true)

    try {
      const res = await fetch(`${API_BASE}/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text, top_k: topK,
          realm_id: activeRealmId ?? undefined,
          external_rag_id: selectedRagId || undefined,
          corpus_id: corpusId || undefined,
          pipeline_id: pipelineId,
        }),
      })
      const data = await res.json()
      setMessages(prev => prev.map(m =>
        m.id === placeholder.id ? {
          ...m,
          text: data.text,
          source_refs: data.source_refs,
          computed_citations: data.computed_citations ?? [],
          trace_id: data.metadata?.trace_id,
          stage_trace: data.stage_trace ?? null,
          rendered_prompt_preview: data.rendered_prompt_preview ?? '',
          loading: false,
        } : m
      ))
    } catch {
      setMessages(prev => prev.map(m =>
        m.id === placeholder.id ? { ...m, text: t('chatPage.connectionError'), loading: false } : m
      ))
    } finally {
      setSending(false)
    }
  }

  function handleKey(e: React.KeyboardEvent) {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send() }
  }

  const activeModel = settings?.active_model ?? '…'

  return (
    <div className="chat-page">
      {/* Header */}
      <div className="chat-header">
        <h1 className="chat-title">{t('chatPage.title')}</h1>
        {/* Chips, and not a row of labelled fields. Six labels with their
            fields did not fit on a line and ran off the right edge; in a chip
            the label sits before the value and takes only the space that
            value needs. */}
        <div className="chat-controls">
          <ChatChip label={t('chatPage.corpusLabel')} title={t('chatPage.corpusTooltip')}>
            <ChatCorpusIdField value={corpusId} onChange={setCorpusId} knownIds={knownCorpusIds} />
          </ChatChip>

          <ChatChip
            label={t('chatPage.implementationLabel')}
            title={t('chatPage.implementationTooltip')}
          >
            <SelectBox value={selectedRagId} onChange={e => setSelectedRagId(e.target.value)}
                       aria-label={t('chatPage.implementationLabel')}>
              <option value="">{t('chatPage.builtinOption')}</option>
              {externalRags.map(r => <option key={r.id} value={r.id}>{r.name}</option>)}
            </SelectBox>
          </ChatChip>

          <ChatChip
            label="Retrieval" dim={!!selectedRagId}
            title={selectedRagId ? t('chatPage.retrievalDisabledTooltip') : t('chatPage.retrievalTypeTooltip')}
          >
            <SelectBox value={pipelineId} onChange={e => setPipelineId(e.target.value)}
                       disabled={!!selectedRagId} aria-label="Retrieval">
              {pipelineIds.map(id => <option key={id} value={id}>{id}</option>)}
            </SelectBox>
          </ChatChip>

          <ChatChip label={t('chatPage.modelLabel')} dim={!!selectedRagId}>
            <SelectBox value={activeModel} onChange={e => setModelMut.mutate(e.target.value)}
                       disabled={!!selectedRagId} aria-label={t('chatPage.modelLabel')}>
              {models.length === 0
                ? <option value={activeModel}>{activeModel}</option>
                : models.map(m => <option key={m.name} value={m.name}>{m.name}</option>)}
            </SelectBox>
          </ChatChip>

          <ChatChip label="top-k">
            <input type="number" min={1} max={20} value={topK}
              onChange={e => setTopK(Number(e.target.value))} className="chat-num" />
          </ChatChip>

          {(settings?.active_packs?.length ?? 0) > 0 && (
            <span className="chat-chip" title={t('chatPage.domainPacksTooltip')}>
              {t('chatPage.domainPacksLabel', { packs: settings!.active_packs.join(', ') })}
            </span>
          )}
        </div>
      </div>

      {/* Messages */}
      <div className="chat-messages">
        {/* An empty chat is a line, and not a forty-pixel emoji in the
            middle of the screen. The subject area is deliberately unnamed
            here: the realm's domain pack or external RAG knows it, and
            the chat shell does not. */}
        {messages.length === 0 && (
          <p className="chat-empty">
            {t('chatPage.emptyTitle')}
            {corpusId && <>{' · '}<span className="mono-sm">{corpusId}</span></>}
          </p>
        )}

        {messages.map(m => (
          <div key={m.id} className={`chat-msg-wrap ${m.role}`}>
            <div className={`chat-bubble ${m.role}`}>
              {m.loading ? <span className="dim-icon">{t('chatPage.generatingAnswer')}</span> : m.text}
            </div>

            {/* Computed citation — read from retrieval metadata (source_refs),
                NOT from the model's own text. Shown separately and labeled as
                such because the model unreliably transcribes citation numbers
                (confuses fragment index with the real number, or invents one
                entirely) — this line is always correct, the model's own
                in-text citation may not be. */}
            {m.computed_citations && m.computed_citations.length > 0 && (
              <div
                className="chat-computed-citation"
                title={t('chatPage.computedCitationTooltip')}
              >
                <span className="chat-computed-citation-label">{t('chatPage.computedCitationLabel')}</span>{' '}
                {m.computed_citations.join('; ')}
              </div>
            )}

            {/* Source refs with scores */}
            {m.source_refs && m.source_refs.length > 0 && (
              <div className="chat-refs">
                {m.source_refs.map((ref, i) => (
                  <span key={i} className="chat-ref-badge" title={
                    `score: ${ref.score.toFixed(3)}` +
                    (ref.dense_score ? `\ndense: ${ref.dense_score}` : '') +
                    (ref.sparse_score ? `\nsparse: ${ref.sparse_score}` : '') +
                    (ref.chunk_text ? `\n\n${ref.chunk_text.slice(0, 120)}…` : '')
                  }>
                    {ref.pinned && (
                      <Pin size={10} className="btn-icon" aria-label={t('chatPage.pinnedRef')} />
                    )}
                    {ref.structural_path || ref.doc_id}
                    <span className="ref-score">{ref.score.toFixed(2)}</span>
                  </span>
                ))}
                {m.trace_id && (
                  <a href={`http://localhost:3001/project/rag-platform/traces/${m.trace_id}`}
                    target="_blank" rel="noreferrer" className="chat-ref-badge trace-link">
                    trace ↗
                  </a>
                )}
              </div>
            )}

            {/* Pipeline trace */}
            {m.stage_trace && !m.loading && (
              <details className="chat-trace-wrap">
                <summary className="chat-trace-summary">⏱ Pipeline trace</summary>
                <PipelineTrace trace={m.stage_trace} promptPreview={m.rendered_prompt_preview} />
              </details>
            )}
          </div>
        ))}
        <div ref={bottomRef} />
      </div>

      {/* Input */}
      {/* Field and button inside one outlined row, as in the design: two
          separate borders read as two different controls, when sending is
          part of typing, and not a thing standing next to it. */}
      <div className="chat-input-bar">
        <textarea value={input} onChange={e => setInput(e.target.value)} onKeyDown={handleKey}
          placeholder={t('chatPage.inputPlaceholder')}
          rows={1} className="chat-textarea" />
        <button onClick={send} disabled={!input.trim() || sending} className="chat-send">
          {sending ? '…' : t('chatPage.sendButton')}
          <kbd className="chat-send-key">⌘⏎</kbd>
        </button>
      </div>
    </div>
  )
}
