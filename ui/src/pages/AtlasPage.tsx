import { useEffect, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import { useTranslation } from 'react-i18next'
import { useQuery } from '@tanstack/react-query'
import { ExternalLink } from 'lucide-react'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import { api, type AtlasCandidate, type AtlasEntry, type Atlas, type AtlasSignal } from '../api/client'
import { Link } from 'react-router-dom'
import { useRealm, useRealmPath } from '../context/RealmContext'

/** What the platform can do about an entry today, decided by the catalogue.
 *
 *  Four, and the difference between them is the whole reason this page exists.
 *  `caught` claims a signal decides it and a bait proved both ends. `visible`
 *  claims the data shows something and a person concludes. `unproven` claims a
 *  signal and points at a bait on a proving ground that does not exist yet, so
 *  nobody has watched it fire. `none` claims nothing catches it, and the entry
 *  has to say what is missing.
 *
 *  The rule lived here for a while, in TypeScript, and a second copy of it
 *  lived in a reporting tool, in Python, with nothing holding the two to the
 *  same answer. It arrives from the server now. */
type State = AtlasEntry['state']

const STATE_KEY: Record<State, string> = {
  caught: 'atlasPage.state.caught',
  visible: 'atlasPage.state.visible',
  unproven: 'atlasPage.state.unproven',
  none: 'atlasPage.state.none',
}

/** A coordinate, and a link to whoever defines it. The codes are borrowed from
 *  a published schema, so a reader meeting one must be able to reach its
 *  definition and not guess at it. */
function Coord({ code, values, url, gap }: {
  code: string; values?: string[]; url: string; gap?: boolean
}) {
  return (
    <span className={`coord${gap ? ' gap' : ''}`}>
      {values && values.length ? `${code} ∈ ${values.join(' · ')}` : code}
      <a href={url} target="_blank" rel="noreferrer" aria-label={code}>
        <ExternalLink size={11} />
      </a>
    </span>
  )
}

function DetectionState({ state }: { state: State }) {
  const { t } = useTranslation()
  return (
    <span className={`det-state det-${state}`}>
      <i className="det-bar" aria-hidden="true" />
      {t(STATE_KEY[state])}
    </span>
  )
}

function EntryDetail({ entry }: { entry: AtlasEntry }) {
  const { t } = useTranslation()
  const s = entry.severity
  return (
    <div className="atlas-detail">
      <div className="stack-8">
        <div className="panel">
          <div className="eyebrow mb-8">
            {entry.detection === 'none'
              ? t('atlasPage.detail.whatIsMissing')
              : t('atlasPage.detail.whatDetects')}
          </div>
          {entry.detection === 'none'
            ? <p className="find-detail">{entry.not_detected_reason}</p>
            : entry.signals.map(signal => (
                <div key={signal.id} className="find-row">
                  <span className="find-dot find-info" aria-hidden="true" />
                  <div>
                    <div className="find-title mono-sm">{signal.id}</div>
                    <p className="find-detail">
                      {signal.side === 'ui'
                        ? t('atlasPage.detail.sideUi')
                        : t('atlasPage.detail.sideCore')}
                    </p>
                  </div>
                </div>
              ))}
          {entry.bait && (
            <p className="hint-line">
              {t('atlasPage.detail.bait')}: <code className="inline-code">{entry.bait}</code>
              {entry.bait_level === 'proving_ground' && `. ${t('atlasPage.detail.baitPending')}`}
            </p>
          )}
          {entry.shares_signals_with.length > 0 && (
            <p className="hint-line">
              {t('atlasPage.detail.sharesSignals', {
                others: entry.shares_signals_with.join(', '),
              })}
            </p>
          )}
        </div>

        <div className="panel">
          <div className="eyebrow mb-8">{t('atlasPage.detail.stages')}</div>
          <p className="find-detail">
            {t('atlasPage.detail.arises')}: <b>{t(`atlasPage.stage.${entry.stage_origin}`, entry.stage_origin)}</b>
            {' · '}
            {t('atlasPage.detail.becomesVisible')}: <b>{t(`atlasPage.stage.${entry.stage_visible}`, entry.stage_visible)}</b>
          </p>
        </div>
      </div>

      <div className="stack-8">
        <div className="panel">
          <div className="eyebrow mb-8">{t('atlasPage.detail.scope')}</div>
          {entry.applies_when.length === 0
            ? <p className="find-detail">{t('atlasPage.detail.scopeAny')}</p>
            : <div className="chip-row">
                {entry.applies_when.map(scope => (
                  <Coord key={scope.code} code={scope.code} values={scope.values} url={scope.url} />
                ))}
              </div>}
          {entry.scope_caveat && (
            <p className="hint-line">{t('atlasPage.detail.scopeCaveat')}: {entry.scope_caveat}</p>
          )}
        </div>

        <div className="panel">
          <div className="eyebrow mb-8">{t('atlasPage.detail.severity')}</div>
          <p className="find-detail">
            {t('atlasPage.detail.quiet')} {s.quiet} · {t('atlasPage.detail.cost')} {s.cost} ·{' '}
            {t('atlasPage.detail.prevalence')} {s.prevalence} = <b>{s.total}</b>
          </p>
          <p className="hint-line">{t('atlasPage.detail.severityCaveat')}</p>
          <p className="hint-line">
            {t('atlasPage.detail.instrument')}: {t(`atlasPage.instrument.${entry.instrument}`, entry.instrument)}
            {entry.atlas_rows.length > 0 && ` · ${t('atlasPage.detail.atlasRow')} ${entry.atlas_rows.join(', ')}`}
          </p>
        </div>
      </div>
    </div>
  )
}

function EntryTable({ entries, openId, onOpen }: {
  entries: AtlasEntry[]
  openId: string | null
  onOpen: (id: string | null) => void
}) {
  const { t } = useTranslation()
  return (
    <div className="table-wrap">
      <table className="table">
        <thead>
          <tr>
            <th>{t('atlasPage.column.id')}</th>
            <th>{t('atlasPage.column.title')}</th>
            <th>{t('atlasPage.column.stage')}</th>
            <th>{t('atlasPage.column.severity')}</th>
            <th>{t('atlasPage.column.state')}</th>
          </tr>
        </thead>
        <tbody>
          {entries.map(entry => {
            const open = openId === entry.id
            return [
              <tr key={entry.id} className={open ? 'atlas-row-open' : undefined}>
                <td>
                  <button
                    className="link-btn mono-sm"
                    aria-expanded={open}
                    onClick={() => onOpen(open ? null : entry.id)}
                  >
                    {entry.id}
                  </button>
                </td>
                <td>{t(entry.title_key, entry.title)}</td>
                <td className="text-muted">{t(`atlasPage.stage.${entry.stage_visible}`, entry.stage_visible)}</td>
                <td className="mono-sm">{entry.severity.total}</td>
                <td><DetectionState state={entry.state} /></td>
              </tr>,
              open ? (
                <tr key={`${entry.id}-detail`} className="atlas-row-open">
                  <td colSpan={5}><EntryDetail entry={entry} /></td>
                </tr>
              ) : null,
            ]
          })}
        </tbody>
      </table>
    </div>
  )
}


/** The catalogue answers "what detects this". The reverse question, "what does
 *  a signal that just fired mean", is asked more often and until now nothing
 *  answered it.
 *
 *  Two facts have to survive this table. A signal standing for two entries is
 *  evidence for either and for neither in particular, and a reader not told
 *  that takes it for evidence about the entry in front of them. And the
 *  platform has two sets of signals, one on each side, which disagree on the
 *  same data; a reference showing one of them would describe half a platform.
 */
function SignalsTab() {
  const { t } = useTranslation()
  const { data, isLoading, error } = useQuery({
    queryKey: ['atlas-signals'],
    queryFn: () => api.atlas.signals(),
  })

  if (isLoading) return <p className="text-muted">{t('common.loading')}</p>
  if (error || !data) return <p className="text-muted">{t('atlasPage.unavailable')}</p>

  const shared = data.signals.filter((s: AtlasSignal) => !s.singles_out)

  return (
    <>
      <p className="page-sub">{t('atlasPage.signals.lead')}</p>
      {shared.length > 0 && (
        <div className="guide-callout warn">
          <div>
            <strong>{t('atlasPage.signals.sharedTitle', { count: shared.length })}</strong>
            <p className="hint-line">{t('atlasPage.signals.sharedDetail')}</p>
          </div>
        </div>
      )}
      <div className="table-wrap">
        <table className="table">
          <thead>
            <tr>
              <th>{t('atlasPage.signals.column.signal')}</th>
              <th>{t('atlasPage.signals.column.side')}</th>
              <th>{t('atlasPage.signals.column.evidenceFor')}</th>
              <th>{t('atlasPage.signals.column.bait')}</th>
            </tr>
          </thead>
          <tbody>
            {data.signals.map((signal: AtlasSignal) => (
              <tr key={signal.id}>
                <td className="mono-sm">{signal.id}</td>
                <td>
                  <span className={`badge ${signal.side === 'ui' ? 'badge-info' : ''}`}>
                    {t(`atlasPage.signals.side.${signal.side}`)}
                  </span>
                </td>
                <td>
                  <span className="mono-sm">{signal.failures.join(', ')}</span>
                  {!signal.singles_out && (
                    <span className="badge badge-warn ml-8">
                      {t('atlasPage.signals.doesNotSeparate')}
                    </span>
                  )}
                </td>
                <td className="text-muted">
                  {signal.bait_level.map(level => t(`atlasPage.signals.bait.${level}`, level)).join(', ')}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  )
}

/** Where a report says it was seen, with the things this platform can open
 *  turned into links.
 *
 *  The field is free text on purpose: somebody who has just met a failure
 *  should not have to know which of the platform's nouns applies to it. So
 *  this recognises and never demands, and anything it does not recognise
 *  stands exactly as it was written. Guessing at an identifier would send a
 *  reader to a page about something else, which is worse than plain text.
 */
const REFERENCE = /\b(run|прогон|corpus|корпус)\s+([A-Za-z0-9][\w-]*)/gi

function ObservedOn({ text }: { text: string }) {
  const realmPath = useRealmPath()
  const parts: Array<string | { label: string; to: string }> = []
  let last = 0
  for (const match of text.matchAll(REFERENCE)) {
    const [whole, noun, id] = match
    const at = match.index ?? 0
    if (at > last) parts.push(text.slice(last, at))
    const run = /^(run|прогон)$/i.test(noun)
    parts.push({
      label: whole,
      to: realmPath(run ? `/experiments/${id}` : `/data/content?corpus_id=${id}`),
    })
    last = at + whole.length
  }
  if (last < text.length) parts.push(text.slice(last))

  return (
    <div className="mono-sm text-muted">
      {parts.map((part, i) =>
        typeof part === 'string'
          ? <span key={i}>{part}</span>
          : <Link key={i} to={part.to}>{part.label}</Link>,
      )}
    </div>
  )
}


/** Deciding what a report is, and recording what it became. Two acts, weeks
 *  apart and by different work, so two panels and never one: a panel offering
 *  both would put a field for an entry that does not exist beside the question
 *  of whether it should.
 */
function TriagePanel({ candidate }: { candidate: AtlasCandidate }) {
  const { t } = useTranslation()
  const { activeRealmId } = useRealm()
  const queryClient = useQueryClient()
  const [note, setNote] = useState('')
  const [entryId, setEntryId] = useState('')

  const done = () =>
    queryClient.invalidateQueries({ queryKey: ['atlas-candidates', activeRealmId] })

  const decide = useMutation({
    mutationFn: (status: 'accepted' | 'rejected') =>
      api.atlas.decide(candidate.candidate_id, { status, note }),
    onSuccess: done,
  })
  const promote = useMutation({
    mutationFn: () => api.atlas.promote(candidate.candidate_id, entryId.trim()),
    onSuccess: done,
  })

  const failed = (decide.error ?? promote.error) as Error | undefined

  if (candidate.status === 'accepted') {
    return (
      <div className="cand-panel">
        <p className="cand-why">{t('atlasPage.candidates.promoteWhy')}</p>
        <pre className="cand-cmd">{`python3 -m tools.atlas_report --scaffold ${candidate.candidate_id}`}</pre>
        <label className="tight">
          {t('atlasPage.candidates.field.entry')}
          <input type="text" value={entryId} maxLength={8} className="cand-entry"
                 onChange={e => setEntryId(e.target.value)} />
        </label>
        <div className="cand-actions">
          <button type="button" className="btn btn-primary"
                  disabled={!entryId.trim() || promote.isPending}
                  onClick={() => promote.mutate()}>
            {t('atlasPage.candidates.recordPointer')}
          </button>
          <span className="text-muted">{t('atlasPage.candidates.refusedUntil')}</span>
        </div>
        {/* The server's own sentence, shown as it arrives. Rewriting it here
            would give the reader a second version of the rule to reconcile
            with the first. */}
        {failed && <p className="hint-line bad">{failed.message}</p>}
      </div>
    )
  }

  return (
    <div className="cand-panel">
      <p className="cand-why">{t('atlasPage.candidates.decideWhy')}</p>
      <label className="tight">
        {t('atlasPage.candidates.field.note')}
        <textarea value={note} maxLength={2000} rows={2}
                  onChange={e => setNote(e.target.value)} />
      </label>
      <div className="cand-actions">
        <button type="button" className="btn btn-primary" disabled={decide.isPending}
                onClick={() => decide.mutate('accepted')}>
          {t('atlasPage.candidates.accept')}
        </button>
        <button type="button" className="btn" disabled={!note.trim() || decide.isPending}
                onClick={() => decide.mutate('rejected')}>
          {t('atlasPage.candidates.reject')}
        </button>
        <span className="text-muted">{t('atlasPage.candidates.rejectNeedsReason')}</span>
      </div>
      {failed && <p className="hint-line bad">{failed.message}</p>}
    </div>
  )
}


/** Failures people have reported, kept apart from the catalogue on purpose.
 *
 *  The catalogue is read-only from here, and that is why its guarantees hold:
 *  an entry claims a signal catches a failure, and the build refuses such a
 *  claim without a bait. A candidate claims nothing of the kind, so it can be
 *  written from the interface, and the refusal keeps holding for everything
 *  that does.
 *
 *  Every open row says on its face that no bait has confirmed it. Not a
 *  column and not a tooltip: a candidate and an entry look alike on a screen,
 *  and a reader who skims has to see the difference without reading a legend.
 */
function CandidatesSection() {
  const { t } = useTranslation()
  const { activeRealmId } = useRealm()
  const queryClient = useQueryClient()
  const [open, setOpen] = useState(false)
  const [form, setForm] = useState({
    title: '', looked_like: '', observed_on: '', suspected_signal: '',
  })

  const { data } = useQuery({
    queryKey: ['atlas-candidates', activeRealmId],
    queryFn: () => api.atlas.candidates(activeRealmId),
  })

  const report = useMutation({
    mutationFn: () => api.atlas.report(form, activeRealmId),
    onSuccess: () => {
      setForm({ title: '', looked_like: '', observed_on: '', suspected_signal: '' })
      setOpen(false)
      queryClient.invalidateQueries({ queryKey: ['atlas-candidates', activeRealmId] })
    },
  })

  const candidates = data?.candidates ?? []
  const promoted = candidates.filter(c => c.promoted_to)
  const openOnes = candidates.filter(c => !c.promoted_to)

  // The title and what looked like it worked are what the server requires, and
  // it says so by refusing a shorter one. Asking here as well means the reader
  // learns it before typing and not after submitting.
  const enough = form.title.trim().length >= 8 && form.looked_like.trim().length >= 8

  return (
    <section className="section">
      <div className="section-rule flush">
        <h2 className="section-title">{t('atlasPage.candidates.title')}</h2>
        <span className="section-meta">
          {t('atlasPage.candidates.counts', { open: openOnes.length, promoted: promoted.length })}
        </span>
      </div>
      <p className="hint-line">{t('atlasPage.candidates.lead')}</p>

      {candidates.length > 0 && (
        <div className="table-wrap mt-8">
          <table className="table">
            <thead>
              <tr>
                <th>{t('atlasPage.candidates.column.reported')}</th>
                <th>{t('atlasPage.candidates.column.what')}</th>
                <th>{t('atlasPage.candidates.column.state')}</th>
              </tr>
            </thead>
            <tbody>
              {candidates.map(c => <CandidateRow key={c.candidate_id} candidate={c} />)}
            </tbody>
          </table>
        </div>
      )}

      <details className="mt-8" open={open}
               onToggle={e => setOpen((e.currentTarget as HTMLDetailsElement).open)}>
        <summary><strong>{t('atlasPage.candidates.report')}</strong></summary>
        <form className="cand-form" onSubmit={e => { e.preventDefault(); report.mutate() }}>
          <label>
            {t('atlasPage.candidates.field.title')}
            <input type="text" value={form.title} maxLength={200}
                   onChange={e => setForm({ ...form, title: e.target.value })} />
          </label>
          <label>
            {t('atlasPage.candidates.field.lookedLike')}
            <textarea value={form.looked_like} maxLength={2000}
                      onChange={e => setForm({ ...form, looked_like: e.target.value })} />
          </label>
          <label>
            {t('atlasPage.candidates.field.observedOn')}
            <input type="text" value={form.observed_on} maxLength={500}
                   onChange={e => setForm({ ...form, observed_on: e.target.value })} />
          </label>
          <label>
            {t('atlasPage.candidates.field.suspectedSignal')}
            <input type="text" value={form.suspected_signal} maxLength={200}
                   onChange={e => setForm({ ...form, suspected_signal: e.target.value })} />
          </label>
          <div className="cand-actions">
            <button type="submit" className="btn btn-primary"
                    disabled={!enough || report.isPending}>
              {t('atlasPage.candidates.submit')}
            </button>
            <span className="text-muted">{t('atlasPage.candidates.harmless')}</span>
          </div>
          {report.isError && (
            <p className="hint-line bad">{String((report.error as Error).message)}</p>
          )}
        </form>
      </details>
    </section>
  )
}

/** One reported failure. A promoted one points at the entry it became and
 *  drops the unconfirmed mark, because an entry arrives with its bait. */
function CandidateRow({ candidate }: { candidate: AtlasCandidate }) {
  const { t } = useTranslation()
  const [open, setOpen] = useState(false)
  const state = candidate.promoted_to ? 'promoted' : candidate.status
  // Deciding a report needs the report in front of you, so the panel opens
  // where the report already is. A page of its own would carry all three of
  // its fields across and be this screen with an extra click.
  const actionable = !candidate.promoted_to && candidate.status !== 'rejected'
  return (
    <tr>
      <td className="mono-sm nowrap">
        {candidate.candidate_id}
        <div className="text-muted">{candidate.created_at.slice(0, 10)}</div>
      </td>
      <td>
        <div className="cand-title">{candidate.title}</div>
        <div className="cand-said">{candidate.looked_like}</div>
        {candidate.observed_on && <ObservedOn text={candidate.observed_on} />}
        {candidate.note && <div className="cand-said">{candidate.note}</div>}
        {!candidate.confirmed_by_a_bait && !candidate.promoted_to && (
          <span className="cand-unbaited">{t('atlasPage.candidates.unbaited')}</span>
        )}
        {actionable && (
          <div>
            <button type="button" className="btn btn-sm cand-open"
                    aria-expanded={open} onClick={() => setOpen(!open)}>
              {t(`atlasPage.candidates.${candidate.status === 'accepted' ? 'promoteIt' : 'decideIt'}`)}
            </button>
          </div>
        )}
        {actionable && open && <TriagePanel candidate={candidate} />}
      </td>
      <td className="nowrap">
        <span className={`det-state cand-${state}`}>
          <i className="det-bar" />
          {t(`atlasPage.candidates.state.${state}`)}
          {candidate.promoted_to && ` → ${candidate.promoted_to}`}
        </span>
      </td>
    </tr>
  )
}


export default function AtlasPage() {
  const { t } = useTranslation()
  const [tab, setTab] = useState<'catalogue' | 'signals'>('catalogue')
  const [point, setPoint] = useState('hybrid')
  // An entry named in the address opens on arrival. Findings elsewhere link
  // here by identifier, and a link landing on the catalogue with nothing opened
  // is a promise the page did not keep.
  const [searchParams] = useSearchParams()
  const requested = searchParams.get('entry')
  const [openId, setOpenId] = useState<string | null>(requested)

  const { data, isLoading, error } = useQuery<Atlas>({
    queryKey: ['atlas', point],
    queryFn: () => api.atlas.read(point),
  })

  // A requested entry that cannot occur in the architecture on screen sits in a
  // list it is filtered out of, so the page moves to one where it can occur.
  // The destination comes from the entry itself. The first version picked "any
  // point other than this one", which sends a reader back and forth for ever
  // between two architectures that both exclude the entry.
  const requestedEntry = data?.entries.find(e => e.id === requested)
  const moveTo =
    requestedEntry && requestedEntry.applies_here !== true
      ? requestedEntry.applies_to_points.find(name => name !== point)
      : undefined
  useEffect(() => {
    if (moveTo) setPoint(moveTo)
  }, [moveTo])

  const tabs = (
    <div className="data-tab-bar" role="tablist">
      <button type="button" role="tab" aria-selected={tab === 'catalogue'}
              className={`data-tab${tab === 'catalogue' ? ' active' : ''}`}
              onClick={() => setTab('catalogue')}>
        {t('atlasPage.tab.catalogue')}
      </button>
      <button type="button" role="tab" aria-selected={tab === 'signals'}
              className={`data-tab${tab === 'signals' ? ' active' : ''}`}
              onClick={() => setTab('signals')}>
        {t('atlasPage.tab.signals')}
      </button>
    </div>
  )

  if (tab === 'signals') {
    return (
      <div className="page">
        <h1 className="page-title">{t('atlasPage.title')}</h1>
        {tabs}
        <SignalsTab />
      </div>
    )
  }

  if (isLoading) return <div className="page">{tabs}<p className="text-muted">{t('common.loading')}</p></div>
  if (error || !data) {
    return <div className="page">{tabs}<p className="text-muted">{t('atlasPage.unavailable')}</p></div>
  }

  // Applicable here, and nothing else. A count over the whole catalogue answers
  // no question anybody has: half the entries cannot occur in a dense system
  // and a different half cannot occur in a graph one.
  const byWeight = (a: AtlasEntry, b: AtlasEntry) =>
    b.severity.total - a.severity.total || a.id.localeCompare(b.id)
  const applicable = data.entries.filter(e => e.applies_here === true).sort(byWeight)
  // Neither applicable nor ruled out: this architecture records nothing about a
  // coordinate the entry depends on. Dropping them silently would turn "never
  // asked" into "does not apply", which is the distinction the whole scope
  // mechanism exists to keep.
  const undetermined = data.entries.filter(e => e.applies_here === null).sort(byWeight)

  const caught = applicable.filter(e => e.state === 'caught')
  const visible = applicable.filter(e => e.state === 'visible')
  const unproven = applicable.filter(e => e.state === 'unproven')
  const notCaught = applicable.filter(e => e.state === 'none')
  const worked = [...caught, ...visible, ...unproven].sort(byWeight)

  return (
    <div className="page">
      <h1 className="page-title">{t('atlasPage.title')}</h1>
      {tabs}
      <p className="page-sub">{t('atlasPage.lead')}</p>

      <section className="section">
        <div className="section-rule flush">
          <h2 className="section-title">{t('atlasPage.architecture')}</h2>
          <span className="section-meta">
            {t('atlasPage.schemaRelease', { release: data.schema.release })}
            {' · '}
            <a className="link-muted" href={data.schema.source} target="_blank" rel="noreferrer">
              {data.schema.source.replace('https://', '')}
            </a>
          </span>
        </div>
        <div className="chip-row">
          {Object.entries(data.points).map(([name, info]) => (
            <button
              key={name}
              className={`chip${name === point ? ' active' : ''}`}
              aria-pressed={name === point}
              onClick={() => { setPoint(name); setOpenId(null) }}
            >
              {t(`atlasPage.point.${name}`, name)}
              <span className="mono-sm">{info.applicable}</span>
            </button>
          ))}
        </div>
        <div className="chip-row mt-8">
          {(data.points[point]?.coordinates ?? []).map(c => (
            <Coord key={c.code} code={`${c.code}=${c.value}`} url={c.url} />
          ))}
        </div>
      </section>

      {data.uncovered_coordinates && data.uncovered_coordinates.length > 0 && (
        <div className="guide-callout warn">
          <div>
            <strong>{t('atlasPage.gapTitle')}</strong>
            <div className="chip-row mt-8">
              {data.uncovered_coordinates.map(g => (
                <Coord key={g.code} code={`${g.code}=${g.value}`} url={g.url} gap />
              ))}
            </div>
            <p className="hint-line">{t('atlasPage.gapDetail')}</p>
          </div>
        </div>
      )}

      <section className="section">
        <div className="section-rule flush">
          <h2 className="section-title">
            {t('atlasPage.applicable', { shown: applicable.length, total: data.entries.length })}
          </h2>
          <span className="section-meta">
            <span className="det-state det-caught"><i className="det-bar" />{caught.length}</span>{' '}
            <span className="det-state det-visible"><i className="det-bar" />{visible.length}</span>{' '}
            <span className="det-state det-unproven"><i className="det-bar" />{unproven.length}</span>{' '}
            <span className="det-state det-none"><i className="det-bar" />{notCaught.length}</span>
          </span>
        </div>

        <EntryTable entries={worked} openId={openId} onOpen={setOpenId} />

        {/* Collapsed, and counted in the header above where the number stays in
            sight. The list of what can be worked on is read more often; the
            share that nothing catches is the honest figure and does not move. */}
        {undetermined.length > 0 && (
          <details className="mt-8">
            <summary>
              <strong>{t('atlasPage.undetermined', { count: undetermined.length })}</strong>{' '}
              <span className="text-muted">{t('atlasPage.undeterminedHint')}</span>
            </summary>
            <EntryTable entries={undetermined} openId={openId} onOpen={setOpenId} />
          </details>
        )}

        {notCaught.length > 0 && (
          <details className="mt-8">
            <summary>
              <span className="det-state det-none"><i className="det-bar" /></span>{' '}
              <strong>{t('atlasPage.notCaught', { count: notCaught.length })}</strong>{' '}
              <span className="text-muted">{t('atlasPage.notCaughtHint')}</span>
            </summary>
            <EntryTable entries={notCaught} openId={openId} onOpen={setOpenId} />
          </details>
        )}
      </section>

      {/* Below the catalogue and never inside it. A candidate and an entry look
          alike on a screen and only one of them has been proven to be caught by
          anything; mixing them would let this page assert what nobody checked,
          which is what the catalogue's read-only rule exists to prevent. */}
      <CandidatesSection />
    </div>
  )
}
