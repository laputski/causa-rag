import { useState } from 'react'
import { useTranslation } from 'react-i18next'
import type { QuestionResult, QuestionStages, RetrievalStage, SourceRefView } from '../api/client'

// Where the source a question needs was at each stage of its retrieval.
//
// Which list a stage is, the place of the source on it and the stage that
// lost it are all decided by the server (core/eval/stages.py). This draws
// them and decides nothing: a second copy of that rule here would be the one
// that disagrees with the findings a run reports.

const STAGE_LABEL: Record<string, string> = {
  dense: 'runPage.stages.stage.dense',
  sparse: 'runPage.stages.stage.sparse',
  graph: 'runPage.stages.stage.graph',
  fusion: 'runPage.stages.stage.fusion',
  retrieval: 'runPage.stages.stage.retrieval',
  rerank: 'runPage.stages.stage.rerank',
  context: 'runPage.stages.stage.context',
}

const SAME_AS: Record<string, string> = {
  fusion: 'runPage.stages.sameAs.fusion',
  retrieval: 'runPage.stages.sameAs.retrieval',
  rerank: 'runPage.stages.sameAs.rerank',
}

// The half that found the source, as the subject of a sentence.
const FINDER: Record<string, string> = {
  dense: 'runPage.stages.finder.dense',
  sparse: 'runPage.stages.finder.sparse',
  graph: 'runPage.stages.finder.graph',
}

const HALVES = ['dense', 'sparse', 'graph']
const ROWS_SHOWN = 6

function useSentence(stages: QuestionStages): string {
  const { t } = useTranslation()
  const list = stages.stages
  if (stages.found === null) return t('runPage.stages.sentence.noSources')
  if (!stages.found) return t('runPage.stages.sentence.nowhere')
  if (stages.lost_at === 'fusion') {
    const half = list.find(s => HALVES.includes(s.stage) && s.rank !== null)
    if (half) {
      return t('runPage.stages.sentence.lostInFusion', { who: t(FINDER[half.stage]), rank: half.rank })
    }
  }
  if (stages.lost_at) {
    const index = list.findIndex(s => s.stage === stages.lost_at)
    const before = list.slice(0, index).reverse().find(s => s.rank !== null)
    const key = stages.lost_at === 'rerank'
      ? 'runPage.stages.sentence.lostInRerank'
      : 'runPage.stages.sentence.lostBeforeContext'
    return t(key, { rank: before?.rank ?? '' })
  }
  const context = list.find(s => s.stage === 'context')
  return t('runPage.stages.sentence.reached', { rank: context?.rank ?? '' })
}

function StageCell({ stage, lost, bounded }: { stage: RetrievalStage; lost: boolean; bounded: boolean }) {
  const { t } = useTranslation()
  return (
    <div className={`stage-cell${lost ? ' lost' : ''}`} data-stage={stage.stage}>
      <span className="eyebrow">{t(STAGE_LABEL[stage.stage] ?? 'runPage.stages.stage.retrieval')}</span>
      {bounded && (
        <span className={`stage-rank${stage.rank === null ? ' absent' : ''}`}>{stage.rank ?? '—'}</span>
      )}
      <span className="stage-size">
        {bounded ? t('runPage.stages.ofSize', { n: stage.size }) : t('runPage.stages.fragments', { count: stage.size })}
      </span>
      {lost && <span className="stage-lost">{t('runPage.stages.lost')}</span>}
      {stage.same_as && SAME_AS[stage.same_as] && (
        <span className="stage-note">{t(SAME_AS[stage.same_as])}</span>
      )}
    </div>
  )
}

/** Every fragment any stage held, the needed ones first, then by how far
 *  each got: its place in the last stage, then in the one before, and so on. */
function orderedFragments(stages: RetrievalStage[], needed: Set<string>): string[] {
  const place = (stage: RetrievalStage, id: string) => {
    const at = (stage.ids ?? []).indexOf(id)
    return at === -1 ? Number.MAX_SAFE_INTEGER : at
  }
  const all = [...new Set(stages.flatMap(s => s.ids ?? []))].filter(Boolean)
  const backwards = [...stages].reverse()
  return all.sort((a, b) => {
    const byNeed = Number(needed.has(b)) - Number(needed.has(a))
    if (byNeed) return byNeed
    for (const stage of backwards) {
      const diff = place(stage, a) - place(stage, b)
      if (diff) return diff
    }
    return 0
  })
}

function FragmentTable({ stages, refs }: { stages: QuestionStages; refs: Map<string, SourceRefView> }) {
  const { t } = useTranslation()
  const [all, setAll] = useState(false)
  const needed = new Set(stages.needed ?? [])
  const rows = orderedFragments(stages.stages, needed)
  const shown = all ? rows : rows.slice(0, ROWS_SHOWN)
  return (
    <>
      <div className="stage-matrix-wrap">
        <table className="stage-matrix">
          <thead>
            <tr>
              <th>{t('runPage.stages.fragment')}</th>
              {stages.stages.map(s => <th key={s.stage}>{t(STAGE_LABEL[s.stage] ?? 'runPage.stages.stage.retrieval')}</th>)}
            </tr>
          </thead>
          <tbody>
            {shown.map(id => {
              const ref = refs.get(id)
              return (
                <tr key={id} className={needed.has(id) ? 'needed' : undefined}>
                  <td title={ref?.chunk_text?.slice(0, 300)}>
                    {ref?.structural_path || id}
                    {needed.has(id) && <> <span className="stage-needed">{t('runPage.stages.needed')}</span></>}
                  </td>
                  {stages.stages.map(s => {
                    const at = (s.ids ?? []).indexOf(id)
                    return (
                      <td key={s.stage} className={`num${at === -1 ? ' absent' : ''}`}>
                        {at === -1 ? '·' : at + 1}
                      </td>
                    )
                  })}
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
      <div className="stage-foot">
        <span>{t('runPage.stages.legend')}</span>
        {rows.length > ROWS_SHOWN && (
          <button type="button" className="btn-sm" onClick={() => setAll(v => !v)}>
            {all ? t('runPage.stages.collapse') : t('runPage.stages.more', { n: rows.length - ROWS_SHOWN })}
          </button>
        )}
      </div>
    </>
  )
}

export default function RetrievalStages({ qr, clearedAt }: { qr: QuestionResult; clearedAt?: string }) {
  const { t } = useTranslation()
  const [open, setOpen] = useState(false)
  const stages = qr.stages
  const sentence = useSentence(stages ?? { stages: [], lost_at: null, found: null })
  if (!stages || stages.stages.length === 0) return null

  // One fragment may stand in several lists; any of them names it and carries
  // its text.
  const refs = new Map<string, SourceRefView>()
  for (const list of [qr.source_refs, qr.pre_rerank_source_refs, qr.candidate_source_refs,
    qr.dense_source_refs, qr.sparse_source_refs, qr.graph_source_refs]) {
    for (const ref of list ?? []) if (!refs.has(ref.chunk_id)) refs.set(ref.chunk_id, ref)
  }
  const hasLists = stages.stages.some(s => (s.ids ?? []).length > 0)
  const bounded = stages.found !== null

  return (
    <div className="qrow-full" data-testid="retrieval-stages">
      <div className="eyebrow">{t('runPage.stages.heading')}</div>
      <div className="stage-strip">
        {stages.stages.map(s => (
          <StageCell key={s.stage} stage={s} lost={stages.lost_at === s.stage} bounded={bounded} />
        ))}
      </div>
      <p className="stage-sentence">{sentence}</p>
      {clearedAt ? (
        <p className="plain-note">
          {t('runPage.stages.cleared', { date: new Date(clearedAt).toLocaleDateString('ru-RU') })}
        </p>
      ) : hasLists && (
        <>
          <button type="button" className="btn-sm" aria-expanded={open} onClick={() => setOpen(v => !v)}>
            {open ? '▼' : '▶'} {t('runPage.stages.matrix')}
          </button>
          {open && <FragmentTable stages={stages} refs={refs} />}
        </>
      )}
    </div>
  )
}
