import { useTranslation } from 'react-i18next'
import { Check, Minus, X } from 'lucide-react'
import type { StatementVerdict } from '../api/client'

// What a right answer must say, and whether this one says it.
//
// Three outcomes and not two: a statement the judge gave no verdict on is
// shown as unchecked. Counting it as absent would turn a judge that was not
// there into an answer that was wrong.

export default function AnswerStatements({ verdicts, wholeCorpus }: {
  verdicts: StatementVerdict[]
  // A whole-corpus question has no retrieval metrics, and says why.
  wholeCorpus: boolean
}) {
  const { t } = useTranslation()
  if (verdicts.length === 0) return null
  const present = verdicts.filter(v => v.stated === true).length
  const judged = verdicts.filter(v => v.stated !== null).length
  return (
    <div className="qrow-full" data-testid="answer-statements">
      <div className="stmt-head">
        <span className="eyebrow">{t('runPage.statements.heading')}</span>
        <span className="stmt-count">
          {judged === 0
            ? t('runPage.statements.notChecked')
            : t('runPage.statements.count', { present, total: verdicts.length })}
        </span>
      </div>
      <ul className="stmt-list">
        {verdicts.map((v, i) => (
          <li key={i} className="stmt-row">
            <span className="stmt-text">{v.statement}</span>
            {v.stated === true && (
              <span className="stmt-mark yes"><Check size={12} aria-hidden="true" /> {t('runPage.statements.yes')}</span>
            )}
            {v.stated === false && (
              <span className="stmt-mark no"><X size={12} aria-hidden="true" /> {t('runPage.statements.no')}</span>
            )}
            {v.stated === null && (
              <span className="stmt-mark none"><Minus size={12} aria-hidden="true" /> {t('runPage.statements.notChecked')}</span>
            )}
          </li>
        ))}
      </ul>
      {wholeCorpus && <p className="hint-line">{t('runPage.statements.noRetrievalMetrics')}</p>}
    </div>
  )
}
