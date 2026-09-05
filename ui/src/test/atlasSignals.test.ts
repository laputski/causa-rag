import { describe, it, expect } from 'vitest'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import {
  SIGNAL_IDS,
  diagnoseMetrics,
  diagnoseRetrieval,
  diagnoseLatency,
} from '../lib/diagnostics'
import type { ExperimentDetail, StageTrace } from '../api/client'

// The interface computes judgements of its own, and the catalogue of failure
// modes in core/eval/atlas.py points at them by name. A Python module cannot
// resolve a TypeScript function, so the catalogue's guard reads the SIGNAL_IDS
// declaration out of the source instead. This file is what stops that
// declaration from drifting away from the functions it claims to describe:
// neither side can move alone.

function run(partial: Partial<ExperimentDetail>): ExperimentDetail {
  return { aggregate_metrics: {}, question_results: [], ...partial } as ExperimentDetail
}

describe('the interface declares the judgements it makes', () => {
  it('every declared id is produced by one of the functions', () => {
    const produced = new Set<string>()

    // Legacy schema: no Phase 0 key present at all.
    diagnoseMetrics(run({ aggregate_metrics: { faithfulness: 0.2 } })).forEach(i => produced.add(i.id))

    // Every band, at both ends, so a branch that lost its id is caught.
    for (const m of [
      { correct_refusal: 0.9, retrieval_recall_at_k: 0.9, answer_similarity: 0.9 },
      { correct_refusal: 0.7, retrieval_recall_at_k: 0.4, answer_similarity: 0.5 },
      { correct_refusal: 0.1, retrieval_recall_at_k: 0.1, answer_similarity: 0.1 },
    ]) {
      diagnoseMetrics(run({ aggregate_metrics: m })).forEach(i => produced.add(i.id))
    }

    diagnoseMetrics(run({
      aggregate_metrics: { correct_refusal: 1 },
      // The refusal verdict arrives from the server now, so the payload
      // carries it instead of a phrase this side would have to re-judge.
      question_results: [{ generated_answer: 'no information', is_refusal: true }] as never,
    })).forEach(i => produced.add(i.id))

    diagnoseRetrieval(run({ question_results: [] })).forEach(i => produced.add(i.id))
    for (const dense of [0.1, 0.8]) {
      diagnoseRetrieval(run({
        question_results: [{ source_refs: [{ dense_score: dense }] }] as never,
      })).forEach(i => produced.add(i.id))
    }

    diagnoseLatency(null).forEach(i => produced.add(i.id))
    for (const t of [
      { total_ms: 1000, generate_ms: 100, embed_ms: 10 },
      { total_ms: 5000, generate_ms: 100, embed_ms: 10 },
      { total_ms: 20000, generate_ms: 19000, embed_ms: 900 },
    ] as StageTrace[]) {
      diagnoseLatency(t).forEach(i => produced.add(i.id))
    }

    const declared = new Set<string>(SIGNAL_IDS)
    const missing = [...declared].filter(id => !produced.has(id)).sort()
    expect(missing, 'declared and never produced: the catalogue would point at nothing').toEqual([])

    const undeclared = [...produced].filter(id => !declared.has(id)).sort()
    expect(undeclared, 'produced and never declared: the catalogue cannot point at it').toEqual([])
  })

  it('the catalogue guard reads this declaration and not a copy of it', () => {
    // The Python side names these ids too. It has to read them from here,
    // because a hand-kept second list is the place a rename gets forgotten.
    const guard = readFileSync(
      resolve(process.cwd(), '../tests/fitness/test_atlas_registry.py'), 'utf8',
    )
    // Checked by the name of the declaration and not by the file name: the
    // file name spelt out here reads as an i18n key to the localisation check,
    // `diagnostics` being one of its namespaces, and reading the declaration is
    // the property that actually matters.
    expect(
      guard.includes('SIGNAL_IDS'),
      'the catalogue guard does not read this declaration, so the two lists can drift apart',
    ).toBe(true)
  })
})
