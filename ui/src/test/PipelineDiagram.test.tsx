import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import PipelineDiagram from '../components/PipelineDiagram'
import type { StageTrace } from '../api/client'

const trace: StageTrace = {
  embed_ms: 5, dense_retrieve_ms: 20, sparse_retrieve_ms: 0, merge_ms: 0,
  generate_ms: 100, total_ms: 125, n_dense: 8, n_sparse: 0, n_merged: 8,
  n_deduped: 0, input_tokens: 40, output_tokens: 60, context_chars: 500,
  rerank_ms: 0, grounding_ms: 0, graph_ms: 0,
  n_reranked: 0, n_unsupported: 0, n_graph: 0,
}

describe('PipelineDiagram', () => {
  it('renders one segment per non-zero stage and skips zero stages', () => {
    render(<PipelineDiagram trace={trace} title="Latency" />)
    // Non-zero stages present.
    expect(screen.getByText('Embed')).toBeInTheDocument()
    expect(screen.getByText('Dense')).toBeInTheDocument()
    expect(screen.getByText('Generate')).toBeInTheDocument()
    // Zero stages omitted.
    expect(screen.queryByText('Sparse')).not.toBeInTheDocument()
    expect(screen.queryByText('Rerank')).not.toBeInTheDocument()
    expect(screen.queryByText('Grounding')).not.toBeInTheDocument()
  })

  it('shows the honest-degradation message when trace is null', () => {
    render(<PipelineDiagram trace={null} />)
    expect(screen.getByText(/unavailable/i)).toBeInTheDocument()
  })

  it('flags the grounding stage when unsupported claims exist', () => {
    const g: StageTrace = { ...trace, grounding_ms: 3, n_unsupported: 2 }
    const { container } = render(<PipelineDiagram trace={g} />)
    const groundSeg = screen.getByText('Grounding').closest('.pipeline-seg')
    // Only a stage worth noticing gets coloured. Everything used to be green,
    // and the colour distinguished nothing.
    expect(groundSeg?.className).toContain('pipeline-seg-hot')
    expect(container.querySelectorAll('.pipeline-seg').length).toBeGreaterThan(0)
  })
})
