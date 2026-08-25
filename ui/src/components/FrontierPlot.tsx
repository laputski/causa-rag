import { useTranslation } from 'react-i18next'
import type { ConfigPoint } from '../api/client'

/**
 * Quality against latency: the frontier points accented, the dominated ones
 * muted.
 *
 * The screen used to show tables only, and the frontier is a claim about the
 * shape of a set: nothing here gets cheaper without costing quality. That shape
 * does not read out of two columns of numbers, and reads instantly out of where
 * the points sit.
 *
 * The axes are linear and labelled with quantities rather than ratios: latency
 * in milliseconds is comparable only within one group, because the platform
 * measures its own work rather than a remote system's, and the caption saying so
 * sits beside it.
 */
export default function FrontierPlot({ frontier, dominated, baselineRunId, metric, source }: {
  frontier: ConfigPoint[]
  dominated: ConfigPoint[]
  baselineRunId?: string
  metric: string
  source: string
}) {
  const { t } = useTranslation()
  const all = [...frontier, ...dominated]
  if (all.length === 0) return null

  const W = 760, H = 250, L = 52, R = 20, T = 26, B = 40
  const maxLat = Math.max(...all.map(p => p.latency_ms), 1)
  const maxQ = Math.max(...all.map(p => p.quality), 0.0001)
  // Both axes start at zero: without that, the difference between 0.80 and
  // 0.82 looks like the difference between failure and success.
  const x = (lat: number) => L + (lat / maxLat) * (W - L - R)
  const y = (q: number) => T + (1 - q / maxQ) * (H - T - B)

  const line = frontier
    .slice()
    .sort((a, b) => a.latency_ms - b.latency_ms)
    .map(p => `${x(p.latency_ms).toFixed(1)},${y(p.quality).toFixed(1)}`)
    .join(' ')

  const fmtLat = (ms: number) => (ms >= 1000 ? `${(ms / 1000).toFixed(1)} s` : `${Math.round(ms)} ms`)

  return (
    <div className="frontier-plot">
      <svg viewBox={`0 0 ${W} ${H}`} xmlns="http://www.w3.org/2000/svg"
           className="plot-svg" role="img"
           aria-label={t('frontierPage.plot.alt')}>
        <line x1={L} y1={T} x2={L} y2={H - B} stroke="var(--diag-line)" />
        <line x1={L} y1={H - B} x2={W - R} y2={H - B} stroke="var(--diag-line)" />

        {[0, 0.5, 1].map(f => (
          <g key={f}>
            <text x={L - 8} y={y(maxQ * f) + 4} textAnchor="end" className="plot-axis">
              {(maxQ * f).toFixed(2)}
            </text>
            {f > 0 && <line x1={L} y1={y(maxQ * f)} x2={W - R} y2={y(maxQ * f)}
                            stroke="var(--diag-line)" strokeOpacity=".3" strokeDasharray="2 4" />}
          </g>
        ))}
        {[0, 0.5, 1].map(f => (
          <text key={f} x={x(maxLat * f)} y={H - B + 16} textAnchor="middle" className="plot-axis">
            {fmtLat(maxLat * f)}
          </text>
        ))}
        <text x={W - R} y={H - 6} textAnchor="end" className="plot-axis">{t('frontierPage.plot.xLabel')}</text>
        {/* The measure's name sits above the axis and to its left rather than
            at the top tick, where it landed on top of a number. */}
        <text x={4} y={T - 14} className="plot-axis">{metric}</text>

        {frontier.length > 1 && (
          <polyline points={line} fill="none" stroke="var(--color-primary)" strokeWidth="1.6" strokeDasharray="4 3" />
        )}

        {dominated.map(p => (
          <circle key={p.run_id} cx={x(p.latency_ms)} cy={y(p.quality)} r={4}
                  fill="var(--diag-muted)" fillOpacity=".5">
            <title>{`${p.run_id.slice(0, 8)} · ${p.quality.toFixed(3)} · ${fmtLat(p.latency_ms)}`}</title>
          </circle>
        ))}
        {frontier.map(p => {
          const isBaseline = p.run_id === baselineRunId
          return (
            <circle key={p.run_id} cx={x(p.latency_ms)} cy={y(p.quality)} r={isBaseline ? 7 : 6}
                    fill={isBaseline ? 'var(--color-success)' : 'var(--color-primary)'}>
              <title>{`${p.run_id.slice(0, 8)} · ${p.quality.toFixed(3)} · ${fmtLat(p.latency_ms)}`}</title>
            </circle>
          )
        })}
      </svg>

      <div className="plot-legend">
        <span><i className="plot-dot on" />{t('frontierPage.plot.onFrontier', { count: frontier.length })}</span>
        <span><i className="plot-dot dim" />{t('frontierPage.plot.dominated', { count: dominated.length })}</span>
        {baselineRunId && <span><i className="plot-dot base" />{t('frontierPage.plot.baseline')}</span>}
        <span className="plot-source">{t('frontierPage.plot.sourceNote', { source })}</span>
      </div>
    </div>
  )
}
