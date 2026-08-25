/**
 * k6 smoke test: a quick check of the tier-1 SLA (≤50 VU, 1 min).
 * For CI and local development.
 * The full test: bash tests/load/run_k6.sh
 *
 * Run: k6 run --env TARGET=http://localhost:8081 tests/load/k6_smoke.js
 */
import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Trend } from 'k6/metrics';

const errorRate = new Rate('errors');
const queryDuration = new Trend('query_duration_ms', true);

const TARGET = __ENV.TARGET || 'http://localhost:8081';

export const options = {
  scenarios: {
    smoke: {
      executor: 'constant-vus',
      vus: 5,
      duration: '1m',
      tags: { profile: 'smoke' },
    },
  },
  thresholds: {
    'query_duration_ms{profile:smoke}': ['p(95)<30000'],
    errors: ['rate<0.05'],
    http_req_failed: ['rate<0.05'],
  },
};

// The demo handbook's subject area, so the load test runs against the corpus
// this repository actually ships. These used to be a different installation's
// own questions, which meant the load test only measured anything on a machine
// holding that corpus.
const QUERIES = [
  'who approves a purchase above ten thousand euro?',
  'how long do I have to submit an expense claim after a trip?',
  'what happens if equipment arrives damaged?',
  'what is the procedure for a lost access badge?',
  'how is a stock discrepancy recorded?',
];

export default function () {
  const query = QUERIES[Math.floor(Math.random() * QUERIES.length)];
  const payload = JSON.stringify({ text: query, top_k: 3 });
  const params = { headers: { 'Content-Type': 'application/json' } };

  const start = Date.now();
  const res = http.post(`${TARGET}/query`, payload, params);
  const elapsed = Date.now() - start;

  queryDuration.add(elapsed, { profile: 'smoke' });

  const ok = check(res, {
    'status 200': (r) => r.status === 200,
    'has text': (r) => {
      try { return JSON.parse(r.body).text && JSON.parse(r.body).text.length > 0; }
      catch { return false; }
    },
    'has trace-id header': (r) => !!r.headers['X-Trace-Id'],
    'response < 30s': (r) => elapsed < 30000,
  });

  errorRate.add(!ok);
  sleep(1);
}
